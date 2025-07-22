import asyncio
import json
import os
from typing import Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from anthropic.types import ToolUseBlock, TextBlock


class MCPClient:
    def __init__(self, model_name=None):
        # Initialize session and client objects
        self.connections = {}  # Dictionary to store connections to multiple servers
        self.exit_stack = AsyncExitStack()

        # Set up model parameters
        self.model_name = (
            model_name or "claude-3-7-sonnet-20250219"
        )  # Default to Claude

        if (
            self.model_name.startswith("gpt")
            or self.model_name.startswith("o1")
            or self.model_name.startswith("chatgpt")
            or self.model_name.startswith("o3")
            or self.model_name.startswith("o4")
        ):
            self.model_provider = "openai"
        elif self.model_name.startswith("claude"):
            self.model_provider = "anthropic"
        elif self.model_name.startswith("gemini"):
            self.model_provider = "gemini"
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        # Initialize appropriate client based on model provider
        self.anthropic = None
        self.openai = None
        self.gemini = None

        if self.model_provider == "anthropic":
            self.anthropic = AsyncAnthropic()
        elif self.model_provider == "openai":
            self.openai = AsyncOpenAI()
        elif self.model_provider == "gemini":
            from google import genai

            self.gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        self.max_tokens = 8191
        self.all_tools = []  # List of all tools from all servers
        self.tool_to_server = {}  # Mapping of tool names to server connections
        self.prompt_to_server = {}  # Mapping of prompt names to server connections
        self.message_history = []  # Store conversation history across queries
        self.tool_outputs = []  # List of tool outputs

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Route a tool call to the appropriate server

        This method attempts to call a tool on the appropriate server.

        Args:
            tool_name (str): Name of the tool to call
            tool_args (dict): Arguments to pass to the tool

        Returns:
            mcp.types.CallToolResult: The result object from the tool call

        Raises:
            ValueError: If no servers are connected or the tool is not found
            TimeoutError: If the tool call times out
            Exception: For connection errors or other errors during tool execution
        """

        # Handle no connections case
        if not self.connections:
            error_msg = (
                "No server connections available. Please connect to a server first."
            )
            raise ValueError(error_msg)

        if tool_name not in self.tool_to_server:
            error_msg = f"Tool '{tool_name}' not found in any connected server"
            raise ValueError(error_msg)

        # Get the server that provides this tool
        server_name = self.tool_to_server[tool_name]
        if server_name not in self.connections:
            error_msg = (
                f"Server '{server_name}' for tool '{tool_name}' is not connected"
            )
            raise ValueError(error_msg)

        session = self.connections[server_name]["session"]

        # Call the tool on the appropriate server with timeout
        try:
            return await asyncio.wait_for(
                session.call_tool(tool_name, tool_args), timeout=30.0
            )
        except asyncio.TimeoutError:
            error_msg = f"Timeout while calling tool '{tool_name}'"
            raise TimeoutError(error_msg)
        except Exception as e:
            if "connection" in str(e).lower() or "pipe" in str(e).lower():
                error_msg = f"Connection to server '{server_name}' lost: {str(e)}"
            else:
                error_msg = str(e)
            raise Exception(error_msg) from e

    async def get_prompt(self, prompt_name: str, args: dict[str, Any]):
        """
        Retrieve the prompt from the appropriate server.

        Args:
            prompt_name (str): Name of the prompt to retrieve
            args (dict[str, Any]): Arguments to pass to the prompt

        Returns:
            mcp.types.GetPromptResult: The prompt result object with messages

        Raises:
            ValueError: If no servers are connected or the prompt is not found
            ConnectionError: If the server connection fails
            TimeoutError: If the prompt retrieval times out
            Exception: For other errors during prompt retrieval
        """

        if not self.connections:
            error_msg = (
                "No server connections available. Please connect to a server first."
            )
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)

        if prompt_name not in self.prompt_to_server:
            error_msg = f"Prompt '{prompt_name}' not found in any connected server"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)

        server_name = self.prompt_to_server[prompt_name]
        if server_name not in self.connections:
            error_msg = (
                f"Server '{server_name}' for prompt '{prompt_name}' is not connected"
            )
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)

        session = self.connections[server_name]["session"]

        # Call the prompt on the appropriate server with timeout
        try:
            return await asyncio.wait_for(
                session.get_prompt(prompt_name, args), timeout=30.0
            )
        except asyncio.TimeoutError:
            error_msg = f"Timeout while getting prompt '{prompt_name}'"
            print(f"Error: {error_msg}")
            raise TimeoutError(error_msg)
        except Exception as e:
            if "connection" in str(e).lower() or "pipe" in str(e).lower():
                error_msg = f"Connection to server '{server_name}' lost: {str(e)}"
                print(f"Error: {error_msg}")
                raise ConnectionError(error_msg) from e
            else:
                error_msg = f"Error getting prompt '{prompt_name}': {str(e)}"
                print(f"Error: {error_msg}")
                raise Exception(error_msg) from e

    def _add_to_message_history(self, message, messages):
        """Add a message to both the persistent history and current message list

        Args:
            message: The message to add
            messages: The current message list for this query
        """
        self.message_history.append(message)
        messages.append(message)

    async def _handle_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
    ):
        """Handle a tool call from any model provider

        Args:
            tool_name (str): Name of the tool to call
            tool_args (dict): Arguments for the tool
            tool_id (str): ID of the tool call (for response mapping)
            messages (list): Current message list for context
            thinking_text (list): List to store thinking output

        Returns:
            tuple[mcp.types.CallToolResult, str]: Tuple containing the tool result object and result text
        """
        print(f"Calling tool {tool_name} with args {tool_args}")

        # Call the tool with retry mechanism
        max_retries = 3
        retry_count = 0
        backoff_time = 1  # seconds

        while retry_count < max_retries:
            try:
                result = await self.call_tool(tool_name, tool_args)
                if result.isError:
                    raise Exception(result.content[0].text)

                # Get result text from the first content item
                result_text = (
                    result.content[0].text if result.content else "No result content"
                )

                print(f"Tool result: {result_text}")

                return result, result_text
            except Exception as e:
                retry_count += 1
                error_msg = f"Error calling tool `{tool_name}`: {str(e)}"
                print(f"Tool error: {error_msg}")

                if retry_count >= max_retries:
                    print(f"Giving up after {max_retries} attempts")
                    # Create a mock result object
                    from mcp.types import CallToolResult, TextContent

                    error_response = f"Error: Failed to call tool `{tool_name}` after {max_retries} attempts. {str(e)} \nModify input args or call another tool. DO NOT call the same tool with the same args."
                    result = CallToolResult(
                        content=[TextContent(type="text", text=error_response)],
                        isError=True,
                    )

                    return result, error_response
                else:
                    # Wait with exponential backoff before retrying
                    print(
                        f"Retrying in {backoff_time} seconds ({retry_count}/{max_retries})"
                    )
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2

    async def _process_gemini_query(self, messages: list, available_tools: list):
        """Process a query using Google's Gemini API

        Args:
            messages (list): Message history for context (in Gemini Content format)
            available_tools (list): Tools descriptions to make available to the model

        Returns:
            str: The final response text from the model or error message
        """
        if not self.gemini:
            error_msg = "Gemini client not initialized"
            print(error_msg)
            return error_msg

        # Convert tools format for Gemini
        from google.genai import types

        gemini_tools = []
        for tool in available_tools:
            # Make a deep copy of the input schema to avoid modifying the original
            import copy

            input_schema = copy.deepcopy(tool["input_schema"])

            # Recursive function to clean schema for Gemini
            def clean_schema_for_gemini(schema):
                if isinstance(schema, dict):
                    # Remove additionalProperties as Gemini doesn't support it
                    if "additionalProperties" in schema:
                        del schema["additionalProperties"]
                    
                    # Process type field
                    if "type" in schema:
                        schema["type"] = schema["type"].upper()
                    
                    # Recursively clean nested structures
                    if "properties" in schema:
                        for prop_name, prop_value in schema["properties"].items():
                            clean_schema_for_gemini(prop_value)
                    
                    if "items" in schema:
                        clean_schema_for_gemini(schema["items"])
                    
                    # Handle anyOf/oneOf/allOf
                    for key in ["anyOf", "oneOf", "allOf"]:
                        if key in schema:
                            if isinstance(schema[key], list):
                                for item in schema[key]:
                                    clean_schema_for_gemini(item)
                
                elif isinstance(schema, list):
                    for item in schema:
                        clean_schema_for_gemini(item)
            
            # Clean the schema
            clean_schema_for_gemini(input_schema)

            func_spec = {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": input_schema,
            }
            function_declaration = types.FunctionDeclaration(**func_spec)
            gemini_tool = types.Tool(function_declarations=[function_declaration])
            gemini_tools.append(gemini_tool)

        # Initial Gemini API call
        try:
            request_params = {
                "temperature": 0,
                "max_output_tokens": self.max_tokens,
                "tools": gemini_tools,
            }

            response = await self.gemini.aio.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=types.GenerateContentConfig(**request_params),
            )

            try:
                response_text = response.text
            except Exception:
                response_text = None

            function_calls = getattr(response, "function_calls", [])
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            print(error_msg)
            return error_msg

        # Process function calls if any
        while function_calls:
            for function_call in function_calls:
                tool_name = function_call.name
                # Handle args which might be a JSON string or already a dictionary
                if isinstance(function_call.args, str):
                    tool_args = json.loads(function_call.args)
                else:
                    tool_args = function_call.args
                tool_id = function_call.name + "_" + str(len(self.tool_outputs))

                # Add tool call to message history
                tool_call_content = response.candidates[0].content
                self._add_to_message_history(tool_call_content, messages)

                # Handle the tool call
                result, result_text = await self._handle_tool_call(tool_name, tool_args)

                # Add tool result to message history
                tool_result_message = types.Content(
                    role="function",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name, response={"result": result_text}
                        )
                    ],
                )
                self._add_to_message_history(tool_result_message, messages)

                # Add tool result to tool outputs
                self.tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                        "result": result_text,
                        "text": response_text,
                    }
                )

            # Get next response from Gemini
            try:
                response = await self.gemini.aio.models.generate_content(
                    model=self.model_name,
                    contents=messages,
                    config=types.GenerateContentConfig(**request_params),
                )

                try:
                    response_text = response.text
                except Exception:
                    response_text = None

                # Extract function calls
                function_calls = getattr(response, "function_calls", [])
            except Exception as e:
                error_msg = f"Error calling Gemini API: {str(e)}"
                print(error_msg)
                return error_msg

            # If no more function calls, break
            if not function_calls:
                break

        # Final response with no tool calls
        final_text = response_text

        # Add final assistant response to message history
        final_message = types.Content(
            role="model", parts=[types.Part.from_text(final_text)]
        )
        self.message_history.append(final_message)

        return final_text

    async def _process_anthropic_query(self, messages: list, available_tools: list):
        """Process a query using Anthropic's API

        Args:
            messages (list): Message history for context
            available_tools (list): Tools descriptions to make available to the model

        Returns:
            str: The final response text from the model or error message
        """
        # Initial Anthropic API call
        try:
            response = await self.anthropic.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=available_tools,
            )
        except Exception as e:
            error_msg = f"Error calling Anthropic API: {str(e)}"
            print(error_msg)
            return error_msg

        while True:
            # Check for tool call in response
            tool_call_block = next(
                (
                    block
                    for block in response.content
                    if isinstance(block, ToolUseBlock)
                ),
                None,
            )
            text_block = next(
                (block for block in response.content if isinstance(block, TextBlock)),
                None,
            )

            if tool_call_block:
                # Extract tool call details
                tool_name = tool_call_block.name
                tool_args = tool_call_block.input
                tool_id = tool_call_block.id

                # Add tool call to message history
                assistant_message = {"role": "assistant", "content": [tool_call_block]}
                self._add_to_message_history(assistant_message, messages)

                # Handle the tool call
                result, result_text = await self._handle_tool_call(tool_name, tool_args)

                # Add tool result to message history
                tool_result_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result.content,
                        }
                    ],
                }
                self._add_to_message_history(tool_result_message, messages)

                # Add tool result to tool outputs
                self.tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                        "result": result_text,
                        "text": text_block.text if text_block else None,
                    }
                )

                # Get next response from Anthropic
                try:
                    response = await self.anthropic.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        messages=messages,
                        tools=available_tools,
                    )
                except Exception as e:
                    error_msg = f"Error calling Anthropic API: {str(e)}"
                    print(error_msg)
                    return error_msg
            else:
                # Final response with no tool calls
                final_text = text_block.text

                # Add final assistant response to message history
                final_message = {"role": "assistant", "content": final_text}
                self.message_history.append(final_message)

                return final_text

    async def _process_openai_query(self, messages: list, available_tools: list):
        """Process a query using OpenAI's API

        Args:
            messages (list): Message history for context
            available_tools (list): Tools descriptions to make available to the model
            thinking_text (list): List to store thinking text

        Returns:
            str: The final response text from the model or error message
        """
        # Convert tools format for OpenAI
        openai_tools = []
        for tool in available_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            openai_tools.append(openai_tool)

        # Initial OpenAI API call
        try:
            response = await self.openai.chat.completions.create(
                model=self.model_name,
                max_completion_tokens=self.max_tokens,
                messages=messages,
                tools=openai_tools,
            )
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            print(error_msg)
            return error_msg

        while True:
            # Check for tool call in response
            tool_call = None
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]

            if tool_call:
                # Extract tool call details
                tool_name = tool_call.function.name

                # Parse args from JSON string
                import json

                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id

                # Add tool call to message history
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
                self._add_to_message_history(assistant_message, messages)

                # Handle the tool call
                result, result_text = await self._handle_tool_call(tool_name, tool_args)

                # Add tool result to message history
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                }
                self._add_to_message_history(tool_result_message, messages)

                # Add tool result to tool outputs
                self.tool_outputs.append(
                    {
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "args": tool_args,
                        "result": result_text,
                        "text": (
                            response.choices[0].message.content
                            if response.choices[0].message.content
                            else None
                        ),
                    }
                )

                # Get next response from OpenAI
                try:
                    response = await self.openai.chat.completions.create(
                        model=self.model_name,
                        max_completion_tokens=self.max_tokens,
                        messages=messages,
                        tools=openai_tools,
                    )
                except Exception as e:
                    error_msg = f"Error calling OpenAI API: {str(e)}"
                    print(error_msg)
                    return error_msg
            else:
                # Final response with no tool calls
                final_text = response.choices[0].message.content

                # Add final assistant response to message history
                final_message = {"role": "assistant", "content": final_text}
                self.message_history.append(final_message)

                return final_text

    async def _process_prompt_templates(self, query: str) -> str:
        """Process prompt templates in the query

        Args:
            query (str): The user's input text that may contain prompt commands

        Returns:
            str: The processed query with prompt templates applied
        """
        # Get any commands from query. Commands are prefixed by /
        commands = [command[1:] for command in query.split() if command.startswith("/")]

        # Filter out non-existent prompts
        commands = [command for command in commands if command in self.prompt_to_server]

        # Only use the first command
        command = commands[0] if commands else ""
        if len(commands) > 1:
            print(
                "Multiple prompt commands found in query. Only using the first command."
            )

        # If no valid command or query is empty, return original query
        if command == "":
            return query

        try:
            if f"/{command}" == query:
                # If query only consists of the command, use previous message as arg of the template
                try:
                    # Get content from the last message in history
                    last_message = self.message_history[-1]
                    if isinstance(last_message.get("content"), str):
                        query = last_message["content"]
                    else:
                        print(
                            f"Warning: Previous message has no text content to use for command /{command}"
                        )
                        return query
                except Exception as e:
                    print(
                        f"Warning: Error accessing message history for command /{command}: {str(e)}"
                    )
                    return query

                # Get prompt template
                try:
                    response = await self.get_prompt(command, {"input": query})
                    if hasattr(response, "messages") and response.messages:
                        if hasattr(response.messages[0], "content") and hasattr(
                            response.messages[0].content, "text"
                        ):
                            query = response.messages[0].content.text
                        else:
                            print(
                                f"Warning: Prompt response for /{command} has invalid structure"
                            )
                            return query
                    else:
                        print(
                            f"Warning: Prompt response for /{command} has no messages"
                        )
                        return query
                except Exception as e:
                    print(f"Error processing prompt template /{command}: {str(e)}")
                    return query

            elif f"/{command}" in query:
                # Remove command from query
                query_text = query.replace(f"/{command}", "").strip()

                # Get prompt template
                try:
                    response = await self.get_prompt(command, {"input": query_text})
                    if hasattr(response, "messages") and response.messages:
                        if hasattr(response.messages[0], "content") and hasattr(
                            response.messages[0].content, "text"
                        ):
                            query = response.messages[0].content.text
                        else:
                            print(
                                f"Warning: Prompt response for /{command} has invalid structure"
                            )
                            return query_text
                    else:
                        print(
                            f"Warning: Prompt response for /{command} has no messages"
                        )
                        return query_text
                except Exception as e:
                    print(f"Error processing prompt template /{command}: {str(e)}")
                    return query_text
        except Exception as e:
            print(f"Unexpected error processing prompt template: {str(e)}")
            # Return original query if anything fails
            return query

        return query

    async def process_query(self, query: str) -> tuple[str, list[str]]:
        """Process a user's query using LLM and available tools

        Args:
            query (str): The user's query to process

        Returns:
            tuple[str, list[str]]: Tuple containing (final_response_text, thinking_logs)

        Raises:
            Exception: If there are errors processing the query
        """
        # Process prompt templates if any
        query = await self._process_prompt_templates(query)
        print(f"Processed query: {query}")

        # Add user query to message history (format depends on provider)
        if self.model_provider == "gemini":
            from google.genai import types

            user_message = types.Content(
                role="user", parts=[types.Part.from_text(query)]
            )
        else:
            user_message = {"role": "user", "content": query}

        self.message_history.append(user_message)

        # Use full message history for context
        messages = self.message_history.copy()

        # Collect all tools from all connected servers
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in self.all_tools
        ]

        # Process based on model provider
        if self.model_provider == "anthropic":
            final_text = await self._process_anthropic_query(messages, available_tools)
        elif self.model_provider == "openai":
            final_text = await self._process_openai_query(messages, available_tools)
        elif self.model_provider == "gemini":
            final_text = await self._process_gemini_query(messages, available_tools)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        return final_text

    async def mcp_chat(self, query: str = None):
        """Run a chat loop for MCP"""

        while True:
            try:
                response = await self.process_query(query)
                return response, self.tool_outputs

            except Exception:
                raise

    async def _connect_to_mcp_sse_server(self, server_name: str, server_url: str):
        """Connect to a single MCP SSE server and register its tools

        Args:
            server_name: Unique name to identify this server
            server_url: URL of the SSE server to connect to

        Returns:
            int: Number of tools registered from this server

        Raises:
            TimeoutError: If server connection times out
            Exception: If connection to server fails
        """
        try:
            streams = await self.exit_stack.enter_async_context(sse_client(server_url))
            session = await self.exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )

            # Initialize the connection with timeout
            try:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Timeout while initializing connection to server '{server_name}'"
                )

            # Store connection details
            self.connections[server_name] = {
                "session": session,
            }

            # List and register available tools
            try:
                response = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                tools = response.tools
                for tool in tools:
                    self.all_tools.append(tool)
                    self.tool_to_server[tool.name] = server_name
            except Exception as e:
                print(f"Failed to list tools from server '{server_name}': {str(e)}")
                raise

            # List and register available prompts
            try:
                response = await asyncio.wait_for(session.list_prompts(), timeout=5.0)
                prompts = response.prompts
            except Exception:
                # If no prompts are available
                prompts = []
            for prompt in prompts:
                self.prompt_to_server[prompt.name] = server_name

            print(
                f"Connected to server '{server_name}' with {len(tools)} tool(s) and {len(prompts)} prompt(s)"
            )

            return len(tools)

        except Exception as e:
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                raise TimeoutError(
                    f"Timeout connecting to server '{server_name}': {str(e)}"
                )
            else:
                print(f"Failed to connect to server '{server_name}':\n{str(e)}")
                raise

    async def _connect_to_mcp_http_server(self, server_name: str, server_url: str):
        """Connect to a single MCP HTTP server and register its tools

        Args:
            server_name: Unique name to identify this server
            server_url: URL of the HTTP server to connect to

        Returns:
            int: Number of tools registered from this server

        Raises:
            TimeoutError: If server connection times out
            Exception: If connection to server fails
        """
        try:
            streams = await self.exit_stack.enter_async_context(
                streamablehttp_client(server_url)
            )
            session = await self.exit_stack.enter_async_context(
                ClientSession(streams[0], streams[1])
            )

            # Initialize the connection with timeout
            try:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Timeout while initializing connection to server '{server_name}'"
                )

            # Store connection details
            self.connections[server_name] = {
                "session": session,
            }

            # List and register available tools
            try:
                response = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                tools = response.tools
                for tool in tools:
                    self.all_tools.append(tool)
                    self.tool_to_server[tool.name] = server_name
            except Exception as e:
                print(f"Failed to list tools from server '{server_name}': {str(e)}")
                raise

            # List and register available prompts
            try:
                response = await asyncio.wait_for(session.list_prompts(), timeout=5.0)
                prompts = response.prompts
            except Exception:
                # If no prompts are available
                prompts = []
            for prompt in prompts:
                self.prompt_to_server[prompt.name] = server_name

            print(
                f"Connected to HTTP server '{server_name}' at {server_url} with {len(tools)} tool(s) and {len(prompts)} prompt(s)"
            )

            return len(tools)

        except Exception as e:
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                raise TimeoutError(
                    f"Timeout connecting to HTTP server '{server_name}': {str(e)}"
                )
            else:
                print(f"Failed to connect to HTTP server '{server_name}':\n{str(e)}")
                raise

    async def _connect_to_mcp_stdio_server(
        self, server_name: str, command: str, args: list, env: Optional[dict] = None
    ):
        """Connect to a single MCP stdio server and register its tools

        Args:
            server_name: Unique name to identify this server
            command: Command to run the server
            args: Arguments for the command
            env: Optional environment variables

        Returns:
            int: Number of tools registered from this server

        Raises:
            TimeoutError: If server connection times out
            Exception: If connection to server fails
        """
        server_params = StdioServerParameters(command=command, args=args, env=env)

        try:
            # Create connection
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            # Initialize the connection with timeout
            try:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Timeout while initializing connection to server '{server_name}'"
                )

            # Store connection details
            self.connections[server_name] = {
                "stdio": stdio,
                "write": write,
                "session": session,
            }

            # List and register available tools
            try:
                response = await asyncio.wait_for(session.list_tools(), timeout=5.0)
                tools = response.tools
                for tool in tools:
                    self.all_tools.append(tool)
                    self.tool_to_server[tool.name] = server_name
            except Exception as e:
                print(f"Failed to list tools from server '{server_name}': {str(e)}")
                raise

            # List and register available prompts
            try:
                response = await asyncio.wait_for(session.list_prompts(), timeout=5.0)
                prompts = response.prompts
            except Exception:
                # If no prompts are available
                prompts = []
            for prompt in prompts:
                self.prompt_to_server[prompt.name] = server_name

            print(
                f"Connected to server '{server_name}' with {len(tools)} tool(s) and {len(prompts)} prompt(s)"
            )

            return len(tools)

        except Exception as e:
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                raise TimeoutError(
                    f"Timeout connecting to server '{server_name}': {str(e)}"
                )
            else:
                print(f"Failed to connect to server '{server_name}': {str(e)}")
                raise

    async def connect_to_server_from_config(self, config: dict):
        """Connect to all MCP servers defined in a config dictionary

        Args:
            config: Dict containing server configuration in the mcpServers format

        Raises:
            ValueError: If config is invalid or missing required fields

        Note:
            Connection errors for individual servers are collected and reported
            but do not stop the function from attempting to connect to other servers.
        """
        # Validate config structure
        if not config:
            raise ValueError("Config cannot be empty")

        if "mcpServers" not in config:
            raise ValueError("Config is missing required 'mcpServers' key")

        if not config["mcpServers"]:
            raise ValueError("No MCP servers defined in config")

        # Connect to all available servers
        print(f"Found {len(config['mcpServers'])} servers in config")
        tool_count = 0
        connection_errors = []

        for server_name, server_info in config["mcpServers"].items():
            # Validate server configuration
            if not server_info:
                connection_errors.append(
                    f"Server '{server_name}' has empty configuration"
                )
                continue

            # Check for transport parameter first (for HTTP)
            transport = server_info.get("transport", None)
            
            if transport == "http":
                # HTTP transport mode
                args = server_info.get("args", [])
                if not args or not args[0]:
                    connection_errors.append(
                        f"Server '{server_name}' with HTTP transport missing URL in args"
                    )
                    continue
            elif "command" not in server_info or not server_info["command"]:
                connection_errors.append(
                    f"Server '{server_name}' missing required 'command' field"
                )
                continue

            command = server_info.get("command", "")
            args = server_info.get("args", [])
            env = server_info.get("env", None)

            # Connect to this server
            try:
                if transport == "http":
                    print(
                        f"Connecting to HTTP server '{server_name}' with URL: {args[0]}"
                    )
                    tools_added = await self._connect_to_mcp_http_server(
                        server_name, args[0]
                    )
                elif command == "sse":
                    print(
                        f"Connecting to SSE server '{server_name}' with URL: {args[0]}"
                    )
                    tools_added = await self._connect_to_mcp_sse_server(
                        server_name, args[0]
                    )
                else:
                    print(
                        f"Connecting to stdio server '{server_name}' with command: {command} and args: {args}"
                    )
                    tools_added = await self._connect_to_mcp_stdio_server(
                        server_name, command, args, env
                    )
                tool_count += tools_added
            except Exception as e:
                error_msg = f"Failed to connect to server '{server_name}': {str(e)}"
                print(f"Error: {error_msg}")
                connection_errors.append(error_msg)

        # Report results and show detailed tool/prompt information
        if tool_count == 0 and connection_errors:
            error_details = "\n- " + "\n- ".join(connection_errors)
            print(f"Warning: Failed to connect to any servers:{error_details}")
            print(
                "The client will continue to function, but tool functionality will be limited."
            )
        elif connection_errors:
            print(
                f"Warning: {len(connection_errors)} server(s) failed to connect. {connection_errors}"
            )

    async def cleanup(self):
        """Clean up resources"""
        # The AsyncExitStack will handle closing all connections
        # that were entered with enter_async_context
        await self.exit_stack.aclose()

        # Clear our connection tracking data
        self.connections = {}
        self.all_tools = []
        self.tool_to_server = {}

        print("All MCP server connections closed.")


async def initialize_mcp_client(config, model):
    """
    Initialize MCP client with config loaded from the specified path or dictionary.

    Args:
        config (Union[str, dict]): Path to the MCP config file or config dictionary
        model (str): Model name to use for the client

    Returns:
        MCPClient: The initialized MCP client

    Raises:
        ValueError: If config format is invalid or missing required fields
        FileNotFoundError: If config file doesn't exist
        PermissionError: If config file can't be accessed due to permissions
        RuntimeError: If there's an error connecting to servers or initializing the client
    """
    config_dict = None

    # Validate inputs
    if model is None or (isinstance(model, str) and not model.strip()):
        raise ValueError("Model name must be provided")

    # Load and validate config
    try:
        # Handle config as either path string or dictionary
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Config file not found: {config}")

            print(f"Loading MCP config from {config}")
            try:
                with open(config, "r") as f:
                    config_dict = json.load(f)
            except json.JSONDecodeError as e:
                line_col = (
                    f" (line {e.lineno}, column {e.colno})"
                    if hasattr(e, "lineno")
                    else ""
                )
                raise ValueError(f"Failed to parse MCP config file{line_col}: {str(e)}")
            except PermissionError:
                raise PermissionError(
                    f"Permission denied when trying to read config file: {config}"
                )

        elif isinstance(config, dict):
            print("Using provided config dictionary")
            config_dict = config
        else:
            raise ValueError(
                f"Config must be a string path or dictionary, got {type(config).__name__}"
            )

        if not config_dict:
            raise ValueError("Config is empty")

    except (ValueError, FileNotFoundError, PermissionError) as e:
        print(f"Error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error loading MCP config: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e

    # Initialize MCP client
    try:
        mcp_client = MCPClient(model_name=model)

        await mcp_client.connect_to_server_from_config(config_dict)

        return mcp_client

    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Failed to initialize MCP client: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
