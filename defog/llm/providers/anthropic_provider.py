import traceback
from defog import config as defog_config
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_anthropic_format
from ..utils_image_support import process_tool_results_with_images
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, api_key: Optional[str] = None, config=None):
        super().__init__(
            api_key or defog_config.get("ANTHROPIC_API_KEY"), config=config
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create Anthropic provider from config."""
        return cls(api_key=config.get_api_key("anthropic"), config=config)

    def get_provider_name(self) -> str:
        return "anthropic"

    def supports_tools(self, model: str) -> bool:
        return True  # All current Claude models support tools

    def supports_response_format(self, model: str) -> bool:
        return True  # All current Claude models support structured output via system prompts

    def convert_content_to_anthropic(self, content: Any) -> Any:
        """Convert message content to Anthropic format."""
        return convert_to_anthropic_format(content)

    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Dict[str, Any]:
        """Create an image message in Anthropic format with validation.

        Args:
            image_base64: Base64 encoded image string or list of strings
            description: Description text for the image(s)
            image_detail: Level of detail (ignored by Anthropic, included for interface consistency)

        Returns:
            Dict containing the formatted message with image(s)

        Raises:
            ValueError: If no valid images are provided or validation fails
        """
        from ..utils_image_support import (
            validate_and_process_image_data,
            safe_extract_media_type_and_data,
        )

        # Validate and process image data
        valid_images, errors = validate_and_process_image_data(image_base64)

        if not valid_images:
            error_summary = "; ".join(errors) if errors else "No valid images provided"
            raise ValueError(f"Cannot create image message: {error_summary}")

        if errors:
            # Log warnings for any invalid images but continue with valid ones
            for error in errors:
                logger.warning(f"Skipping invalid image: {error}")

        content = []

        # Add description text first
        if description:
            content.append({"type": "text", "text": description})

        # Add each validated image
        for img_data in valid_images:
            media_type, clean_data = safe_extract_media_type_and_data(img_data)

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": clean_data,
                    },
                }
            )

        return {"role": "user", "content": content}

    def build_params(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Create the parameter dict for Anthropic's .messages.create()."""
        # Convert messages to support multimodal content
        converted_messages = []
        system_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                if isinstance(msg["content"], str):
                    system_messages.append(msg["content"])
                elif isinstance(msg["content"], list):
                    system_messages.append(
                        "\n\n".join(
                            [item["text"] for item in msg["content"] if "text" in item]
                        )
                    )
            else:
                # Convert message content to Anthropic format
                converted_msg = msg.copy()
                converted_msg["content"] = self.convert_content_to_anthropic(
                    msg["content"]
                )
                converted_messages.append(converted_msg)

        # Concatenate all system messages into a single string
        sys_msg = "\n\n".join(system_messages) if system_messages else ""

        messages = converted_messages

        if reasoning_effort is not None and ("3-7" in model or "-4-" in model):
            temperature = 1.0
            if reasoning_effort == "low":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 2048,
                }
            elif reasoning_effort == "medium":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 4096,
                }
            elif reasoning_effort == "high":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": 8192,
                }
        else:
            thinking = {
                "type": "disabled",
            }

        # Anthropic does not allow `None` as a value for max_completion_tokens
        if max_completion_tokens is None:
            max_completion_tokens = 32000

        params = {
            "system": sys_msg,
            "messages": messages,
            "model": model,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "timeout": timeout,
            "thinking": thinking,
        }

        # Handle structured output for Anthropic models
        # When tools are provided, we'll set response_format later in the final call
        if response_format and not (tools and len(tools) > 0):
            # Add instructions to the latest user message to enforce structured output
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                schema = response_format.model_json_schema()
                schema_str = json.dumps(schema, indent=2)

                # Append structured output instructions to the latest user message
                structured_instruction = f"""

IMPORTANT: You must respond with ONLY a valid, properly formatted JSON object that conforms to the following JSON schema:
{schema_str}

RESPONSE FORMAT INSTRUCTIONS:
1. Your entire response must be ONLY the JSON object, with no additional text before or after.
2. Format the JSON properly with no line breaks within property values.
3. Use double quotes for all property names and string values.
4. Do not add comments or explanations outside the JSON structure.
5. Ensure all required properties in the schema are included.
6. Make sure the JSON is properly formatted and can be parsed by standard JSON parsers.

THE RESPONSE SHOULD START WITH '{{' AND END WITH '}}' WITH NO OTHER CHARACTERS BEFORE OR AFTER.
"""
                # Find the latest user message and append the structured instruction
                if messages and len(messages) > 0:
                    # Find the last user message
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            # Handle both string content and list content
                            if isinstance(messages[i]["content"], str):
                                messages[i]["content"] += structured_instruction
                            elif isinstance(messages[i]["content"], list):
                                # For list content, append a text block with the instruction
                                messages[i]["content"].append(
                                    {"type": "text", "text": structured_instruction}
                                )
                            break

        if tools:
            function_specs = get_function_specs(tools, model)
            params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                params["tool_choice"] = tool_choice
            else:
                params["tool_choice"] = {"type": "auto"}

            # Add parallel tool calls configuration
            if "tool_choice" in params and isinstance(params["tool_choice"], dict):
                if not parallel_tool_calls:
                    params["tool_choice"]["disable_parallel_tool_use"] = True

        return params, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from Anthropic response.
        Handles chaining of tool calls and structured output parsing.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        # Note: We check block.type property instead of using isinstance with specific block classes
        # This ensures compatibility with both regular and beta API responses

        if response.stop_reason == "max_tokens":
            raise MaxTokensError("Max tokens reached")
        if len(response.content) == 0:
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Handle tool processing for both local tools and MCP server tools
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens
                # Check if the response contains a tool call
                # Collect all blocks by type - check type property instead of isinstance
                # Handle both regular tool_use and MCP mcp_tool_use blocks
                tool_call_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type")
                    and block.type in ["tool_use", "mcp_tool_use"]
                ]
                # Collect MCP tool result blocks (these contain results from MCP server execution)
                mcp_tool_result_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "mcp_tool_result"
                ]
                thinking_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "thinking"
                ]
                text_blocks = [
                    block
                    for block in response.content
                    if hasattr(block, "type") and block.type == "text"
                ]

                # call this at the start of the while loop
                # to ensure we also log the first message (that comes in the function arg)
                await self.call_post_response_hook(
                    post_response_hook=post_response_hook,
                    response=response,
                    messages=request_params.get("messages", []),
                )

                if len(tool_call_blocks) > 0:
                    try:
                        # Separate MCP tools from regular tools
                        mcp_tool_calls = []
                        regular_tool_calls = []

                        for tool_call_block in tool_call_blocks:
                            try:
                                func_name = tool_call_block.name
                                args = tool_call_block.input
                                tool_id = tool_call_block.id

                                # Check if this is an MCP tool call
                                is_mcp_tool = (
                                    hasattr(tool_call_block, "type")
                                    and tool_call_block.type == "mcp_tool_use"
                                )

                                tool_call_info = {
                                    "id": tool_id,
                                    "function": {
                                        "name": func_name,
                                        "arguments": args,
                                    },
                                }

                                if is_mcp_tool:
                                    # Add MCP-specific info if available
                                    if hasattr(tool_call_block, "server_name"):
                                        tool_call_info["server_name"] = (
                                            tool_call_block.server_name
                                        )
                                    mcp_tool_calls.append(tool_call_info)
                                else:
                                    regular_tool_calls.append(tool_call_info)

                            except Exception as e:
                                raise ProviderError(
                                    self.get_provider_name(),
                                    f"Error parsing tool call: {e}",
                                    e,
                                )

                        # Execute regular tool calls (not MCP tools, which are already executed by the API)
                        results = []
                        if regular_tool_calls:
                            (
                                results,
                                consecutive_exceptions,
                            ) = await self.execute_tool_calls_with_retry(
                                regular_tool_calls,
                                tool_dict,
                                request_params["messages"],
                                post_tool_function,
                                consecutive_exceptions,
                                tool_handler=tool_handler,
                                parallel_tool_calls=kwargs.get(
                                    "parallel_tool_calls", True
                                ),
                            )

                        # For MCP tools, extract results from mcp_tool_result blocks
                        mcp_results = []
                        for mcp_result_block in mcp_tool_result_blocks:
                            try:
                                # Extract result content
                                result_content = ""
                                if (
                                    hasattr(mcp_result_block, "content")
                                    and mcp_result_block.content
                                ):
                                    for content_item in mcp_result_block.content:
                                        if (
                                            hasattr(content_item, "type")
                                            and content_item.type == "text"
                                        ):
                                            result_content += content_item.text
                                mcp_results.append(result_content)
                            except Exception as e:
                                print(f"Warning: Failed to parse MCP tool result: {e}")
                                mcp_results.append("Error parsing MCP result")

                        # Combine results in the order they were called
                        all_results = []
                        regular_idx = 0
                        mcp_idx = 0
                        for tool_call_block in tool_call_blocks:
                            is_mcp_tool = (
                                hasattr(tool_call_block, "type")
                                and tool_call_block.type == "mcp_tool_use"
                            )
                            if is_mcp_tool:
                                if mcp_idx < len(mcp_results):
                                    all_results.append(mcp_results[mcp_idx])
                                    mcp_idx += 1
                                else:
                                    all_results.append("MCP result not found")
                            else:
                                if regular_idx < len(results):
                                    all_results.append(results[regular_idx])
                                    regular_idx += 1
                                else:
                                    all_results.append("Regular tool result not found")

                        results = all_results

                        # Reset consecutive_exceptions when tool calls are successful
                        consecutive_exceptions = 0

                        # Store tool outputs for all tools (both MCP and regular)
                        for tool_call_block, result in zip(tool_call_blocks, results):
                            func_name = tool_call_block.name
                            args = tool_call_block.input
                            tool_id = tool_call_block.id

                            # Store the tool call, result, and text
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                }
                            )

                        # Check stop_reason to determine if we should continue
                        if response.stop_reason == "end_turn":
                            # Conversation is complete, extract final content and break
                            content = "\n".join([block.text for block in text_blocks])
                            break
                        elif response.stop_reason == "tool_use":
                            # Need to continue conversation with tool results (for regular tools only)
                            # MCP tools are already executed, so this shouldn't apply to them
                            if (
                                regular_tool_calls
                            ):  # Only continue if we have regular tools to execute
                                # Build assistant content with all tool calls
                                assistant_content = []
                                if len(thinking_blocks) > 0:
                                    assistant_content += thinking_blocks

                                for tool_call_block in tool_call_blocks:
                                    assistant_content.append(tool_call_block)

                                # Append the tool calls as an assistant response
                                request_params["messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": assistant_content,
                                    }
                                )

                                # Build user response with all tool results and handle images
                                tool_results_data = process_tool_results_with_images(
                                    tool_call_blocks,
                                    results,
                                    tool_handler.image_result_keys,
                                )

                                # Build tool results content
                                tool_results_content = []
                                for tool_data in tool_results_data:
                                    tool_result = {
                                        "type": "tool_result",
                                        "tool_use_id": tool_data.tool_id,
                                        "content": tool_data.tool_result_text,
                                    }

                                    # If there are images, add them to the content
                                    if tool_data.image_data:
                                        tool_result["content"] = []
                                        # Add text content first
                                        tool_result["content"].append(
                                            {
                                                "type": "text",
                                                "text": tool_data.tool_result_text,
                                            }
                                        )
                                        # Add image content - handle both string and list with validation
                                        from ..utils_image_support import (
                                            validate_and_process_image_data,
                                            safe_extract_media_type_and_data,
                                        )

                                        valid_images, errors = (
                                            validate_and_process_image_data(
                                                tool_data.image_data
                                            )
                                        )

                                        # Log any validation errors but continue with valid images
                                        for error in errors:
                                            logger.warning(
                                                f"Invalid image in tool result: {error}"
                                            )

                                        for image_base64 in valid_images:
                                            media_type, clean_image_data = (
                                                safe_extract_media_type_and_data(
                                                    image_base64
                                                )
                                            )

                                            tool_result["content"].append(
                                                {
                                                    "type": "image",
                                                    "source": {
                                                        "type": "base64",
                                                        "media_type": media_type,
                                                        "data": clean_image_data,
                                                    },
                                                }
                                            )

                                    tool_results_content.append(tool_result)

                                # Append all tool results in a single user message
                                request_params["messages"].append(
                                    {
                                        "role": "user",
                                        "content": tool_results_content,
                                    }
                                )

                                # Set tool_choice to "auto" so that the next message will be generated normally
                                request_params["tool_choice"] = (
                                    {"type": "auto"}
                                    if request_params["tool_choice"] != "auto"
                                    else None
                                )

                                # Update available tools based on budget after successful tool execution
                                tools, tool_dict = self.update_tools_with_budget(
                                    tools,
                                    tool_handler,
                                    request_params,
                                    request_params.get("model"),
                                )
                            else:
                                # Only MCP tools, conversation is complete
                                content = "\n".join(
                                    [block.text for block in text_blocks]
                                )
                                break
                        else:
                            # For other stop reasons, extract content and break
                            content = "\n".join([block.text for block in text_blocks])
                            break
                    except ProviderError:
                        # Re-raise provider errors from base class
                        raise
                    except Exception as e:
                        # For other exceptions, use the same retry logic
                        consecutive_exceptions += 1
                        if (
                            consecutive_exceptions
                            >= tool_handler.max_consecutive_errors
                        ):
                            raise ProviderError(
                                self.get_provider_name(),
                                f"Consecutive errors during tool chaining: {e}",
                                e,
                            )
                        print(
                            f"{e}. Retries left: {tool_handler.max_consecutive_errors - consecutive_exceptions}"
                        )
                        request_params["messages"].append(
                            {"role": "assistant", "content": str(e)}
                        )

                    # Update available tools based on budget before making next call
                    tools, tool_dict = self.update_tools_with_budget(
                        tools, tool_handler, request_params, request_params.get("model")
                    )

                    response = await client.messages.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = "\n".join([block.text for block in text_blocks])

                    # If we have response_format and tools were being used, make one more call
                    if response_format and request_params.get("tools"):
                        # Add the assistant's response to messages
                        assistant_content = []
                        if len(thinking_blocks) > 0:
                            assistant_content += thinking_blocks
                        for block in text_blocks:
                            assistant_content.append(block)

                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "content": assistant_content,
                            }
                        )

                        # Remove tools and add structured output instructions
                        request_params.pop("tools", None)
                        request_params.pop("tool_choice", None)

                        # Re-add structured output instructions to the conversation
                        if isinstance(response_format, type) and hasattr(
                            response_format, "model_json_schema"
                        ):
                            schema = response_format.model_json_schema()
                            schema_str = json.dumps(schema, indent=2)

                            structured_instruction = f"""

IMPORTANT: You must respond with ONLY a valid, properly formatted JSON object that conforms to the following JSON schema:
{schema_str}

RESPONSE FORMAT INSTRUCTIONS:
1. Your entire response must be ONLY the JSON object, with no additional text before or after.
2. Format the JSON properly with no line breaks within property values.
3. Use double quotes for all property names and string values.
4. Do not add comments or explanations outside the JSON structure.
5. Ensure all required properties in the schema are included.
6. Make sure the JSON is properly formatted and can be parsed by standard JSON parsers.

THE RESPONSE SHOULD START WITH '{{' AND END WITH '}}' WITH NO OTHER CHARACTERS BEFORE OR AFTER.
"""
                            # Add a new user message with the structured output request
                            request_params["messages"].append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Based on the tool results above, please provide the final answer in the requested JSON format."
                                            + structured_instruction,
                                        }
                                    ],
                                }
                            )

                        # Make final call for structured output
                        response = await client.messages.create(**request_params)

                        await self.call_post_response_hook(
                            post_response_hook=post_response_hook,
                            response=response,
                            messages=request_params.get("messages", []),
                        )

                        # Extract final content
                        content = ""
                        for block in response.content:
                            if hasattr(block, "type") and block.type == "text":
                                content = block.text
                                break

                        # Update token counts
                        total_input_tokens += (
                            response.usage.input_tokens
                            + response.usage.cache_creation_input_tokens
                        )
                        total_output_tokens += response.usage.output_tokens

                    break
        else:
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=request_params.get("messages", []),
            )
            # No tools provided
            content = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    content = block.text
                    break

        # Parse structured output if response_format is provided
        if response_format:
            # Use base class method for structured response parsing
            content = self.parse_structured_response(content, response_format)

        usage = response.usage
        total_input_tokens += usage.input_tokens + usage.cache_creation_input_tokens
        total_output_tokens += usage.output_tokens
        cached_input_tokens = usage.cache_read_input_tokens

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            cached_input_tokens,
            None,
        )

    async def execute_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format=None,
        tools: Optional[List[Callable]] = None,
        tool_choice: Optional[str] = None,
        store: bool = True,
        metadata: Optional[Dict[str, str]] = None,
        timeout: int = 600,
        prediction: Optional[Dict[str, str]] = None,
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Anthropic."""
        from anthropic import AsyncAnthropic

        # Create a ToolHandler instance with tool_budget and image_result_keys if provided
        tool_handler = self.create_tool_handler_with_budget(
            tool_budget, image_result_keys, kwargs.get("tool_output_max_tokens")
        )

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()

        # Set up headers based on whether MCP servers are provided
        headers = {}
        headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        client = AsyncAnthropic(
            api_key=self.api_key,
            default_headers=headers,
        )

        # Filter tools based on budget before building params
        tools = self.filter_tools_by_budget(tools, tool_handler)

        params, _ = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            parallel_tool_calls=kwargs.get("parallel_tool_calls", True),
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in params:
            tool_dict = tool_handler.build_tool_dict(tools)

        func_to_call = client.messages.create

        try:
            response = await func_to_call(**params)

            (
                content,
                tool_outputs,
                input_toks,
                output_toks,
                cached_toks,
                output_details,
            ) = await self.process_response(
                client=client,
                response=response,
                request_params=params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                tool_handler=tool_handler,
                **kwargs,
            )
        except Exception as e:
            traceback.print_exc()
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_toks, output_toks, cached_toks
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_toks,
            output_tokens=output_toks,
            cached_input_tokens=cached_toks,
            output_tokens_details=output_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
        )
