from defog import config as defog_config
import time
import json
import base64
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider implementation using OpenAI-compatible API."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or defog_config.get("DEEPSEEK_API_KEY"),
            base_url or "https://api.deepseek.com",
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create DeepSeek provider from config."""
        return cls(
            api_key=config.get_api_key("deepseek"),
            base_url=config.get_base_url("deepseek") or "https://api.deepseek.com",
            config=config,
        )

    def get_provider_name(self) -> str:
        return "deepseek"

    def _get_media_type(self, img_data: str) -> str:
        """Detect media type from base64 image data."""
        try:
            decoded = base64.b64decode(img_data[:100])
            if decoded.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            elif decoded.startswith(b"GIF8"):
                return "image/gif"
            elif decoded.startswith(b"RIFF"):
                return "image/webp"
            else:
                return "image/png"  # Default
        except Exception:
            return "image/png"

    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Dict[str, Any]:
        """
        Create a message with image content.
        Note: DeepSeek's vision models (VL2) are not yet fully integrated into their API.
        This is a placeholder implementation for future support.

        Args:
            image_base64: Base64-encoded image data - can be single string or list of strings
            description: Description of the image(s)
            image_detail: Level of detail (ignored by DeepSeek, included for interface consistency)

        Returns:
            Message dict with text description only (images not yet supported)

        Raises:
            ValueError: If image validation fails (for consistency with other providers)
        """
        from ..utils_image_support import validate_and_process_image_data

        # Validate image data even though we won't use it
        valid_images, errors = validate_and_process_image_data(image_base64)

        if errors:
            logger.warning(
                f"DeepSeek provider received invalid images: {'; '.join(errors)}"
            )

        # For now, just return a text message since DeepSeek API doesn't fully support images yet
        image_count = len(valid_images) if valid_images else 0
        if image_count > 0:
            return {
                "role": "user",
                "content": f"{description} [Note: {image_count} image(s) received but not displayed - DeepSeek API image support is limited]",
            }
        else:
            return {"role": "user", "content": description}

    def build_params(
        self,
        messages: List[Dict[str, str]],
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
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        """
        Build the parameter dictionary for DeepSeek's chat.completions.create().
        """
        # Preprocess messages using the base class method
        messages = self.preprocess_messages(messages, model)

        request_params = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_completion_tokens,
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }

        # DeepSeek-reasoner doesn't support temperature
        if model != "deepseek-reasoner":
            request_params["temperature"] = temperature

        # Tools are supported for deepseek-chat
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

            # Set parallel_tool_calls based on configuration
            request_params["parallel_tool_calls"] = (
                self.config.enable_parallel_tool_calls
            )

        # Handle structured output for DeepSeek models
        if response_format:
            # Check if response_format is a Pydantic model
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                # Convert Pydantic model to JSON schema and add instructions
                schema = response_format.model_json_schema()
                schema_str = json.dumps(schema, indent=2)

                # Set JSON mode for DeepSeek
                request_params["response_format"] = {"type": "json_object"}

                # Append structured output instructions to the latest user message
                # DeepSeek requires the word "json" in the prompt
                structured_instruction = f"""

IMPORTANT: You must respond with ONLY a valid JSON object that conforms to the following JSON schema:
{schema_str}

Please format your response as a JSON object. Make sure to:
1. Include all required properties from the schema
2. Use proper JSON formatting with double quotes
3. Ensure the JSON is valid and parseable
4. Do not include any text outside the JSON object

Respond with JSON only.
"""
                # Find the latest user message and append the structured instruction
                if messages and len(messages) > 0:
                    # Find the last user message
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i]["content"] += structured_instruction
                            break
            else:
                # For non-Pydantic response formats, use the standard approach
                request_params["response_format"] = response_format

        return request_params, messages

    async def process_response(
        self,
        client,
        response,
        request_params: Dict[str, Any],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format=None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from DeepSeek response.
        Handles chaining of tool calls.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        if len(response.choices) == 0:
            raise ProviderError(self.get_provider_name(), "No response from DeepSeek")
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        model_for_tokens = model or "gpt-4.1"

        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                # Use base class method for token calculation
                input_tokens, output_tokens, cached_tokens, _ = (
                    self.calculate_token_usage(response)
                )
                total_input_tokens += input_tokens
                total_cached_input_tokens += cached_tokens
                total_output_tokens += output_tokens
                message = response.choices[0].message

                if message.tool_calls:
                    try:
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call in message.tool_calls:
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            tool_calls_batch.append(
                                {
                                    "id": tool_call.id,
                                    "function": {"name": func_name, "arguments": args},
                                }
                            )

                        # Use base class method for tool execution with retry
                        (
                            results,
                            consecutive_exceptions,
                        ) = await self.execute_tool_calls_with_retry(
                            tool_calls_batch,
                            tool_dict,
                            request_params["messages"],
                            post_tool_function,
                            consecutive_exceptions,
                            tool_handler,
                        )

                        # Append the tool calls as an assistant response
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "tool_calls": message.tool_calls,
                            }
                        )

                        # Process results and append tool messages
                        for tool_call, result in zip(message.tool_calls, results):
                            func_name = tool_call.function.name
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {}

                            sampled_result = await tool_handler.sample_tool_result(
                                func_name,
                                result,
                                args,
                                tool_id=tool_call.id,
                                tool_sample_functions=tool_sample_functions,
                            )
                            text_for_llm, was_truncated, _ = (
                                tool_handler.prepare_result_for_llm(
                                    sampled_result,
                                    preview_max_tokens=tool_result_preview_max_tokens,
                                    model=model_for_tokens,
                                )
                            )

                            # Store the tool call, result, and text
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "result_for_llm": text_for_llm,
                                    "result_truncated_for_llm": was_truncated,
                                    "sampling_applied": tool_handler.is_sampler_configured(
                                        func_name, tool_sample_functions
                                    ),
                                    "text": (
                                        message.content if message.content else None
                                    ),
                                }
                            )

                            # Append the tool message
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": text_for_llm,
                                }
                            )

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params, model
                        )

                        # Set tool_choice to "auto" so that the next message will be generated normally
                        request_params["tool_choice"] = (
                            "auto" if request_params["tool_choice"] != "auto" else None
                        )
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

                    # Make next call
                    response = await client.chat.completions.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    content = message.content
                    break
        else:
            # No tools provided
            content = response.choices[0].message.content

            # Parse structured output if response_format is provided
            if response_format:
                try:
                    # Try to get parsed content from OpenAI SDK first
                    parsed_content = response.choices[0].message.parsed
                    if parsed_content is not None:
                        content = parsed_content
                    else:
                        # Use base class method for structured response parsing
                        content = self.parse_structured_response(
                            content, response_format
                        )
                except Exception:
                    # Use base class method for structured response parsing
                    content = self.parse_structured_response(content, response_format)

        # Final token calculation
        input_tokens, output_tokens, cached_tokens, output_tokens_details = (
            self.calculate_token_usage(response)
        )
        total_input_tokens += input_tokens
        total_cached_input_tokens += cached_tokens
        total_output_tokens += output_tokens
        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            output_tokens_details,
        )

    async def execute_chat(
        self,
        messages: List[Dict[str, str]],
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
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with DeepSeek."""
        from openai import AsyncOpenAI

        # Create a ToolHandler instance with tool_budget if provided
        sample_functions = tool_sample_functions or kwargs.get("tool_sample_functions")
        preview_max_tokens = (
            tool_result_preview_max_tokens
            if tool_result_preview_max_tokens is not None
            else kwargs.get("tool_result_preview_max_tokens")
        )
        tool_handler = self.create_tool_handler_with_budget(
            tool_budget,
            None,
            kwargs.get("tool_output_max_tokens"),
            tool_sample_functions=sample_functions,
            tool_result_preview_max_tokens=preview_max_tokens,
        )

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_deepseek = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        # Filter tools based on budget before building params
        tools = self.filter_tools_by_budget(tools, tool_handler)

        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            prediction=prediction,
            reasoning_effort=reasoning_effort,
            store=store,
            metadata=metadata,
            timeout=timeout,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            # Use regular Chat Completions API
            if request_params.get("response_format"):
                response = await client_deepseek.beta.chat.completions.parse(
                    **request_params
                )
            else:
                response = await client_deepseek.chat.completions.create(
                    **request_params
                )

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=client_deepseek,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                post_tool_function=post_tool_function,
                tool_handler=tool_handler,
                tool_sample_functions=sample_functions,
                tool_result_preview_max_tokens=preview_max_tokens,
            )
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_tokens, output_tokens, cached_input_tokens
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            output_tokens_details=completion_token_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
        )
