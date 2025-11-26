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
from ..utils_function_calling import get_function_specs
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""

    def __init__(self, api_key: Optional[str] = None, config=None):
        super().__init__(api_key or defog_config.get("MISTRAL_API_KEY"), config=config)

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create Mistral provider from config."""
        return cls(api_key=config.get_api_key("mistral"), config=config)

    def get_provider_name(self) -> str:
        return "mistral"

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
        Create a message with image content in Mistral's format with validation.

        Args:
            image_base64: Base64-encoded image data - can be single string or list of strings
            description: Description of the image(s)
            image_detail: Level of detail (ignored by Mistral, included for interface consistency)

        Returns:
            Message dict in Mistral's format

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

        content = [{"type": "text", "text": description}]

        # Handle validated images
        for img_data in valid_images:
            media_type, clean_data = safe_extract_media_type_and_data(img_data)
            content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:{media_type};base64,{clean_data}",
                }
            )

        return {"role": "user", "content": content}

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
        """Create the parameter dict for Mistral's chat.complete()."""

        # Build base params
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add max_tokens if specified
        if max_completion_tokens is not None:
            params["max_tokens"] = max_completion_tokens

        # Handle structured output
        if response_format:
            if isinstance(response_format, type) and hasattr(
                response_format, "model_json_schema"
            ):
                schema = response_format.model_json_schema()

                # Mistral requires additionalProperties: False in all object schemas
                def add_additional_properties_false(obj):
                    if isinstance(obj, dict):
                        if obj.get("type") == "object":
                            obj["additionalProperties"] = False
                        for key, value in obj.items():
                            add_additional_properties_false(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            add_additional_properties_false(item)

                add_additional_properties_false(schema)

                # Mistral expects response_format with type and json_schema
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("title", "Response"),
                        "strict": True,
                        "schema": schema,
                    },
                }

        # Handle tools
        if tools:
            function_specs = get_function_specs(tools, model)
            # Convert to Mistral format
            mistral_tools = []
            for spec in function_specs:
                mistral_tools.append({"type": "function", "function": spec["function"]})
            params["tools"] = mistral_tools

            # Disable parallel tool calls to ensure sequential execution
            # This is important for dependent tool calls
            params["parallel_tool_calls"] = False

            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                # Mistral uses "any", "none", or specific function name
                if tool_choice == "auto":
                    params["tool_choice"] = "auto"
                elif tool_choice == "none":
                    params["tool_choice"] = "none"
                elif tool_choice in tool_names_list:
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = "auto"
            else:
                # Set default tool choice to encourage tool usage
                params["tool_choice"] = "auto"

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
        tool_handler: Optional[ToolHandler] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from Mistral response.
        Handles chaining of tool calls and structured output parsing.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        # Check for max tokens
        if response.choices[0].finish_reason == "length":
            raise MaxTokensError("Max tokens reached")

        tool_outputs = []
        total_input_tokens = response.usage.prompt_tokens
        total_output_tokens = response.usage.completion_tokens
        model_for_tokens = request_params.get("model") or "gpt-4.1"

        # Handle tool calls if present
        if tools and response.choices[0].message.tool_calls:
            consecutive_exceptions = 0

            while True:
                message = response.choices[0].message

                if message.tool_calls:
                    try:
                        # Process tool calls
                        tool_calls = []
                        for tool_call in message.tool_calls:
                            tool_calls.append(
                                {
                                    "id": tool_call.id,
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": json.loads(
                                            tool_call.function.arguments
                                        ),
                                    },
                                }
                            )

                        # Use base class method for tool execution with retry
                        (
                            results,
                            consecutive_exceptions,
                        ) = await self.execute_tool_calls_with_retry(
                            tool_calls,
                            tool_dict,
                            request_params["messages"],
                            post_tool_function,
                            consecutive_exceptions,
                            tool_handler,
                        )

                        prepared_results = []

                        # Store tool outputs
                        for tool_call, result in zip(message.tool_calls, results):
                            parsed_args = json.loads(tool_call.function.arguments)
                            sampled_result = await tool_handler.sample_tool_result(
                                tool_call.function.name,
                                result,
                                parsed_args,
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
                            prepared_results.append((tool_call, text_for_llm))

                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "args": parsed_args,
                                    "result": result,
                                    "result_for_llm": text_for_llm,
                                    "result_truncated_for_llm": was_truncated,
                                    "sampling_applied": tool_handler.is_sampler_configured(
                                        tool_call.function.name, tool_sample_functions
                                    ),
                                }
                            )

                        # Add assistant message with tool calls
                        request_params["messages"].append(
                            {
                                "role": "assistant",
                                "content": message.content or "",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in message.tool_calls
                                ],
                            }
                        )

                        # Add tool results as tool messages
                        for tool_call, text_for_llm in prepared_results:
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "name": tool_call.function.name,
                                    "content": text_for_llm,
                                    "tool_call_id": tool_call.id,
                                }
                            )

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools,
                            tool_handler,
                            request_params,
                            request_params.get("model"),
                        )

                        # Make next call
                        response = await client.chat.complete_async(**request_params)
                        total_input_tokens += response.usage.prompt_tokens
                        total_output_tokens += response.usage.completion_tokens

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
                        response = await client.chat.complete_async(**request_params)
                else:
                    # No more tool calls, extract final content
                    content = message.content
                    break
        else:
            # No tools, just extract content
            content = response.choices[0].message.content

        # Parse structured output if response_format is provided
        if response_format and not tools:
            # Use base class method for structured response parsing
            content = self.parse_structured_response(content, response_format)

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            None,  # Mistral doesn't provide cached tokens info
            None,  # Mistral doesn't provide output token details
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
        """Execute a chat completion with Mistral."""
        from mistralai import Mistral

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

        client = Mistral(api_key=self.api_key)

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
            timeout=timeout,
        )

        # Construct tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            response = await client.chat.complete_async(**params)
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
                tool_handler=tool_handler,
                tool_sample_functions=sample_functions,
                tool_result_preview_max_tokens=preview_max_tokens,
            )
        except Exception as e:
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
