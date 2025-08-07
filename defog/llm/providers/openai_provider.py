from defog import config as defog_config
import time
import json
import base64
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_openai_format
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, config=None
    ):
        super().__init__(
            api_key or defog_config.get("OPENAI_API_KEY"),
            base_url or "https://api.openai.com/v1/",
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create OpenAI provider from config."""
        return cls(
            api_key=config.get_api_key("openai"),
            base_url=config.get_base_url("openai") or "https://api.openai.com/v1/",
            config=config,
        )

    def get_provider_name(self) -> str:
        return "openai"

    def convert_content_to_openai(self, content: Any) -> Any:
        """Convert message content to OpenAI format."""
        return convert_to_openai_format(content)

    def preprocess_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """Preprocess messages for OpenAI-specific requirements."""
        messages = deepcopy(messages)

        # Convert multimodal content
        for msg in messages:
            msg["content"] = self.convert_content_to_openai(msg["content"])

            # Keep roles as-is; we'll map system/developer into `instructions` for Responses API

        return messages

    def _messages_to_responses_input(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert chat-style messages into Responses API input + instructions.

        - Concatenates any system/developer content into `instructions`.
        - Converts content blocks to `input_*` where appropriate.
        - Leaves plain strings as-is for compatibility.
        """
        instructions_parts: List[str] = []
        input_items: List[Dict[str, Any]] = []

        def convert_parts(parts: Any) -> Any:
            if isinstance(parts, str):
                return [{"type": "input_text", "text": parts}]
            converted: List[Dict[str, Any]] = []
            for block in parts or []:
                btype = block.get("type")
                if btype == "text":
                    converted.append(
                        {"type": "input_text", "text": block.get("text", "")}
                    )
                elif btype == "image_url":
                    # Map Chat Completions-style images to Responses input images
                    img = block.get("image_url", {})
                    conv = {"type": "input_image"}
                    # Support both url and data URLs
                    if isinstance(img, dict) and img.get("url"):
                        conv["image_url"] = img["url"]
                    else:
                        # Fallback to raw structure if unexpected
                        conv["image_url"] = img
                    converted.append(conv)
                elif btype and btype.startswith("input_"):
                    # Already in Responses format (e.g., input_file)
                    converted.append(block)
                else:
                    # Fallback: treat as raw text
                    text = block.get("text") if isinstance(block, dict) else str(block)
                    converted.append({"type": "input_text", "text": text})
            return converted

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role in ("system", "developer"):
                # Extract only text portions for instructions
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if texts:
                        instructions_parts.append("\n".join(texts))
                continue

            # user/assistant messages become input items with role-appropriate content types
            if role == "assistant":
                # Map assistant text to output_text
                if isinstance(content, str):
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": content}],
                        }
                    )
                else:
                    out_parts: List[Dict[str, Any]] = []
                    for block in content or []:
                        if block.get("type") == "text":
                            out_parts.append(
                                {"type": "output_text", "text": block.get("text", "")}
                            )
                    if out_parts:
                        input_items.append({"role": "assistant", "content": out_parts})
                continue

            # Default: treat as user input
            if isinstance(content, str):
                input_items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
            else:
                input_items.append({"role": "user", "content": convert_parts(content)})

        instructions = (
            "\n\n".join([p for p in instructions_parts if p.strip()])
            if instructions_parts
            else None
        )
        return instructions if instructions else None, input_items

    def _coalesce_output_text(self, response: Any) -> str:
        """Best-effort extraction of assistant text from a Responses API response."""
        # Prefer SDK-provided aggregate text if present
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Fall back to walking output items and collecting text parts
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                # SDK objects may have a .text attribute; dicts will have ['text']
                if hasattr(block, "text") and isinstance(getattr(block, "text"), str):
                    parts.append(getattr(block, "text"))
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block.get("text"))
        return "".join(parts)

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
        Create a message with image content in OpenAI's format with validation.

        Args:
            image_base64: Base64-encoded image data - can be single string or list of strings
            description: Description of the image(s)
            image_detail: Level of detail for image analysis - "low" or "high" (default: "low")

        Returns:
            Message dict in OpenAI's format

        Raises:
            ValueError: If no valid images are provided or validation fails
        """
        from ..utils_image_support import (
            validate_and_process_image_data,
            safe_extract_media_type_and_data,
        )

        # Validate image_detail parameter
        if image_detail not in ["low", "high"]:
            raise ValueError(
                f"Invalid image_detail value: {image_detail}. Must be 'low' or 'high'"
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
                    "image_url": {
                        "url": f"data:{media_type};base64,{clean_data}",
                        "detail": image_detail,
                    },
                }
            )

        return {"role": "user", "content": content}

    def supports_tools(self, model: str) -> bool:
        return True

    def supports_response_format(self, model: str) -> bool:
        return True

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
        """
        Build the parameter dictionary for OpenAI's Responses API.
        Also handles special logic for o-series and GPT-5 reasoning models.
        """
        # Preprocess messages using the base class method
        messages = self.preprocess_messages(messages, model)

        # Convert messages to Responses input + instructions
        instructions, input_items = self._messages_to_responses_input(messages)

        request_params: Dict[str, Any] = {
            "model": model,
            "input": input_items if input_items else (""),
            "max_output_tokens": max_completion_tokens,
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }
        if instructions:
            request_params["instructions"] = instructions

        # Tools are only supported for certain models
        if tools and len(tools) > 0:
            function_specs = get_function_specs(tools, model)
            # Responses API expects function tools with a top-level name field
            flat_specs = []
            for spec in function_specs:
                if (
                    isinstance(spec, dict)
                    and spec.get("type") == "function"
                    and isinstance(spec.get("function"), dict)
                ):
                    f = spec["function"]
                    flat_specs.append(
                        {
                            "type": "function",
                            "name": f.get("name"),
                            "description": f.get("description"),
                            "parameters": f.get("parameters"),
                        }
                    )
                else:
                    flat_specs.append(spec)
            request_params["tools"] = flat_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
                # Flatten function choice for Responses API
                if (
                    isinstance(tool_choice, dict)
                    and tool_choice.get("type") == "function"
                    and isinstance(tool_choice.get("function"), dict)
                ):
                    tool_choice = {
                        "type": "function",
                        "name": tool_choice["function"].get("name"),
                    }
                request_params["tool_choice"] = tool_choice
            else:
                request_params["tool_choice"] = "auto"

            # Set parallel_tool_calls based on parameter
            if model not in ["o3-mini", "o4-mini", "o3"]:
                request_params["parallel_tool_calls"] = parallel_tool_calls

        # Temperature not supported by reasoning models; keep for others
        if (
            model.startswith("o")
            or model.startswith("gpt-5")
            or model == "deepseek-reasoner"
        ):
            pass
        else:
            request_params["temperature"] = temperature

        # Reasoning effort
        if (
            model.startswith("o") or model.startswith("gpt-5")
        ) and reasoning_effort is not None:
            # Responses API expects reasoning.effort, not reasoning_effort
            request_params["reasoning"] = {"effort": reasoning_effort}

        # Verbosity (Responses-only)
        verbosity = kwargs.get("verbosity")
        if verbosity is not None:
            request_params["verbosity"] = verbosity

        # Special case: prediction only applies to certain models; omit for Responses by default

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
        post_response_hook: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """
        Extract content (including any tool calls) and usage info from OpenAI response.
        Handles chaining of tool calls.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        # Responses API: basic validation
        if not hasattr(response, "output") and not getattr(
            response, "output_text", None
        ):
            raise ProviderError(self.get_provider_name(), "No response from OpenAI")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            iteration_count = 0
            while True:
                print("TEST")
                # Token usage for Responses API
                usage = getattr(response, "usage", None)
                if usage:
                    total_input_tokens += getattr(usage, "input_tokens", 0) or 0
                    total_output_tokens += getattr(usage, "output_tokens", 0) or 0

                # Post-response hook
                await self.call_post_response_hook(
                    post_response_hook=post_response_hook,
                    response=response,
                    messages=request_params.get("input", []),
                )

                # Detect tool/function calls within Responses output
                function_calls = []
                for item in getattr(response, "output", []) or []:
                    # SDK types may expose .type and .name/.arguments
                    itype = getattr(item, "type", None)
                    if itype and "function" in itype:
                        # function_call item
                        fname = getattr(item, "name", None) or getattr(
                            getattr(item, "function", None), "name", None
                        )
                        fargs = getattr(item, "arguments", None) or getattr(
                            getattr(item, "function", None), "arguments", None
                        )
                        function_calls.append(
                            {
                                "id": getattr(item, "id", None),
                                "function": {"name": fname, "arguments": fargs},
                            }
                        )

                if function_calls:
                    try:
                        iteration_count += 1
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call in function_calls:
                            func_name = tool_call["function"]["name"]
                            try:
                                args = (
                                    json.loads(tool_call["function"]["arguments"])
                                    if isinstance(
                                        tool_call["function"].get("arguments"), str
                                    )
                                    else tool_call["function"].get("arguments", {})
                                )
                            except json.JSONDecodeError:
                                args = {}

                            tool_calls_batch.append(
                                {
                                    "id": tool_call.get("id"),
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
                            request_params["input"],
                            post_tool_function,
                            consecutive_exceptions,
                            tool_handler,
                            parallel_tool_calls=parallel_tool_calls,
                        )

                        # Do not append an assistant tool_calls placeholder in Responses input

                        # Process results and handle images with provider-specific format
                        tool_call_blocks = function_calls

                        # Store tool outputs for tracking
                        for tool_call, result in zip(function_calls, results):
                            func_name = tool_call["function"]["name"]
                            args = tool_call["function"].get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}

                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_call.get("id"),
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "text": None,
                                }
                            )

                        # Use provider-specific image processing
                        from ..utils_image_support import (
                            process_tool_results_with_images,
                        )

                        tool_data_list = process_tool_results_with_images(
                            tool_call_blocks, results, tool_handler.image_result_keys
                        )

                        # Create OpenAI-specific messages (separate tool and image messages)
                        for tool_data in tool_data_list:
                            # Add tool result as a user message for Responses API
                            tool_text = (
                                f"Tool {tool_data.tool_name} (id={tool_data.tool_id}) output:\n"
                                f"{tool_data.tool_result_text}"
                            )
                            request_params["input"].append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": tool_text}
                                    ],
                                }
                            )

                            # Add image message immediately after if present
                            if tool_data.image_data:
                                image_message = self.create_image_message(
                                    tool_data.image_data,
                                    f"Image generated by {tool_data.tool_name} tool",
                                )
                                # Convert created image_message (Chat style) into Responses input style
                                # image_message is {role, content:[{type:'text'},{type:'image_url'}]}
                                converted_parts = []
                                for part in image_message.get("content", []):
                                    if part.get("type") == "text":
                                        converted_parts.append(
                                            {
                                                "type": "input_text",
                                                "text": part.get("text", ""),
                                            }
                                        )
                                    elif part.get("type") == "image_url":
                                        converted_parts.append(
                                            {
                                                "type": "input_image",
                                                "image_url": part.get("image_url", {}),
                                            }
                                        )
                                request_params["input"].append(
                                    {"role": "user", "content": converted_parts}
                                )

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params, model
                        )

                        # Heuristic: after several tool iterations, force finalization by disabling tools
                        if tools:
                            request_params.pop("tools", None)
                            request_params.pop("tool_choice", None)
                            request_params.pop("parallel_tool_calls", None)
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
                        request_params["input"].append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": str(e)}],
                            }
                        )

                    # Make next call
                    response = await client.responses.create(**request_params)

                else:
                    # No more tool calls, prepare final response
                    if response_format:
                        # Need to make one more call without tools but with response_format
                        # Add last assistant message content if any (from output_text)
                        last_text = self._coalesce_output_text(response)
                        if last_text:
                            request_params["input"].append(
                                {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": last_text}
                                    ],
                                }
                            )

                        # Remove tools-related parameters
                        request_params.pop("tools", None)
                        request_params.pop("tool_choice", None)
                        request_params.pop("parallel_tool_calls", None)

                        # Add response format and make final call (exclude SDK-unsupported fields if any)
                        _parse_params = {
                            **request_params,
                            "text_format": response_format,
                        }
                        _parse_params.pop("reasoning_effort", None)
                        # Some SDKs may not accept nested reasoning in parse; remove if present
                        _parse_params.pop("reasoning", None)
                        response = await client.responses.parse(**_parse_params)

                        # Extract parsed content
                        try:
                            parsed_content = getattr(response, "output_parsed", None)
                            if parsed_content is not None:
                                content = parsed_content
                            else:
                                content = self.parse_structured_response(
                                    getattr(response, "output_text", "") or "",
                                    response_format,
                                )
                        except Exception:
                            content = self.parse_structured_response(
                                getattr(response, "output_text", "") or "",
                                response_format,
                            )
                    else:
                        # No response format needed, just use the content
                        content = self._coalesce_output_text(response)
                    break
        else:
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=request_params.get("input", []),
            )

            # No tools provided
            if response_format:
                try:
                    parsed_content = getattr(response, "output_parsed", None)
                    if parsed_content is not None:
                        content = parsed_content
                    else:
                        content = self.parse_structured_response(
                            getattr(response, "output_text", "") or "",
                            response_format,
                        )
                except Exception:
                    content = self.parse_structured_response(
                        getattr(response, "output_text", "") or "",
                        response_format,
                    )
            else:
                content = self._coalesce_output_text(response)

        # Final token calculation for Responses API
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        cached_tokens = (
            getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
            if usage
            else 0
        )
        output_tokens_details = None
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
        parallel_tool_calls: bool = False,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with OpenAI."""
        from openai import AsyncOpenAI

        # Create a ToolHandler instance with tool_budget and image_result_keys if provided
        tool_handler = self.create_tool_handler_with_budget(
            tool_budget, image_result_keys, kwargs.get("tool_output_max_tokens")
        )

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client_openai = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

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
            parallel_tool_calls=parallel_tool_calls,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            # Use Responses API
            if response_format and not (tools and len(tools) > 0):
                _parse_params = {**request_params, "text_format": response_format}
                _parse_params.pop("reasoning_effort", None)
                _parse_params.pop("reasoning", None)
                response = await client_openai.responses.parse(**_parse_params)
            else:
                response = await client_openai.responses.create(**request_params)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
            ) = await self.process_response(
                client=client_openai,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                tool_handler=tool_handler,
                parallel_tool_calls=parallel_tool_calls,
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
