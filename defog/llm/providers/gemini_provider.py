from defog import config as defog_config
import traceback
import time
import base64
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from google import genai
from google.genai.types import (
    Part,
    Content,
    AutomaticFunctionCallingConfig,
    ToolConfig,
    FunctionCallingConfig,
    GenerateContentConfig,
)

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_gemini_parts
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""

    def __init__(
        self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None
    ):
        super().__init__(api_key or defog_config.get("GEMINI_API_KEY"), config=config)

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create Gemini provider from config."""
        return cls(api_key=config.get_api_key("gemini"), config=config)

    def get_provider_name(self) -> str:
        return "gemini"

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
    ) -> Content:
        """
        Create a message with image content in Gemini's format with validation.

        Args:
            image_base64: Base64-encoded image data - can be single string or list of strings
            description: Description of the image(s)
            image_detail: Level of detail (ignored by Gemini, included for interface consistency)

        Returns:
            Content object in Gemini's format

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

        parts = [Part.from_text(text=description)]

        # Handle validated images
        for img_data in valid_images:
            media_type, clean_data = safe_extract_media_type_and_data(img_data)
            # Convert base64 to bytes for Gemini's format
            try:
                image_bytes = base64.b64decode(clean_data, validate=True)
                parts.append(
                    Part.from_bytes(
                        data=image_bytes,
                        mime_type=media_type,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to decode image for Gemini: {e}")

        return Content(role="user", parts=parts)

    def supports_tools(self, model: str) -> bool:
        return True  # All current Gemini models support tools

    def supports_response_format(self, model: str) -> bool:
        return True  # All current Gemini models support structured output

    def convert_content_to_gemini_parts(self, content: Any, genai_types) -> List[Any]:
        """Convert message content to Gemini Part objects."""
        return convert_to_gemini_parts(content, genai_types)

    def build_params(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format: Optional[Any] = None,
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
        """Construct parameters for Gemini's generate_content call."""

        from google.genai import types

        # Extract all system messages
        system_messages = []
        non_system_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if not isinstance(content, str):
                    # System message should always be text
                    content = " ".join(
                        [
                            block.get("text", "")
                            for block in content
                            if block.get("type") == "text"
                        ]
                    )
                system_messages.append(content)
            else:
                non_system_messages.append(msg)

        # Concatenate all system messages into a single string
        system_msg = "\n\n".join(system_messages) if system_messages else None
        messages = non_system_messages

        # Convert messages to Gemini Content objects

        # For now, Gemini's conversational model expects a single user prompt
        # We'll combine all messages into a single user message with multimodal parts
        all_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Add role prefix to help maintain conversation context
            if role == "assistant":
                all_parts.append(types.Part.from_text(text="\nAssistant: "))
            elif role == "user" and len(all_parts) > 0:
                all_parts.append(types.Part.from_text(text="\nUser: "))

            # Convert content to parts
            parts = self.convert_content_to_gemini_parts(content, types)
            all_parts.extend(parts)

        # Create a single user content with all parts
        user_prompt_content = types.Content(
            role="user",
            parts=all_parts,
        )
        messages = [user_prompt_content]
        request_params = {
            "temperature": temperature,
            "system_instruction": system_msg,
            "max_output_tokens": max_completion_tokens,
        }

        if tools:
            function_specs = get_function_specs(tools, model)
            request_params["tools"] = function_specs

            # Set up automatic_function_calling and tool_config based on tool_choice
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
            if tool_choice:
                request_params["automatic_function_calling"] = (
                    AutomaticFunctionCallingConfig(disable=True)
                )
                request_params["tool_config"] = tool_choice

            # Note: Gemini handles parallel tool calling automatically
            # The model decides when to call multiple functions in parallel
            # This is controlled internally and cannot be disabled

        # When tools are provided, we'll set response_format later in the final call
        if response_format and not (tools and len(tools) > 0):
            # If we want a JSON / Pydantic format
            # "response_schema" is only recognized if the google.genai library supports it
            request_params["response_mime_type"] = "application/json"
            request_params["response_schema"] = response_format

        return request_params, messages

    async def process_response(
        self,
        client: genai.Client,
        response: Any,  # Gemini response type
        request_params: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format: Optional[Any] = None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Extract content (including any tool calls) and usage info from Gemini response.
        Handles chaining of tool calls.
        """
        # Use provided tool_handler or fall back to self.tool_handler
        if tool_handler is None:
            tool_handler = self.tool_handler

        if len(response.candidates) == 0:
            raise ProviderError(self.get_provider_name(), "No response from Gemini")
        if response.candidates[0].finish_reason == "MAX_TOKENS":
            raise MaxTokensError("Max tokens reached")

        # If we have tools, handle dynamic chaining:
        tool_outputs = []
        total_input_tokens = 0
        total_output_tokens = 0
        if tools and len(tools) > 0:
            consecutive_exceptions = 0
            while True:
                # this can sometimes be none
                total_input_tokens += response.usage_metadata.prompt_token_count or 0

                # this can sometimes be none
                total_output_tokens += (
                    response.usage_metadata.candidates_token_count or 0
                )

                # call this at the start of the while loop
                # to ensure we also log the first message (that comes in the function arg)
                await self.call_post_response_hook(
                    post_response_hook=post_response_hook,
                    response=response,
                    messages=request_params.get("messages", []),
                )

                if response.function_calls:
                    try:
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tool_call in response.function_calls:
                            func_name = tool_call.name
                            args = tool_call.args
                            # set tool_id to None, as Gemini models do not return a tool_id by default
                            tool_id = getattr(tool_call, "id", None)

                            tool_calls_batch.append(
                                {
                                    "id": tool_id,
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
                            messages,
                            post_tool_function,
                            consecutive_exceptions,
                            tool_handler,
                        )

                        # Try to get text if available
                        try:
                            # Note that this will throw a warning:
                            # Warning: there are non-text parts in the response: ['function_call'], returning concatenated text result from text parts. Check the full candidates.content.parts accessor to get the full model response.
                            # this seems intentional: https://github.com/googleapis/python-genai/issues/850
                            # happens when accessing .text for responses that also contain function calls
                            text = response.text
                        except Exception:
                            text = None

                        # Append the tool call content to messages
                        tool_call_content = response.candidates[0].content
                        messages.append(tool_call_content)

                        # Store tool outputs for tracking
                        for tool_call, result in zip(response.function_calls, results):
                            func_name = tool_call.name
                            args = tool_call.args
                            tool_id = getattr(tool_call, "id", None)

                            tool_outputs.append(
                                {
                                    "tool_id": tool_id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "text": text,
                                }
                            )

                        # Use provider-specific image processing
                        from ..utils_image_support import (
                            process_tool_results_with_images,
                        )

                        # print(results, image_result_keys)

                        tool_data_list = process_tool_results_with_images(
                            response.function_calls,
                            results,
                            tool_handler.image_result_keys,
                        )

                        # Create Gemini-specific messages
                        for tool_data in tool_data_list:
                            # For Gemini, we need to combine function response and images in one message
                            parts = [
                                Part.from_function_response(
                                    name=tool_data.tool_name,
                                    response={"result": tool_data.tool_result_text},
                                )
                            ]

                            # Add images to the same message if present
                            if tool_data.image_data:
                                # Use the create_image_message method to get properly formatted parts
                                image_message = self.create_image_message(
                                    image_base64=tool_data.image_data,
                                    description=f"Image(s) generated by {tool_data.tool_name} tool:",
                                )
                                # Extract parts from the image message and add them to our parts list
                                # Skip the first part which is the description text, as we'll add it separately
                                parts.append(
                                    Part.from_text(
                                        text=f"Image(s) generated by {tool_data.tool_name} tool:"
                                    )
                                )
                                parts.extend(
                                    image_message.parts[1:]
                                )  # Add all image parts

                            messages.append(Content(role="user", parts=parts))

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params, model
                        )

                        # Set tool_choice to AUTO so that the next message will be generated normally without required tool calls
                        request_params["automatic_function_calling"] = (
                            AutomaticFunctionCallingConfig(disable=False)
                        )
                        request_params["tool_config"] = ToolConfig(
                            function_calling_config=FunctionCallingConfig(mode="AUTO")
                        )

                        # If no more tools are available and we have response_format, set it now
                        if not tools and response_format:
                            request_params["response_mime_type"] = "application/json"
                            request_params["response_schema"] = response_format
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
                        messages.append(
                            Content(
                                role="model",
                                parts=[Part.from_text(text=str(e))],
                            )
                        )

                    # Make next call
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=messages,
                        config=GenerateContentConfig(**request_params),
                    )
                else:
                    # Break out of loop when tool calls are finished
                    content = response.text.strip() if response.text else None

                    # If we have response_format and tools were being used, make one more call
                    if response_format and request_params.get("tools"):
                        # Add the assistant's response to messages
                        messages.append(
                            Content(
                                role="model",
                                parts=[Part.from_text(text=content)] if content else [],
                            )
                        )

                        # Remove tools and set response format
                        request_params.pop("tools", None)
                        request_params.pop("tool_config", None)
                        request_params.pop("automatic_function_calling", None)
                        request_params["response_mime_type"] = "application/json"
                        request_params["response_schema"] = response_format

                        # Make final call for structured output
                        response = await client.aio.models.generate_content(
                            model=model,
                            contents=messages,
                            config=GenerateContentConfig(**request_params),
                        )

                        await self.call_post_response_hook(
                            post_response_hook=post_response_hook,
                            response=response,
                            messages=request_params.get("messages", []),
                        )

                        content = response.text.strip() if response.text else None
                    break
        else:
            # call this at the start of the while loop
            # to ensure we also log the first message (that comes in the function arg)
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=request_params.get("messages", []),
            )

            # No tools provided
            content = response.text.strip() if response.text else None

        # Parse structured output if response_format is provided
        if response_format and content:
            # Use base class method for structured response parsing
            content = self.parse_structured_response(content, response_format)

        usage = response.usage_metadata
        total_input_tokens += usage.prompt_token_count or 0
        total_output_tokens += usage.candidates_token_count or 0
        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            None,
            None,
        )

    async def execute_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_completion_tokens: Optional[int] = None,
        temperature: float = 0.0,
        response_format: Optional[Any] = None,
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
        """Execute a chat completion with Gemini."""
        # Create a ToolHandler instance with tool_budget and image_result_keys if provided
        tool_handler = self.create_tool_handler_with_budget(
            tool_budget, image_result_keys, kwargs.get("tool_output_max_tokens")
        )

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client = genai.Client(api_key=self.api_key)

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
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

            # Set up automatic_function_calling and tool_config based on tool_choice
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(tool_choice, tool_names_list, model)
            if tool_choice:
                request_params["automatic_function_calling"] = (
                    AutomaticFunctionCallingConfig(disable=True)
                )
                request_params["tool_config"] = tool_choice

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=messages,
                config=GenerateContentConfig(**request_params),
            )

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
                request_params=request_params,
                messages=messages,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                tool_handler=tool_handler,
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
