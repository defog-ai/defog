import inspect
import json
import traceback
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from defog import config as defog_config

from google import genai
from pydantic import BaseModel

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, ToolError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation using Interactions API."""

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

    def _get_or_assign_tool_id(self, tool_call: Any) -> str:
        """Ensure tool call has a stable id even when Gemini omits it."""
        tool_id = getattr(tool_call, "id", None)
        if tool_id:
            return tool_id

        tool_id = f"gemini-tool-{uuid.uuid4().hex}"
        try:
            setattr(tool_call, "id", tool_id)
        except Exception:
            logger.debug(
                "Unable to assign generated tool_id to Gemini function call instance"
            )
        return tool_id

    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Dict[str, Any]:
        """
        Create a message with image content in Gemini's format.
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
            for error in errors:
                logger.warning(f"Skipping invalid image: {error}")

        parts = [{"type": "text", "text": description}]

        # Handle validated images
        for img_data in valid_images:
            media_type, clean_data = safe_extract_media_type_and_data(img_data)
            parts.append(
                {"type": "image_bytes", "data": clean_data, "mime_type": media_type}
            )

        return {"role": "user", "content": parts}

    def _messages_to_interactions_input(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Union[str, List[Any]]]:
        """
        Convert standard messages to Interactions API input format.

        Extracts system messages and returns them separately so they can be
        passed via the ``system_instruction`` parameter of
        ``interactions.create()``.  The Interactions API requires alternating
        user/model turns and rejects consecutive same-role turns with a 400
        error, so system messages must not be mapped to the ``"user"`` role.

        Returns:
            A tuple of (system_instruction, input_contents) where
            system_instruction is the concatenated text of all system messages
            (or None if there are none), and input_contents is the list of
            Turn dicts for the ``input`` parameter.
        """
        system_parts: List[str] = []
        steps: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            # System messages are sent via the ``system_instruction`` parameter,
            # not as an input step.
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    texts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if texts:
                        system_parts.append("\n\n".join(texts))
                continue

            # Assistant tool calls each become their own ``function_call`` step.
            if tool_calls:
                for tc in tool_calls:
                    function = tc.get("function", {})
                    args = function.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    call_step: Dict[str, Any] = {
                        "type": "function_call",
                        "name": function.get("name"),
                        "arguments": args or {},
                    }
                    if tc.get("id"):
                        call_step["id"] = tc["id"]
                    steps.append(call_step)

            # Tool results become ``function_result`` steps.
            if role == "tool":
                tool_name = msg.get("name")
                call_id = msg.get("tool_call_id") or msg.get("call_id")
                try:
                    result_content = (
                        json.loads(content) if isinstance(content, str) else content
                    )
                except (json.JSONDecodeError, TypeError):
                    result_content = content
                result_step: Dict[str, Any] = {
                    "type": "function_result",
                    "result": {"result": result_content},
                }
                if call_id:
                    result_step["call_id"] = call_id
                if tool_name:
                    result_step["name"] = tool_name
                steps.append(result_step)
                continue

            # Text / image content becomes a ``user_input`` or ``model_output``
            # step. The Interactions API keys turns by step type, not a ``role``
            # field — a role-shaped dict is accepted but its content is ignored.
            parts = []
            if isinstance(content, str):
                if content:
                    parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        parts.append({"type": "text", "text": block})
                    elif isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            parts.append(
                                {"type": "text", "text": block.get("text", "")}
                            )
                        elif btype == "image_bytes":
                            parts.append(
                                {
                                    "type": "image",
                                    "data": block["data"],
                                    "mime_type": block.get("mime_type"),
                                }
                            )
                        elif btype == "image_url":
                            url = (block.get("image_url") or {}).get("url", "")
                            if url.startswith("data:"):
                                header, _, data = url.partition(",")
                                mime = (
                                    header[len("data:") :].split(";")[0] or "image/jpeg"
                                )
                                parts.append(
                                    {"type": "image", "data": data, "mime_type": mime}
                                )
                            elif url:
                                parts.append({"type": "image", "uri": url})
                        elif btype == "image":
                            # Anthropic-style image block.
                            source = block.get("source") or {}
                            if source.get("type") == "base64":
                                parts.append(
                                    {
                                        "type": "image",
                                        "data": source.get("data", ""),
                                        "mime_type": source.get(
                                            "media_type", "image/jpeg"
                                        ),
                                    }
                                )
                            elif source.get("type") == "url":
                                parts.append(
                                    {"type": "image", "uri": source.get("url", "")}
                                )

            if parts:
                step_type = "model_output" if role == "assistant" else "user_input"
                steps.append({"type": step_type, "content": parts})

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, steps

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
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: bool = True,
        previous_response_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        """Construct parameters for Gemini's interactions.create call."""

        # 1. Handle History / New Messages
        # When previous_response_id is supplied, the server holds prior turns
        # (including any thought blocks / signatures), so we only need to send
        # messages after the last assistant turn.
        if previous_response_id:
            last_assistant_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            if last_assistant_idx != -1:
                new_messages = messages[last_assistant_idx + 1 :]
            else:
                new_messages = messages
        else:
            new_messages = messages

        # 2. Convert messages to Interactions Input
        system_instruction, interactions_input = self._messages_to_interactions_input(
            new_messages
        )

        # 3. Build Configuration
        generation_config_dict = {"temperature": temperature}
        if max_completion_tokens is not None:
            generation_config_dict["max_output_tokens"] = max_completion_tokens

        request_params = {
            "model": model,
            "input": interactions_input,
        }

        if system_instruction:
            request_params["system_instruction"] = system_instruction

        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                schema = response_format.model_json_schema()
            else:
                schema = response_format
            # The Interactions API expects the JSON schema itself as
            # response_format (a {"type": "text", ...} wrapper is accepted
            # but the schema inside it is not enforced).
            request_params["response_format"] = schema

        if previous_response_id:
            request_params["previous_interaction_id"] = previous_response_id

        # 4. Tools
        if tools:
            function_specs = get_function_specs(tools, "gemini")
            request_params["tools"] = function_specs

            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice_config = convert_tool_choice(
                    tool_choice, tool_names_list, "gemini"
                )
                if tool_choice_config:
                    generation_config_dict["tool_choice"] = tool_choice_config

        request_params["generation_config"] = generation_config_dict

        if reasoning_effort:
            if not model.startswith("gemini-3"):
                raise ValueError(
                    f"reasoning_effort is not supported for model {model}. It is only supported for gemini-3 models."
                )

            if reasoning_effort not in ["minimal", "low", "medium", "high"]:
                raise ValueError(
                    "reasoning_effort must be one of 'minimal', 'low', 'medium', or 'high'"
                )

            if reasoning_effort not in ["low", "high"] and model.startswith(
                "gemini-3-pro"
            ):
                raise ValueError(
                    f"reasoning_effort must be 'low' or 'high' for model {model}."
                )

            # Add thinking level directly to generation_config for Interactions API
            generation_config_dict["thinking_level"] = reasoning_effort

        return request_params, new_messages

    async def extract_reasoning_text(
        self,
        response: Any,
        post_tool_function: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Extract thinking/reasoning text from Gemini thought blocks and call post_tool_function."""
        reasoning_summaries = []
        if not response.steps:
            return []
        for part in response.steps:
            if getattr(part, "type", None) == "thought" and part.summary:
                for summary_block in part.summary:
                    if (
                        getattr(summary_block, "type", None) == "text"
                        and summary_block.text
                    ):
                        reasoning_summaries.append(summary_block.text)
                        if post_tool_function:
                            if inspect.iscoroutinefunction(post_tool_function):
                                await post_tool_function(
                                    function_name="reasoning",
                                    input_args={},
                                    tool_result=summary_block.text,
                                    tool_id=None,
                                )
                            else:
                                post_tool_function(
                                    function_name="reasoning",
                                    input_args={},
                                    tool_result=summary_block.text,
                                    tool_id=None,
                                )

        return [
            {
                "tool_call_id": None,
                "name": "reasoning",
                "args": {},
                "result": summary,
                "text": None,
            }
            for summary in reasoning_summaries
        ]

    async def process_response(
        self,
        client: genai.Client,
        response: Any,  # Gemini Interaction object
        request_params: Dict[str, Any],
        messages: List[Dict[str, Any]],  # These are the messages sent in THIS turn
        tools: Optional[List[Callable]],
        tool_dict: Dict[str, Callable],
        response_format: Optional[Any] = None,
        model: str = "",
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        return_tool_outputs_only: bool = False,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        request_usages: list[dict[str, int]] | None = None,
        **kwargs,
    ) -> Tuple[
        Any, List[Dict[str, Any]], int, int, Optional[int], Optional[Dict[str, int]]
    ]:
        """Process the response from the Interactions API.

        When ``request_usages`` is given, the token counts of each model
        request are appended to it, so that each request can be priced on
        its own prompt length.
        """

        if tool_handler is None:
            tool_handler = self.tool_handler

        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        total_reasoning_tokens = 0

        def add_usage(usage: Any) -> None:
            """Add the token counts of one model request to the totals."""
            nonlocal total_input_tokens, total_output_tokens
            nonlocal total_cached_tokens, total_reasoning_tokens
            # Try new fields first, then fallback to old ones
            input_tokens = getattr(usage, "total_input_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(usage, "prompt_token_count", 0)
            input_tokens = input_tokens or 0

            output_tokens = getattr(usage, "total_output_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(usage, "candidates_token_count", 0)
            # SDK >= 2.0 renamed reasoning usage to ``total_thought_tokens``.
            reasoning_tokens = (
                getattr(usage, "total_thought_tokens", None)
                or getattr(usage, "total_reasoning_tokens", 0)
                or 0
            )
            # also add reasoning tokens to output tokens for cost calculation
            output_tokens = (output_tokens or 0) + reasoning_tokens
            cached_tokens = getattr(usage, "total_cached_tokens", 0) or 0

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_cached_tokens += cached_tokens
            total_reasoning_tokens += reasoning_tokens
            if request_usages is not None:
                request_usages.append(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_input_tokens": cached_tokens,
                        # Gemini's input count is the whole prompt, cached
                        # part included.
                        "prompt_tokens": input_tokens,
                    }
                )

        if hasattr(response, "usage"):
            add_usage(response.usage)

        tool_outputs = []
        tool_calls_executed = False

        # We need to loop if there are tool calls
        while True:
            # Check for function calls
            function_calls = []
            if response.steps:
                for part in response.steps:
                    # Check if part is a function call
                    # It could be an object or dict depending on how the client returns it
                    # Assuming object based on previous errors
                    if getattr(part, "type", None) == "function_call":
                        function_calls.append(part)
                    # Also handle if it's a dict (just in case)
                    elif isinstance(part, dict) and part.get("type") == "function_call":
                        # Wrap in SimpleNamespace or similar if needed, or just append
                        # But downstream code expects object access (fc.name, fc.id)
                        # Let's assume object for now as the client returns Pydantic models
                        function_calls.append(part)

            # Call post_response_hook
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=messages,
            )

            # Extract reasoning text from thought blocks
            reasoning_blocks = await self.extract_reasoning_text(
                response, post_tool_function
            )
            tool_outputs.extend(reasoning_blocks)

            if function_calls:
                tool_calls_executed = True

                # Prepare tool calls
                tool_calls_batch = []
                for fc in function_calls:
                    tool_id = self._get_or_assign_tool_id(fc)

                    # Sanitize tool name (strip namespace if present)
                    tool_name = fc.name
                    if ":" in tool_name:
                        tool_name = tool_name.split(":")[-1]

                    # Parse arguments if they're a JSON string (like OpenAI provider does)
                    try:
                        args = (
                            json.loads(fc.arguments)
                            if isinstance(fc.arguments, str)
                            else (fc.arguments or {})
                        )
                    except json.JSONDecodeError:
                        args = fc.arguments if fc.arguments else {}

                    tool_calls_batch.append(
                        {
                            "id": tool_id,
                            "function": {"name": tool_name, "arguments": args},
                        }
                    )

                # Execute tools
                consecutive_exceptions = 0
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

                # Process results and prepare next input
                next_input_parts = []

                for fc, result, tool_call_dict in zip(
                    function_calls, results, tool_calls_batch
                ):
                    tool_id = tool_call_dict["id"]

                    # Sample and prepare for LLM
                    parsed_args = tool_call_dict["function"]["arguments"]
                    sampled_result = await tool_handler.sample_tool_result(
                        fc.name,
                        result,
                        parsed_args,
                        tool_id=tool_id,
                        tool_sample_functions=tool_sample_functions,
                    )

                    text_for_llm, was_truncated, _ = (
                        tool_handler.prepare_result_for_llm(
                            sampled_result,
                            preview_max_tokens=tool_result_preview_max_tokens,
                            model=model,
                        )
                    )

                    tool_outputs.append(
                        {
                            "tool_call_id": tool_id,
                            "name": tool_call_dict["function"]["name"],
                            "args": parsed_args,
                            "result": result,
                            "result_for_llm": text_for_llm,
                            "result_truncated_for_llm": was_truncated,
                            "sampling_applied": tool_handler.is_sampler_configured(
                                fc.name, tool_sample_functions
                            ),
                            "thought_signature": getattr(fc, "thought_signature", None),
                        }
                    )

                    next_input_parts.append(
                        {
                            "type": "function_result",
                            "name": fc.name,
                            "call_id": fc.id,
                            "result": {"result": text_for_llm},
                        }
                    )

                # Send results back to Gemini
                try:
                    next_interaction_kwargs = {
                        "model": model,
                        "previous_interaction_id": response.id,
                        "input": next_input_parts,
                    }

                    # Pass through configuration from request_params
                    if "tools" in request_params:
                        next_interaction_kwargs["tools"] = request_params["tools"]

                    if "response_format" in request_params:
                        next_interaction_kwargs["response_format"] = request_params[
                            "response_format"
                        ]

                    if "generation_config" in request_params:
                        next_interaction_kwargs["generation_config"] = request_params[
                            "generation_config"
                        ]

                    if "system_instruction" in request_params:
                        next_interaction_kwargs["system_instruction"] = request_params[
                            "system_instruction"
                        ]

                    response = await client.aio.interactions.create(
                        **next_interaction_kwargs
                    )

                    # Update usage
                    if hasattr(response, "usage"):
                        add_usage(response.usage)

                except Exception as e:
                    raise ProviderError(
                        self.get_provider_name(), f"Failed to send tool results: {e}", e
                    )

            else:
                # No function calls, we are done
                break

        if tool_calls_executed:
            await self.emit_tool_phase_complete(
                post_tool_function, message=tool_phase_complete_message
            )

        # Extract final content. The Interactions API concatenates the model's
        # text into ``output_text``; fall back to reading text blocks off the
        # ``model_output`` steps if that convenience field is unavailable.
        content = getattr(response, "output_text", None) or ""
        if not content and response.steps:
            content_parts = []
            for step in response.steps:
                if getattr(step, "type", None) == "model_output":
                    for block in step.content or []:
                        if getattr(block, "type", None) == "text":
                            content_parts.append(block.text or "")
            content = "".join(content_parts)

        if return_tool_outputs_only and tool_outputs:
            content = ""

        # Parse structured output if needed
        if response_format and content:
            content = self.parse_structured_response(content, response_format)

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            total_cached_tokens,
            {"reasoning_tokens": total_reasoning_tokens}
            if total_reasoning_tokens > 0
            else None,
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
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Gemini Interactions API."""

        # Create ToolHandler
        sample_functions = tool_sample_functions or kwargs.get("tool_sample_functions")
        preview_max_tokens = (
            tool_result_preview_max_tokens
            if tool_result_preview_max_tokens is not None
            else kwargs.get("tool_result_preview_max_tokens")
        )
        tool_handler = self.create_tool_handler_with_budget(
            tool_budget,
            image_result_keys,
            kwargs.get("tool_output_max_tokens"),
            tool_sample_functions=sample_functions,
            tool_result_preview_max_tokens=preview_max_tokens,
        )
        return_tool_outputs_only = kwargs.pop("return_tool_outputs_only", False)

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)

        t = time.time()
        client = genai.Client(api_key=self.api_key)

        # Filter tools
        tools = self.filter_tools_by_budget(tools, tool_handler)

        # We don't need to load cached history anymore
        conversation_messages = messages

        request_params, new_messages = self.build_params(
            messages=conversation_messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id,
            reasoning_effort=reasoning_effort,
        )

        # Build tool dict
        tool_dict = {}
        if tools:
            tool_dict = tool_handler.build_tool_dict(tools)

        request_usages: list[dict[str, int]] = []
        try:
            response = await client.aio.interactions.create(**request_params)

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
                messages=new_messages,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                tool_handler=tool_handler,
                return_tool_outputs_only=return_tool_outputs_only,
                tool_sample_functions=sample_functions,
                tool_result_preview_max_tokens=preview_max_tokens,
                tool_phase_complete_message=tool_phase_complete_message,
                request_usages=request_usages,
            )
        except (ProviderError, ToolError):
            raise
        except Exception as e:
            traceback.print_exc()
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Handle ID and caching
        api_response_id = getattr(response, "id", None)
        response_id = api_response_id
        # Price each request on its own, so that a price that depends on the
        # prompt length (Gemini 2.5 Pro and 3.1 Pro) sees one prompt at a time.
        if request_usages:
            cost = CostCalculator.calculate_requests_cost(model, request_usages)
        else:
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
            response_id=response_id,
        )
