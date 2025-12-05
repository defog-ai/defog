from defog import config as defog_config
import time
import json
import inspect
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError
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

    def create_image_message(
        self,
        image_base64: Union[str, List[str]],
        description: str = "Tool generated image",
        image_detail: str = "low",
    ) -> Dict[str, Any]:
        return convert_to_openai_format(image_base64)

    def preprocess_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> List[Dict[str, Any]]:
        """Preprocess messages for OpenAI-specific requirements."""
        messages = deepcopy(messages)

        # Ensure that images are in OpenAI format
        for msg in messages:
            msg["content"] = convert_to_openai_format(msg["content"])

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
        empty_assistant_placeholder = (
            "No content was generated for this turn. "
            "Refer to the preceding tool outputs for context."
        )

        def _content_is_empty(content: Any) -> bool:
            """Detect empty/whitespace-only content (including text blocks)."""
            if content is None:
                return True
            if isinstance(content, str):
                return content.strip() == ""
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        return False
                    btype = block.get("type")
                    if btype and btype != "text":
                        # Non-text blocks (images, files, etc.) count as content
                        return False
                    if btype == "text" and block.get("text", "").strip():
                        return False
                return True
            return False

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

            # Replace empty assistant turns with a small placeholder to avoid empty content
            if role == "assistant" and _content_is_empty(content):
                content = empty_assistant_placeholder

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
        previous_response_id: Optional[str] = None,
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
            "max_tool_calls": kwargs.get("max_tool_calls"),
            "store": store,
            "metadata": metadata,
            "timeout": timeout,
        }
        if instructions:
            request_params["instructions"] = instructions

        # If a previous response id is provided, add it for continuation
        if previous_response_id:
            request_params["previous_response_id"] = previous_response_id

        if tools:
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
        if model.startswith("o") or model.startswith("gpt-5"):
            if reasoning_effort is not None:
                request_params["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": "auto",
                }
            else:
                request_params["reasoning"] = {
                    "effort": "medium",
                    "summary": "auto",
                }

        # Verbosity
        verbosity = kwargs.get("verbosity")
        if verbosity is not None:
            request_params["verbosity"] = verbosity

        return request_params, messages

    async def extract_reasoning_text(
        self, response: Dict[str, Any], post_tool_function: Optional[Callable] = None
    ) -> List[str]:
        reasoning_summaries = []
        for item in response.output:
            if item.type == "reasoning":
                for reasoning_summary_block in item.summary:
                    if reasoning_summary_block.type == "summary_text":
                        reasoning_summary = reasoning_summary_block.text
                        reasoning_summaries.append(reasoning_summary)
                        if post_tool_function:
                            if inspect.iscoroutinefunction(post_tool_function):
                                await post_tool_function(
                                    function_name="reasoning",
                                    input_args={},
                                    tool_result=reasoning_summary,
                                    tool_id=None,
                                )
                            else:
                                post_tool_function(
                                    function_name="reasoning",
                                    input_args={},
                                    tool_result=reasoning_summary,
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
        return_tool_outputs_only: bool = False,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        **kwargs,
    ) -> Tuple[
        Any,
        List[Dict[str, Any]],
        int,
        int,
        Optional[int],
        Optional[Dict[str, int]],
        str,
    ]:
        """
        Extract content (including any tool calls) and usage info from OpenAI response.
        Handles chaining of tool calls.
        """

        # Helper to split usage into billable input, cached, and output tokens.
        def _extract_usage(usage_obj: Any) -> Tuple[int, int, int]:
            if not usage_obj:
                return 0, 0, 0
            raw_input = getattr(usage_obj, "input_tokens", 0) or 0
            cached = (
                getattr(
                    getattr(usage_obj, "input_tokens_details", None), "cached_tokens", 0
                )
                or 0
            )
            # input_tokens already includes cached tokens for OpenAI Responses
            billable_input = max(raw_input - cached, 0)
            output = getattr(usage_obj, "output_tokens", 0) or 0
            return billable_input, cached, output

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
        tool_calls_executed = False
        if tools:
            # this is important, as tools might go to 0 if we run out of tool budget
            consecutive_exceptions = 0
            iteration_count = 0

            while True:
                # Token usage for Responses API
                billable_input, cached_tokens, output_tokens = _extract_usage(
                    response.usage
                )
                total_input_tokens += billable_input
                total_cached_input_tokens += cached_tokens
                total_output_tokens += output_tokens

                # Post-response hook
                await self.call_post_response_hook(
                    post_response_hook=post_response_hook,
                    response=response,
                    messages=request_params.get("input", []),
                )

                # Extract reasoning text
                reasoning_blocks = await self.extract_reasoning_text(
                    response, post_tool_function
                )
                tool_outputs.extend(reasoning_blocks)

                # Detect function calls and reasoning blocks within Responses output
                function_calls = []
                for item in response.output:
                    itype = item.type

                    # FUNCTION CALLS
                    if itype and "function" in itype:
                        # function_call item
                        fname = item.name or getattr(item.function, "name", None)
                        fargs = item.arguments or getattr(
                            item.function, "arguments", None
                        )
                        call_id = item.call_id
                        function_calls.append(
                            {
                                "call_id": call_id,
                                "function": {"name": fname, "arguments": fargs},
                            }
                        )

                if function_calls:
                    tool_calls_executed = True
                    # if there is any function call left to make, we will let the model decide what do do next
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
                                    "id": tool_call["call_id"],
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

                        # Store tool outputs for tracking
                        for tool_call, result in zip(function_calls, results):
                            func_name = tool_call["function"]["name"]
                            args = tool_call["function"].get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}

                            sampled_result = await tool_handler.sample_tool_result(
                                func_name,
                                result,
                                args,
                                tool_id=tool_call["call_id"],
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
                                    "tool_call_id": tool_call["call_id"],
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "result_for_llm": text_for_llm,
                                    "result_truncated_for_llm": was_truncated,
                                    "sampling_applied": tool_handler.is_sampler_configured(
                                        func_name, tool_sample_functions
                                    ),
                                    "text": None,
                                }
                            )

                            # Add tool result as a user message for Responses API
                            request_params["input"].append(
                                {
                                    "type": "function_call_output",
                                    "call_id": tool_call["call_id"],
                                    "output": text_for_llm,
                                }
                            )

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params, model
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
                        request_params["input"].append(
                            {
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": str(e)}],
                            }
                        )

                    # Make next call
                    response = await client.responses.create(**request_params)
                    request_params["input"] = []
                    request_params["previous_response_id"] = response.id
                else:
                    break

            if tool_calls_executed:
                await self.emit_tool_phase_complete(
                    post_tool_function, message=tool_phase_complete_message
                )
            # After processing tool calls (or if none were made), extract final content
            has_tool_call_outputs = any(
                output.get("tool_call_id") for output in tool_outputs or []
            )
            skip_final_completion = return_tool_outputs_only and has_tool_call_outputs

            if skip_final_completion:
                content = ""
            elif response_format:
                response = await client.responses.create(
                    **request_params,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": response_format.schema()["title"],
                            "schema": response_format.model_json_schema()
                            | {"additionalProperties": False},
                        }
                    },
                )
                content = self.parse_structured_response(
                    getattr(response, "output_text", "") or "",
                    response_format,
                )
            else:
                content = getattr(response, "output_text", "") or ""
        else:
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=request_params.get("input", []),
            )

            # No tools provided or we have run out of tool budget
            if response_format:
                content = self.parse_structured_response(
                    response.output_text or "",
                    response_format,
                )
            else:
                content = response.output_text or ""

        # Final token calculation for Responses API
        billable_input, cached_tokens, output_tokens = _extract_usage(response.usage)
        input_tokens = billable_input
        output_tokens_details = None

        # When tools were used, usage for the final response has already been aggregated
        # inside the tool-chaining loop. Only add usage here for the no-tools path.
        if not tools and response.usage:
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
            response.id,
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
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with OpenAI."""
        from openai import AsyncOpenAI

        # Create a ToolHandler instance with tool_budget and image_result_keys if provided
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
        return_tool_outputs_only = kwargs.get("return_tool_outputs_only", False)

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
            previous_response_id=previous_response_id,
        )

        # Build a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            # Use Responses API
            if response_format and not tools:
                response = await client_openai.responses.create(
                    **request_params,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": response_format.schema()["title"],
                            "schema": response_format.model_json_schema()
                            | {"additionalProperties": False},
                        }
                    },
                )
            else:
                response = await client_openai.responses.create(**request_params)

            request_params["previous_response_id"] = response.id
            request_params["input"] = []

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                completion_token_details,
                response_id,
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
                return_tool_outputs_only=return_tool_outputs_only,
                tool_sample_functions=sample_functions,
                tool_result_preview_max_tokens=preview_max_tokens,
                tool_phase_complete_message=tool_phase_complete_message,
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
            response_id=response_id,
        )
