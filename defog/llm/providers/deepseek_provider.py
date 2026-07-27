from defog import config as defog_config
import time
import json
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, ToolError
from ..config import LLMConfig
from ..memory.conversation_cache import ConversationCache
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_openai_format
from ..tools.handler import ToolHandler


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# DeepSeek serves the OpenAI-compatible Chat Completions wire format. The
# "openai" provider string in utils_function_calling yields the standard
# OpenAI Chat Completions tool/tool_choice shape, which DeepSeek accepts.
_FUNCTION_SPEC_PROVIDER = "openai"


class DeepSeekProvider(BaseLLMProvider):
    """Native DeepSeek provider using the OpenAI-compatible Chat Completions API.

    DeepSeek is served natively at ``https://api.deepseek.com/v1`` and speaks
    the OpenAI Chat Completions wire format. The native integration path is the
    official ``openai`` SDK (``AsyncOpenAI``) pointed at that base URL — there
    is no separate DeepSeek SDK. This provider is self-contained and extends
    :class:`BaseLLMProvider` directly.

    Structured output:
        DeepSeek rejects ``response_format={"type": "json_schema", ...}`` with
        ``400 invalid_request_error`` ("This response_format type is
        unavailable now"). It only supports ``{"type": "json_object"}`` and
        requires the literal word "json" in at least one message plus a
        schema/example to shape the output. ``_build_response_format`` emits
        json_object mode and ``build_params`` appends the Pydantic schema as a
        system instruction so the parse/repair path downstream still
        materializes the target model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config=None,
    ):
        super().__init__(
            api_key or defog_config.get("DEEPSEEK_API_KEY"),
            base_url or DEEPSEEK_BASE_URL,
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create DeepSeek provider from config."""
        return cls(
            api_key=config.get_api_key("deepseek"),
            base_url=config.get_base_url("deepseek") or DEEPSEEK_BASE_URL,
            config=config,
        )

    def get_provider_name(self) -> str:
        return "deepseek"

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
        """Preprocess messages for DeepSeek (OpenAI-compatible format)."""
        messages = deepcopy(messages)
        for msg in messages:
            content = msg.get("content")
            if content is not None:
                msg["content"] = convert_to_openai_format(content)
        return messages

    # ------------------------------------------------------------------
    # Structured output (DeepSeek-specific)
    # ------------------------------------------------------------------

    def _build_response_format(self, response_format) -> Optional[Dict[str, Any]]:
        """DeepSeek only supports json_object mode; it 400s on json_schema."""
        if response_format is None:
            return None
        if hasattr(response_format, "model_json_schema"):
            return {"type": "json_object"}
        return None

    @staticmethod
    def _inject_system_instruction(
        messages: List[Dict[str, Any]], instruction: str
    ) -> List[Dict[str, Any]]:
        """Append *instruction* to the first system message, or prepend a new one.

        DeepSeek's json_object mode requires the literal word "json" in a
        message; the instruction text already contains it.
        """
        new_messages = [dict(m) for m in messages]
        for m in new_messages:
            if m.get("role") == "system":
                existing = m.get("content") or ""
                if isinstance(existing, list):
                    m["content"] = existing + [
                        {"type": "text", "text": "\n\n" + instruction}
                    ]
                else:
                    m["content"] = f"{existing}\n\n{instruction}"
                return new_messages
        return [{"role": "system", "content": instruction}, *new_messages]

    # ------------------------------------------------------------------
    # JSON repair helpers (json_object output is loose)
    # ------------------------------------------------------------------

    async def _parse_with_repair(
        self,
        raw_content: str,
        response_format,
        client,
        model: str,
        request_params: Optional[Dict[str, Any]] = None,
        repair_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return await super()._parse_with_repair(
            raw_content,
            response_format,
            client,
            model,
            request_params,
            repair_metadata=repair_metadata,
        )

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
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: bool = True,
        previous_response_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Build parameters for the OpenAI-compatible Chat Completions API."""
        messages = self.preprocess_messages(messages, model)

        request_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Default max_tokens so we do not exhaust the model's full output.
        request_params["max_tokens"] = max_completion_tokens or 4096

        # Tools
        if tools:
            function_specs = get_function_specs(tools, _FUNCTION_SPEC_PROVIDER)
            request_params["tools"] = function_specs

            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                request_params["tool_choice"] = convert_tool_choice(
                    tool_choice, tool_names_list, _FUNCTION_SPEC_PROVIDER
                )
            else:
                request_params["tool_choice"] = "auto"

            request_params["parallel_tool_calls"] = parallel_tool_calls

        # Structured output (only when not using tools — after tool chaining we
        # make a separate structured-output call). DeepSeek requires
        # json_object mode plus an in-prompt schema/instruction.
        if response_format and not tools:
            rf = self._build_response_format(response_format)
            if rf:
                request_params["response_format"] = rf

            if hasattr(response_format, "model_json_schema"):
                schema = response_format.model_json_schema()
                instruction = (
                    "Respond with a single valid JSON object that conforms to "
                    "the schema below. Output only the JSON — no prose, no "
                    "markdown fences, no commentary.\n\n"
                    f"JSON schema:\n{json.dumps(schema, indent=2)}"
                )
                injected = self._inject_system_instruction(messages, instruction)
                request_params["messages"] = injected
                return request_params, injected

        return request_params, messages

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

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
        parallel_tool_calls: bool = True,
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
        Optional[Dict[str, Any]],
    ]:
        """Process Chat Completions response, handling tool call chaining."""
        if tool_handler is None:
            tool_handler = self.tool_handler

        if not hasattr(response, "choices") or not response.choices:
            raise ProviderError(self.get_provider_name(), "No response from DeepSeek")

        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        repair_metadata: Dict[str, Any] = {}
        tool_calls_executed = False

        if tools:
            consecutive_exceptions = 0

            while True:
                # Token usage
                input_tokens, output_tokens, cached_tokens, _ = (
                    self.calculate_token_usage(response)
                )
                total_input_tokens += input_tokens
                total_cached_input_tokens += cached_tokens
                total_output_tokens += output_tokens

                # Post-response hook
                await self.call_post_response_hook(
                    post_response_hook=post_response_hook,
                    response=response,
                    messages=request_params.get("messages", []),
                )

                message = response.choices[0].message

                if message.tool_calls:
                    tool_calls_executed = True

                    try:
                        # Prepare tool calls for batch execution
                        tool_calls_batch = []
                        for tc in message.tool_calls:
                            try:
                                args = (
                                    json.loads(tc.function.arguments)
                                    if isinstance(tc.function.arguments, str)
                                    else tc.function.arguments or {}
                                )
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls_batch.append(
                                {
                                    "id": tc.id,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": args,
                                    },
                                }
                            )

                        # Execute tools
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
                            parallel_tool_calls=parallel_tool_calls,
                        )

                        # Append assistant message with tool_calls to conversation
                        assistant_msg: Dict[str, Any] = {"role": "assistant"}
                        if message.content:
                            assistant_msg["content"] = message.content
                        else:
                            assistant_msg["content"] = None
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in message.tool_calls
                        ]
                        request_params["messages"].append(assistant_msg)

                        # Store tool outputs and append results to messages
                        for tc, result in zip(message.tool_calls, results):
                            try:
                                args = (
                                    json.loads(tc.function.arguments)
                                    if isinstance(tc.function.arguments, str)
                                    else tc.function.arguments or {}
                                )
                            except json.JSONDecodeError:
                                args = {}

                            sampled_result = await tool_handler.sample_tool_result(
                                tc.function.name,
                                result,
                                args,
                                tool_id=tc.id,
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
                                    "tool_call_id": tc.id,
                                    "name": tc.function.name,
                                    "args": args,
                                    "result": result,
                                    "result_for_llm": text_for_llm,
                                    "result_truncated_for_llm": was_truncated,
                                    "sampling_applied": tool_handler.is_sampler_configured(
                                        tc.function.name, tool_sample_functions
                                    ),
                                    "text": None,
                                }
                            )

                            # Add tool result message
                            request_params["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": text_for_llm,
                                }
                            )

                        # Update available tools based on budget
                        tools, tool_dict = self.update_tools_with_budget(
                            tools, tool_handler, request_params
                        )

                    except (ProviderError, ToolError):
                        raise
                    except Exception as e:
                        consecutive_exceptions += 1
                        if (
                            consecutive_exceptions
                            >= tool_handler.max_consecutive_errors
                        ):
                            raise ToolError(
                                "batch",
                                f"Consecutive errors during tool chaining: {e}",
                                e,
                            )
                        print(
                            f"{e}. Retries left: {tool_handler.max_consecutive_errors - consecutive_exceptions}"
                        )
                        request_params["messages"].append(
                            {"role": "assistant", "content": str(e)}
                        )

                    # Make next API call
                    response = await client.chat.completions.create(**request_params)
                    if not hasattr(response, "choices") or not response.choices:
                        raise ProviderError(
                            self.get_provider_name(),
                            "No response from DeepSeek",
                        )

                else:
                    # No more tool calls
                    break

            if tool_calls_executed:
                await self.emit_tool_phase_complete(
                    post_tool_function, message=tool_phase_complete_message
                )

            # Extract final content
            has_tool_call_outputs = any(o.get("tool_call_id") for o in tool_outputs)
            skip_final = return_tool_outputs_only and has_tool_call_outputs

            if skip_final:
                content = ""
            elif response_format:
                # Make a final structured-output call after tool chaining
                final_params = {
                    k: v
                    for k, v in request_params.items()
                    if k not in ("tools", "tool_choice", "parallel_tool_calls")
                }
                rf = self._build_response_format(response_format)
                if rf:
                    final_params["response_format"] = rf
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()
                    instruction = (
                        "Respond with a single valid JSON object that conforms "
                        "to the schema below. Output only the JSON — no prose, "
                        "no markdown fences, no commentary.\n\n"
                        f"JSON schema:\n{json.dumps(schema, indent=2)}"
                    )
                    final_params["messages"] = self._inject_system_instruction(
                        final_params["messages"], instruction
                    )
                response = await client.chat.completions.create(**final_params)
                if not hasattr(response, "choices") or not response.choices:
                    raise ProviderError(
                        self.get_provider_name(),
                        "No response from DeepSeek",
                    )
                raw_content = response.choices[0].message.content or ""
                content = await self._parse_with_repair(
                    raw_content,
                    response_format,
                    client,
                    model,
                    final_params,
                    repair_metadata=repair_metadata,
                )
            else:
                content = response.choices[0].message.content or ""
        else:
            # No tools path
            await self.call_post_response_hook(
                post_response_hook=post_response_hook,
                response=response,
                messages=request_params.get("messages", []),
            )

            if response_format:
                # Already called with response_format in build_params
                raw_content = response.choices[0].message.content or ""
                content = await self._parse_with_repair(
                    raw_content,
                    response_format,
                    client,
                    model,
                    request_params,
                    repair_metadata=repair_metadata,
                )
            else:
                content = response.choices[0].message.content or ""

        # Final token usage (for no-tools path)
        input_tokens, output_tokens, cached_tokens, output_tokens_details = (
            self.calculate_token_usage(response)
        )

        if not tools and response.usage:
            total_input_tokens += input_tokens
            total_cached_input_tokens += cached_tokens
            total_output_tokens += output_tokens

        total_input_tokens += repair_metadata.get("input_tokens", 0)
        total_cached_input_tokens += repair_metadata.get("cached_input_tokens", 0)
        total_output_tokens += repair_metadata.get("output_tokens", 0)

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            output_tokens_details,
            response.id or "",
            repair_metadata or None,
        )

    # ------------------------------------------------------------------
    # Chat execution
    # ------------------------------------------------------------------

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
        reasoning_effort: Optional[str] = None,
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        parallel_tool_calls: bool = True,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        conversation_cache: Optional[ConversationCache] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion via DeepSeek."""
        from openai import AsyncOpenAI

        # Tool handler setup
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

        # Handle conversation continuation via base class cache
        messages = await self.prepare_conversation_messages(
            messages, previous_response_id, conversation_cache
        )

        # Create OpenAI client pointed at DeepSeek
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Filter tools by budget
        tools = self.filter_tools_by_budget(tools, tool_handler)

        request_params, messages = self.build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            reasoning_effort=reasoning_effort,
            store=store,
            metadata=metadata,
            timeout=timeout,
            parallel_tool_calls=parallel_tool_calls,
            previous_response_id=previous_response_id,
        )

        # Build tool dict
        tool_dict = {}
        if tools and "tools" in request_params:
            tool_dict = tool_handler.build_tool_dict(tools)

        try:
            response = await client.chat.completions.create(**request_params)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                output_tokens_details,
                response_id,
                repair_metadata,
            ) = await self.process_response(
                client=client,
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
        except (ProviderError, ToolError):
            raise
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Generate response ID for conversation continuation
        gen_response_id = self.generate_response_id()

        # Persist conversation history for follow-up calls
        history = self.append_assistant_message_to_history(messages, content)
        await self.persist_conversation_history(
            gen_response_id, history, conversation_cache
        )

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
            output_tokens_details=output_tokens_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
            response_id=gen_response_id,
            structured_output_repairs=repair_metadata,
        )
