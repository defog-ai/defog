from defog import config as defog_config
import time
import json
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, ToolError
from ..config import LLMConfig
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_openai_format
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider using OpenAI-compatible Chat Completions API.

    OpenRouter is a unified gateway that routes requests to multiple LLM
    providers (Anthropic, OpenAI, Google, Meta, etc.) via a single
    OpenAI-compatible API.

    Key features:
    - Prompt caching: auto-enabled for most providers; explicit cache_control
      for Anthropic models.
    - Tool/function calling: standard OpenAI Chat Completions tool format.
    - Structured outputs: via response_format with json_schema type.
    - Provider routing: control which upstream provider handles the request.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config=None,
    ):
        super().__init__(
            api_key or defog_config.get("OPENROUTER_API_KEY"),
            base_url or OPENROUTER_BASE_URL,
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create OpenRouter provider from config."""
        return cls(
            api_key=config.get_api_key("openrouter"),
            base_url=config.get_base_url("openrouter") or OPENROUTER_BASE_URL,
            config=config,
        )

    def get_provider_name(self) -> str:
        return "openrouter"

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
        """Preprocess messages for OpenRouter (OpenAI-compatible format)."""
        messages = deepcopy(messages)
        for msg in messages:
            content = msg.get("content")
            if content is not None:
                msg["content"] = convert_to_openai_format(content)
        return messages

    @staticmethod
    def _is_anthropic_model(model: str) -> bool:
        """Check if the model is an Anthropic model (for cache_control)."""
        model_lower = model.lower()
        return "claude" in model_lower or model_lower.startswith("anthropic/")

    @staticmethod
    def _is_pydantic_model(response_format) -> bool:
        """Check if response_format is a Pydantic model class."""
        return (
            response_format is not None
            and isinstance(response_format, type)
            and hasattr(response_format, "model_validate")
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
        parallel_tool_calls: bool = False,
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

        # OpenRouter defaults to the model's max output if max_tokens is
        # omitted, which can exhaust credits quickly.  Use a sensible default.
        request_params["max_tokens"] = max_completion_tokens or 4096

        # Tools
        if tools:
            function_specs = get_function_specs(tools, "openrouter")
            request_params["tools"] = function_specs

            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                request_params["tool_choice"] = convert_tool_choice(
                    tool_choice, tool_names_list, "openrouter"
                )
            else:
                request_params["tool_choice"] = "auto"

            request_params["parallel_tool_calls"] = parallel_tool_calls

        # Structured output is handled via client.beta.chat.completions.parse()
        # in process_response / execute_chat, not via request_params.

        # Extra body for OpenRouter-specific features
        extra_body: Dict[str, Any] = {}

        # Prompt caching for Anthropic models
        if self._is_anthropic_model(model):
            extra_body["cache_control"] = {"type": "ephemeral"}

        if extra_body:
            request_params["extra_body"] = extra_body

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
        return_tool_outputs_only: bool = False,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        extra_body: Optional[Dict[str, Any]] = None,
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
        """Process Chat Completions response, handling tool call chaining."""
        if tool_handler is None:
            tool_handler = self.tool_handler

        if not hasattr(response, "choices") or not response.choices:
            raise ProviderError(self.get_provider_name(), "No response from OpenRouter")

        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
        total_openrouter_cost = None
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

                # Accumulate OpenRouter-reported cost
                _cost = getattr(response.usage, "cost", None) if response.usage else None
                if _cost is not None:
                    total_openrouter_cost = (total_openrouter_cost or 0) + _cost

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
                    call_params = {
                        k: v for k, v in request_params.items() if k != "extra_body"
                    }
                    if extra_body:
                        call_params["extra_body"] = extra_body
                    response = await client.chat.completions.create(**call_params)

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
                    if k
                    not in (
                        "tools",
                        "tool_choice",
                        "parallel_tool_calls",
                        "extra_body",
                    )
                }
                if extra_body:
                    final_params["extra_body"] = extra_body
                if self._is_pydantic_model(response_format):
                    response = await client.beta.chat.completions.parse(
                        **final_params, response_format=response_format
                    )
                    content = response.choices[0].message.parsed
                else:
                    response = await client.chat.completions.create(**final_params)
                    raw_content = response.choices[0].message.content or ""
                    content = self.parse_structured_response(
                        raw_content, response_format
                    )
                _cost = getattr(response.usage, "cost", None) if response.usage else None
                if _cost is not None:
                    total_openrouter_cost = (total_openrouter_cost or 0) + _cost
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
                # .parse() was used in execute_chat; read the parsed result
                if self._is_pydantic_model(response_format):
                    content = response.choices[0].message.parsed
                else:
                    raw_content = response.choices[0].message.content or ""
                    content = self.parse_structured_response(
                        raw_content, response_format
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
            _cost = getattr(response.usage, "cost", None)
            if _cost is not None:
                total_openrouter_cost = (total_openrouter_cost or 0) + _cost

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            output_tokens_details,
            response.id or "",
            total_openrouter_cost,
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
        """Execute a chat completion via OpenRouter."""
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
            messages, previous_response_id
        )

        # Create OpenAI client pointed at OpenRouter
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "X-Title": "defog",
            },
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
            # Separate extra_body from regular params for the API call
            extra_body = request_params.pop("extra_body", None)

            create_kwargs = {**request_params}
            if extra_body:
                create_kwargs["extra_body"] = extra_body

            # Use .parse() for structured output (no tools path) — the SDK
            # handles schema transformation, $ref resolution, and
            # additionalProperties automatically.
            if self._is_pydantic_model(response_format) and not tools:
                response = await client.beta.chat.completions.parse(
                    **create_kwargs, response_format=response_format
                )
            else:
                response = await client.chat.completions.create(**create_kwargs)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                output_tokens_details,
                response_id,
                openrouter_cost,
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
                extra_body=extra_body,
            )
        except (ProviderError, ToolError):
            raise
        except Exception as e:
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        # Generate response ID for conversation continuation
        gen_response_id = self.generate_response_id()

        # Persist conversation history for follow-up calls
        history = self.append_assistant_message_to_history(messages, content)
        await self.persist_conversation_history(gen_response_id, history)

        # Use OpenRouter-reported cost if available, fall back to local price table
        if openrouter_cost is not None:
            cost = round(openrouter_cost * 100, 6)  # Convert USD to cents
        else:
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
        )
