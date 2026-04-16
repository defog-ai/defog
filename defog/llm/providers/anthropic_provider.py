import inspect
import traceback
from defog import config as defog_config
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

from anthropic import AsyncAnthropic, transform_schema

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, MaxTokensError, ToolError
from ..config import LLMConfig
from ..memory.conversation_cache import ConversationCache
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_anthropic_format
from ..utils_image_support import process_tool_results_with_images, ToolResultData
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config=None,
    ):
        super().__init__(
            api_key or defog_config.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create Anthropic provider from config."""
        return cls(
            api_key=config.get_api_key("anthropic"),
            base_url=config.get_base_url("anthropic"),
            config=config,
        )

    def get_provider_name(self) -> str:
        return "anthropic"

    def convert_content_to_anthropic(self, content: Any) -> Any:
        """Convert message content to Anthropic format."""
        return convert_to_anthropic_format(content)

    async def _apply_pre_model_call_hook(
        self,
        request_params: Dict[str, Any],
        model: str,
        pre_model_call_hook: Optional[Callable],
        *,
        checkpoint_kind: str,
    ) -> None:
        injected_messages = await self.call_pre_model_call_hook(
            pre_model_call_hook,
            checkpoint_kind=checkpoint_kind,
            model=model,
        )
        if not injected_messages:
            return

        request_params.setdefault("messages", [])
        for message in injected_messages:
            role = message.get("role")
            if role == "system":
                raise ValueError(
                    "pre_model_call_hook may not inject system messages into "
                    "Anthropic conversations."
                )
            request_params["messages"].append(
                {
                    "role": role,
                    "content": self.convert_content_to_anthropic(
                        message.get("content")
                    ),
                }
            )

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
        reasoning_effort: Optional[str] = None,
        parallel_tool_calls: bool = True,
        strict_tools: bool = True,
        server_tools: Optional[List[Dict[str, Any]]] = None,
        programmatic_tool_calling: bool = False,
        container_id: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Create the parameter dict for Anthropic's .messages.create()."""
        # Convert messages to support multimodal content
        converted_messages = []
        system_messages = []

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
                        # Non-text blocks (tool_use, tool_result, image) count as content
                        return False
                    if btype == "text" and block.get("text", "").strip():
                        return False
                # All text blocks were empty/whitespace
                return True
            return False

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
                converted_content = self.convert_content_to_anthropic(msg["content"])

                # Ensure user messages are never empty
                if msg.get("role") == "assistant" and _content_is_empty(
                    converted_content
                ):
                    converted_content = (
                        "No content was generated for this turn. "
                        "Refer to the preceding tool outputs for context."
                    )

                converted_msg["content"] = converted_content
                converted_messages.append(converted_msg)

        # Concatenate all system messages into a single string
        sys_msg = "\n\n".join(system_messages) if system_messages else ""

        messages = converted_messages

        # Extended thinking: "-4" catches all Claude 4+ models (sonnet-4,
        # opus-4-1, opus-4-5, opus-4-6, haiku-4-5, etc.), "3-7" catches
        # Claude 3.7 Sonnet. Using "-4" instead of "-4-" so that
        # "claude-sonnet-4" (no date suffix) is also matched.
        supports_thinking = "3-7" in model or "-4" in model
        # Claude 4.6+ models use adaptive thinking (type: "adaptive") with
        # effort via output_config, replacing the deprecated budget_tokens
        # param. Update this tuple when new models add adaptive support.
        supports_adaptive = any(
            p in model for p in ("opus-4-6", "opus-4-7", "sonnet-4-6")
        )
        # effort: "max" is only available on Opus models. Sending it to
        # Sonnet returns an API error.
        supports_max_effort = "opus-4-6" in model or "opus-4-7" in model
        # Opus models require adaptive thinking always on. For other
        # adaptive models (e.g. Sonnet 4.6), only enable it when
        # reasoning_effort is explicitly requested.
        requires_adaptive = "opus-4-6" in model or "opus-4-7" in model
        use_adaptive = requires_adaptive or (
            supports_adaptive and reasoning_effort is not None
        )

        if use_adaptive:
            # Adaptive thinking: the model decides how much to think
            # based on query complexity. The effort level is optionally
            # set via output_config below.
            thinking = {
                "type": "adaptive",
            }
            temperature = 1.0
        elif reasoning_effort is not None and supports_thinking:
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
            elif reasoning_effort in ("high", "max"):
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
            max_completion_tokens = 64000

        params = {
            "system": sys_msg,
            "messages": messages,
            "model": model,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "timeout": timeout,
            "thinking": thinking,
        }

        # Build output_config: may include adaptive thinking effort and/or
        # structured output format (json_schema).
        output_config = {}
        if use_adaptive and reasoning_effort is not None:
            # Cap effort to "high" for models that don't support "max".
            if reasoning_effort == "max" and not supports_max_effort:
                output_config["effort"] = "high"
            else:
                output_config["effort"] = reasoning_effort
        # Native structured outputs via constrained decoding.
        # The Anthropic API supports output_config.format alongside tools
        # with strict: true in the same request, so we always set it here.
        if (
            response_format
            and isinstance(response_format, type)
            and hasattr(response_format, "model_json_schema")
        ):
            output_config["format"] = {
                "type": "json_schema",
                "schema": transform_schema(response_format),
            }
        if output_config:
            params["output_config"] = output_config

        # Use Anthropic's automatic caching: a single top-level cache_control
        # on the request body. The API automatically applies the cache breakpoint
        # to the last cacheable block and moves it forward as conversations grow.
        params["cache_control"] = {"type": "ephemeral"}

        # Programmatic tool calling constraint validation. We do these before
        # building tool specs so callers get clear errors instead of silent
        # downstream API failures.
        if programmatic_tool_calling:
            if response_format is not None:
                raise ValueError(
                    "programmatic_tool_calling=True is incompatible with "
                    "response_format / structured outputs."
                )
            if tool_choice is not None and tool_choice not in ("auto",):
                raise ValueError(
                    "programmatic_tool_calling=True only supports tool_choice='auto' "
                    f"(or None); got {tool_choice!r}."
                )
            if not parallel_tool_calls:
                raise ValueError(
                    "programmatic_tool_calling=True is incompatible with "
                    "parallel_tool_calls=False (Anthropic disallows "
                    "disable_parallel_tool_use with programmatic tool calling)."
                )
            if strict_tools:
                logger.warning(
                    "programmatic_tool_calling=True forces strict_tools=False; "
                    "ignoring strict_tools=True."
                )
                strict_tools = False

        if tools:
            function_specs = get_function_specs(
                tools, self.get_provider_name(), strict=strict_tools
            )

            # When programmatic tool calling is enabled, every user-supplied
            # callable becomes invokable from the code execution sandbox via
            # ``await my_tool(...)``. We mark each spec with the allowed_callers
            # field so the API knows it can be called from code execution.
            if programmatic_tool_calling:
                for spec in function_specs:
                    spec["allowed_callers"] = ["code_execution_20260120"]

            params["tools"] = function_specs
            if tool_choice:
                tool_names_list = [func.__name__ for func in tools]
                tool_choice = convert_tool_choice(
                    tool_choice, tool_names_list, self.get_provider_name()
                )
                params["tool_choice"] = tool_choice
            else:
                params["tool_choice"] = {"type": "auto"}

            # Add parallel tool calls configuration. Skip when programmatic
            # tool calling is on or when server tools are present because
            # Anthropic forbids disable_parallel_tool_use in those modes.
            if (
                "tool_choice" in params
                and isinstance(params["tool_choice"], dict)
                and not programmatic_tool_calling
                and not server_tools
            ):
                if not parallel_tool_calls:
                    params["tool_choice"]["disable_parallel_tool_use"] = True

        # Append Anthropic server-side tool specs (web_search, web_fetch,
        # code_execution, advisor) to the same tools array. They can coexist
        # with user callables and MCP tools.
        if server_tools:
            params.setdefault("tools", []).extend(server_tools)
            # If we only have server tools (no user callables), still set a
            # default tool_choice so the model knows it can use them.
            if "tool_choice" not in params:
                params["tool_choice"] = {"type": "auto"}

        # Continue an existing code-execution / programmatic-calling container.
        if container_id:
            params["container"] = container_id

        return params, messages

    async def extract_reasoning_text(
        self,
        thinking_blocks: list,
        post_tool_function: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Extract thinking/reasoning text from Anthropic thinking blocks and call post_tool_function."""
        reasoning_summaries = []
        for block in thinking_blocks:
            thinking_text = getattr(block, "thinking", None)
            if thinking_text:
                reasoning_summaries.append(thinking_text)
                if post_tool_function:
                    if inspect.iscoroutinefunction(post_tool_function):
                        await post_tool_function(
                            function_name="reasoning",
                            input_args={},
                            tool_result=thinking_text,
                            tool_id=None,
                        )
                    else:
                        post_tool_function(
                            function_name="reasoning",
                            input_args={},
                            tool_result=thinking_text,
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
        post_tool_function: Optional[Callable] = None,
        post_response_hook: Optional[Callable] = None,
        pre_model_call_hook: Optional[Callable] = None,
        tool_handler: Optional[ToolHandler] = None,
        return_tool_outputs_only: bool = False,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        server_tools: Optional[List[Dict[str, Any]]] = None,
        programmatic_tool_calling: bool = False,
        **kwargs,
    ) -> Tuple[
        Any,
        List[Dict[str, Any]],
        int,
        int,
        Optional[int],
        Optional[int],
        Optional[Dict[str, int]],
        Optional[List[Dict[str, Any]]],
        Optional[Dict[str, int]],
        Optional[str],
        Optional[str],
    ]:
        """
        Extract content (including any tool calls) and usage info from Anthropic response.
        Handles chaining of tool calls and structured output parsing.

        Returns a tuple of:
            content,
            tool_outputs,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            output_tokens_details,
            server_tool_outputs,
            server_tool_usage,
            container_id,
            container_expires_at,
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
        cached_input_tokens = 0
        cache_creation_input_tokens = 0
        return_tool_outputs_only = bool(return_tool_outputs_only)
        model_for_tokens = request_params.get("model") or "gpt-4.1"
        tool_calls_executed = False

        # Anthropic server-side tool state.
        server_tool_outputs: List[Dict[str, Any]] = []
        server_tool_usage: Dict[str, int] = {}
        container_id: Optional[str] = None
        container_expires_at: Optional[str] = None

        # Block types that come from Anthropic's server-side tool execution.
        # The model emits server_tool_use as the "call", and the API streams
        # back result blocks with one of these types.
        SERVER_TOOL_RESULT_TYPES = {
            "web_search_tool_result",
            "web_fetch_tool_result",
            "code_execution_tool_result",
            "bash_code_execution_tool_result",
            "text_editor_code_execution_tool_result",
            "advisor_tool_result",
        }

        def _block_to_dict(block: Any) -> Any:
            """Best-effort serialize an Anthropic SDK block to a plain dict.

            We use this only for the user-visible ``server_tool_outputs`` list,
            so it's OK if some nested values fall back to ``str(...)``.
            """
            if block is None or isinstance(block, (str, int, float, bool)):
                return block
            if isinstance(block, dict):
                return {k: _block_to_dict(v) for k, v in block.items()}
            if isinstance(block, (list, tuple)):
                return [_block_to_dict(v) for v in block]
            if hasattr(block, "model_dump"):
                try:
                    return block.model_dump()
                except Exception:
                    pass
            if hasattr(block, "__dict__"):
                return {k: _block_to_dict(v) for k, v in vars(block).items()}
            return str(block)

        async def collect_server_tool_blocks(resp) -> None:
            """Pull server-side tool result blocks out of a response and
            call ``post_tool_function`` for each one so callers can observe
            server-tool activity the same way they observe user-tool activity.
            """
            # Build a lookup from tool_use_id → (name, input) so we can
            # report meaningful names/args to post_tool_function.
            server_use_map: Dict[str, tuple] = {}
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "server_tool_use":
                    use_id = getattr(block, "id", None)
                    if use_id:
                        server_use_map[use_id] = (
                            getattr(block, "name", "server_tool"),
                            _block_to_dict(getattr(block, "input", {})),
                        )

            for block in getattr(resp, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype in SERVER_TOOL_RESULT_TYPES:
                    tool_use_id = getattr(block, "tool_use_id", None)
                    result = _block_to_dict(getattr(block, "content", None))
                    server_tool_outputs.append(
                        {
                            "type": btype,
                            "tool_use_id": tool_use_id,
                            "result": result,
                        }
                    )
                    if post_tool_function:
                        tool_name, input_args = server_use_map.get(
                            tool_use_id, (btype, {})
                        )
                        if inspect.iscoroutinefunction(post_tool_function):
                            await post_tool_function(
                                function_name=tool_name,
                                input_args=input_args,
                                tool_result=result,
                                tool_id=tool_use_id,
                            )
                        else:
                            post_tool_function(
                                function_name=tool_name,
                                input_args=input_args,
                                tool_result=result,
                                tool_id=tool_use_id,
                            )

        def collect_server_tool_usage(resp) -> None:
            """Pull cumulative server tool usage counters off response.usage."""
            usage = getattr(resp, "usage", None)
            if usage is None:
                return
            stu = getattr(usage, "server_tool_use", None)
            if stu is None:
                return
            mapping: Optional[Dict[str, Any]] = None
            if isinstance(stu, dict):
                mapping = stu
            elif hasattr(stu, "model_dump"):
                try:
                    mapping = stu.model_dump()
                except Exception:
                    mapping = None
            if mapping is None and hasattr(stu, "__dict__"):
                mapping = vars(stu)
            if not isinstance(mapping, dict):
                return
            for key, value in mapping.items():
                if isinstance(value, int):
                    # Anthropic reports cumulative counters per response, so
                    # we replace rather than sum within a single response and
                    # take the max across loop iterations (since each new
                    # response usually reports the new total).
                    server_tool_usage[key] = max(server_tool_usage.get(key, 0), value)

        def collect_container(resp) -> None:
            """Capture container id/expiry off the response, if any."""
            nonlocal container_id, container_expires_at
            container = getattr(resp, "container", None)
            if container is None:
                return
            cid = getattr(container, "id", None)
            cexp = getattr(container, "expires_at", None)
            if cid:
                container_id = cid
            if cexp:
                container_expires_at = cexp

        def is_programmatic_tool_use(block: Any) -> bool:
            """Return True iff a tool_use block was emitted from code exec."""
            caller = getattr(block, "caller", None)
            if caller is None:
                return False
            ctype = getattr(caller, "type", None)
            if ctype is None and isinstance(caller, dict):
                ctype = caller.get("type")
            return bool(ctype and ctype.startswith("code_execution_"))

        def has_tool_call_outputs() -> bool:
            return any(output.get("tool_call_id") for output in tool_outputs)

        def add_usage(usage_obj):
            nonlocal \
                total_input_tokens, \
                total_output_tokens, \
                cached_input_tokens, \
                cache_creation_input_tokens
            # Anthropic sometimes returns ``None`` for cache fields (e.g. when
            # the response did not interact with the cache), so we coerce.
            total_input_tokens += getattr(usage_obj, "input_tokens", 0) or 0
            total_output_tokens += getattr(usage_obj, "output_tokens", 0) or 0
            cached_input_tokens += getattr(usage_obj, "cache_read_input_tokens", 0) or 0
            cache_creation_input_tokens += (
                getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
            )

        # Handle tool processing for both local tools and MCP server tools.
        # We also enter this branch when ``server_tools`` are present (so we
        # can handle pause_turn iterations and collect server tool outputs)
        # or when programmatic_tool_calling is enabled.
        if (tools and len(tools) > 0) or server_tools or programmatic_tool_calling:
            consecutive_exceptions = 0
            while True:
                add_usage(response.usage)
                await collect_server_tool_blocks(response)
                collect_server_tool_usage(response)
                collect_container(response)

                # Handle pause_turn (long-running code execution): echo the
                # assistant content verbatim with no user reply, then re-call
                # the API with the same container.
                if response.stop_reason == "pause_turn":
                    request_params["messages"].append(
                        {
                            "role": "assistant",
                            "content": list(response.content),
                        }
                    )
                    if container_id:
                        request_params["container"] = container_id
                    await self._apply_pre_model_call_hook(
                        request_params,
                        request_params.get("model") or model_for_tokens,
                        pre_model_call_hook,
                        checkpoint_kind="pause_turn",
                    )
                    response = await client.messages.create(**request_params)
                    continue

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

                # Extract reasoning text from thinking blocks and call post_tool_function
                reasoning_outputs = await self.extract_reasoning_text(
                    thinking_blocks, post_tool_function
                )
                tool_outputs.extend(reasoning_outputs)

                # Detect whether any tool_use blocks were emitted from the
                # code execution sandbox (programmatic tool calling). When at
                # least one is, the user reply we send back must contain ONLY
                # tool_result blocks — no text, no thinking, no extras — and
                # the assistant echo must include every block verbatim.
                programmatic_call_present = any(
                    is_programmatic_tool_use(block) for block in tool_call_blocks
                )

                if len(tool_call_blocks) > 0:
                    tool_calls_executed = True
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

                        sampled_results = []
                        text_previews = []

                        # Store tool outputs for all tools (both MCP and regular)
                        for tool_call_block, result in zip(tool_call_blocks, results):
                            func_name = tool_call_block.name
                            args = tool_call_block.input
                            tool_id = tool_call_block.id

                            sampled_result = await tool_handler.sample_tool_result(
                                func_name,
                                result,
                                args,
                                tool_id=tool_id,
                                tool_sample_functions=tool_sample_functions,
                            )
                            text_for_llm, was_truncated, _ = (
                                tool_handler.prepare_result_for_llm(
                                    sampled_result,
                                    preview_max_tokens=tool_result_preview_max_tokens,
                                    model=model_for_tokens,
                                )
                            )
                            sampled_results.append(sampled_result)
                            text_previews.append(text_for_llm)

                            # Store the tool call, result, and text
                            tool_outputs.append(
                                {
                                    "tool_call_id": tool_id,
                                    "name": func_name,
                                    "args": args,
                                    "result": result,
                                    "result_for_llm": text_for_llm,
                                    "result_truncated_for_llm": was_truncated,
                                    "sampling_applied": tool_handler.is_sampler_configured(
                                        func_name, tool_sample_functions
                                    ),
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
                                # Build assistant content. For programmatic
                                # tool calling, Anthropic requires the full
                                # transcript verbatim (text, thinking,
                                # server_tool_use, server tool results, and
                                # tool_use). For the legacy direct path we
                                # echo just thinking + tool_use blocks, which
                                # is what Anthropic accepts and matches the
                                # existing tests.
                                has_server_blocks = any(
                                    getattr(b, "type", "") in ("server_tool_use",)
                                    or getattr(b, "type", "")
                                    in SERVER_TOOL_RESULT_TYPES
                                    for b in response.content
                                )
                                if programmatic_call_present or has_server_blocks:
                                    assistant_content = list(response.content)
                                else:
                                    assistant_content = []
                                    if len(thinking_blocks) > 0:
                                        assistant_content += thinking_blocks
                                    for tool_call_block in tool_call_blocks:
                                        assistant_content.append(tool_call_block)

                                request_params["messages"].append(
                                    {
                                        "role": "assistant",
                                        "content": assistant_content,
                                    }
                                )

                                # Build user response. For programmatic tool
                                # calls, the user message must contain ONLY
                                # tool_result blocks (no text, thinking, or
                                # images) — this is required by Anthropic
                                # when responding to a code-execution-issued
                                # tool_use.
                                if programmatic_call_present:
                                    tool_results_content = []
                                    for tool_call_block, preview_text in zip(
                                        tool_call_blocks, text_previews
                                    ):
                                        tool_results_content.append(
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_call_block.id,
                                                "content": preview_text,
                                            }
                                        )
                                    request_params["messages"].append(
                                        {
                                            "role": "user",
                                            "content": tool_results_content,
                                        }
                                    )
                                    # Programmatic path: skip image handling
                                    # and the legacy tool_results_data branch
                                    # entirely.
                                    request_params["tool_choice"] = (
                                        {"type": "auto"}
                                        if request_params.get("tool_choice") != "auto"
                                        else None
                                    )
                                    if container_id:
                                        request_params["container"] = container_id
                                    tools, tool_dict = self.update_tools_with_budget(
                                        tools,
                                        tool_handler,
                                        request_params,
                                    )
                                    await self._apply_pre_model_call_hook(
                                        request_params,
                                        request_params.get("model") or model_for_tokens,
                                        pre_model_call_hook,
                                        checkpoint_kind="post_tool_batch",
                                    )
                                    response = await client.messages.create(
                                        **request_params
                                    )
                                    continue

                                # Build user response with all tool results and handle images
                                tool_results_data = process_tool_results_with_images(
                                    tool_call_blocks,
                                    sampled_results if sampled_results else results,
                                    tool_handler.image_result_keys,
                                )

                                if sampled_results:
                                    adjusted_results = []
                                    for tool_data, preview_text in zip(
                                        tool_results_data, text_previews
                                    ):
                                        adjusted_results.append(
                                            ToolResultData(
                                                tool_id=tool_data.tool_id,
                                                tool_name=tool_data.tool_name,
                                                tool_result_text=preview_text,
                                                image_data=tool_data.image_data,
                                            )
                                        )
                                    tool_results_data = adjusted_results

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
                    except (ProviderError, ToolError):
                        # Re-raise provider/tool errors from base class
                        raise
                    except Exception as e:
                        # For other exceptions, use the same retry logic
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

                    # Update available tools based on budget before making next call
                    tools, tool_dict = self.update_tools_with_budget(
                        tools, tool_handler, request_params
                    )

                    if container_id:
                        request_params["container"] = container_id

                    await self._apply_pre_model_call_hook(
                        request_params,
                        request_params.get("model") or model_for_tokens,
                        pre_model_call_hook,
                        checkpoint_kind="post_tool_batch",
                    )
                    response = await client.messages.create(**request_params)
                else:
                    # Break out of loop when tool calls are finished
                    skip_final_response = (
                        return_tool_outputs_only and has_tool_call_outputs()
                    )
                    if skip_final_response:
                        content = ""
                        break

                    content = "\n".join([block.text for block in text_blocks])
                    break
            if tool_calls_executed:
                await self.emit_tool_phase_complete(
                    post_tool_function, message=tool_phase_complete_message
                )
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
            if response.usage:
                add_usage(response.usage)
            await collect_server_tool_blocks(response)
            collect_server_tool_usage(response)
            collect_container(response)

        if return_tool_outputs_only and has_tool_call_outputs():
            content = ""

        # Parse structured output if response_format is provided
        if response_format:
            # Use base class method for structured response parsing
            content = self.parse_structured_response(content, response_format)

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            None,
            server_tool_outputs if server_tool_outputs else None,
            server_tool_usage if server_tool_usage else None,
            container_id,
            container_expires_at,
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
        pre_model_call_hook: Optional[Callable] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_budget: Optional[Dict[str, int]] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        tool_phase_complete_message: str = "exploration done, generating answer",
        server_tools: Optional[
            Union[List[str], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
        ] = None,
        programmatic_tool_calling: bool = False,
        container_id: Optional[str] = None,
        conversation_cache: Optional[ConversationCache] = None,
        **kwargs,
    ) -> LLMResponse:
        """Execute a chat completion with Anthropic."""
        from .anthropic_server_tools import normalize_server_tools

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
        return_tool_outputs_only = kwargs.pop("return_tool_outputs_only", False)

        if post_tool_function:
            tool_handler.validate_post_tool_function(post_tool_function)
        if pre_model_call_hook:
            self.validate_pre_model_call_hook(pre_model_call_hook)

        # Normalize server tools first so version validation errors surface
        # before any API call.
        normalized_server_tools: List[Dict[str, Any]] = []
        beta_headers_needed: set = set()
        if server_tools is not None or programmatic_tool_calling:
            normalized_server_tools, beta_headers_needed = normalize_server_tools(
                server_tools or [],
                model=model,
                programmatic_tool_calling=programmatic_tool_calling,
            )

        # Programmatic tool calling silently flips strict_tools off (with a
        # warning emitted in build_params) and forces parallel tool use to
        # ON. Override the kwargs we pass to build_params accordingly.
        if programmatic_tool_calling:
            kwargs.setdefault("parallel_tool_calls", True)
            # If the caller explicitly passed False at the chat_async level,
            # we still want to surface the conflict via build_params.

        t = time.time()

        # Interleaved thinking beta header: required for Claude 4/4.5 models,
        # automatic for Opus 4.6+ (adaptive thinking enables it implicitly).
        # Safe to always include — the API ignores it for unsupported models.
        beta_header_values = ["interleaved-thinking-2025-05-14"]
        for extra in sorted(beta_headers_needed):
            if extra not in beta_header_values:
                beta_header_values.append(extra)
        headers = {"anthropic-beta": ",".join(beta_header_values)}

        client_kwargs = {
            "api_key": self.api_key,
            "default_headers": headers,
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = AsyncAnthropic(**client_kwargs)

        # Store strict_tools setting for use in update_tools_with_budget
        self._strict_tools = kwargs.get("strict_tools", True)
        if programmatic_tool_calling:
            self._strict_tools = False

        # Filter tools based on budget before building params
        tools = self.filter_tools_by_budget(tools, tool_handler)

        conversation_messages = await self.prepare_conversation_messages(
            messages, previous_response_id, conversation_cache
        )

        params, _ = self.build_params(
            messages=conversation_messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            parallel_tool_calls=kwargs.get("parallel_tool_calls", True),
            strict_tools=kwargs.get("strict_tools", True),
            server_tools=normalized_server_tools or None,
            programmatic_tool_calling=programmatic_tool_calling,
            container_id=container_id,
        )

        # Construct a tool dict if needed
        tool_dict = {}
        if tools and len(tools) > 0 and "tools" in params:
            tool_dict = tool_handler.build_tool_dict(tools)

        func_to_call = client.messages.create

        try:
            await self._apply_pre_model_call_hook(
                params,
                model,
                pre_model_call_hook,
                checkpoint_kind="initial_request",
            )
            response = await func_to_call(**params)

            (
                content,
                tool_outputs,
                input_toks,
                output_toks,
                cached_toks,
                cache_creation_toks,
                output_details,
                server_tool_outputs_resp,
                server_tool_usage_resp,
                response_container_id,
                response_container_expires_at,
            ) = await self.process_response(
                client=client,
                response=response,
                request_params=params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                post_tool_function=post_tool_function,
                post_response_hook=post_response_hook,
                pre_model_call_hook=pre_model_call_hook,
                tool_handler=tool_handler,
                return_tool_outputs_only=return_tool_outputs_only,
                tool_sample_functions=sample_functions,
                tool_result_preview_max_tokens=preview_max_tokens,
                tool_phase_complete_message=tool_phase_complete_message,
                server_tools=normalized_server_tools or None,
                programmatic_tool_calling=programmatic_tool_calling,
                **kwargs,
            )
        except Exception as e:
            traceback.print_exc()
            raise ProviderError(self.get_provider_name(), f"API call failed: {e}", e)

        api_response_id = getattr(response, "id", None)
        response_id = api_response_id
        if store:
            cache_response_id = api_response_id or self.generate_response_id()
            history_for_cache = self.append_assistant_message_to_history(
                conversation_messages, content
            )
            await self.persist_conversation_history(
                cache_response_id, history_for_cache, conversation_cache
            )
            response_id = cache_response_id

        # Calculate cost
        cost = CostCalculator.calculate_cost(
            model, input_toks, output_toks, cached_toks, cache_creation_toks
        )

        return LLMResponse(
            model=model,
            content=content,
            time=round(time.time() - t, 3),
            input_tokens=input_toks,
            output_tokens=output_toks,
            cached_input_tokens=cached_toks,
            cache_creation_input_tokens=cache_creation_toks,
            output_tokens_details=output_details,
            cost_in_cents=cost,
            tool_outputs=tool_outputs,
            response_id=response_id,
            server_tool_outputs=server_tool_outputs_resp,
            server_tool_usage=server_tool_usage_resp,
            container_id=response_container_id,
            container_expires_at=response_container_expires_at,
        )
