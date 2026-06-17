from defog import config as defog_config
import re
import time
import json
import logging
from types import SimpleNamespace
from copy import deepcopy
from typing import Dict, List, Any, Optional, Callable, Tuple, Union

import httpx

from .base import BaseLLMProvider, LLMResponse
from ..exceptions import ProviderError, ToolError
from ..config import LLMConfig
from ..memory.conversation_cache import ConversationCache
from ..cost import CostCalculator
from ..utils_function_calling import get_function_specs, convert_tool_choice
from ..image_utils import convert_to_openai_format
from ..tools.handler import ToolHandler

logger = logging.getLogger(__name__)

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"

# ZAI serves the OpenAI-compatible Chat Completions wire format. The
# "openai" provider string in utils_function_calling yields the standard
# OpenAI Chat Completions tool/tool_choice shape, which ZAI accepts.
_FUNCTION_SPEC_PROVIDER = "openai"


def _to_namespace(value: Any) -> Any:
    """Convert response dictionaries into attribute-access objects."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _normalize_completion_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Fill optional Chat Completions fields that provider code reads directly."""
    data.setdefault("id", data.get("request_id"))
    data.setdefault("usage", {})
    usage = data["usage"] or {}
    usage.setdefault("prompt_tokens", 0)
    usage.setdefault("completion_tokens", 0)
    usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
    data["usage"] = usage

    for choice in data.get("choices") or []:
        message = choice.setdefault("message", {})
        message.setdefault("content", None)
        message.setdefault("tool_calls", None)
    return data


class ZAIProvider(BaseLLMProvider):
    """Native ZAI provider using the ZAI Chat Completions HTTP API.

    ZAI is served natively at ``https://api.z.ai/api/paas/v4`` and speaks an
    OpenAI-style Chat Completions wire format. This provider calls the HTTP
    endpoint directly with ``httpx``.

    Structured output:
        ZAI rejects ``response_format={"type": "json_schema", ...}`` with
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
            api_key or defog_config.get("ZAI_API_KEY"),
            base_url or ZAI_BASE_URL,
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        """Create ZAI provider from config."""
        return cls(
            api_key=config.get_api_key("zai"),
            base_url=config.get_base_url("zai") or ZAI_BASE_URL,
            config=config,
        )

    def get_provider_name(self) -> str:
        return "zai"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Accept-Language": "en-US,en",
            "Content-Type": "application/json; charset=UTF-8",
        }

    async def _create_completion(
        self, client: httpx.AsyncClient, request_params: Dict[str, Any]
    ) -> Any:
        response = await client.post("/chat/completions", json=request_params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                self.get_provider_name(),
                f"API request failed ({response.status_code}): {response.text}",
                e,
            ) from e

        try:
            data = response.json()
        except ValueError as e:
            raise ProviderError(
                self.get_provider_name(),
                f"API returned non-JSON response: {response.text}",
                e,
            ) from e

        return _to_namespace(_normalize_completion_response(data))

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
        """Preprocess messages for ZAI (OpenAI-compatible format)."""
        messages = deepcopy(messages)
        for msg in messages:
            content = msg.get("content")
            if content is not None:
                msg["content"] = convert_to_openai_format(content)
        return messages

    # ------------------------------------------------------------------
    # Structured output (ZAI-specific)
    # ------------------------------------------------------------------

    def _build_response_format(self, response_format) -> Optional[Dict[str, Any]]:
        """ZAI only supports json_object mode; it 400s on json_schema."""
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

        ZAI's json_object mode requires the literal word "json" in a
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

    @staticmethod
    def _deterministic_json_repair(content: str) -> str:
        """Apply deterministic fixes for common JSON issues from loose output.

        Handles: markdown fences, Python literals (True/False/None),
        JS constants (NaN/Infinity/undefined), trailing commas, comments,
        single-quoted strings, and unbalanced brackets.
        """
        s = content.strip()

        # Strip markdown code fences
        if s.startswith("```"):
            first_nl = s.find("\n")
            s = s[first_nl + 1 :] if first_nl != -1 else s[3:]
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()

        # Remove single-line JS/Python comments (best effort)
        s = re.sub(r"(?m)^\s*//[^\n]*$", "", s)
        s = re.sub(r"//[^\n]*", "", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)

        # Replace Python/JS literals with JSON equivalents
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        s = re.sub(r"\bNaN\b", "null", s)
        s = re.sub(r"\bundefined\b", "null", s)
        s = re.sub(r"\bInfinity\b", "null", s)
        s = re.sub(r"-\s*null\b", "null", s)  # fix -Infinity → -null → null

        # Remove trailing commas before } or ]
        s = re.sub(r",(\s*[}\]])", r"\1", s)

        # Try single-quote → double-quote conversion if no double quotes present
        # in the structural positions (keys/values). Best-effort heuristic.
        if "'" in s and not re.search(r'(?<=[{\[,:])\s*"', s):
            candidate = s.replace("'", '"')
            try:
                json.loads(candidate)
                s = candidate
            except json.JSONDecodeError:
                pass  # didn't help, keep original

        # Balance unclosed brackets using a character-level scan
        open_braces = 0
        open_brackets = 0
        in_string = False
        escape_next = False
        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                open_braces += 1
            elif ch == "}":
                open_braces -= 1
            elif ch == "[":
                open_brackets += 1
            elif ch == "]":
                open_brackets -= 1

        # If we ended inside a string, close it
        if in_string:
            s += '"'

        # If last non-whitespace is a colon, append null (truncated value)
        stripped_tail = s.rstrip()
        if stripped_tail and stripped_tail[-1] == ":":
            s = stripped_tail + " null"
        # If last non-whitespace is a comma, strip it
        elif stripped_tail and stripped_tail[-1] == ",":
            s = stripped_tail[:-1]

        # Append missing closing brackets
        if open_brackets > 0:
            s += "]" * open_brackets
        if open_braces > 0:
            s += "}" * open_braces

        # Final pass: remove any remaining trailing commas before closers
        s = re.sub(r",(\s*[}\]])", r"\1", s)

        return s

    async def _llm_json_repair(
        self,
        broken_content: str,
        response_format,
        client,
        model: str,
        request_params: Dict[str, Any],
    ) -> Any:
        """Use a follow-on LLM call to fix broken JSON.

        Appends the model's broken response as an assistant turn and adds a
        user follow-up asking it to return valid JSON, preserving the full
        conversation context.
        """
        schema = None
        if hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()

        schema_text = json.dumps(schema, indent=2) if schema else "Not available"

        follow_up = (
            "Your previous response was not valid JSON or did not match the "
            "expected schema. Please return ONLY the corrected JSON — no "
            "markdown fences, no explanations, no extra text.\n\n"
            f"Expected JSON schema:\n{schema_text}"
        )

        logger.info("Attempting LLM-based JSON repair for model=%s", model)

        messages = list(request_params.get("messages", []))
        messages.append({"role": "assistant", "content": broken_content})
        messages.append({"role": "user", "content": follow_up})

        repair_params = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        rf = self._build_response_format(response_format)
        if rf:
            repair_params["response_format"] = rf

        response = await self._create_completion(client, repair_params)
        if not hasattr(response, "choices") or not response.choices:
            raise ProviderError(
                self.get_provider_name(),
                "No response from ZAI",
            )
        fixed_content = response.choices[0].message.content or ""
        return self.parse_structured_response(fixed_content, response_format)

    async def _parse_with_repair(
        self,
        raw_content: str,
        response_format,
        client,
        model: str,
        request_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Parse structured response with deterministic repair and LLM fallback.

        Sequence:
        1. Try base parse_structured_response (markdown strip + json.loads + regex)
        2. Apply deterministic fixes and re-parse
        3. Make a follow-on LLM call to repair the JSON
        4. Return original content if everything fails
        """
        if not response_format or not raw_content:
            return raw_content

        has_pydantic = hasattr(response_format, "model_validate")

        def _is_parsed(result):
            """True when the result is a parsed object, not raw text."""
            if has_pydantic:
                return not isinstance(result, str)
            return isinstance(result, (dict, list))

        # --- Step 1: standard parsing ---
        try:
            result = self.parse_structured_response(raw_content, response_format)
            if _is_parsed(result):
                return result
        except Exception:
            pass

        # --- Step 2: deterministic repair ---
        try:
            repaired = self._deterministic_json_repair(raw_content)
            result = self.parse_structured_response(repaired, response_format)
            if _is_parsed(result):
                logger.info("Deterministic JSON repair succeeded")
                return result
        except Exception:
            pass

        # --- Step 3: LLM-based repair ---
        try:
            result = await self._llm_json_repair(
                raw_content,
                response_format,
                client,
                model,
                dict(request_params or {}),
            )
            if _is_parsed(result):
                logger.info("LLM-based JSON repair succeeded")
                return result
        except Exception as e:
            logger.warning("LLM-based JSON repair failed: %s", e)

        # --- All repair attempts exhausted ---
        logger.warning(
            "All JSON repair attempts failed for structured output; "
            "returning raw content"
        )
        return raw_content

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

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

        if reasoning_effort is not None:
            request_params["reasoning_effort"] = reasoning_effort
        else:
            request_params["thinking"] = {"type": "disabled"}

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

        # Structured output (only when not using tools — after tool chaining we
        # make a separate structured-output call). ZAI requires
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
    ]:
        """Process Chat Completions response, handling tool call chaining."""
        if tool_handler is None:
            tool_handler = self.tool_handler

        if not hasattr(response, "choices") or not response.choices:
            raise ProviderError(self.get_provider_name(), "No response from ZAI")

        tool_outputs = []
        total_input_tokens = 0
        total_cached_input_tokens = 0
        total_output_tokens = 0
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
                    response = await self._create_completion(client, request_params)
                    if not hasattr(response, "choices") or not response.choices:
                        raise ProviderError(
                            self.get_provider_name(),
                            "No response from ZAI",
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
                    if k not in ("tools", "tool_choice")
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
                response = await self._create_completion(client, final_params)
                if not hasattr(response, "choices") or not response.choices:
                    raise ProviderError(
                        self.get_provider_name(),
                        "No response from ZAI",
                    )
                raw_content = response.choices[0].message.content or ""
                content = await self._parse_with_repair(
                    raw_content,
                    response_format,
                    client,
                    model,
                    final_params,
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

        return (
            content,
            tool_outputs,
            total_input_tokens,
            total_cached_input_tokens,
            total_output_tokens,
            output_tokens_details,
            response.id or "",
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
        """Execute a chat completion via ZAI."""
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
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers(),
                timeout=timeout,
            )
            response = await self._create_completion(client, request_params)

            (
                content,
                tool_outputs,
                input_tokens,
                cached_input_tokens,
                output_tokens,
                output_tokens_details,
                response_id,
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
        finally:
            if "client" in locals():
                await client.aclose()

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
        )
