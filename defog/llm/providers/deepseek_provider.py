from defog import config as defog_config
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .openrouter_provider import OpenRouterProvider
from ..config import LLMConfig

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(OpenRouterProvider):
    """DeepSeek provider using the OpenAI-compatible Chat Completions API.

    DeepSeek exposes the same wire format as OpenRouter (OpenAI Chat
    Completions), so we subclass OpenRouterProvider and only swap the base
    URL, API key env var, and provider name.

    Structured output:
        DeepSeek rejects ``response_format={"type": "json_schema", ...}``
        with ``400 invalid_request_error`` ("This response_format type is
        unavailable now"). It only supports ``{"type": "json_object"}`` and
        requires the literal word "json" in at least one message plus a
        schema/example to shape the output. We override
        ``_build_response_format`` to emit json_object mode and
        ``build_params`` to append the Pydantic schema as a system
        instruction so the existing parse/repair path downstream still
        materializes the target model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config=None,
    ):
        super().__init__(
            api_key=api_key or defog_config.get("DEEPSEEK_API_KEY"),
            base_url=base_url or DEEPSEEK_BASE_URL,
            config=config,
        )

    @classmethod
    def from_config(cls, config: LLMConfig):
        return cls(
            api_key=config.get_api_key("deepseek"),
            base_url=config.get_base_url("deepseek") or DEEPSEEK_BASE_URL,
            config=config,
        )

    def get_provider_name(self) -> str:
        return "deepseek"

    def _build_response_format(self, response_format) -> Optional[Dict[str, Any]]:
        if response_format is None:
            return None
        if hasattr(response_format, "model_json_schema"):
            return {"type": "json_object"}
        return None

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
        request_params, processed_messages = super().build_params(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            store=store,
            metadata=metadata,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=parallel_tool_calls,
            previous_response_id=previous_response_id,
            **kwargs,
        )

        if (
            response_format is not None
            and not tools
            and hasattr(response_format, "model_json_schema")
        ):
            schema = response_format.model_json_schema()
            instruction = (
                "Respond with a single valid JSON object that conforms to the "
                "schema below. Output only the JSON — no prose, no markdown "
                "fences, no commentary.\n\n"
                f"JSON schema:\n{json.dumps(schema, indent=2)}"
            )
            injected = self._inject_system_instruction(processed_messages, instruction)
            request_params["messages"] = injected
            return request_params, injected

        return request_params, processed_messages

    @staticmethod
    def _inject_system_instruction(
        messages: List[Dict[str, Any]], instruction: str
    ) -> List[Dict[str, Any]]:
        """Append *instruction* to the first system message, or prepend a new one.

        DeepSeek's json_object mode requires the literal word "json" in a
        message; the instruction text above already contains it.
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
