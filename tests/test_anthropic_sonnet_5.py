"""Offline regression coverage for Sonnet 5 thinking and structured output."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from defog.llm import chat_async
from defog.llm.config import LLMConfig
from defog.llm.providers.anthropic_provider import AnthropicProvider


class Answer(BaseModel):
    text: str


ANSWER_FORMAT = {
    "type": "json_schema",
    "schema": {
        "properties": {"text": {"title": "Text", "type": "string"}},
        "required": ["text"],
        "title": "Answer",
        "type": "object",
        "additionalProperties": False,
    },
}


def _params(model="claude-sonnet-5", **kwargs):
    params, _ = AnthropicProvider(api_key="sk-test").build_params(
        model=model,
        messages=[{"role": "user", "content": "Say hello."}],
        **kwargs,
    )
    return params


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_sonnet_5_uses_adaptive_thinking(effort):
    params = _params(reasoning_effort=effort, response_format=Answer)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": effort, "format": ANSWER_FORMAT}
    assert "temperature" not in params


@pytest.mark.parametrize(
    "kwargs", [{}, {"reasoning_effort": None}, {"reasoning_effort": "none"}]
)
def test_sonnet_5_preserves_thinking_off_without_effort(kwargs):
    params = _params(response_format=Answer, **kwargs)
    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"] == {"format": ANSWER_FORMAT}
    assert "temperature" not in params


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
def test_sonnet_5_omits_sampling_parameters(temperature):
    params = _params(temperature=temperature)
    assert "temperature" not in params
    assert "output_config" not in params


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
)
@pytest.mark.parametrize(
    "effort,budget", [("low", 2048), ("medium", 4096), ("high", 8192)]
)
def test_legacy_thinking_budgets_unchanged(model, effort, budget):
    params = _params(model, reasoning_effort=effort)
    assert params["thinking"] == {"type": "enabled", "budget_tokens": budget}
    assert params["temperature"] == 1.0
    assert "output_config" not in params


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        ("claude-sonnet-4-6", "low", "low"),
        ("claude-sonnet-4-6", "max", "high"),
        ("claude-sonnet-4-6", "xhigh", "high"),
        ("claude-opus-4-6", "xhigh", "max"),
        ("claude-opus-4-7", "xhigh", "xhigh"),
        ("claude-opus-4-8", "max", "max"),
        ("claude-opus-5", "xhigh", "xhigh"),
        ("claude-fable-5", "max", "max"),
    ],
)
def test_existing_adaptive_effort_handling_unchanged(model, effort, expected):
    params = _params(model, reasoning_effort=effort)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": expected}


@pytest.mark.parametrize(
    "model,thinking",
    [
        ("claude-sonnet-4-6", "disabled"),
        ("claude-opus-4-6", "adaptive"),
        ("claude-opus-4-7", "adaptive"),
        ("claude-opus-4-8", "adaptive"),
        ("claude-opus-5", "adaptive"),
        ("claude-fable-5", "adaptive"),
    ],
)
def test_existing_adaptive_defaults_unchanged(model, thinking):
    params = _params(model)
    assert params["thinking"] == {"type": thinking}
    assert "output_config" not in params


@pytest.mark.asyncio
async def test_reported_chat_async_structured_output_request():
    create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"text":"Hello!"}')],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            container=None,
            model="claude-sonnet-5",
            id="msg_test",
        )
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    with patch(
        "defog.llm.providers.anthropic_provider.AsyncAnthropic", return_value=client
    ):
        response = await chat_async(
            provider="anthropic",
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello."}],
            reasoning_effort="low",
            response_format=Answer,
            config=LLMConfig(api_keys={"anthropic": "sk-test"}),
            max_retries=1,
        )

    create.assert_awaited_once()
    params = create.call_args.kwargs
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "low", "format": ANSWER_FORMAT}
    assert "temperature" not in params
    assert response.content == Answer(text="Hello!")
