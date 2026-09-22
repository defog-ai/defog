"""Mocked unit tests for Claude Opus 5.5 request building."""

from __future__ import annotations

import pytest

from defog.llm.providers.anthropic_provider import (
    AnthropicProvider,
    rejects_forced_tool_choice,
)

OPUS_55 = "claude-opus-5-5"


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


def _params(model: str, **kwargs):
    provider = AnthropicProvider(api_key="sk-test")
    params, _ = provider.build_params(
        messages=[{"role": "user", "content": "hi"}],
        model=model,
        **kwargs,
    )
    return params


def test_rejects_forced_tool_choice():
    assert rejects_forced_tool_choice(OPUS_55)
    assert rejects_forced_tool_choice("claude-fable-5-1")
    assert not rejects_forced_tool_choice("claude-opus-5")
    assert not rejects_forced_tool_choice("claude-fable-5")


@pytest.mark.parametrize("tool_choice", ["required", "get_weather"])
def test_forced_tool_choice_becomes_auto(tool_choice):
    params = _params(OPUS_55, tools=[get_weather], tool_choice=tool_choice)
    assert params["tool_choice"] == {"type": "auto"}


def test_forced_tool_choice_kept_on_opus_5():
    params = _params("claude-opus-5", tools=[get_weather], tool_choice="required")
    assert params["tool_choice"] == {"type": "any"}


def test_parallel_tool_calls_flag_survives_downgrade():
    params = _params(
        OPUS_55,
        tools=[get_weather],
        tool_choice="required",
        parallel_tool_calls=False,
    )
    assert params["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }


def test_thinking_is_adaptive_without_effort():
    # Thinking cannot be disabled on Opus 5.5.
    params = _params(OPUS_55)
    assert params["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_all_effort_levels_pass_through(effort):
    params = _params(OPUS_55, reasoning_effort=effort)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == effort
