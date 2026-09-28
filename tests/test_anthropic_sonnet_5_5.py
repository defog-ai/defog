"""Mocked unit tests for Claude Sonnet 5.5 request building and pricing."""

import pytest

from defog.llm.cost.calculator import CostCalculator
from defog.llm.providers.anthropic_provider import (
    AnthropicProvider,
    rejects_forced_tool_choice,
)

SONNET_55 = "claude-sonnet-5-5"


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


def _params(model=SONNET_55, **kwargs):
    params, _ = AnthropicProvider(api_key="sk-test").build_params(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )
    return params


@pytest.mark.parametrize("effort", [None, "none"])
def test_thinking_off_uses_between_tools(effort):
    params = _params(reasoning_effort=effort)
    assert params["thinking"] == {"type": "between_tools"}
    assert "temperature" not in params
    assert "effort" not in params.get("output_config", {})


def test_sonnet_5_still_uses_disabled():
    assert _params("claude-sonnet-5")["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_effort_uses_adaptive_thinking(effort):
    params = _params(reasoning_effort=effort)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == effort
    assert "temperature" not in params


def test_rejects_forced_tool_choice():
    assert rejects_forced_tool_choice(SONNET_55)
    assert not rejects_forced_tool_choice("claude-sonnet-5")


@pytest.mark.parametrize("tool_choice", ["required", "get_weather"])
def test_forced_tool_choice_becomes_auto(tool_choice):
    params = _params(tools=[get_weather], tool_choice=tool_choice)
    assert params["tool_choice"] == {"type": "auto"}


def test_pricing():
    cost = CostCalculator.calculate_cost(SONNET_55, 1_000_000, 1_000_000, 0)
    assert cost == pytest.approx(1200.0)  # cents: $2 in + $10 out
