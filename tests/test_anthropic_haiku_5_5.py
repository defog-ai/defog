"""Mocked unit tests for Claude Haiku 5.5 request building and pricing."""

import pytest

from defog.llm.cost.calculator import CostCalculator, _find_match
from defog.llm.providers.anthropic_provider import (
    AnthropicProvider,
    adaptive_thinking_config,
    rejects_forced_tool_choice,
)

HAIKU_55 = "claude-haiku-5-5"


def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return "sunny"


def _params(model=HAIKU_55, **kwargs):
    params, _ = AnthropicProvider(api_key="sk-test").build_params(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )
    return params


@pytest.mark.parametrize("effort", [None, "none"])
def test_thinking_off_uses_disabled(effort):
    params = _params(reasoning_effort=effort)
    assert params["thinking"] == {"type": "disabled"}
    assert "temperature" not in params
    assert "effort" not in params.get("output_config", {})


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_effort_uses_adaptive_thinking(effort):
    params = _params(reasoning_effort=effort)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == effort
    assert "temperature" not in params


def test_forced_tool_choice_is_kept():
    assert not rejects_forced_tool_choice(HAIKU_55)
    params = _params(tools=[get_weather], tool_choice="required")
    assert params["tool_choice"] == {"type": "any"}


def test_pricing():
    cost = CostCalculator.calculate_cost(HAIKU_55, 1_000_000, 1_000_000, 0)
    assert cost == pytest.approx(60.0)  # cents: $0.10 in + $0.50 out


def test_pricing_cache_rates():
    cost = CostCalculator.calculate_cost(HAIKU_55, 0, 0, 1_000_000, 1_000_000)
    assert cost == pytest.approx(13.5)  # cents: $0.01 read + $0.125 write


def test_dated_model_id_matches():
    assert _find_match("claude-haiku-5-5-20261007") == HAIKU_55
    assert _find_match("claude-haiku-4-5-20251001") == "claude-haiku-4-5"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_adaptive_thinking_config(effort):
    assert adaptive_thinking_config(HAIKU_55, effort) == (
        {"type": "adaptive"},
        {"effort": effort},
    )


def test_adaptive_thinking_config_none_effort():
    assert adaptive_thinking_config(HAIKU_55, "none") == ({"type": "disabled"}, None)


def test_haiku_4_5_unchanged():
    assert adaptive_thinking_config("claude-haiku-4-5", "high") is None
    params = _params("claude-haiku-4-5")
    assert params["thinking"] == {"type": "disabled"}
    assert params["temperature"] == 0.0
    params = _params("claude-haiku-4-5", reasoning_effort="high")
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert params["temperature"] == 1.0
