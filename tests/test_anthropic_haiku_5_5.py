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
    cost = CostCalculator.calculate_cost(HAIKU_55, 100_000, 100_000, 0)
    assert cost == pytest.approx(6.0)  # cents: $0.10 in + $0.50 out per 1M


def test_pricing_cache_rates():
    cost = CostCalculator.calculate_cost(HAIKU_55, 0, 0, 50_000, 50_000)
    assert cost == pytest.approx(0.675)  # cents: $0.01 read + $0.125 write per 1M


def test_pricing_long_prompt():
    # Over 100,000 prompt tokens, every token of the request costs five times as much.
    cost = CostCalculator.calculate_cost(HAIKU_55, 100_001, 100_000, 0)
    assert cost == pytest.approx((100_001 * 0.50 + 100_000 * 2.50) / 1_000_000 * 100)


def test_pricing_long_prompt_counts_cache_tokens():
    # Cache reads and cache writes are part of the prompt.
    cost = CostCalculator.calculate_cost(HAIKU_55, 20_000, 0, 80_000, 1)
    assert cost == pytest.approx(
        (20_000 * 0.50 + 80_000 * 0.05 + 1 * 0.625) / 1_000_000 * 100
    )


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
    assert "temperature" not in params
    params = _params("claude-haiku-4-5", reasoning_effort="high")
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert "temperature" not in params
