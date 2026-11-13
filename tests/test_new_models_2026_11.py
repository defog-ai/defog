"""Mocked unit tests for Claude Fable 5.2, GPT-6.1 and Gemini 4 / 3.x Flash."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from defog.llm.cost import CostCalculator
from defog.llm.providers.anthropic_provider import (
    AnthropicProvider,
    adaptive_thinking_config,
    rejects_forced_tool_choice,
)
from defog.llm.providers.gemini_provider import validate_thinking_level

FABLE_52 = "claude-fable-5-2"
MILLION = 1_000_000


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


# A fixed time before every scheduled Gemini price change, so that the
# tests below do not depend on the day they run.
BEFORE_CHANGES = datetime(2026, 11, 11, tzinfo=timezone.utc)


def _dollars(model: str, **kwargs) -> float:
    """Cost in USD (calculate_cost returns cents)."""
    kwargs.setdefault("calculation_time", BEFORE_CHANGES)
    return CostCalculator.calculate_cost(model, **kwargs) / 100


# --- Claude Fable 5.2 -------------------------------------------------------


def test_fable_5_2_rejects_forced_tool_choice():
    assert rejects_forced_tool_choice(FABLE_52)
    assert not rejects_forced_tool_choice("claude-fable-5")


@pytest.mark.parametrize("tool_choice", ["required", "get_weather"])
def test_fable_5_2_forced_tool_choice_becomes_auto(tool_choice):
    params = _params(FABLE_52, tools=[get_weather], tool_choice=tool_choice)
    assert params["tool_choice"] == {"type": "auto"}


def test_fable_5_2_thinking_is_adaptive_without_effort():
    params = _params(FABLE_52)
    assert params["thinking"] == {"type": "adaptive"}
    assert "temperature" not in params


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_fable_5_2_effort_levels_pass_through(effort):
    params = _params(FABLE_52, reasoning_effort=effort)
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == effort
    assert adaptive_thinking_config(FABLE_52, effort) == (
        {"type": "adaptive"},
        {"effort": effort},
    )


def test_fable_5_2_prices_per_million_tokens():
    assert _dollars(FABLE_52, input_tokens=MILLION, output_tokens=0) == pytest.approx(8)
    assert _dollars(FABLE_52, input_tokens=0, output_tokens=MILLION) == pytest.approx(
        40
    )
    assert _dollars(
        FABLE_52, input_tokens=0, output_tokens=0, cached_input_tokens=MILLION
    ) == pytest.approx(0.20)
    assert _dollars(
        FABLE_52, input_tokens=0, output_tokens=0, cache_creation_input_tokens=MILLION
    ) == pytest.approx(10)
    assert _dollars(
        FABLE_52,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=MILLION,
        cache_creation_1h_input_tokens=MILLION,
    ) == pytest.approx(16)


def test_fable_5_2_batch_is_half_price():
    assert _dollars(
        FABLE_52, input_tokens=MILLION, output_tokens=MILLION, batch=True
    ) == pytest.approx(24)


# --- GPT-6.1 ----------------------------------------------------------------


@pytest.mark.parametrize(
    "model,input_usd,cached_usd,output_usd",
    [
        ("gpt-6.1-sol", 2, 0.10, 10),
        ("gpt-6.1-terra", 1, 0.05, 5),
    ],
)
def test_gpt_6_1_standard_prices(model, input_usd, cached_usd, output_usd):
    # 100K tokens each: below the 272K long-context threshold.
    n = 100_000
    scale = n / MILLION
    assert _dollars(model, input_tokens=n, output_tokens=0) == pytest.approx(
        input_usd * scale
    )
    assert _dollars(model, input_tokens=0, output_tokens=n) == pytest.approx(
        output_usd * scale
    )
    assert _dollars(
        model, input_tokens=0, output_tokens=0, cached_input_tokens=n
    ) == pytest.approx(cached_usd * scale)


@pytest.mark.parametrize(
    "model,expected_usd",
    [
        # 300K input at the 2x long-context rate plus 1K output at 1.5x.
        ("gpt-6.1-sol", 0.3 * 4 + 0.001 * 15),
        ("gpt-6.1-terra", 0.3 * 2 + 0.001 * 7.5),
    ],
)
def test_gpt_6_1_long_context_prices(model, expected_usd):
    assert _dollars(model, input_tokens=300_000, output_tokens=1000) == pytest.approx(
        expected_usd
    )


@pytest.mark.parametrize(
    "model,expected_usd",
    [
        # 100K input, 100K cached input and 100K output at the Batch/Flex rates.
        ("gpt-6.1-sol", 0.1 * (1 + 0.05 + 5)),
        ("gpt-6.1-terra", 0.1 * (0.5 + 0.025 + 2.5)),
    ],
)
@pytest.mark.parametrize("tier,batch", [("flex", False), (None, True)])
def test_gpt_6_1_flex_and_batch_prices(model, expected_usd, tier, batch):
    assert _dollars(
        model,
        input_tokens=100_000,
        output_tokens=100_000,
        cached_input_tokens=100_000,
        service_tier=tier,
        batch=batch,
    ) == pytest.approx(expected_usd)


def test_gpt_6_1_does_not_use_gpt_6_cached_rate():
    # gpt-6-sol caches at 10% of input, gpt-6.1-sol at 5%.
    kwargs = dict(input_tokens=0, output_tokens=0, cached_input_tokens=100_000)
    assert _dollars("gpt-6.1-sol", **kwargs) == pytest.approx(
        _dollars("gpt-6-sol", **kwargs) / 2
    )


# --- Gemini -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model,input_usd,cached_usd,output_usd",
    [
        ("gemini-4-carbon", 2, 0.10, 10),
        ("gemini-4-argon", 2, 0.10, 10),
        ("gemini-3.8-flash", 0.75, 0.075, 3.75),
        ("gemini-3.6-flash", 0.75, 0.075, 3.75),
        ("gemini-3.5-flash-lite", 0.30, 0.03, 2.50),
    ],
)
def test_gemini_prices_per_million_tokens(model, input_usd, cached_usd, output_usd):
    assert _dollars(model, input_tokens=MILLION, output_tokens=0) == pytest.approx(
        input_usd
    )
    assert _dollars(model, input_tokens=0, output_tokens=MILLION) == pytest.approx(
        output_usd
    )
    assert _dollars(
        model, input_tokens=0, output_tokens=0, cached_input_tokens=MILLION
    ) == pytest.approx(cached_usd)


@pytest.mark.parametrize(
    "model,change_date,input_usd,cached_usd,output_usd",
    [
        ("gemini-4-carbon", datetime(2026, 12, 1), 4, 0.20, 20),
        ("gemini-4-argon", datetime(2026, 12, 1), 4, 0.20, 20),
        ("gemini-3.8-flash", datetime(2027, 1, 1), 1.50, 0.15, 7.50),
        ("gemini-3.6-flash", datetime(2027, 1, 1), 1.50, 0.15, 7.50),
    ],
)
def test_gemini_scheduled_price_change(
    model, change_date, input_usd, cached_usd, output_usd
):
    tokens = dict(
        input_tokens=MILLION, output_tokens=MILLION, cached_input_tokens=MILLION
    )
    new_total = input_usd + cached_usd + output_usd
    start = change_date.replace(tzinfo=timezone.utc)
    last_old = start - timedelta(seconds=1)
    assert _dollars(model, calculation_time=last_old, **tokens) == pytest.approx(
        new_total / 2
    )
    assert _dollars(model, calculation_time=start, **tokens) == pytest.approx(new_total)
    # A naive datetime is read as UTC.
    assert _dollars(model, calculation_time=change_date, **tokens) == pytest.approx(
        new_total
    )


def test_price_change_uses_current_time_by_default(monkeypatch):
    import defog.llm.cost.calculator as calculator

    class _After(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2027, 6, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(calculator, "datetime", _After)
    assert CostCalculator.calculate_cost("gemini-4-carbon", MILLION, 0) == (
        pytest.approx(400)
    )


def test_model_without_price_change_ignores_date():
    late = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert _dollars(
        "gemini-3.5-flash-lite",
        input_tokens=MILLION,
        output_tokens=0,
        calculation_time=late,
    ) == pytest.approx(0.30)


def test_flash_lite_does_not_use_flash_price():
    assert _dollars(
        "gemini-3.5-flash-lite", input_tokens=MILLION, output_tokens=0
    ) != pytest.approx(
        _dollars("gemini-3.5-flash", input_tokens=MILLION, output_tokens=0)
    )


@pytest.mark.parametrize(
    "model",
    [
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ],
)
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_gemini_thinking_levels_accepted(model, effort):
    validate_thinking_level(model, effort)


@pytest.mark.parametrize("model", ["gemini-3.6-flash", "gemini-3.5-flash-lite"])
def test_gemini_minimal_accepted(model):
    validate_thinking_level(model, "minimal")


@pytest.mark.parametrize(
    "model", ["gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash"]
)
def test_gemini_minimal_rejected(model):
    with pytest.raises(ValueError, match="'low', 'medium' or 'high'"):
        validate_thinking_level(model, "minimal")


def test_gemini_older_rules_unchanged():
    with pytest.raises(ValueError, match="only supported for"):
        validate_thinking_level("gemini-2.5-flash", "low")
    with pytest.raises(ValueError, match="'low' or 'high'"):
        validate_thinking_level("gemini-3-pro-preview", "medium")
    with pytest.raises(ValueError, match="must be one of"):
        validate_thinking_level("gemini-3-flash-preview", "max")


# --- Gemini 4 Hydrogen ------------------------------------------------------


def test_gemini_4_hydrogen_prices_per_million_tokens():
    model = "gemini-4-hydrogen"
    assert _dollars(model, input_tokens=MILLION, output_tokens=0) == pytest.approx(1.50)
    assert _dollars(model, input_tokens=0, output_tokens=MILLION) == pytest.approx(7.50)
    assert _dollars(
        model, input_tokens=0, output_tokens=0, cached_input_tokens=MILLION
    ) == pytest.approx(0.075)


def test_gemini_4_hydrogen_has_no_scheduled_price_change():
    # Carbon and Argon double on 2026-12-01. Hydrogen has no published change.
    after = datetime(2026, 12, 2, tzinfo=timezone.utc)
    assert _dollars(
        "gemini-4-hydrogen",
        input_tokens=MILLION,
        output_tokens=0,
        calculation_time=after,
    ) == pytest.approx(1.50)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_gemini_4_hydrogen_thinking_levels_accepted(effort):
    validate_thinking_level("gemini-4-hydrogen", effort)


@pytest.mark.parametrize("effort", ["minimal", "max"])
def test_gemini_4_hydrogen_other_levels_rejected(effort):
    with pytest.raises(ValueError, match="'low', 'medium', 'high' or 'xhigh'"):
        validate_thinking_level("gemini-4-hydrogen", effort)


@pytest.mark.parametrize(
    "model",
    ["gemini-4-carbon", "gemini-4-argon", "gemini-3.8-flash", "gemini-3-pro-preview"],
)
def test_gemini_xhigh_rejected_outside_hydrogen(model):
    with pytest.raises(ValueError, match="only supported for gemini-4-hydrogen"):
        validate_thinking_level(model, "xhigh")
