"""Tests for defog.llm.cost.calculator model matching and pricing."""

from datetime import datetime, timedelta, timezone

import pytest

from defog.llm.cost import calculator
from defog.llm.cost.calculator import (
    CostCalculator,
    _find_match,
    _is_deepseek_peak_time,
)
from defog.llm.cost.models import MODEL_COSTS


DEEPSEEK_FLASH_MODELS = (
    "deepseek-flash",
    "deepseek-v4.1-flash",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
)


def _cost(
    model: str,
    input_t: int = 1000,
    output_t: int = 1000,
    cached: int = 0,
    calculation_time: datetime | None = None,
):
    return CostCalculator.calculate_cost(
        model=model,
        input_tokens=input_t,
        output_tokens=output_t,
        cached_input_tokens=cached,
        calculation_time=calculation_time,
    )


def test_exact_match_uses_own_entry():
    assert _find_match("gpt-5-mini") == "gpt-5-mini"
    assert _find_match("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_gpt_5_4_entries_are_explicit():
    # These have distinct prices from gpt-5/gpt-5-mini/gpt-5-nano and must
    # not silently fall back to gpt-5 tier pricing.
    assert "gpt-5.4" in MODEL_COSTS
    assert "gpt-5.4-mini" in MODEL_COSTS
    assert "gpt-5.4-nano" in MODEL_COSTS
    assert _find_match("gpt-5.4-mini") == "gpt-5.4-mini"


def test_gpt_5_5_entries_are_explicit():
    # gpt-5.5 has distinct pricing from gpt-5/gpt-5.4 and must not silently
    # fall back to an older GPT-5 family price.
    assert "gpt-5.5" in MODEL_COSTS
    assert "gpt-5.5-pro" in MODEL_COSTS
    assert _find_match("gpt-5.5") == "gpt-5.5"
    assert _find_match("gpt-5.5-2026-04-23") == "gpt-5.5"
    assert _find_match("gpt-5.5-pro") == "gpt-5.5-pro"
    assert _find_match("gpt-5.5-pro-2026-04-23") == "gpt-5.5-pro"


def test_gpt_5_6_entries_are_explicit():
    # GPT-5.6 uses named tiers rather than the mini/nano/pro tier names.
    assert "gpt-5.6" in MODEL_COSTS
    assert "gpt-5.6-sol" in MODEL_COSTS
    assert "gpt-5.6-terra" in MODEL_COSTS
    assert "gpt-5.6-luna" in MODEL_COSTS
    assert _find_match("gpt-5.6") == "gpt-5.6"
    assert _find_match("gpt-5.6-sol") == "gpt-5.6-sol"
    assert _find_match("gpt-5.6-terra") == "gpt-5.6-terra"
    assert _find_match("gpt-5.6-luna") == "gpt-5.6-luna"
    assert _find_match("gpt-5.6-terra-2026-07-09") == "gpt-5.6-terra"


def test_gpt_6_and_opus_5_5_entries_are_explicit():
    # Without their own entries, claude-opus-5-5 and claude-fable-5-1 would
    # fall back to claude-opus-5 / claude-fable-5 pricing through the
    # family-prefix match.
    assert _find_match("gpt-6-astra") == "gpt-6-astra"
    assert _find_match("gpt-6-sol") == "gpt-6-sol"
    assert _find_match("gpt-6-luna") == "gpt-6-luna"
    assert _find_match("claude-opus-5-5") == "claude-opus-5-5"
    assert _find_match("claude-opus-5") == "claude-opus-5"
    assert _find_match("claude-fable-5-1") == "claude-fable-5-1"
    assert _find_match("claude-fable-5") == "claude-fable-5"


def test_unknown_mini_does_not_fall_back_to_base_pricing():
    # Regression: previously `gpt-5.4-mini` fell back to `gpt-5` (full) pricing
    # via loose substring match, inflating cost ~5x. With size-suffix parity,
    # unknown *-mini names must only match *-mini entries.
    resolved = _find_match("gpt-9.9-mini")
    if resolved is not None:
        assert resolved.endswith("-mini")


def test_unknown_nano_does_not_fall_back_to_base_pricing():
    resolved = _find_match("gpt-9.9-nano")
    if resolved is not None:
        assert resolved.endswith("-nano")


def test_unknown_version_with_known_family_routes_by_prefix():
    # A brand-new gpt-5.9-mini should route to gpt-5-mini (family prefix
    # match with matching size suffix), not to gpt-5 base pricing.
    assert _find_match("gpt-5.9-mini") == "gpt-5-mini"
    assert _find_match("gpt-5.9-nano") == "gpt-5-nano"
    assert _find_match("gpt-5.9") == "gpt-5"


def test_claude_dated_suffix_still_matches():
    # Anthropic returns model ids with date suffixes; these should still
    # resolve via the substring fallback.
    assert _find_match("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert _find_match("claude-sonnet-4-6-20250922") == "claude-sonnet-4-6"


def test_gpt_5_4_pricing_matches_openai_rate_card():
    # Per https://developers.openai.com/api/docs/pricing as of 2026-04-16:
    # gpt-5.4-mini: $0.75 / $4.50 per 1M tokens
    # 1000 input + 1000 output = (1 * 0.00075 + 1 * 0.0045) * 100 cents
    assert _cost("gpt-5.4-mini") == pytest.approx(0.525)
    # gpt-5.4-nano: $0.20 / $1.25 per 1M tokens
    assert _cost("gpt-5.4-nano") == pytest.approx(0.145)
    # gpt-5.4: $2.50 / $15.00 per 1M tokens
    assert _cost("gpt-5.4") == pytest.approx(1.75)


def test_gpt_5_5_pricing_matches_openai_rate_card():
    # Per https://openai.com/api/pricing/ and
    # https://openai.com/index/introducing-gpt-5-5/ as of 2026-04-24:
    # gpt-5.5: $5.00 input / $0.50 cached input / $30.00 output per 1M tokens
    assert _cost("gpt-5.5") == pytest.approx(3.5)
    assert _cost("gpt-5.5", cached=1000) == pytest.approx(3.55)
    # gpt-5.5-pro: $30.00 input / $180.00 output per 1M tokens
    assert _cost("gpt-5.5-pro") == pytest.approx(21.0)


def test_gpt_5_6_pricing_matches_openai_rate_card():
    # Per https://developers.openai.com/api/docs/pricing as of 2026-09-15.
    # Sol (and its gpt-5.6 alias): $4 / $0.40 cached / $20 per 1M tokens
    # (not the Flex rate doubled).
    assert _cost("gpt-5.6") == pytest.approx(2.4)
    assert _cost("gpt-5.6-sol") == pytest.approx(2.4)
    assert _cost("gpt-5.6-sol", cached=1000) == pytest.approx(2.44)
    assert _cost("gpt-5.6", cached=1000) == pytest.approx(2.44)
    # Terra: $2 / $0.20 cached / $12 per 1M tokens.
    assert _cost("gpt-5.6-terra") == pytest.approx(1.4)
    assert _cost("gpt-5.6-terra", cached=1000) == pytest.approx(1.42)
    # Luna: $0.20 / $0.02 cached / $1.20 per 1M tokens.
    assert _cost("gpt-5.6-luna") == pytest.approx(0.14)
    assert _cost("gpt-5.6-luna", cached=1000) == pytest.approx(0.142)


def test_gpt_6_pricing_matches_openai_rate_card():
    # Per https://developers.openai.com/api/docs/pricing as of 2026-09-23.
    # Astra: $10 / $1 cached / $50 per 1M tokens.
    assert _cost("gpt-6-astra") == pytest.approx(6.0)
    assert _cost("gpt-6-astra", cached=1000) == pytest.approx(6.1)
    # Sol: $2 / $0.20 cached / $10 per 1M tokens.
    assert _cost("gpt-6-sol") == pytest.approx(1.2)
    assert _cost("gpt-6-sol", cached=1000) == pytest.approx(1.22)
    # Luna: $0.10 / $0.01 cached / $0.50 per 1M tokens.
    assert _cost("gpt-6-luna") == pytest.approx(0.06)
    assert _cost("gpt-6-luna", cached=1000) == pytest.approx(0.061)


def test_claude_opus_5_5_pricing():
    # Per https://platform.claude.com/docs/en/models/opus-5-5/overview:
    # $4 input, $0.20 cache read, $5 5m cache write, $20 output per 1M tokens.
    assert _cost("claude-opus-5-5") == pytest.approx(2.4)
    assert _cost("claude-opus-5-5", cached=1000) == pytest.approx(2.42)
    assert CostCalculator.calculate_cost(
        "claude-opus-5-5", 0, 0, cache_creation_input_tokens=1000
    ) == pytest.approx(0.5)


def test_claude_fable_5_1_pricing():
    # Per https://platform.claude.com/docs/en/about-claude/pricing:
    # $10 input, $0.25 cache read (2.5% of input), $12.50 5m cache write,
    # $50 output per 1M tokens. Fable 5 cache reads stay at $1.
    assert _cost("claude-fable-5-1") == pytest.approx(6.0)
    assert _cost("claude-fable-5-1", cached=1000) == pytest.approx(6.025)
    assert _cost("claude-fable-5", cached=1000) == pytest.approx(6.1)
    assert CostCalculator.calculate_cost(
        "claude-fable-5-1", 0, 0, cache_creation_input_tokens=1000
    ) == pytest.approx(1.25)


@pytest.mark.parametrize("hour", [1, 2, 3, 6, 7, 8, 9])
@pytest.mark.parametrize("flash_model", DEEPSEEK_FLASH_MODELS)
def test_deepseek_peak_pricing(hour: int, flash_model: str):
    calculation_time = datetime(2026, 9, 10, hour, tzinfo=timezone.utc)

    # Per https://api-docs.deepseek.com/quick_start/pricing/ as of 2026-09-10.
    # Pro peak: $1.32 input / $0.044 cached / $3.96 output per 1M tokens.
    assert _cost(
        "deepseek-v4-pro", cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.5324)
    # V4.1 Flash peak: $0.30 input / $0.006 cached / $1.20 output per 1M tokens.
    assert _cost(
        flash_model, cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.1506)


@pytest.mark.parametrize("hour", [0, 4, 5, 10, 16, 23])
@pytest.mark.parametrize("flash_model", DEEPSEEK_FLASH_MODELS)
def test_deepseek_off_peak_pricing(hour: int, flash_model: str):
    calculation_time = datetime(2026, 9, 10, hour, tzinfo=timezone.utc)

    # Off-peak prices are half of peak prices.
    assert _cost(
        "deepseek-v4-pro", cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.2662)
    assert _cost(
        flash_model, cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.0753)


@pytest.mark.parametrize("day", [12, 13])
@pytest.mark.parametrize("hour", [1, 2, 3, 6, 7, 8, 9])
@pytest.mark.parametrize("flash_model", DEEPSEEK_FLASH_MODELS)
def test_deepseek_weekend_pricing_is_off_peak(day: int, hour: int, flash_model: str):
    calculation_time = datetime(2026, 9, day, hour, tzinfo=timezone.utc)

    assert _cost(
        "deepseek-v4-pro", cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.2662)
    assert _cost(
        flash_model, cached=1000, calculation_time=calculation_time
    ) == pytest.approx(0.0753)


@pytest.mark.parametrize("model", DEEPSEEK_FLASH_MODELS)
def test_deepseek_flash_names_have_explicit_pricing(model: str):
    assert model in MODEL_COSTS
    assert _find_match(model) == model
    assert CostCalculator.is_model_supported(model)


@pytest.mark.parametrize("model", DEEPSEEK_FLASH_MODELS)
@pytest.mark.parametrize(
    ("input_t", "output_t", "cached", "peak_cents"),
    [
        (1_000_000, 0, 0, 30.0),
        (0, 1_000_000, 0, 120.0),
        (0, 0, 1_000_000, 0.6),
        (0, 0, 0, 0.0),
    ],
)
@pytest.mark.parametrize(("hour", "multiplier"), [(1, 1), (4, 0.5)])
def test_deepseek_flash_token_rates(
    model: str,
    input_t: int,
    output_t: int,
    cached: int,
    peak_cents: float,
    hour: int,
    multiplier: float,
):
    # Check each published USD-per-million rate independently, including
    # cache-only requests and conversion to the calculator's cents unit.
    assert _cost(
        model,
        input_t,
        output_t,
        cached,
        calculation_time=datetime(2026, 9, 10, hour, tzinfo=timezone.utc),
    ) == pytest.approx(peak_cents * multiplier)


@pytest.mark.parametrize(
    ("calculation_time", "expected"),
    [
        (datetime(2026, 8, 28, 16, 30, tzinfo=timezone.utc), False),
        (datetime(2026, 8, 30, 16, 30, tzinfo=timezone.utc), True),
    ],
)
def test_deepseek_weekday_uses_beijing_calendar(
    monkeypatch: pytest.MonkeyPatch,
    calculation_time: datetime,
    expected: bool,
):
    # Use a hypothetical 16:00-17:00 UTC peak window to make the Beijing and
    # UTC weekday classifications observable at the UTC date boundary.
    monkeypatch.setattr(calculator, "_DEEPSEEK_PEAK_WINDOWS_UTC", ((16, 17),))

    assert _is_deepseek_peak_time(calculation_time) is expected


def test_deepseek_pricing_converts_calculation_time_to_utc():
    singapore = timezone(timedelta(hours=8))

    # 09:00 Singapore is 01:00 UTC and therefore peak time.
    assert _cost(
        "deepseek-v4-pro",
        calculation_time=datetime(2026, 8, 17, 9, tzinfo=singapore),
    ) == pytest.approx(0.528)


def test_is_model_supported():
    assert CostCalculator.is_model_supported("gpt-5-mini") is True
    assert CostCalculator.is_model_supported("gpt-5.4-mini") is True
    assert CostCalculator.is_model_supported("gpt-5.5") is True
    assert CostCalculator.is_model_supported("gpt-5.5-pro") is True
    assert CostCalculator.is_model_supported("gpt-5.6") is True
    assert CostCalculator.is_model_supported("gpt-5.6-sol") is True
    assert CostCalculator.is_model_supported("gpt-5.6-terra") is True
    assert CostCalculator.is_model_supported("gpt-5.6-luna") is True
    assert CostCalculator.is_model_supported("gpt-6-astra") is True
    assert CostCalculator.is_model_supported("gpt-6-sol") is True
    assert CostCalculator.is_model_supported("gpt-6-luna") is True
    assert CostCalculator.is_model_supported("claude-opus-5-5") is True
    assert CostCalculator.is_model_supported("claude-fable-5-1") is True
    assert CostCalculator.is_model_supported("gpt-5.9-mini") is True
    assert CostCalculator.is_model_supported("claude-sonnet-4-6") is True
    assert CostCalculator.is_model_supported("totally-made-up-xyz") is False


@pytest.mark.parametrize("model", sorted(MODEL_COSTS.keys()))
def test_calculator_matches_models_json(model: str) -> None:
    """Every entry in MODEL_COSTS should cost exactly what its dict says.

    Catches two regressions at once:
      - Typos or unit errors in defog/llm/cost/models.py (the dict keys
        must use the per-1k convention the calculator reads).
      - Matcher changes that accidentally route an exact model id to a
        different entry.
    """
    costs = MODEL_COSTS[model]
    input_t = 1_000
    output_t = 1_000
    cached_t = 1_000 if "cached_input_cost_per1k" in costs else 0
    cache_creation_t = 1_000 if "cache_creation_input_cost_per1k" in costs else 0

    expected_cents = (
        input_t / 1000 * costs["input_cost_per1k"]
        + output_t / 1000 * costs["output_cost_per1k"]
    ) * 100
    if cached_t:
        expected_cents += (cached_t / 1000 * costs["cached_input_cost_per1k"]) * 100
    if cache_creation_t:
        expected_cents += (
            cache_creation_t / 1000 * costs["cache_creation_input_cost_per1k"]
        ) * 100

    actual = CostCalculator.calculate_cost(
        model=model,
        input_tokens=input_t,
        output_tokens=output_t,
        cached_input_tokens=cached_t,
        cache_creation_input_tokens=cache_creation_t,
        calculation_time=datetime(2026, 8, 17, 1, tzinfo=timezone.utc),
    )
    assert actual == pytest.approx(expected_cents), (
        f"calculate_cost for {model!r} disagreed with its MODEL_COSTS entry"
    )
