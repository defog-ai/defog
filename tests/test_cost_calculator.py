"""Tests for defog.llm.cost.calculator model matching and pricing."""

import pytest

from defog.llm.cost.calculator import CostCalculator, _find_match
from defog.llm.cost.models import MODEL_COSTS


def _cost(model: str, input_t: int = 1000, output_t: int = 1000, cached: int = 0):
    return CostCalculator.calculate_cost(
        model=model,
        input_tokens=input_t,
        output_tokens=output_t,
        cached_input_tokens=cached,
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


def test_is_model_supported():
    assert CostCalculator.is_model_supported("gpt-5-mini") is True
    assert CostCalculator.is_model_supported("gpt-5.4-mini") is True
    assert CostCalculator.is_model_supported("gpt-5.9-mini") is True
    assert CostCalculator.is_model_supported("claude-sonnet-4-6") is True
    assert CostCalculator.is_model_supported("totally-made-up-xyz") is False


def test_print_overcharge_table(capsys):
    """Print the before/after table for the gpt-5.4 fallback regression.

    Run with ``pytest -s`` to see the output unconditionally. The asserts at
    the end make sure the magnitudes we quote in the PR description stay
    accurate if someone touches pricing later.
    """
    gpt5 = MODEL_COSTS["gpt-5"]

    def tokens_cost(costs: dict, input_t: int, output_t: int) -> float:
        return (
            input_t / 1000 * costs["input_cost_per1k"]
            + output_t / 1000 * costs["output_cost_per1k"]
        ) * 100

    # 1k input + 1k output is the size used throughout this file.
    input_t = output_t = 1000

    rows = []
    for model_id in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"):
        actual = tokens_cost(MODEL_COSTS[model_id], input_t, output_t)
        # Old behaviour: longest-substring fallback picked `gpt-5` (since
        # `gpt-5-mini`/`gpt-5-nano` aren't substrings of `gpt-5.4-*`).
        old_logged = tokens_cost(gpt5, input_t, output_t)
        ratio = old_logged / actual if actual else float("inf")
        rows.append((model_id, old_logged, actual, ratio))

    with capsys.disabled():
        print()
        print("Fallback-matcher overcharge for gpt-5.4-* (1k input + 1k output):")
        print(f"  {'model':<16} {'old (gpt-5 tier)':>20} {'new (actual)':>16} {'ratio':>10}")
        for model_id, old_logged, actual, ratio in rows:
            direction = "over" if ratio > 1 else "under"
            print(
                f"  {model_id:<16} {old_logged:>18.4f}¢ {actual:>14.4f}¢ "
                f"{ratio:>8.2f}x {direction}"
            )

    by_model = {row[0]: row for row in rows}
    # gpt-5.4 is slightly under-billed at gpt-5 pricing (gpt-5.4 is the
    # more expensive model). The others are over-billed substantially.
    assert by_model["gpt-5.4"][3] < 1.0
    assert by_model["gpt-5.4-mini"][3] > 2.0
    assert by_model["gpt-5.4-nano"][3] > 7.0
