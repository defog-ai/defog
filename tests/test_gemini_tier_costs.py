"""Offline tests for Gemini Batch, Flex and Priority token costs."""

from datetime import datetime, timezone

import pytest

from defog.llm.cost import CostCalculator

MILLION = 1_000_000
BEFORE = datetime(2026, 11, 13, tzinfo=timezone.utc)


def _usd(model, **kwargs):
    kwargs.setdefault("calculation_time", BEFORE)
    cents = CostCalculator.calculate_cost(model, **kwargs)
    return None if cents is None else cents / 100


def _prices(model, **kwargs):
    """(input, cached input, output) in USD per 1M tokens."""
    return (
        _usd(model, input_tokens=MILLION, output_tokens=0, **kwargs),
        _usd(
            model,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=MILLION,
            prompt_tokens=1,
            **kwargs,
        ),
        _usd(model, input_tokens=0, output_tokens=MILLION, **kwargs),
    )


# Short prompts only: one request of 1M input tokens is over the 200,000
# token limit of the Pro models, so those are tested separately below.
@pytest.mark.parametrize(
    "model,batch_usd,priority_usd",
    [
        ("gemini-4-carbon", (1, 0.05, 5), (3.60, 0.18, 18)),
        ("gemini-4-argon", (1, 0.05, 5), (3.60, 0.18, 18)),
        ("gemini-4-hydrogen", (0.75, 0.0375, 3.75), (2.70, 0.135, 13.50)),
        ("gemini-3.8-flash", (0.375, 0.0375, 1.875), (1.35, 0.135, 6.75)),
        ("gemini-3.6-flash", (0.375, 0.0375, 1.875), (1.35, 0.135, 6.75)),
        ("gemini-3.1-flash-lite", (0.125, 0.0125, 0.75), (0.45, 0.045, 2.70)),
        ("gemini-3-flash-preview", (0.25, 0.05, 1.50), (0.90, 0.09, 5.40)),
        ("gemini-2.5-flash", (0.15, 0.03, 1.25), (0.54, 0.054, 4.50)),
    ],
)
def test_tier_prices_per_million_tokens(model, batch_usd, priority_usd):
    assert _prices(model, batch=True) == pytest.approx(batch_usd)
    assert _prices(model, service_tier="flex") == pytest.approx(batch_usd)
    assert _prices(model, service_tier="priority") == pytest.approx(priority_usd)


def test_flash_lite_3_5_cached_price_differs_by_tier():
    model = "gemini-3.5-flash-lite"
    assert _prices(model, batch=True) == pytest.approx((0.15, 0.015, 1.25))
    assert _prices(model, service_tier="flex") == pytest.approx((0.15, 0.02, 1.25))
    assert _prices(model, service_tier="priority") == pytest.approx((0.54, 0.05, 4.50))


@pytest.mark.parametrize(
    "model,kwargs,short_usd,long_usd",
    [
        # (input, cached input, output) per 1M, at or under and over 200k.
        ("gemini-3.1-pro-preview", dict(batch=True), (1, 0.20, 6), (2, 0.40, 9)),
        (
            "gemini-3.1-pro-preview",
            dict(service_tier="flex"),
            (1, 0.20, 6),
            (2, 0.40, 9),
        ),
        (
            "gemini-3.1-pro-preview",
            dict(service_tier="priority"),
            (3.60, 0.36, 21.60),
            (7.20, 0.72, 32.40),
        ),
        ("gemini-2.5-pro", dict(batch=True), (0.625, 0.125, 5), (1.25, 0.25, 7.50)),
        (
            "gemini-2.5-pro",
            dict(service_tier="priority"),
            (2.25, 0.225, 18),
            (4.50, 0.45, 27),
        ),
    ],
)
def test_pro_tier_prices_by_prompt_length(model, kwargs, short_usd, long_usd):
    tokens = dict(input_tokens=100_000, cached_input_tokens=50_000, output_tokens=1_000)
    for prompt_tokens, (input_usd, cached_usd, output_usd) in (
        (200_000, short_usd),
        (200_001, long_usd),
    ):
        expected = (100_000 * input_usd + 50_000 * cached_usd + 1_000 * output_usd) / (
            MILLION
        )
        assert _usd(
            model, prompt_tokens=prompt_tokens, **tokens, **kwargs
        ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "model,change_date",
    [
        ("gemini-4-carbon", datetime(2026, 12, 1, tzinfo=timezone.utc)),
        ("gemini-4-argon", datetime(2026, 12, 1, tzinfo=timezone.utc)),
        ("gemini-3.8-flash", datetime(2027, 1, 1, tzinfo=timezone.utc)),
        ("gemini-3.6-flash", datetime(2027, 1, 1, tzinfo=timezone.utc)),
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [dict(batch=True), dict(service_tier="flex"), dict(service_tier="priority")],
)
def test_tier_prices_double_on_the_change_date(model, change_date, kwargs):
    before = _prices(model, **kwargs)
    after = _prices(model, calculation_time=change_date, **kwargs)
    assert after == pytest.approx(tuple(2 * price for price in before))


def test_hydrogen_tier_prices_do_not_change():
    late = datetime(2027, 6, 1, tzinfo=timezone.utc)
    for kwargs in (dict(batch=True), dict(service_tier="priority")):
        assert _prices("gemini-4-hydrogen", calculation_time=late, **kwargs) == _prices(
            "gemini-4-hydrogen", **kwargs
        )


def test_batch_takes_precedence_over_service_tier():
    assert _prices(
        "gemini-4-hydrogen", batch=True, service_tier="priority"
    ) == pytest.approx((0.75, 0.0375, 3.75))


@pytest.mark.parametrize("tier", [None, "default", "standard", "auto"])
def test_other_service_tiers_use_the_standard_price(tier):
    assert _prices("gemini-4-hydrogen", service_tier=tier) == pytest.approx(
        (1.50, 0.075, 7.50)
    )


@pytest.mark.parametrize(
    "model",
    [
        # Not on Google's pricing page.
        "gemini-3.5-flash",
        "gemini-3-pro-preview",
        "gemini-2.0-flash",
        # Resolve to another model's standard price through a loose match.
        "gemini-2.5-flash-lite",
        "gemini-4-hydrogen-image",
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [dict(batch=True), dict(service_tier="flex"), dict(service_tier="priority")],
)
def test_unpublished_tier_price_returns_none(model, kwargs):
    assert _usd(model, input_tokens=1000, output_tokens=1000, **kwargs) is None


def test_tiers_do_not_change_other_providers():
    for model in ("claude-sonnet-4-6", "deepseek-v4-flash"):
        tokens = dict(input_tokens=1000, output_tokens=500)
        assert _usd(model, service_tier="priority", **tokens) == _usd(model, **tokens)
