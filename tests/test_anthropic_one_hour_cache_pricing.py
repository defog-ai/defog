"""Offline tests for pricing writes to Anthropic's 1-hour prompt cache.

Rates come from https://platform.claude.com/docs/en/about-claude/pricing
(checked 2026-10-08): a 1-hour cache write costs 2x the base input price,
a 5-minute cache write 1.25x. No test calls a live API.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from defog.llm.cost.calculator import CostCalculator
from defog.llm.cost.models import MODEL_COSTS
from defog.llm.providers.anthropic_provider import AnthropicProvider

HAIKU_55 = "claude-haiku-5-5"
M = 1_000_000


def _cents(usd_per_million_by_count):
    """Sum of tokens x USD per million tokens, in cents."""
    return sum(n * rate for n, rate in usd_per_million_by_count) / M * 100


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


def test_one_hour_writes_use_the_one_hour_rate():
    cost = CostCalculator.calculate_cost(
        HAIKU_55, 0, 0, 0, 50_000, cache_creation_1h_input_tokens=50_000
    )
    assert cost == pytest.approx(_cents([(50_000, 0.20)]))


def test_mixed_writes_split_between_the_two_rates():
    # Opus 5.5: 5-minute writes $5, 1-hour writes $8 per million.
    cost = CostCalculator.calculate_cost(
        "claude-opus-5-5", 0, 0, 0, 1_000_000, cache_creation_1h_input_tokens=400_000
    )
    assert cost == pytest.approx(_cents([(600_000, 5.0), (400_000, 8.0)]))


@pytest.mark.parametrize("one_hour", [None, 0])
def test_without_a_split_all_writes_use_the_five_minute_rate(one_hour):
    expected = _cents([(1_000, 4.0), (500, 20.0), (2_000, 0.20), (3_000, 5.0)])
    assert CostCalculator.calculate_cost(
        "claude-opus-5-5", 1_000, 500, 2_000, 3_000
    ) == pytest.approx(expected)
    assert CostCalculator.calculate_cost(
        "claude-opus-5-5",
        1_000,
        500,
        2_000,
        3_000,
        cache_creation_1h_input_tokens=one_hour,
    ) == pytest.approx(expected)


def test_one_hour_count_is_capped_at_the_total_writes():
    cost = CostCalculator.calculate_cost(
        "claude-opus-5-5", 0, 0, 0, 1_000, cache_creation_1h_input_tokens=5_000
    )
    assert cost == pytest.approx(_cents([(1_000, 8.0)]))


def test_long_haiku_prompt_uses_the_long_one_hour_rate():
    # Over 100,000 prompt tokens: input $0.50, output $2.50, read $0.05,
    # 5-minute write $0.625, 1-hour write $1.00 per million.
    cost = CostCalculator.calculate_cost(
        HAIKU_55,
        20_000,
        1_000,
        30_000,
        60_000,
        cache_creation_1h_input_tokens=40_000,
    )
    assert cost == pytest.approx(
        _cents(
            [
                (20_000, 0.50),
                (1_000, 2.50),
                (30_000, 0.05),
                (20_000, 0.625),
                (40_000, 1.00),
            ]
        )
    )


def test_short_haiku_prompt_uses_the_short_one_hour_rate():
    cost = CostCalculator.calculate_cost(
        HAIKU_55, 20_000, 1_000, 30_000, 50_000, cache_creation_1h_input_tokens=40_000
    )
    assert cost == pytest.approx(
        _cents(
            [
                (20_000, 0.10),
                (1_000, 0.50),
                (30_000, 0.01),
                (10_000, 0.125),
                (40_000, 0.20),
            ]
        )
    )


def test_model_without_a_one_hour_rate_keeps_the_five_minute_rate():
    rates = MODEL_COSTS["claude-3-haiku"]
    assert "cache_creation_1h_input_cost_per1k" not in rates
    cost = CostCalculator.calculate_cost(
        "claude-3-haiku", 0, 0, 0, 1_000, cache_creation_1h_input_tokens=1_000
    )
    assert cost == pytest.approx(rates["cache_creation_input_cost_per1k"] * 100)


def test_every_listed_one_hour_rate_is_twice_the_input_rate():
    checked = 0
    for model, rates in MODEL_COSTS.items():
        for table in (rates, rates.get("long_prompt") or {}):
            if "cache_creation_1h_input_cost_per1k" in table:
                assert table["cache_creation_1h_input_cost_per1k"] == pytest.approx(
                    2 * table["input_cost_per1k"]
                ), model
                checked += 1
    assert checked == 17  # 16 Claude rows plus Haiku 5.5's long-prompt rates


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.4", "gemini-2.5-flash"])
def test_other_providers_are_unchanged(model):
    before = CostCalculator.calculate_cost(model, 150_000, 1_000, 10_000)
    after = CostCalculator.calculate_cost(
        model, 150_000, 1_000, 10_000, cache_creation_1h_input_tokens=5_000
    )
    assert after == pytest.approx(before)


def test_requests_cost_passes_the_one_hour_count_through():
    cost = CostCalculator.calculate_requests_cost(
        "claude-opus-5-5",
        [
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 1_000,
                "cache_creation_1h_input_tokens": 1_000,
            },
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 1_000,
            },
        ],
    )
    assert cost == pytest.approx(_cents([(1_000, 8.0), (1_000, 5.0)]))


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


def _usage(input_t, output_t=0, read_t=0, write_t=0, cache_creation=None):
    return SimpleNamespace(
        input_tokens=input_t,
        output_tokens=output_t,
        cache_read_input_tokens=read_t,
        cache_creation_input_tokens=write_t,
        cache_creation=cache_creation,
    )


def _split(five_minute, one_hour):
    return SimpleNamespace(
        ephemeral_5m_input_tokens=five_minute, ephemeral_1h_input_tokens=one_hour
    )


def _response(usage):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        usage=usage,
        container=None,
        model=HAIKU_55,
        id="msg_01",
    )


async def _process(usage):
    request_usages = []
    await AnthropicProvider(api_key="sk-test").process_response(
        client=AsyncMock(),
        response=_response(usage),
        request_params={"messages": [], "model": HAIKU_55},
        tools=None,
        tool_dict={},
        request_usages=request_usages,
    )
    return request_usages


@pytest.mark.asyncio
async def test_provider_records_the_one_hour_count():
    usages = await _process(_usage(1_000, 500, 2_000, 30_000, _split(10_000, 20_000)))
    assert usages == [
        {
            "input_tokens": 1_000,
            "output_tokens": 500,
            "cached_input_tokens": 2_000,
            "cache_creation_input_tokens": 30_000,
            "cache_creation_1h_input_tokens": 20_000,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [None, _split(30_000, 0)])
async def test_provider_without_one_hour_writes_records_no_split(split):
    usages = await _process(_usage(1_000, 500, 2_000, 30_000, split))
    assert "cache_creation_1h_input_tokens" not in usages[0]


@pytest.mark.asyncio
async def test_provider_ignores_a_mock_usage_object():
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 20
    usage.cache_creation_input_tokens = 30
    usages = await _process(usage)
    assert "cache_creation_1h_input_tokens" not in usages[0]


@pytest.mark.asyncio
async def test_execute_chat_reports_the_one_hour_price():
    response = _response(_usage(1_000, 500, 2_000, 30_000, _split(10_000, 20_000)))

    class FakeAsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = SimpleNamespace(create=AsyncMock(return_value=response))

    with patch(
        "defog.llm.providers.anthropic_provider.AsyncAnthropic", FakeAsyncAnthropic
    ):
        result = await AnthropicProvider(api_key="sk-test").execute_chat(
            messages=[{"role": "user", "content": "hi"}], model=HAIKU_55
        )

    assert result.cache_creation_input_tokens == 30_000
    assert result.cost_in_cents == pytest.approx(
        _cents(
            [
                (1_000, 0.10),
                (500, 0.50),
                (2_000, 0.01),
                (10_000, 0.125),
                (20_000, 0.20),
            ]
        )
    )
