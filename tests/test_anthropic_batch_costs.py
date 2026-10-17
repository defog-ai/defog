"""Offline tests for Anthropic Message Batches prices.

Anthropic's Batch discount is 50% of every standard rate, and it stacks with
the prompt-caching rates. Token counts of 100,000 keep Claude Haiku 5.5 at
its standard (not long-prompt) rates. Each result in cents is then ten times
the published price in USD per million tokens.
"""

import pytest

from defog.llm.cost import CostCalculator

N = 100_000


@pytest.mark.parametrize(
    "tokens,standard,batch",
    [
        # Input: USD 0.10 -> 0.05 per million tokens.
        ({"input_tokens": N, "output_tokens": 0}, 1.0, 0.5),
        # Output: USD 0.50 -> 0.25 per million tokens.
        ({"input_tokens": 0, "output_tokens": N}, 5.0, 2.5),
        # Cache read: USD 0.01 -> 0.005 per million tokens.
        (
            {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": N},
            0.1,
            0.05,
        ),
        # 5-minute cache write: USD 0.125 -> 0.0625 per million tokens.
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": N,
            },
            1.25,
            0.625,
        ),
        # 1-hour cache write: USD 0.20 -> 0.10 per million tokens.
        (
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": N,
                "cache_creation_1h_input_tokens": N,
            },
            2.0,
            1.0,
        ),
    ],
)
def test_haiku_5_5_batch_halves_each_rate(tokens, standard, batch):
    assert CostCalculator.calculate_cost("claude-haiku-5-5", **tokens) == pytest.approx(
        standard
    )
    assert CostCalculator.calculate_cost(
        "claude-haiku-5-5", **tokens, batch=True
    ) == pytest.approx(batch)


def test_haiku_5_5_long_prompt_batch_is_half_of_long_prompt_price():
    # A prompt of more than 100,000 tokens pays the long-prompt rates on every
    # token: USD 0.50 input and 2.50 output per million. Batch halves them.
    assert CostCalculator.calculate_cost(
        "claude-haiku-5-5", 150_000, 1_000
    ) == pytest.approx(7.75)
    assert CostCalculator.calculate_cost(
        "claude-haiku-5-5", 150_000, 1_000, batch=True
    ) == pytest.approx(3.875)


def test_long_prompt_limit_counts_cached_tokens_in_batch():
    # 60,000 uncached + 50,000 cache-read tokens is a 110,000-token prompt,
    # so the long-prompt rates apply, then the Batch discount.
    standard = CostCalculator.calculate_cost("claude-haiku-5-5", 60_000, 1_000, 50_000)
    assert standard == pytest.approx(3.0 + 0.25 + 0.25)
    assert CostCalculator.calculate_cost(
        "claude-haiku-5-5", 60_000, 1_000, 50_000, batch=True
    ) == pytest.approx(standard / 2)


def test_other_claude_models_get_the_batch_price():
    # Claude Sonnet 4.6: USD 3 input, 0.30 cache read, 15 output per million.
    assert CostCalculator.calculate_cost(
        "claude-sonnet-4-6", 1000, 500, 1000, batch=True
    ) == pytest.approx(0.54)


def test_claude_flex_tier_keeps_the_standard_price():
    assert CostCalculator.calculate_cost(
        "claude-haiku-5-5", N, N, service_tier="flex"
    ) == pytest.approx(6.0)


# Values computed on the code before the Anthropic Batch price was added. A
# call without batch=True must return exactly these numbers.
@pytest.mark.parametrize(
    "model,args,kwargs,expected",
    [
        ("gpt-5-mini", (1000, 500, 1000), {}, 0.1275),
        ("gpt-5-mini", (1000, 500, 1000), {"batch": True}, 0.06375),
        ("claude-sonnet-4-6", (1000, 500, 1000), {}, 1.08),
        ("claude-haiku-5-5", (1000, 500, 1000, 200), {}, 0.0385),
        (
            "claude-haiku-5-5",
            (1000, 500, 1000, 200),
            {"cache_creation_1h_input_tokens": 100},
            0.03925,
        ),
        ("claude-haiku-5-5", (150_000, 1_000), {}, 7.75),
    ],
)
def test_prices_without_anthropic_batch_are_unchanged(model, args, kwargs, expected):
    assert CostCalculator.calculate_cost(model, *args, **kwargs) == pytest.approx(
        expected, rel=1e-12
    )
