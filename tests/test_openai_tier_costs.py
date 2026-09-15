"""Offline acceptance tests for OpenAI token costs by processing tier."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from defog.llm.cost import CostCalculator
from defog.llm.exceptions import PauseToolExecution
from defog.llm.providers.openai_provider import OpenAIProvider


@pytest.mark.parametrize(
    "tier,batch,expected",
    [
        (None, False, 0.1275),
        ("default", False, 0.1275),
        ("auto", False, 0.1275),
        ("flex", False, 0.06375),
        (None, True, 0.06375),
        ("flex", True, 0.06375),
    ],
)
def test_calculator_tiers(tier, batch, expected):
    assert CostCalculator.calculate_cost(
        "gpt-5-mini", 1000, 500, 1000, service_tier=tier, batch=batch
    ) == pytest.approx(expected)


@pytest.mark.parametrize("tier,batch", [("flex", False), (None, True)])
@pytest.mark.parametrize(
    "tokens,expected",
    [
        ((1000, 0, 0), 0.0125),
        ((0, 0, 1000), 0.00125),
        ((0, 1000, 0), 0.1),
    ],
)
def test_each_token_rate(tier, batch, tokens, expected):
    assert CostCalculator.calculate_cost(
        "gpt-5-mini-2025-08-07", *tokens, service_tier=tier, batch=batch
    ) == pytest.approx(expected)


def test_batch_without_cache_rate_bills_all_input():
    assert CostCalculator.calculate_cost(
        "gpt-4o", 1000, 500, 1000, batch=True
    ) == pytest.approx(0.5)


def test_flex_uses_published_rates_even_if_standard_table_is_old():
    assert CostCalculator.calculate_cost(
        "o3", 1000, 500, 1000, service_tier="flex"
    ) == pytest.approx(0.325)


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "gemini-3.1-pro"])
def test_other_provider_costs_are_unchanged(model):
    original = CostCalculator.calculate_cost(model, 1000, 500, 1000)
    assert (
        CostCalculator.calculate_cost(
            model, 1000, 500, 1000, service_tier="flex", batch=True
        )
        == original
    )


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.3-chat-latest", "gpt-5-unknown"])
def test_unlisted_flex_prices_are_not_guessed(model):
    assert CostCalculator.calculate_cost(model, 1000, 500, service_tier="flex") is None


# Requests above 272,000 total input tokens use OpenAI's long-context rates on
# every tier. Amounts are cents for gpt-5.6-luna unless stated otherwise.
@pytest.mark.parametrize(
    "tokens,tier,batch,expected",
    [
        ((300_000, 0, 1000), None, False, 12.18),
        ((300_000, 0, 1000), "default", False, 12.18),
        ((300_000, 0, 1000), "flex", False, 6.09),
        ((300_000, 0, 1000), None, True, 6.09),
        # Zero output tokens: the input alone is priced at the long rate.
        ((300_000, 0, 0), None, False, 12.0),
        ((300_000, 0, 0), "flex", False, 6.0),
        # Cached input counts towards the threshold and uses the long cached rate.
        ((200_000, 100_000, 1000), None, False, 8.58),
        ((200_000, 100_000, 1000), "flex", False, 4.29),
        # Exactly 272,000 total input tokens is still the short-context price.
        ((272_000, 0, 1000), None, False, 5.56),
        ((172_000, 100_000, 1000), None, False, 3.76),
        ((272_001, 0, 0), None, False, 10.88004),
    ],
)
def test_long_context_prices(tokens, tier, batch, expected):
    input_tokens, cached_tokens, output_tokens = tokens
    assert CostCalculator.calculate_cost(
        "gpt-5.6-luna",
        input_tokens,
        output_tokens,
        cached_tokens,
        service_tier=tier,
        batch=batch,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.6-luna-2026-05-01", 12.18),
        ("gpt-5.6", 243.0),
        ("gpt-5.6-sol", 243.0),
        ("gpt-5.6-terra", 121.8),
        ("gpt-5.5", 304.5),
        ("gpt-5.4", 152.25),
    ],
)
def test_long_context_standard_prices_by_model(model, expected):
    assert CostCalculator.calculate_cost(model, 300_000, 1000) == pytest.approx(
        expected
    )


def test_long_context_pro_model_bills_cached_input_at_the_input_rate():
    assert CostCalculator.calculate_cost(
        "gpt-5.4-pro", 250_000, 0, 50_000
    ) == pytest.approx(1800.0)
    assert CostCalculator.calculate_cost(
        "gpt-5.4-pro", 250_000, 0, 50_000, batch=True
    ) == pytest.approx(900.0)


@pytest.mark.parametrize("model", ["gpt-5.5-pro", "gpt-5.4-pro"])
def test_pro_models_bill_cached_input_at_the_input_rate_on_every_tier(model):
    """These models publish no cached input discount: 10,000 uncached and
    200,000 cached input tokens are 210,000 input tokens at $30/1M, plus
    5,000 output tokens at $180/1M, on the standard tier; Flex halves it."""
    assert CostCalculator.calculate_cost(model, 10_000, 5000, 200_000) == pytest.approx(
        720.0
    )
    assert CostCalculator.calculate_cost(
        model, 10_000, 5000, 200_000, service_tier="flex"
    ) == pytest.approx(360.0)
    assert CostCalculator.calculate_cost(
        model, 10_000, 5000, 200_000, batch=True
    ) == pytest.approx(360.0)


def test_gpt_5_5_pro_keeps_its_published_rate_above_the_threshold():
    # Its model page publishes no long-context price, unlike gpt-5.4-pro.
    assert CostCalculator.calculate_cost("gpt-5.5-pro", 300_000, 1000) == pytest.approx(
        918.0
    )
    assert CostCalculator.calculate_cost(
        "gpt-5.5-pro", 300_000, 1000, batch=True
    ) == pytest.approx(459.0)


@pytest.mark.parametrize(
    "model", ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1", "gpt-5.5-pro"]
)
def test_models_without_a_long_context_price_keep_the_standard_price(model):
    # The price stays linear in the token counts: no rate change above 272,000.
    assert CostCalculator.calculate_cost(model, 300_000, 1000) == pytest.approx(
        2 * CostCalculator.calculate_cost(model, 150_000, 500)
    )


def test_long_context_leaves_other_providers_unchanged():
    per_token = CostCalculator.calculate_cost("claude-sonnet-4-6", 1000, 0) / 1000
    assert CostCalculator.calculate_cost(
        "claude-sonnet-4-6", 300_000, 0
    ) == pytest.approx(per_token * 300_000)


def response(tier="flex", tool=None, response_id="response-1"):
    result = SimpleNamespace(
        id=response_id,
        output=[]
        if tool is None
        else [
            SimpleNamespace(
                type="function_call",
                name=tool.__name__,
                arguments="{}",
                call_id=response_id,
            )
        ],
        output_text="done",
        output_parsed={"answer": "done"},
        usage=SimpleNamespace(
            input_tokens=2000,
            output_tokens=500,
            input_tokens_details=SimpleNamespace(cached_tokens=1000),
        ),
    )
    if tier != "missing":
        result.service_tier = tier
    return result


@pytest.fixture
def sdk(monkeypatch):
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(), parse=AsyncMock())
    )
    context = AsyncMock()
    context.__aenter__.return_value = client
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: context)
    return client.responses


def run_chat(**kwargs):
    return OpenAIProvider(api_key="test-only").execute_chat(
        model="gpt-5-mini", messages=[{"role": "user", "content": "Hello"}], **kwargs
    )


@pytest.mark.parametrize(
    "tier,flex,expected",
    [
        ("flex", True, 0.06375),
        ("flex", False, 0.06375),
        ("default", True, 0.1275),
        ("auto", True, 0.1275),
        ("missing", True, 0.06375),
        (None, True, 0.06375),
        ("missing", False, 0.1275),
    ],
)
def test_completed_call_uses_actual_tier(sdk, tier, flex, expected):
    sdk.create.return_value = response(tier)
    result = asyncio.run(run_chat(flex_processing=flex))
    assert result.cost_in_cents == pytest.approx(expected)
    assert result.list_cost_in_cents == pytest.approx(0.1275)
    assert (
        result.service_tier
        == {
            "flex": "flex",
            "default": "default",
            "auto": "auto",
            "missing": "flex" if flex else "default",
            None: "flex" if flex else "default",
        }[tier]
    )
    assert (result.input_tokens, result.cached_input_tokens, result.output_tokens) == (
        1000,
        1000,
        500,
    )
    assert sdk.create.call_args.kwargs.get("service_tier") == ("flex" if flex else None)


def lookup():
    """Return a fixed tool result."""
    return "found"


def ask_user():
    """Ask for user input."""
    raise PauseToolExecution({"question": "Continue?"})


@pytest.mark.parametrize("budget", [None, {"lookup": 1}])
def test_mixed_tool_turns_count_each_response_once(sdk, budget):
    sdk.create.side_effect = [
        response("flex", lookup),
        response("default", response_id="response-2"),
    ]
    result = asyncio.run(
        run_chat(tools=[lookup], tool_budget=budget, flex_processing=True)
    )
    assert result.cost_in_cents == pytest.approx(0.19125)
    assert result.list_cost_in_cents == pytest.approx(0.255)
    assert result.service_tier == "mixed"
    assert (result.input_tokens, result.cached_input_tokens, result.output_tokens) == (
        2000,
        2000,
        1000,
    )


def test_pause_and_resume_costs(sdk):
    sdk.create.side_effect = [
        response("default", lookup),
        response("flex", ask_user, "response-2"),
        response("default", response_id="response-3"),
    ]
    paused = asyncio.run(run_chat(tools=[lookup, ask_user], flex_processing=True))
    assert paused.status == "paused"
    assert paused.cost_in_cents == pytest.approx(0.19125)
    assert paused.list_cost_in_cents == pytest.approx(0.255)
    assert paused.service_tier == "mixed"
    assert (paused.input_tokens, paused.cached_input_tokens, paused.output_tokens) == (
        2000,
        2000,
        1000,
    )
    resumed = asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-5-mini",
            messages=paused.messages,
            previous_response_id=paused.response_id,
            resume_tool_results={paused.pending_tool_use["id"]: "yes"},
            flex_processing=True,
        )
    )
    assert resumed.cost_in_cents == pytest.approx(0.1275)
    assert resumed.list_cost_in_cents == pytest.approx(0.1275)
    assert resumed.service_tier == "default"
    assert paused.cost_in_cents + resumed.cost_in_cents == pytest.approx(0.31875)
    assert sdk.create.call_args.kwargs["previous_response_id"] == "response-2"


def test_long_context_threshold_applies_to_each_request(sdk):
    """Two 200,000-token requests are short-context requests; their summed
    400,000 tokens must not be priced at the long-context rate."""
    first = response("default", lookup)
    second = response("default", response_id="response-2")
    for item in (first, second):
        item.usage = SimpleNamespace(
            input_tokens=200_000,
            output_tokens=1000,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        )
    sdk.create.side_effect = [first, second]
    result = asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[lookup],
        )
    )
    assert (result.input_tokens, result.output_tokens) == (400_000, 2000)
    # 2 x (200,000 x $0.20/1M + 1,000 x $1.20/1M) = 8.24 cents; the summed
    # tokens at long-context rates would be 16.36 cents.
    assert result.cost_in_cents == pytest.approx(8.24)
    assert result.list_cost_in_cents == pytest.approx(8.24)
    assert result.service_tier == "default"


def luna_request(uncached, cached, output, tier, tool=None, response_id="response-1"):
    item = response(tier, tool, response_id)
    item.usage = SimpleNamespace(
        input_tokens=uncached + cached,
        output_tokens=output,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )
    return item


def run_luna(sdk, requests, **kwargs):
    sdk.create.side_effect = requests
    return asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hello"}],
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "second_tier,paid,served",
    [("flex", 4.12, "flex"), ("default", 6.18, "mixed")],
)
def test_list_price_and_served_tier_add_across_short_requests(
    sdk, second_tier, paid, served
):
    """Two Flex requests of 200,000 tokens: paid 4.12, list 8.24. When the
    second request is served on the default tier the amount paid is 6.18,
    the list price is unchanged and the tier is mixed. Repricing the summed
    400,000/2,000 tokens would wrongly give a 16.36-cent list price."""
    result = run_luna(
        sdk,
        [
            luna_request(200_000, 0, 1000, "flex", lookup),
            luna_request(200_000, 0, 1000, second_tier, response_id="response-2"),
        ],
        tools=[lookup],
        flex_processing=True,
    )
    assert result.cost_in_cents == pytest.approx(paid)
    assert result.list_cost_in_cents == pytest.approx(8.24)
    assert result.service_tier == served


@pytest.mark.parametrize(
    "uncached,cached,output,standard,flex",
    [
        (300_000, 0, 1000, 12.18, 6.09),
        (100_000, 200_000, 1000, 4.98, 2.49),
        (300_000, 0, 0, 12.0, 6.0),
        (272_000, 0, 1000, 5.56, 2.78),
        (272_000, 1, 1000, 11.060004, 5.530002),
    ],
)
@pytest.mark.parametrize("tier", ["default", "flex"])
def test_list_price_is_the_standard_price_of_one_request(
    sdk, uncached, cached, output, standard, flex, tier
):
    result = run_luna(
        sdk, [luna_request(uncached, cached, output, tier)], flex_processing=True
    )
    assert result.cost_in_cents == pytest.approx(
        standard if tier == "default" else flex
    )
    assert result.list_cost_in_cents == pytest.approx(standard)
    assert result.service_tier == tier


@pytest.mark.parametrize("tier,paid", [("default", 720.0), ("flex", 360.0)])
def test_pro_model_list_price_includes_cached_input(sdk, tier, paid):
    item = luna_request(10_000, 200_000, 5000, tier)
    sdk.create.side_effect = [item]
    result = asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-5.5-pro",
            messages=[{"role": "user", "content": "Hello"}],
            flex_processing=True,
        )
    )
    assert (result.input_tokens, result.cached_input_tokens) == (10_000, 200_000)
    assert result.cost_in_cents == pytest.approx(paid)
    assert result.list_cost_in_cents == pytest.approx(720.0)
    assert result.service_tier == tier


def test_returned_tier_has_priority_over_the_requested_tier(sdk):
    result = run_luna(sdk, [luna_request(300_000, 0, 0, "flex")], flex_processing=False)
    assert result.cost_in_cents == pytest.approx(6.0)
    assert result.list_cost_in_cents == pytest.approx(12.0)
    assert result.service_tier == "flex"


def test_paused_response_reports_list_price_and_tier_of_each_request(sdk):
    paused = run_luna(
        sdk,
        [
            luna_request(200_000, 0, 1000, "flex", lookup),
            luna_request(200_000, 0, 1000, "flex", ask_user, "response-2"),
        ],
        tools=[lookup, ask_user],
        flex_processing=True,
    )
    assert paused.status == "paused"
    assert paused.cost_in_cents == pytest.approx(4.12)
    assert paused.list_cost_in_cents == pytest.approx(8.24)
    assert paused.service_tier == "flex"


def test_one_long_context_request_is_priced_at_the_long_rate(sdk):
    only = response("default")
    only.usage = SimpleNamespace(
        input_tokens=300_000,
        output_tokens=1000,
        input_tokens_details=SimpleNamespace(cached_tokens=100_000),
    )
    sdk.create.return_value = only
    result = asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-5.6-luna",
            messages=[{"role": "user", "content": "Hello"}],
        )
    )
    assert (result.input_tokens, result.cached_input_tokens) == (200_000, 100_000)
    assert result.cost_in_cents == pytest.approx(8.58)


class Answer(BaseModel):
    answer: str


def test_structured_completion_after_tools_is_included(sdk):
    sdk.create.side_effect = [
        response("flex", lookup),
        response("flex", response_id="response-2"),
    ]
    sdk.parse.return_value = response("default", response_id="response-3")
    result = asyncio.run(
        run_chat(tools=[lookup], response_format=Answer, flex_processing=True)
    )
    assert result.cost_in_cents == pytest.approx(0.255)
    assert (result.input_tokens, result.cached_input_tokens, result.output_tokens) == (
        3000,
        3000,
        1500,
    )


def test_structured_completion_without_tools(sdk):
    sdk.parse.return_value = response("flex")
    result = asyncio.run(run_chat(response_format=Answer, flex_processing=True))
    assert result.cost_in_cents == pytest.approx(0.06375)


def test_concurrent_standard_requests_are_not_batch(sdk):
    sdk.create.return_value = response("default")

    async def run():
        return await asyncio.gather(run_chat(), run_chat())

    results = asyncio.run(run())
    assert [r.cost_in_cents for r in results] == pytest.approx([0.1275, 0.1275])


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize(
    "uncached,cached,expected",
    [
        (172000, 100000, 47.0),
        (172001, 100000, 93.2505),
    ],
)
def test_long_context_threshold_includes_cached_tokens(
    batch, uncached, cached, expected
):
    assert CostCalculator.calculate_cost(
        "gpt-5.5", uncached, 1000, cached, service_tier="flex", batch=batch
    ) == pytest.approx(expected)


def test_zero_tokens_and_unknown_model():
    assert CostCalculator.calculate_cost("gpt-5-mini", 0, 0, batch=True) == 0
    assert CostCalculator.calculate_cost("unknown", 1000, 500, batch=True) is None


def test_existing_positional_arguments_keep_their_meaning():
    from datetime import datetime

    when = datetime.fromisoformat("2026-09-14T02:00:00+00:00")
    expected = CostCalculator.calculate_cost(
        "deepseek-v4-pro", 1000, 500, 1000, 0, when
    )
    assert (
        CostCalculator.calculate_cost(
            "deepseek-v4-pro", 1000, 500, 1000, 0, when, service_tier="flex", batch=True
        )
        == expected
    )
    expected = CostCalculator.calculate_cost("claude-sonnet-4-6", 1000, 500, 1000, 1000)
    assert (
        CostCalculator.calculate_cost(
            "claude-sonnet-4-6", 1000, 500, 1000, 1000, service_tier="flex", batch=True
        )
        == expected
    )


def test_tool_outputs_only_does_not_charge_for_an_unused_parse(sdk):
    sdk.create.side_effect = [
        response("flex", lookup),
        response("default", response_id="response-2"),
    ]
    result = asyncio.run(
        run_chat(
            tools=[lookup],
            response_format=Answer,
            return_tool_outputs_only=True,
            flex_processing=True,
        )
    )
    assert result.cost_in_cents == pytest.approx(0.19125)
    assert result.list_cost_in_cents == pytest.approx(0.255)
    assert result.service_tier == "mixed"
    sdk.parse.assert_not_called()


def test_unlisted_tier_price_remains_unknown_in_response(sdk):
    sdk.create.return_value = response("flex")
    result = asyncio.run(
        OpenAIProvider(api_key="test-only").execute_chat(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            flex_processing=True,
        )
    )
    assert result.cost_in_cents is None
    # The standard price and the served tier are still known.
    assert result.list_cost_in_cents == CostCalculator.calculate_cost(
        "gpt-4o", 1000, 500, 1000
    )
    assert result.service_tier == "flex"


def test_other_providers_leave_the_new_fields_unset():
    from defog.llm.providers.base import LLMResponse

    plain = LLMResponse(
        content="x",
        model="claude-sonnet-4-6",
        time=0.1,
        input_tokens=1,
        output_tokens=1,
    )
    assert plain.list_cost_in_cents is None
    assert plain.service_tier is None
