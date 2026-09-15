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
    assert paused.cost_in_cents + resumed.cost_in_cents == pytest.approx(0.31875)
    assert sdk.create.call_args.kwargs["previous_response_id"] == "response-2"


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
