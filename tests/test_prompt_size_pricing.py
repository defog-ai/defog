"""Providers record each model request so that it is priced on its own prompt."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from defog.llm.cost.calculator import CostCalculator
from defog.llm.providers.anthropic_provider import AnthropicProvider
from defog.llm.providers.gemini_provider import GeminiProvider


def _tool_handler():
    tool_handler = MagicMock()
    tool_handler.sample_tool_result = AsyncMock(return_value="result")
    tool_handler.prepare_result_for_llm = MagicMock(
        return_value=("result", False, None)
    )
    tool_handler.is_sampler_configured = MagicMock(return_value=False)
    tool_handler.image_result_keys = []
    tool_handler.max_consecutive_errors = 3
    return tool_handler


def _anthropic_usage(input_tokens, output_tokens, cache_read=0, cache_creation=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )


@pytest.mark.asyncio
async def test_anthropic_tool_loop_records_each_request():
    provider = AnthropicProvider(api_key="test")

    tool_use_block = MagicMock(type="tool_use", id="tool1", input={})
    tool_use_block.name = "test_tool"
    tool_use_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tool1",
        "name": "test_tool",
        "input": {},
    }
    first = MagicMock(stop_reason="tool_use", content=[tool_use_block])
    first.usage = _anthropic_usage(60_000, 500, cache_read=0, cache_creation=10)
    final = MagicMock(
        stop_reason="end_turn", content=[MagicMock(type="text", text="done")]
    )
    final.usage = _anthropic_usage(1_000, 700, cache_read=60_000)

    client = AsyncMock()
    client.messages.create.side_effect = [final]
    provider.execute_tool_calls_with_retry = AsyncMock(return_value=(["result"], 0))
    provider.update_tools_with_budget = MagicMock(
        return_value=([lambda: None], {"test_tool": lambda: "result"})
    )

    request_usages = []
    result = await provider.process_response(
        client=client,
        response=first,
        request_params={
            "messages": [{"role": "user", "content": "hello"}],
            "model": "claude-haiku-5-5",
            "tool_choice": {"type": "auto"},
        },
        tools=[lambda: None],
        tool_dict={"test_tool": lambda: "result"},
        tool_handler=_tool_handler(),
        request_usages=request_usages,
    )

    assert request_usages == [
        {
            "input_tokens": 60_000,
            "output_tokens": 500,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 10,
        },
        {
            "input_tokens": 1_000,
            "output_tokens": 700,
            "cached_input_tokens": 60_000,
            "cache_creation_input_tokens": 0,
        },
    ]
    # The totals are unchanged: they are the sums of the requests.
    assert result[2:6] == (61_000, 1_200, 60_000, 10)

    # Each prompt is about 60,000 tokens, so both requests pay the standard
    # rate. Priced as one 121,010-token prompt, all of it would cost 5x.
    per_request = AnthropicProvider._requests_cost(
        "claude-haiku-5-5", request_usages, 61_000, 1_200, 60_000, 10
    )
    as_one_prompt = CostCalculator.calculate_cost(
        "claude-haiku-5-5", 61_000, 1_200, 60_000, 10
    )
    assert per_request == pytest.approx(as_one_prompt / 5)


def test_anthropic_requests_cost_falls_back_to_totals():
    cost = AnthropicProvider._requests_cost("claude-haiku-5-5", [], 1_000, 1_000, 0, 0)
    assert cost == CostCalculator.calculate_cost("claude-haiku-5-5", 1_000, 1_000)


def _gemini_usage(input_tokens, output_tokens, thought_tokens, cached_tokens=0):
    return SimpleNamespace(
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_thought_tokens=thought_tokens,
        total_cached_tokens=cached_tokens,
    )


@pytest.mark.asyncio
async def test_gemini_tool_loop_records_each_request_and_counts_thinking_once():
    provider = GeminiProvider(api_key="test")
    provider.call_post_response_hook = AsyncMock()
    provider.extract_reasoning_text = AsyncMock(return_value=[])
    provider.emit_tool_phase_complete = AsyncMock()
    provider.execute_tool_calls_with_retry = AsyncMock(
        return_value=(["result", "result"], 0)
    )

    def call(call_id):
        return SimpleNamespace(
            type="function_call", id=call_id, name="test_tool", arguments={}
        )

    first = SimpleNamespace(
        id="i1", steps=[call("c1")], usage=_gemini_usage(150_000, 100, 300)
    )
    second = SimpleNamespace(
        id="i2",
        steps=[call("c2")],
        usage=_gemini_usage(150_500, 100, 200, cached_tokens=150_000),
    )
    final = SimpleNamespace(
        id="i3",
        steps=[],
        output_text="done",
        usage=_gemini_usage(151_000, 50, 0, cached_tokens=150_000),
    )
    client = MagicMock()
    client.aio.interactions.create = AsyncMock(side_effect=[second, final])

    request_usages = []
    result = await provider.process_response(
        client=client,
        response=first,
        request_params={},
        messages=[],
        tools=[lambda: None],
        tool_dict={"test_tool": lambda: "result"},
        model="gemini-3.1-pro-preview",
        tool_handler=_tool_handler(),
        request_usages=request_usages,
    )

    assert [u["output_tokens"] for u in request_usages] == [400, 300, 50]
    assert [u["prompt_tokens"] for u in request_usages] == [150_000, 150_500, 151_000]
    content, _, input_toks, output_toks, cached_toks, details = result
    assert content == "done"
    assert input_toks == 451_500
    # Thinking tokens are added once per request: 250 + 500 thinking.
    assert output_toks == 750
    assert cached_toks == 300_000
    assert details == {"reasoning_tokens": 500}

    # Every prompt is under 200,000 tokens, so every request pays the
    # standard Gemini 3.1 Pro rate.
    cost = CostCalculator.calculate_requests_cost(
        "gemini-3.1-pro-preview", request_usages
    )
    assert cost == pytest.approx(
        (451_500 * 2 + 750 * 12 + 300_000 * 0.2) / 1_000_000 * 100
    )
