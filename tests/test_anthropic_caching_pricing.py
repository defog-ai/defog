import pytest
from unittest.mock import MagicMock, AsyncMock
from defog.llm.cost.calculator import CostCalculator
from defog.llm.providers.anthropic_provider import AnthropicProvider


def test_cost_calculator_with_cache_creation():
    # Test with Claude 3.5 Sonnet
    model = "claude-3-5-sonnet"
    input_tokens = 1000
    output_tokens = 1000
    cached_tokens = 1000
    cache_creation_tokens = 1000

    # Prices for claude-3-5-sonnet:
    # input: 0.003 per 1k
    # output: 0.015 per 1k
    # cached read: 0.0003 per 1k
    # cache creation: 0.00375 per 1k

    expected_cost = (
        (1000 / 1000 * 0.003)
        + (1000 / 1000 * 0.015)
        + (1000 / 1000 * 0.0003)
        + (1000 / 1000 * 0.00375)
    ) * 100

    cost = CostCalculator.calculate_cost(
        model, input_tokens, output_tokens, cached_tokens, cache_creation_tokens
    )

    assert cost == pytest.approx(expected_cost)


def test_anthropic_build_params_cache_control():
    provider = AnthropicProvider(api_key="test")

    # Test system message caching
    # User requested to remove system prompt caching
    # messages = [
    #     {"role": "system", "content": "system prompt"},
    #     {"role": "user", "content": "msg1"},
    #     {"role": "assistant", "content": "msg2"},
    #     {"role": "user", "content": "msg3"}
    # ]
    # But we need messages for the next test
    messages = [
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "msg2"},
        {"role": "user", "content": "msg3"},
    ]
    params, _ = provider.build_params(
        messages=messages, model="claude-3-5-sonnet", system="system prompt"
    )

    # assert isinstance(params["system"], list)
    # assert params["system"][-1]["cache_control"] == {"type": "ephemeral"}

    # Test message caching (last 2 messages)
    # msg3 (last) should be cached
    assert params["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    # msg2 (2nd last) should be cached
    assert params["messages"][-2]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    # msg1 (3rd last) should NOT be cached
    # Note: msg1 content might be string or list depending on implementation details of build_params
    # But checking if cache_control is absent is tricky if it's a string.
    # build_params converts strings to list of dicts ONLY if it adds cache_control?
    # No, it converts everything to list of dicts? Let's check build_params implementation.
    # It converts content to anthropic format.
    # If it's just text, it might remain string if no cache control added?
    # Let's just check that msg1 doesn't have cache_control if it's a dict, or is a string.
    msg1_content = params["messages"][0]["content"]
    if isinstance(msg1_content, list):
        assert "cache_control" not in msg1_content[-1]

    # Test tool caching
    def dummy_tool():
        """Dummy tool"""
        pass

    params, _ = provider.build_params(
        messages=messages, model="claude-3-5-sonnet", tools=[dummy_tool]
    )

    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_anthropic_process_response_cache_tokens():
    provider = AnthropicProvider(api_key="test")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="response")]
    mock_response.stop_reason = "end_turn"
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50
    mock_response.usage.cache_read_input_tokens = 20
    mock_response.usage.cache_creation_input_tokens = 30

    client = AsyncMock()

    (
        content,
        tool_outputs,
        input_toks,
        output_toks,
        cached_toks,
        cache_creation_toks,
        output_details,
    ) = await provider.process_response(
        client=client,
        response=mock_response,
        request_params={"messages": []},
        tools=None,
        tool_dict={},
    )

    assert input_toks == 100
    assert output_toks == 50
    assert cached_toks == 20
    assert cache_creation_toks == 30


@pytest.mark.asyncio
async def test_anthropic_process_response_tool_caching():
    provider = AnthropicProvider(api_key="test")

    # Mock response with tool use
    tool_use_block = MagicMock(type="tool_use", id="tool1", name="test_tool", input={})
    tool_use_block.model_dump.return_value = {
        "type": "tool_use",
        "id": "tool1",
        "name": "test_tool",
        "input": {},
    }

    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [tool_use_block]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 10

    # Mock client
    client = AsyncMock()
    # Setup client.messages.create to return a final response on the second call
    final_response = MagicMock()
    final_response.stop_reason = "end_turn"
    final_response.content = [MagicMock(type="text", text="final answer")]
    final_response.usage = MagicMock()

    client.messages.create.side_effect = [final_response]

    # Mock tool handler
    tool_handler = MagicMock()
    tool_handler.sample_tool_result = AsyncMock(return_value="result")
    tool_handler.prepare_result_for_llm = MagicMock(
        return_value=("result", False, None)
    )
    tool_handler.is_sampler_configured = MagicMock(return_value=False)
    tool_handler.image_result_keys = []
    tool_handler.max_consecutive_errors = 3

    # Request params
    request_params = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-3-5-sonnet",
        "tool_choice": {"type": "auto"},
    }

    # Mock execute_tool_calls_with_retry to return results directly
    provider.execute_tool_calls_with_retry = AsyncMock(return_value=(["result"], 0))

    # Mock update_tools_with_budget to return tools as is
    provider.update_tools_with_budget = MagicMock(
        return_value=([lambda: None], {"test_tool": lambda: "result"})
    )

    await provider.process_response(
        client=client,
        response=mock_response,
        request_params=request_params,
        tools=[lambda: None],
        tool_dict={"test_tool": lambda: "result"},
        tool_handler=tool_handler,
    )

    # Check that client.messages.create was called with cached tool messages
    call_args = client.messages.create.call_args
    assert call_args is not None
    params = call_args[1]  # kwargs

    # The messages list should now contain:
    # 0: User (hello)
    # 1: Assistant (ToolUse)
    # 2: User (ToolResult)

    messages = params["messages"]

    assert len(messages) == 3

    # Check Assistant ToolUse message caching
    asst_msg = messages[1]
    assert asst_msg["role"] == "assistant"
    # The content is a list of blocks. The last block should have cache_control.
    assert asst_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}

    # Check User ToolResult message caching
    user_msg = messages[2]
    assert user_msg["role"] == "user"
    # The content is a list of tool_result blocks. The last block should have cache_control.
    assert user_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}
