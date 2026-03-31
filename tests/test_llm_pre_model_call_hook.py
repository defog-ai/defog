#!/usr/bin/env python3
"""Tests for the pre_model_call_hook functionality."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from defog.llm.providers.anthropic_provider import AnthropicProvider
from defog.llm.providers.base import LLMResponse
from defog.llm.providers.openai_provider import OpenAIProvider
from defog.llm.utils import chat_async


class DummyToolHandler:
    max_consecutive_errors = 3
    image_result_keys = None

    async def sample_tool_result(self, _func_name, result, _args, **_kwargs):
        return result

    def prepare_result_for_llm(self, sampled_result, **_kwargs):
        return str(sampled_result), False, None

    def is_sampler_configured(self, *_args, **_kwargs):
        return False


def test_chat_async_forwards_pre_model_call_hook():
    """chat_async should validate and pass the hook to the provider."""
    calls = []

    async def mock_pre_model_call_hook(*, checkpoint_kind, provider_name, model):
        calls.append((checkpoint_kind, provider_name, model))
        return None

    async def mock_execute_chat(**kwargs):
        await kwargs["pre_model_call_hook"](
            checkpoint_kind="initial_request",
            provider_name="anthropic",
            model=kwargs["model"],
        )
        return LLMResponse(
            content="ok",
            model=kwargs["model"],
            time=0.0,
            input_tokens=0,
            output_tokens=0,
        )

    with patch("defog.llm.utils.get_provider_instance") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.execute_chat = mock_execute_chat
        mock_provider.validate_post_response_hook = MagicMock()
        mock_provider.validate_pre_model_call_hook = MagicMock()
        mock_get_provider.return_value = mock_provider

        response = asyncio.run(
            chat_async(
                provider="anthropic",
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "test"}],
                pre_model_call_hook=mock_pre_model_call_hook,
                max_retries=1,
            )
        )

    mock_provider.validate_pre_model_call_hook.assert_called_once()
    assert response.content == "ok"
    assert calls == [("initial_request", "anthropic", "claude-haiku-4-5")]


def test_openai_pre_model_call_hook_applies_before_next_tool_batch_call():
    """OpenAI provider should inject hook messages before the next post-tool-batch request."""
    provider = OpenAIProvider(api_key="test")
    provider.execute_tool_calls_with_retry = AsyncMock(return_value=(["tool-result"], 0))
    provider.update_tools_with_budget = MagicMock(
        side_effect=lambda tools, _tool_handler, request_params: (
            tools,
            request_params.get("tools", {}),
        )
    )

    hook_calls = []

    async def mock_pre_model_call_hook(*, checkpoint_kind, provider_name, model):
        hook_calls.append((checkpoint_kind, provider_name, model))
        return [{"role": "user", "content": "please include trade headwinds"}]

    first_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=6,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        output=[
            SimpleNamespace(
                type="function_call",
                name="lookup",
                arguments='{"q":"singapore"}',
                call_id="call_1",
            )
        ],
        output_text="",
    )
    second_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=4,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
        output=[],
        output_text="final answer",
        id="resp_2",
    )
    client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=second_response))
    )

    asyncio.run(
        provider.process_response(
            client=client,
            response=first_response,
            request_params={"input": [], "model": "gpt-5.4-mini", "tool_choice": "auto"},
            tools=[lambda q: q],
            tool_dict={"lookup": lambda q: q},
            model="gpt-5.4-mini",
            pre_model_call_hook=mock_pre_model_call_hook,
            tool_handler=DummyToolHandler(),
            parallel_tool_calls=False,
        )
    )

    assert hook_calls == [("post_tool_batch", "openai", "gpt-5.4-mini")]
    request_input = client.responses.create.await_args.kwargs["input"]
    assert any(item.get("type") == "function_call_output" for item in request_input)
    assert any(
        item.get("role") == "user"
        and "trade headwinds" in str(item.get("content", ""))
        for item in request_input
    )


def test_anthropic_pre_model_call_hook_applies_before_next_tool_batch_call():
    """Anthropic provider should inject hook messages before the next post-tool-batch request."""
    provider = AnthropicProvider(api_key="test")
    provider.execute_tool_calls_with_retry = AsyncMock(return_value=(["tool-result"], 0))
    provider.update_tools_with_budget = MagicMock(
        side_effect=lambda tools, _tool_handler, request_params: (
            tools,
            request_params.get("tools", {}),
        )
    )

    hook_calls = []

    async def mock_pre_model_call_hook(*, checkpoint_kind, provider_name, model):
        hook_calls.append((checkpoint_kind, provider_name, model))
        return [{"role": "user", "content": "please include trade headwinds"}]

    first_response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                name="lookup",
                input={"q": "singapore"},
                id="tool_1",
            )
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=6,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    second_response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="final answer")],
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=4,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=second_response))
    )

    asyncio.run(
        provider.process_response(
            client=client,
            response=first_response,
            request_params={"messages": [], "model": "claude-3-7-sonnet", "tool_choice": {"type": "auto"}},
            tools=[lambda q: q],
            tool_dict={"lookup": lambda q: q},
            pre_model_call_hook=mock_pre_model_call_hook,
            tool_handler=DummyToolHandler(),
            parallel_tool_calls=False,
        )
    )

    assert hook_calls == [("post_tool_batch", "anthropic", "claude-3-7-sonnet")]
    request_messages = client.messages.create.await_args.kwargs["messages"]
    assert request_messages[-1]["role"] == "user"
    assert "trade headwinds" in str(request_messages[-1]["content"])
