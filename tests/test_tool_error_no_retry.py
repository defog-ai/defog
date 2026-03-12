"""Test that ToolError from tool execution does not trigger chat_async retry loop,
while API-level errors (ProviderError, generic Exception) still do."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from defog.llm.exceptions import ToolError, ProviderError, LLMError
from defog.llm.utils import chat_async


@pytest.mark.asyncio
async def test_tool_error_not_retried():
    """ToolError should propagate immediately without triggering the retry loop."""
    call_count = 0

    async def mock_execute_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ToolError(
            "batch",
            "Tool execution failed after 3 consecutive errors: some tool error",
        )

    with patch("defog.llm.utils.get_provider_instance") as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.execute_chat = mock_execute_chat
        mock_provider.validate_post_response_hook = MagicMock()
        mock_get_provider.return_value = mock_provider

        with pytest.raises(ToolError):
            await chat_async(
                provider="anthropic",
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "test"}],
                max_retries=3,
            )

    # Should only be called once — ToolError should NOT trigger retries
    assert call_count == 1, (
        f"execute_chat was called {call_count} times; "
        f"ToolError should not trigger retries"
    )


@pytest.mark.asyncio
async def test_provider_error_still_retried():
    """ProviderError (non-tool) should still trigger the retry loop."""
    call_count = 0

    async def mock_execute_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ProviderError("anthropic", "API rate limit exceeded")

    with (
        patch("defog.llm.utils.get_provider_instance") as mock_get_provider,
        patch("defog.llm.utils.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_provider = MagicMock()
        mock_provider.execute_chat = mock_execute_chat
        mock_provider.validate_post_response_hook = MagicMock()
        mock_get_provider.return_value = mock_provider

        with pytest.raises(ProviderError):
            await chat_async(
                provider="anthropic",
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "test"}],
                max_retries=3,
            )

    # Should be called 3 times — ProviderError should trigger retries
    assert call_count == 3, (
        f"execute_chat was called {call_count} times; "
        f"ProviderError should trigger all {3} retries"
    )


@pytest.mark.asyncio
async def test_generic_api_error_still_retried():
    """Generic exceptions (e.g. from LLM API calls) should still trigger retries."""
    call_count = 0

    async def mock_execute_chat(**kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("API connection reset")

    with (
        patch("defog.llm.utils.get_provider_instance") as mock_get_provider,
        patch("defog.llm.utils.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_provider = MagicMock()
        mock_provider.execute_chat = mock_execute_chat
        mock_provider.validate_post_response_hook = MagicMock()
        mock_get_provider.return_value = mock_provider

        with pytest.raises(LLMError):
            await chat_async(
                provider="anthropic",
                model="claude-haiku-4-5",
                messages=[{"role": "user", "content": "test"}],
                max_retries=3,
            )

    # Generic API errors should trigger all retries
    assert call_count == 3, (
        f"execute_chat was called {call_count} times; "
        f"generic API errors should trigger all {3} retries"
    )
