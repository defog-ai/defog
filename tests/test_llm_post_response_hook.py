#!/usr/bin/env python3
"""Test for post_response_hook hook functionality."""

import pytest
from typing import Any, Dict, List

from defog.llm.utils import chat_async
from tests.conftest import skip_if_no_api_key

from dotenv import load_dotenv

load_dotenv()
import logging

logging.basicConfig(level=logging.INFO)

# Mock response hook for testing
response_hook_calls = []


async def mock_response_hook(
    response: Any,
    messages: List[Dict[str, Any]],
) -> None:
    """Mock hook that records calls."""
    response_hook_calls.append(
        {
            "response_type": type(response).__name__,
            "message_count": len(messages),
        }
    )

    logging.info(response)


def calculate(expression: str) -> str:
    """Simple calculation tool for testing."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is {result}"
    except Exception:
        return f"Error evaluating {expression}"


@pytest.mark.asyncio
@skip_if_no_api_key("openai")
async def test_post_response_hook_simple():
    """Test the hook is called for simple chat completion."""
    global response_hook_calls
    response_hook_calls = []

    response = await chat_async(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello, hooks!' and nothing else."}],
        post_response_hook=mock_response_hook,
        max_completion_tokens=50,
    )

    # Verify hook was called
    assert len(response_hook_calls) == 1
    assert response_hook_calls[0]["message_count"] == 1
    assert "hooks" in response.content.lower()


@pytest.mark.asyncio
@skip_if_no_api_key("openai")
async def test_post_response_hook_with_tools():
    """Test the hook is called multiple times during tool use."""
    global response_hook_calls
    response_hook_calls = []

    response = await chat_async(
        provider="openai",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Calculate 25 * 4"}],
        tools=[calculate],
        post_response_hook=mock_response_hook,
        max_completion_tokens=200,
    )

    # Verify hook was called multiple times (at least twice - initial + after tool)
    assert len(response_hook_calls) >= 2
    assert response.tool_outputs is not None
    assert len(response.tool_outputs) > 0
    assert "100" in response.content


@pytest.mark.asyncio
@skip_if_no_api_key("gemini")
async def test_post_response_hook_gemini():
    """Test the hook works with gemini provider."""
    global response_hook_calls
    response_hook_calls = []

    response = await chat_async(
        provider="gemini",
        model="gemini-2.5-flash-lite-preview-06-17",
        messages=[
            {"role": "user", "content": "Say 'gemini hook test' and nothing else."}
        ],
        post_response_hook=mock_response_hook,
        max_completion_tokens=50,
    )

    # Verify hook was called
    assert len(response_hook_calls) == 1
    assert "gemini hook test" in response.content.lower()


@pytest.mark.asyncio
@skip_if_no_api_key("anthropic")
async def test_post_response_hook_anthropic():
    """Test the hook works with anthropic provider."""
    global response_hook_calls
    response_hook_calls = []

    response = await chat_async(
        provider="anthropic",
        model="claude-haiku-4-5",
        messages=[
            {"role": "user", "content": "Say 'anthropic hook test' and nothing else."}
        ],
        post_response_hook=mock_response_hook,
        max_completion_tokens=50,
    )

    # Verify hook was called
    assert len(response_hook_calls) == 1
    assert "anthropic hook test" in response.content.lower()


@pytest.mark.asyncio
@skip_if_no_api_key("gemini")
async def test_normal():
    """Test normal api without the post response hook."""
    global response_hook_calls
    response_hook_calls = []

    response = await chat_async(
        provider="gemini",
        model="gemini-2.5-flash-lite-preview-06-17",
        messages=[
            {"role": "user", "content": "Say 'gemini hook test' and nothing else."}
        ],
        max_completion_tokens=50,
    )

    # Verify this worked
    assert "gemini hook test" in response.content.lower()
