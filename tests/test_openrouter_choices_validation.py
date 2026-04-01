"""Unit tests for OpenRouter response.choices validation (issue #261).

Verifies that process_response and _llm_json_repair raise ProviderError
when OpenRouter returns choices=None on follow-up API calls, instead of
crashing with a TypeError.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from defog.llm.providers.openrouter_provider import OpenRouterProvider
from defog.llm.exceptions import ProviderError


# ---------------------------------------------------------------------------
# Test schema and helpers
# ---------------------------------------------------------------------------


class StructuredOutput(BaseModel):
    answer: str
    score: int


def _make_response(
    content="hello",
    tool_calls=None,
    choices_present=True,
    input_tokens=10,
    output_tokens=5,
    cost=None,
):
    """Build a mock Chat Completions response object."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.prompt_tokens_details = None
    usage.completion_tokens_details = None
    usage.cost = cost

    response = MagicMock()
    response.id = "resp_test"
    if choices_present:
        response.choices = [choice]
    else:
        response.choices = None
    response.usage = usage

    return response


def _make_tool_call(name="numsum", arguments='{"a": 3, "b": 5}', tc_id="tc_1"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Tests: initial response validation (already existed, sanity check)
# ---------------------------------------------------------------------------


class TestInitialResponseValidation(unittest.IsolatedAsyncioTestCase):
    """The initial response validation at the top of process_response should
    raise ProviderError when choices is None."""

    async def test_choices_none_raises_provider_error(self):
        provider = OpenRouterProvider(api_key="test-key")
        response = _make_response(choices_present=False)
        client = AsyncMock()

        with self.assertRaises(ProviderError) as ctx:
            await provider.process_response(
                client=client,
                response=response,
                request_params={"messages": []},
                tools=None,
                tool_dict={},
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))

    async def test_choices_empty_list_raises_provider_error(self):
        provider = OpenRouterProvider(api_key="test-key")
        response = _make_response()
        response.choices = []
        client = AsyncMock()

        with self.assertRaises(ProviderError) as ctx:
            await provider.process_response(
                client=client,
                response=response,
                request_params={"messages": []},
                tools=None,
                tool_dict={},
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: follow-up call in tool-chaining loop (line ~632)
# ---------------------------------------------------------------------------


class TestToolChainingFollowUpValidation(unittest.IsolatedAsyncioTestCase):
    """After tool calls execute, the follow-up API call may return
    choices=None. This must raise ProviderError, not TypeError."""

    async def test_followup_choices_none_raises_provider_error(self):
        """Simulate: initial response has tool_calls, tools execute, but
        the follow-up call returns choices=None."""
        provider = OpenRouterProvider(api_key="test-key")

        # Initial response with a tool call
        tool_call = _make_tool_call()
        initial_response = _make_response(tool_calls=[tool_call])

        # Follow-up response with choices=None
        bad_followup = _make_response(choices_present=False)

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=bad_followup)

        # Minimal tool dict
        def numsum(input):
            return 8

        tool_dict = {"numsum": numsum}
        tools = [numsum]

        request_params = {"messages": [{"role": "user", "content": "What is 3+5?"}]}

        with self.assertRaises(ProviderError) as ctx:
            await provider.process_response(
                client=client,
                response=initial_response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))

    async def test_followup_choices_empty_raises_provider_error(self):
        """Same as above but choices is an empty list instead of None."""
        provider = OpenRouterProvider(api_key="test-key")

        tool_call = _make_tool_call()
        initial_response = _make_response(tool_calls=[tool_call])

        bad_followup = _make_response()
        bad_followup.choices = []

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=bad_followup)

        def numsum(input):
            return 8

        tool_dict = {"numsum": numsum}
        tools = [numsum]

        request_params = {"messages": [{"role": "user", "content": "What is 3+5?"}]}

        with self.assertRaises(ProviderError) as ctx:
            await provider.process_response(
                client=client,
                response=initial_response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: structured-output final call after tool chaining (line ~667)
# ---------------------------------------------------------------------------


class TestStructuredOutputFinalCallValidation(unittest.IsolatedAsyncioTestCase):
    """After tool chaining completes, a final API call is made for structured
    output. If this returns choices=None, it must raise ProviderError."""

    async def test_structured_output_choices_none_raises_provider_error(self):
        """Simulate: tools execute, loop ends (no more tool_calls), then
        the structured-output final call returns choices=None."""
        provider = OpenRouterProvider(api_key="test-key")

        # Initial response with tool call
        tool_call = _make_tool_call()
        initial_response = _make_response(tool_calls=[tool_call])

        # Second response: no tool calls (loop exits)
        no_tools_response = _make_response(content="Done with tools")

        # Third response: structured output call returns choices=None
        bad_structured = _make_response(choices_present=False)

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Follow-up after tool execution - no more tool calls
                return no_tools_response
            else:
                # Structured output final call - bad response
                return bad_structured

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(side_effect=mock_create)

        def numsum(input):
            return 8

        tool_dict = {"numsum": numsum}
        tools = [numsum]

        request_params = {"messages": [{"role": "user", "content": "What is 3+5?"}]}

        with self.assertRaises(ProviderError) as ctx:
            await provider.process_response(
                client=client,
                response=initial_response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=StructuredOutput,
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: LLM JSON repair call (line ~287)
# ---------------------------------------------------------------------------


class TestLlmJsonRepairValidation(unittest.IsolatedAsyncioTestCase):
    """_llm_json_repair makes an API call to fix broken JSON. If that call
    returns choices=None, it must raise ProviderError."""

    async def test_repair_choices_none_raises_provider_error(self):
        provider = OpenRouterProvider(api_key="test-key")

        bad_response = _make_response(choices_present=False)
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=bad_response)

        with self.assertRaises(ProviderError) as ctx:
            await provider._llm_json_repair(
                broken_content='{"answer": "hello", "score": }',
                response_format=StructuredOutput,
                client=client,
                model="test-model",
                request_params={"messages": []},
            )
        self.assertIn("No response from OpenRouter", str(ctx.exception))


# ---------------------------------------------------------------------------
# Tests: happy path still works (regression check)
# ---------------------------------------------------------------------------


class TestValidResponsesStillWork(unittest.IsolatedAsyncioTestCase):
    """Ensure valid responses with proper choices are not broken by the fix."""

    async def test_no_tools_no_format_returns_content(self):
        provider = OpenRouterProvider(api_key="test-key")
        response = _make_response(content="Hello world")
        client = AsyncMock()

        result = await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": []},
            tools=None,
            tool_dict={},
        )
        content = result[0]
        self.assertEqual(content, "Hello world")

    async def test_no_tools_with_format_parses_structured(self):
        provider = OpenRouterProvider(api_key="test-key")
        json_content = json.dumps({"answer": "test", "score": 42})
        response = _make_response(content=json_content)
        client = AsyncMock()

        result = await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": []},
            tools=None,
            tool_dict={},
            response_format=StructuredOutput,
        )
        content = result[0]
        self.assertIsInstance(content, StructuredOutput)
        self.assertEqual(content.answer, "test")
        self.assertEqual(content.score, 42)

    async def test_tool_chaining_with_valid_followup(self):
        """Tool call followed by valid follow-up should work normally."""
        provider = OpenRouterProvider(api_key="test-key")

        tool_call = _make_tool_call()
        initial_response = _make_response(tool_calls=[tool_call])

        followup_response = _make_response(content="The sum is 8")

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=followup_response)

        def numsum(input):
            return 8

        tool_dict = {"numsum": numsum}
        tools = [numsum]

        request_params = {"messages": [{"role": "user", "content": "What is 3+5?"}]}

        result = await provider.process_response(
            client=client,
            response=initial_response,
            request_params=request_params,
            tools=tools,
            tool_dict=tool_dict,
        )
        content = result[0]
        self.assertEqual(content, "The sum is 8")

    async def test_llm_json_repair_with_valid_response(self):
        provider = OpenRouterProvider(api_key="test-key")

        good_json = json.dumps({"answer": "fixed", "score": 99})
        good_response = _make_response(content=good_json)
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=good_response)

        result = await provider._llm_json_repair(
            broken_content='{"answer": "hello", "score": }',
            response_format=StructuredOutput,
            client=client,
            model="test-model",
            request_params={"messages": []},
        )
        self.assertIsInstance(result, StructuredOutput)
        self.assertEqual(result.answer, "fixed")
        self.assertEqual(result.score, 99)
