"""Tests for the human-in-the-loop pause/resume primitive.

Covers:
  * PauseToolExecution propagating out of parallel + sequential tool execution
    (Limitation 1 from issue #282) with tool identity and sibling results
    preserved.
  * Anthropic message capture / resume-injection helpers (offline).
  * chat_async validation around resume_tool_results.
  * Live end-to-end pause -> resume for Anthropic and OpenAI (gated on keys).
"""

import asyncio
import pytest

from defog.llm import chat_async, PauseToolExecution
from defog.llm.utils_function_calling import execute_tools_parallel
from defog.llm.tools.handler import ToolHandler
from defog.llm.providers.anthropic_provider import AnthropicProvider
from tests.conftest import skip_if_no_api_key


# ---------------------------------------------------------------------------
# Tools used across tests
# ---------------------------------------------------------------------------
async def ask_user_question(question: str) -> dict:
    """Ask the end user a clarifying question and wait for their answer.

    Use this whenever a request is ambiguous and you need more information
    before you can proceed.
    """
    raise PauseToolExecution(payload={"questions": [question]})


def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b


# ---------------------------------------------------------------------------
# Offline: propagation through the tool-execution layer (Limitation 1)
# ---------------------------------------------------------------------------
class TestPausePropagation:
    def setup_method(self):
        self.tool_dict = {
            "ask_user_question": ask_user_question,
            "add_numbers": add_numbers,
        }
        self.calls = [
            {"id": "call_add", "name": "add_numbers", "arguments": {"a": 2, "b": 3}},
            {
                "id": "call_ask",
                "name": "ask_user_question",
                "arguments": {"question": "which db?"},
            },
        ]

    def test_pause_is_base_exception(self):
        # Must not be caught by generic `except Exception` handlers.
        assert issubclass(PauseToolExecution, BaseException)
        assert not issubclass(PauseToolExecution, Exception)

    def test_parallel_propagates_with_identity_and_siblings(self):
        async def run():
            return await execute_tools_parallel(
                self.calls, self.tool_dict, enable_parallel=True
            )

        with pytest.raises(PauseToolExecution) as exc_info:
            asyncio.run(run())

        pause = exc_info.value
        assert pause.tool_use_id == "call_ask"
        assert pause.tool_name == "ask_user_question"
        assert pause.tool_input == {"question": "which db?"}
        assert pause.payload == {"questions": ["which db?"]}
        # Sibling that finished normally is preserved.
        assert pause.completed_results == {"call_add": 5}

    def test_sequential_propagates_with_identity_and_siblings(self):
        async def run():
            return await execute_tools_parallel(
                self.calls, self.tool_dict, enable_parallel=False
            )

        with pytest.raises(PauseToolExecution) as exc_info:
            asyncio.run(run())

        pause = exc_info.value
        assert pause.tool_use_id == "call_ask"
        assert pause.completed_results == {"call_add": 5}

    def test_tool_handler_batch_parallel_propagates(self):
        async def run():
            return await ToolHandler().execute_tool_calls_batch(
                self.calls, self.tool_dict, parallel_tool_calls=True
            )

        with pytest.raises(PauseToolExecution) as exc_info:
            asyncio.run(run())
        assert exc_info.value.tool_use_id == "call_ask"
        assert exc_info.value.completed_results == {"call_add": 5}

    def test_tool_handler_batch_sequential_propagates(self):
        async def run():
            return await ToolHandler().execute_tool_calls_batch(
                self.calls, self.tool_dict, parallel_tool_calls=False
            )

        with pytest.raises(PauseToolExecution) as exc_info:
            asyncio.run(run())
        assert exc_info.value.tool_use_id == "call_ask"
        assert exc_info.value.completed_results == {"call_add": 5}


# ---------------------------------------------------------------------------
# Offline: Anthropic capture / resume-injection helpers
# ---------------------------------------------------------------------------
class TestAnthropicResumeHelpers:
    def setup_method(self):
        self.provider = AnthropicProvider(api_key="test-key")
        self.handler = ToolHandler()

    def test_serialize_messages_passes_dicts_through(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "tool_use", "id": "tu_1", "name": "ask", "input": {}},
                ],
            },
        ]
        out = self.provider._serialize_messages(messages)
        assert out == messages  # plain dicts unchanged, thinking preserved

    def test_inject_resume_tool_results_appends_user_turn(self):
        messages = [
            {"role": "user", "content": "help me"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me ask"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "ask_user_question",
                        "input": {"question": "x"},
                    },
                ],
            },
        ]
        out = self.provider._inject_resume_tool_results(
            messages,
            {"tu_1": "use the prod db"},
            self.handler,
            None,
            "claude-haiku-4-5",
        )
        assert out[-1]["role"] == "user"
        assert out[-1]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "use the prod db",
            }
        ]

    def test_inject_completes_partial_sibling_turn(self):
        # Parallel pause: a sibling tool_result is already present; the resume
        # must fill in only the still-pending tool call.
        messages = [
            {"role": "user", "content": "help me"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_add",
                        "name": "add_numbers",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_ask",
                        "name": "ask_user_question",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_add", "content": "5"}
                ],
            },
        ]
        out = self.provider._inject_resume_tool_results(
            messages, {"tu_ask": "answer"}, self.handler, None, "claude-haiku-4-5"
        )
        # The trailing user turn now has results for both tool calls.
        result_ids = {
            b["tool_use_id"] for b in out[-1]["content"] if b["type"] == "tool_result"
        }
        assert result_ids == {"tu_add", "tu_ask"}

    def test_inject_missing_result_raises(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "ask", "input": {}}
                ],
            }
        ]
        with pytest.raises(ValueError, match="missing results"):
            self.provider._inject_resume_tool_results(
                messages, {"other_id": "x"}, self.handler, None, "claude-haiku-4-5"
            )

    def test_inject_without_tool_use_turn_raises(self):
        messages = [{"role": "user", "content": "hi"}]
        with pytest.raises(ValueError, match="no assistant tool_use turn"):
            self.provider._inject_resume_tool_results(
                messages, {"tu_1": "x"}, self.handler, None, "claude-haiku-4-5"
            )


# ---------------------------------------------------------------------------
# Offline: chat_async validation
# ---------------------------------------------------------------------------
class TestResumeValidation:
    def test_unsupported_provider_rejected(self):
        with pytest.raises(ValueError, match="only supported for"):
            asyncio.run(
                chat_async(
                    provider="gemini",
                    model="gemini-2.5-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[ask_user_question],
                    resume_tool_results={"tu_1": "x"},
                )
            )

    def test_resume_requires_tools(self):
        with pytest.raises(ValueError, match="requires the same `tools`"):
            asyncio.run(
                chat_async(
                    provider="anthropic",
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hi"}],
                    resume_tool_results={"tu_1": "x"},
                )
            )


# ---------------------------------------------------------------------------
# Live: end-to-end pause -> resume
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful assistant. Before giving any recommendation you MUST "
    "call the ask_user_question tool exactly once to clarify the user's intent. "
    "After you receive their answer, give your final recommendation as text and "
    "do not ask any further questions."
)


@pytest.mark.asyncio
@skip_if_no_api_key("anthropic")
async def test_anthropic_pause_and_resume():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Recommend a programming language for me to learn.",
        },
    ]
    tools = [ask_user_question]

    paused = await chat_async(
        provider="anthropic",
        model="claude-haiku-4-5",
        messages=messages,
        tools=tools,
        max_completion_tokens=1024,
    )

    assert paused.status == "paused"
    assert paused.pending_tool_use is not None
    assert paused.pending_tool_use["name"] == "ask_user_question"
    assert paused.pending_tool_use["id"]
    assert paused.pause_payload and "questions" in paused.pause_payload
    assert isinstance(paused.messages, list) and len(paused.messages) >= 2

    tool_use_id = paused.pending_tool_use["id"]
    resumed = await chat_async(
        provider="anthropic",
        model="claude-haiku-4-5",
        messages=paused.messages,
        tools=tools,
        max_completion_tokens=1024,
        resume_tool_results={
            tool_use_id: "I want to build web backends and value job prospects."
        },
    )

    assert resumed.status != "paused"
    assert isinstance(resumed.content, str) and len(resumed.content) > 0


@pytest.mark.asyncio
@skip_if_no_api_key("anthropic")
async def test_anthropic_pause_resume_preserves_thinking():
    """The captured assistant turn must keep thinking blocks intact so the
    signed thinking round-trips back into the API on resume."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Recommend a database for my project."},
    ]
    tools = [ask_user_question]

    paused = await chat_async(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=messages,
        tools=tools,
        reasoning_effort="low",
        max_completion_tokens=2048,
    )

    assert paused.status == "paused"
    assistant_turn = [m for m in paused.messages if m["role"] == "assistant"][-1]
    block_types = [
        b.get("type") for b in assistant_turn["content"] if isinstance(b, dict)
    ]
    assert "thinking" in block_types or "redacted_thinking" in block_types
    assert "tool_use" in block_types

    tool_use_id = paused.pending_tool_use["id"]
    resumed = await chat_async(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=paused.messages,
        tools=tools,
        reasoning_effort="low",
        max_completion_tokens=2048,
        resume_tool_results={
            tool_use_id: "A high-write, time-series analytics workload."
        },
    )

    assert resumed.status != "paused"
    assert isinstance(resumed.content, str) and len(resumed.content) > 0


@pytest.mark.asyncio
@skip_if_no_api_key("openai")
async def test_openai_pause_and_resume():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Recommend a programming language for me to learn.",
        },
    ]
    tools = [ask_user_question]

    paused = await chat_async(
        provider="openai",
        model="gpt-4.1",
        messages=messages,
        tools=tools,
        max_completion_tokens=1024,
    )

    assert paused.status == "paused"
    assert paused.pending_tool_use is not None
    assert paused.pending_tool_use["name"] == "ask_user_question"
    assert paused.pending_tool_use["id"]
    assert paused.pause_payload and "questions" in paused.pause_payload
    # OpenAI resumes via previous_response_id (assistant turn is server-side).
    assert paused.response_id

    tool_use_id = paused.pending_tool_use["id"]
    resumed = await chat_async(
        provider="openai",
        model="gpt-4.1",
        messages=paused.messages,
        tools=tools,
        max_completion_tokens=1024,
        previous_response_id=paused.response_id,
        resume_tool_results={
            tool_use_id: "I want to build web backends and value job prospects."
        },
    )

    assert resumed.status != "paused"
    assert isinstance(resumed.content, str) and len(resumed.content) > 0
