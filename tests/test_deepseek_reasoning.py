"""DeepSeek reasoning controls and history replay, without live API calls."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from defog.llm.config import LLMConfig
from defog.llm.memory.conversation_cache import load_messages
from defog.llm.utils import chat_async


class Answer(BaseModel):
    answer: int


def lookup(value: int) -> str:
    """Look up a value."""
    return str(value)


def _completion(content="Done", reasoning="Thinking", tool_id=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_id:
        message["tool_calls"] = [
            {
                "id": tool_id,
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"value": 42}'},
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "deepseek-test",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_id else "stop",
                    "message": message,
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
            },
        }
    )


@pytest.fixture
def api(monkeypatch, tmp_path):
    state = SimpleNamespace(requests=[], responses=[])

    async def create(**params):
        # Snapshot what would go over the wire before later history mutations.
        state.requests.append(deepcopy(params))
        return state.responses.pop(0)

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.chat.completions.create.side_effect = create
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr("defog.config._env_vars", None)
    return state


async def _chat(**kwargs):
    return await chat_async(
        provider="deepseek",
        model="deepseek-v4.1-flash",
        config=LLMConfig(api_keys={"deepseek": "sk-test-not-real"}),
        max_retries=1,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effort",
    [None, "none", "low", "high", "max", "minimal", "medium", "xhigh", "ultra"],
)
async def test_effort_reaches_api(api, effort):
    api.responses.append(_completion())
    await _chat(
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort=effort,
    )
    if effort is None:
        assert "reasoning_effort" not in api.requests[0]
    else:
        assert api.requests[0]["reasoning_effort"] == effort


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", [None, "none", "low", "max"])
@pytest.mark.parametrize("mode", ["structured", "full_repair", "field_repair"])
async def test_structured_output_and_repairs_keep_effort(api, effort, mode):
    initial = {
        "structured": '{"answer": 42}',
        "full_repair": "not JSON",
        "field_repair": '{"answer": "invalid"}',
    }[mode]
    api.responses.append(_completion(initial, reasoning="Original reasoning"))
    if mode != "structured":
        repair = (
            '{"repairs": [{"path": "/answer", "value": 42}]}'
            if mode == "field_repair"
            else '{"answer": 42}'
        )
        api.responses.append(_completion(repair))

    response = await _chat(
        messages=[{"role": "user", "content": "Return 42"}],
        reasoning_effort=effort,
        response_format=Answer,
    )

    assert response.content == Answer(answer=42)
    assert len(api.requests) == (1 if mode == "structured" else 2)
    for request in api.requests:
        assert request["model"] == "deepseek-flash"
        assert request["response_format"] == {"type": "json_object"}
        assert request.get("reasoning_effort") == effort
        if effort is None:
            assert "reasoning_effort" not in request
    history = await load_messages(response.response_id)
    assert history[-1]["reasoning_content"] == "Original reasoning"
    assert Answer.model_validate_json(history[-1]["content"]) == response.content


@pytest.mark.asyncio
async def test_tool_chain_and_cached_followup_preserve_reasoning(api):
    api.responses.extend(
        [
            _completion(None, "First tool reasoning", "tool-1"),
            _completion(None, "Second tool reasoning", "tool-2"),
            _completion("42", "Final reasoning"),
            _completion("Confirmed", "Follow-up reasoning"),
        ]
    )
    messages = [{"role": "user", "content": "Look up 42 twice"}]
    original = deepcopy(messages)
    response = await _chat(messages=messages, tools=[lookup], reasoning_effort="low")

    assert response.content == "42"
    assert len(response.tool_outputs) == 2
    assert messages == original
    for index, request in enumerate(api.requests):
        assert request["reasoning_effort"] == "low"
        assistants = [m for m in request["messages"] if m["role"] == "assistant"]
        assert [m["reasoning_content"] for m in assistants] == [
            "First tool reasoning",
            "Second tool reasoning",
        ][:index]
    history = await load_messages(response.response_id)
    assert [m["role"] for m in history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert history[-1]["reasoning_content"] == "Final reasoning"

    await _chat(
        messages=[{"role": "user", "content": "Are you sure?"}],
        previous_response_id=response.response_id,
        tools=[lookup],
        reasoning_effort="max",
    )
    assert api.requests[-1]["reasoning_effort"] == "max"
    assert api.requests[-1]["messages"][:-1] == history


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", ["Plain chat reasoning", "", None])
async def test_plain_chat_history_can_be_reused_with_tools(api, reasoning):
    api.responses.extend([_completion(reasoning=reasoning), _completion()])
    response = await _chat(messages=[{"role": "user", "content": "Hello"}])
    await _chat(
        messages=[{"role": "user", "content": "Use tools if needed"}],
        previous_response_id=response.response_id,
        tools=[lookup],
    )
    assistants = [m for m in api.requests[-1]["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 1
    if reasoning is None:
        assert "reasoning_content" not in assistants[0]
    else:
        assert assistants[0]["reasoning_content"] == reasoning


@pytest.mark.asyncio
async def test_structured_answer_after_tools_keeps_effort_and_final_reasoning(api):
    api.responses.extend(
        [
            _completion(None, "Tool reasoning", "tool-1"),
            _completion("42", "Tool summary reasoning"),
            _completion('{"answer": 42}', "Structured answer reasoning"),
        ]
    )
    response = await _chat(
        messages=[{"role": "user", "content": "Look up 42"}],
        tools=[lookup],
        reasoning_effort="low",
        response_format=Answer,
    )
    assert response.content == Answer(answer=42)
    assert len(api.requests) == 3
    assert all(request["reasoning_effort"] == "low" for request in api.requests)
    assert "tools" not in api.requests[-1]
    assert api.requests[-1]["response_format"] == {"type": "json_object"}
    history = await load_messages(response.response_id)
    assert history[-1]["reasoning_content"] == "Structured answer reasoning"
    assert Answer.model_validate_json(history[-1]["content"]) == response.content
