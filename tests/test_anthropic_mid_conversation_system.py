"""Tests for Anthropic mid-conversation system messages in chat_async /
AnthropicProvider.

A ``{"role": "system"}`` message that appears *after* the conversation has
started is normally hoisted into the top-level ``system`` field (which
invalidates the cached prefix). On Claude Opus 4.8 the provider instead keeps
such a message in place as a system turn — automatically, with no flag —
whenever its position is unambiguously legal per Anthropic's placement rules
(directly follows a user turn; is the last message or directly precedes an
assistant turn). In any other position, or on any other model, it falls back
to the historical hoist-into-``system`` behaviour, so no previously valid
request changes.

The unit + mocked-wiring tests below do not hit the live API. The final class
is an end-to-end test against the real Anthropic API and is skipped unless
``ANTHROPIC_API_KEY`` is set. Run it with:

    PYTHONPATH=. python -m pytest tests/test_anthropic_mid_conversation_system.py \
        -v --envfile .env
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from defog.llm import chat_async
from defog.llm.providers.anthropic_provider import AnthropicProvider


OPUS_48 = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"


def _leading_then_mid():
    """A conversation with a leading system prompt plus a mid-conversation
    system instruction appended after the latest user turn (a legal in-place
    position)."""
    return [
        {"role": "system", "content": "You are a code review assistant."},
        {"role": "user", "content": "Review process() in utils.py."},
        {"role": "assistant", "content": "Looks fine for small inputs."},
        {"role": "user", "content": "Now review the calling code."},
        {"role": "system", "content": "From now on, require type annotations."},
    ]


class TestMergeSystemContent:
    def test_two_strings_join_with_blank_line(self):
        assert AnthropicProvider._merge_system_content("a", "b") == "a\n\nb"

    def test_mixed_str_and_blocks_normalize_to_blocks(self):
        merged = AnthropicProvider._merge_system_content(
            "a", [{"type": "text", "text": "b"}]
        )
        assert merged == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_two_block_lists_concatenate(self):
        merged = AnthropicProvider._merge_system_content(
            [{"type": "text", "text": "a"}],
            [{"type": "text", "text": "b"}],
        )
        assert merged == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]


class TestCoalesceSystemMessages:
    def test_consecutive_systems_merged(self):
        out = AnthropicProvider._coalesce_system_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "a"},
                {"role": "system", "content": "b"},
            ]
        )
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "a\n\nb"},
        ]

    def test_non_adjacent_systems_not_merged(self):
        msgs = [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "b"},
        ]
        assert AnthropicProvider._coalesce_system_messages(msgs) == msgs

    def test_input_not_mutated(self):
        msgs = [
            {"role": "system", "content": "a"},
            {"role": "system", "content": "b"},
        ]
        AnthropicProvider._coalesce_system_messages(msgs)
        assert len(msgs) == 2


class TestNonOpus48HoistsEverything:
    """On any non-Opus-4.8 model every system message is hoisted — the
    historical behaviour is unchanged."""

    def test_sonnet_hoists_mid_system(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=_leading_then_mid(),
            model=SONNET,
        )
        assert "You are a code review assistant." in params["system"]
        assert "require type annotations" in params["system"]
        assert all(m["role"] != "system" for m in params["messages"])


class TestOpus48KeepsLegalMidSystemInPlace:
    def test_leading_hoisted_mid_kept_in_place(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=_leading_then_mid(),
            model=OPUS_48,
        )
        # Leading system prompt hoisted...
        assert params["system"] == "You are a code review assistant."
        # ...mid-conversation instruction NOT hoisted...
        assert "type annotations" not in params["system"]
        # ...it stays in place as the final system turn.
        assert params["messages"][-1] == {
            "role": "system",
            "content": "From now on, require type annotations.",
        }

    def test_no_leading_system_means_empty_top_level(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "bye"},
                {"role": "system", "content": "Be terse now."},
            ],
            model=OPUS_48,
        )
        assert params["system"] == ""
        assert params["messages"][-1]["role"] == "system"

    def test_list_content_kept_in_place_unchanged(self):
        provider = AnthropicProvider(api_key="sk-test")
        block_content = [{"type": "text", "text": "Be terse."}]
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": block_content},
            ],
            model=OPUS_48,
        )
        assert params["messages"][-1] == {
            "role": "system",
            "content": block_content,
        }

    def test_consecutive_mid_system_messages_merged_and_kept(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "First instruction."},
                {"role": "system", "content": "Second instruction."},
            ],
            model=OPUS_48,
        )
        system_turns = [m for m in params["messages"] if m["role"] == "system"]
        assert len(system_turns) == 1
        assert system_turns[0]["content"] == "First instruction.\n\nSecond instruction."
        assert params["system"] == ""

    def test_mid_system_followed_by_assistant_kept(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be terse."},
                {"role": "assistant", "content": "ok"},
            ],
            model=OPUS_48,
        )
        roles = [m["role"] for m in params["messages"]]
        assert roles == ["user", "system", "assistant"]
        assert params["system"] == ""


class TestOpus48FallsBackToHoistInIllegalPositions:
    """When the position is not unambiguously legal, the message is hoisted —
    never an error. This guarantees previously valid requests keep working."""

    def test_followed_by_user_is_hoisted(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "more"},
            ],
            model=OPUS_48,
        )
        # Not kept in place; hoisted instead. No exception raised.
        assert "Be terse." in params["system"]
        assert all(m["role"] != "system" for m in params["messages"])

    def test_preceded_by_assistant_is_hoisted(self):
        # A system message directly after a normal assistant turn is not a legal
        # in-place position (Anthropic requires the preceding turn to be a user
        # turn or an assistant turn ending in server tool use), so it hoists.
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "Be terse."},
            ],
            model=OPUS_48,
        )
        assert "Be terse." in params["system"]
        assert all(m["role"] != "system" for m in params["messages"])


class TestExecuteChatWiring:
    """End-to-end wiring: patch AsyncAnthropic and confirm the in-place system
    message reaches the request payload's messages array while the top-level
    ``system`` field holds only the leading prompt."""

    @staticmethod
    def _fake_client_factory(captured: Dict[str, Any]):
        def fake_client_ctor(**client_kwargs: Any) -> SimpleNamespace:
            captured["client_kwargs"] = client_kwargs

            async def _create(**params: Any) -> SimpleNamespace:
                captured["request_params"] = params
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=1,
                        cache_read_input_tokens=0,
                        cache_creation_input_tokens=0,
                    ),
                    container=None,
                    model=OPUS_48,
                    id="msg_01test",
                )

            messages_ns = SimpleNamespace(create=AsyncMock(side_effect=_create))
            return SimpleNamespace(messages=messages_ns)

        return fake_client_ctor

    @pytest.mark.asyncio
    async def test_opus48_keeps_mid_system_in_payload(self):
        captured: Dict[str, Any] = {}
        with patch(
            "defog.llm.providers.anthropic_provider.AsyncAnthropic",
            side_effect=self._fake_client_factory(captured),
        ):
            await chat_async(
                provider="anthropic",
                model=OPUS_48,
                messages=_leading_then_mid(),
                max_retries=1,
            )

        request_params = captured["request_params"]
        # Leading prompt hoisted; mid-conversation instruction is NOT.
        assert request_params["system"] == "You are a code review assistant."
        assert "type annotations" not in request_params["system"]
        # Mid-conversation instruction is carried as an in-place system turn.
        assert request_params["messages"][-1] == {
            "role": "system",
            "content": "From now on, require type annotations.",
        }

    @pytest.mark.asyncio
    async def test_sonnet_hoists_everything_in_payload(self):
        captured: Dict[str, Any] = {}
        with patch(
            "defog.llm.providers.anthropic_provider.AsyncAnthropic",
            side_effect=self._fake_client_factory(captured),
        ):
            await chat_async(
                provider="anthropic",
                model=SONNET,
                messages=_leading_then_mid(),
                max_retries=1,
            )

        request_params = captured["request_params"]
        assert "type annotations" in request_params["system"]
        assert all(m["role"] != "system" for m in request_params["messages"])


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live Anthropic API test",
)
class TestLiveMidConversationSystem:
    """Live end-to-end test against the real Anthropic API (Opus 4.8).

    Appends a mid-conversation system instruction that strongly determines the
    model's output, then asserts the instruction was applied — proving the
    in-place system turn carries system-level priority through the real API.
    """

    @pytest.mark.asyncio
    async def test_mid_system_instruction_is_applied(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And the capital of Japan?"},
            {
                "role": "system",
                "content": (
                    "From now on, end every reply with the exact token "
                    "<<ACK>> on its own line."
                ),
            },
        ]
        response = await chat_async(
            provider="anthropic",
            model=OPUS_48,
            messages=messages,
            max_completion_tokens=256,
            max_retries=2,
        )
        assert response.content is not None
        # The mid-conversation system instruction must have been honoured.
        assert "<<ACK>>" in response.content
