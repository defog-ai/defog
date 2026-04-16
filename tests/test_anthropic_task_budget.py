"""Mocked unit tests for Anthropic task_budget support in chat_async /
AnthropicProvider.

These tests do not hit the live Anthropic API. They exercise the
``_normalize_task_budget`` helper, the ``build_params`` output_config
wiring, the top-level ``chat_async`` validation path, and the beta-header
wiring via a patched ``AsyncAnthropic`` client.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from defog.llm import chat_async
from defog.llm.providers.anthropic_provider import AnthropicProvider


OPUS_47 = "claude-opus-4-7"


class TestNormalizeTaskBudget:
    def test_none_returns_none(self):
        assert AnthropicProvider._normalize_task_budget(None, OPUS_47) is None

    def test_int_sugar(self):
        result = AnthropicProvider._normalize_task_budget(64000, OPUS_47)
        assert result == {"type": "tokens", "total": 64000}

    def test_dict_passthrough(self):
        result = AnthropicProvider._normalize_task_budget(
            {"type": "tokens", "total": 64000, "remaining": 30000},
            OPUS_47,
        )
        assert result == {"type": "tokens", "total": 64000, "remaining": 30000}

    def test_dict_defaults_type(self):
        result = AnthropicProvider._normalize_task_budget({"total": 50000}, OPUS_47)
        assert result == {"type": "tokens", "total": 50000}

    def test_dict_does_not_mutate_caller(self):
        original = {"total": 50000}
        AnthropicProvider._normalize_task_budget(original, OPUS_47)
        assert "type" not in original

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="at least 20000"):
            AnthropicProvider._normalize_task_budget(10000, OPUS_47)

    def test_dict_without_total_raises(self):
        with pytest.raises(ValueError, match="must include 'total'"):
            AnthropicProvider._normalize_task_budget({"type": "tokens"}, OPUS_47)

    def test_wrong_model_raises(self):
        with pytest.raises(ValueError, match="only supported on claude-opus-4-7"):
            AnthropicProvider._normalize_task_budget(64000, "claude-opus-4-6")

    def test_bool_rejected(self):
        with pytest.raises(ValueError, match="not bool"):
            AnthropicProvider._normalize_task_budget(True, OPUS_47)

    def test_non_int_total_rejected(self):
        with pytest.raises(ValueError, match="'total' must be an int"):
            AnthropicProvider._normalize_task_budget({"total": "64000"}, OPUS_47)

    def test_remaining_exceeds_total_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed 'total'"):
            AnthropicProvider._normalize_task_budget(
                {"total": 50000, "remaining": 60000}, OPUS_47
            )


class TestBuildParamsTaskBudget:
    def test_task_budget_lands_in_output_config(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model=OPUS_47,
            task_budget={"type": "tokens", "total": 64000},
        )
        assert params["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": 64000,
        }

    def test_task_budget_merges_with_effort(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model=OPUS_47,
            reasoning_effort="high",
            task_budget={"type": "tokens", "total": 64000},
        )
        oc = params["output_config"]
        assert oc["effort"] == "high"
        assert oc["task_budget"] == {"type": "tokens", "total": 64000}

    def test_no_task_budget_no_key(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model=OPUS_47,
        )
        assert "task_budget" not in params.get("output_config", {})


class TestChatAsyncValidation:
    @pytest.mark.asyncio
    async def test_non_anthropic_provider_rejected(self):
        with pytest.raises(ValueError, match="task_budget"):
            await chat_async(
                provider="openai",
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                task_budget=30000,
            )

    @pytest.mark.asyncio
    async def test_wrong_anthropic_model_rejected(self):
        # Model check happens inside execute_chat via _normalize_task_budget.
        # chat_async wraps errors in max_retries loop, so we set max_retries=1
        # to surface the first raise.
        with pytest.raises(Exception, match="only supported on claude-opus-4-7"):
            await chat_async(
                provider="anthropic",
                model="claude-opus-4-6",
                messages=[{"role": "user", "content": "hi"}],
                task_budget=30000,
                max_retries=1,
            )

    @pytest.mark.asyncio
    async def test_below_minimum_rejected(self):
        with pytest.raises(Exception, match="at least 20000"):
            await chat_async(
                provider="anthropic",
                model=OPUS_47,
                messages=[{"role": "user", "content": "hi"}],
                task_budget=10000,
                max_retries=1,
            )


class TestExecuteChatWiring:
    """End-to-end wiring test: patch AsyncAnthropic and confirm that when a
    caller supplies ``task_budget``, the beta header is present and
    ``output_config.task_budget`` appears in the request payload."""

    @pytest.mark.asyncio
    async def test_task_budget_wires_header_and_output_config(self):
        captured: Dict[str, Any] = {}

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
                    model=OPUS_47,
                    id="msg_01test",
                )

            messages_ns = SimpleNamespace(create=AsyncMock(side_effect=_create))
            return SimpleNamespace(messages=messages_ns)

        with patch(
            "defog.llm.providers.anthropic_provider.AsyncAnthropic",
            side_effect=fake_client_ctor,
        ):
            await chat_async(
                provider="anthropic",
                model=OPUS_47,
                messages=[{"role": "user", "content": "hi"}],
                task_budget=64000,
                max_retries=1,
            )

        headers = captured["client_kwargs"]["default_headers"]
        assert "task-budgets-2026-03-13" in headers["anthropic-beta"]

        request_params = captured["request_params"]
        assert request_params["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": 64000,
        }

    @pytest.mark.asyncio
    async def test_task_budget_dict_form_wires_through(self):
        captured: Dict[str, Any] = {}

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
                    model=OPUS_47,
                    id="msg_01test",
                )

            messages_ns = SimpleNamespace(create=AsyncMock(side_effect=_create))
            return SimpleNamespace(messages=messages_ns)

        with patch(
            "defog.llm.providers.anthropic_provider.AsyncAnthropic",
            side_effect=fake_client_ctor,
        ):
            await chat_async(
                provider="anthropic",
                model=OPUS_47,
                messages=[{"role": "user", "content": "hi"}],
                task_budget={
                    "type": "tokens",
                    "total": 64000,
                    "remaining": 30000,
                },
                max_retries=1,
            )

        request_params = captured["request_params"]
        assert request_params["output_config"]["task_budget"] == {
            "type": "tokens",
            "total": 64000,
            "remaining": 30000,
        }
