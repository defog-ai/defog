from __future__ import annotations

from typing import Any, Dict

import pytest

from defog.llm.config import LLMConfig
from defog.llm.providers.base import LLMResponse
from defog.llm.providers.openai_provider import OpenAIProvider
from defog.llm.utils import chat_async


def test_build_params_enables_flex_service_tier():
    provider = OpenAIProvider(api_key="sk-test")

    request_params, _ = provider.build_params(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5-mini",
        flex_processing=True,
    )

    assert request_params["service_tier"] == "flex"


def test_build_params_omits_service_tier_by_default():
    provider = OpenAIProvider(api_key="sk-test")

    request_params, _ = provider.build_params(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5-mini",
    )

    assert "service_tier" not in request_params


@pytest.mark.asyncio
async def test_chat_async_forwards_flex_processing(monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_execute_chat(self, **kwargs):
        captured.update(kwargs)
        return LLMResponse(
            content="ok",
            model=kwargs["model"],
            time=0.0,
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr(OpenAIProvider, "execute_chat", fake_execute_chat)

    response = await chat_async(
        provider="openai",
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        flex_processing=True,
        config=LLMConfig(api_keys={"openai": "sk-test"}),
        max_retries=1,
    )

    assert response.content == "ok"
    assert captured["flex_processing"] is True


@pytest.mark.asyncio
async def test_chat_async_rejects_flex_processing_for_non_openai_provider():
    with pytest.raises(ValueError, match="only supported.*openai"):
        await chat_async(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hello"}],
            flex_processing=True,
        )
