from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from defog.llm.config import LLMConfig
from defog.llm.providers.base import LLMResponse
from defog.llm.providers.openrouter_provider import OpenRouterProvider
from defog.llm.utils import chat_async


def test_build_params_providers_list_maps_to_only():
    provider = OpenRouterProvider(api_key="sk-test")

    request_params, _ = provider.build_params(
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/gpt-5-mini",
        providers=["azure", "openai"],
    )

    assert request_params["extra_body"]["provider"] == {"only": ["azure", "openai"]}


def test_build_params_provider_routing_dict_is_passed_through_and_normalized():
    provider = OpenRouterProvider(api_key="sk-test")

    request_params, _ = provider.build_params(
        messages=[{"role": "user", "content": "Hello"}],
        model="anthropic/claude-sonnet-4.6",
        providers={"order": ["anthropic"], "allowFallbacks": False},
    )

    assert request_params["extra_body"]["provider"] == {
        "order": ["anthropic"],
        "allow_fallbacks": False,
    }
    assert request_params["extra_body"]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("providers", [[], ["anthropic", ""], "anthropic"])
def test_build_params_rejects_invalid_providers(providers):
    provider = OpenRouterProvider(api_key="sk-test")

    with pytest.raises(ValueError, match="providers"):
        provider.build_params(
            messages=[{"role": "user", "content": "Hello"}],
            model="openai/gpt-5-mini",
            providers=providers,
        )


@pytest.mark.asyncio
async def test_chat_async_forwards_openrouter_providers(monkeypatch):
    captured = {}

    async def fake_execute_chat(self, **kwargs):
        captured.update(kwargs)
        return LLMResponse(
            content="ok",
            model=kwargs["model"],
            time=0.0,
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr(OpenRouterProvider, "execute_chat", fake_execute_chat)

    response = await chat_async(
        provider="openrouter",
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "Hello"}],
        providers=["azure"],
        config=LLMConfig(api_keys={"openrouter": "sk-test"}),
        max_retries=1,
    )

    assert response.content == "ok"
    assert captured["providers"] == ["azure"]


@pytest.mark.asyncio
async def test_chat_async_rejects_providers_for_non_openrouter():
    with pytest.raises(ValueError, match="openrouter"):
        await chat_async(
            provider="openai",
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Hello"}],
            providers=["azure"],
            config=LLMConfig(api_keys={"openai": "sk-test"}),
            max_retries=1,
        )


@pytest.mark.asyncio
async def test_llm_json_repair_forwards_provider_extra_body():
    provider = OpenRouterProvider(api_key="sk-test")
    extra_body = {"provider": {"only": ["azure"]}}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="fixed"))]
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )

    result = await provider._llm_json_repair(
        broken_content="broken",
        response_format=None,
        client=client,
        model="openai/gpt-5-mini",
        request_params={"messages": [], "extra_body": extra_body},
    )

    assert result == "fixed"
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == extra_body
