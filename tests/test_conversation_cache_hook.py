"""Unit tests for the supplemental ConversationCache hook on BaseLLMProvider.

These exercise the cache semantics without hitting a real provider:
    - Store writes to both the pickle cache and the supplemental cache.
    - Load tries the supplemental cache first; falls back to pickle on miss.
    - With no cache passed, behavior is identical to the pre-hook pickle-only path.
"""

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from defog.llm.memory.conversation_cache import ConversationCache
from defog.llm.providers.base import BaseLLMProvider


class _DummyProvider(BaseLLMProvider):
    """Minimal concrete provider; only needed so we can instantiate BaseLLMProvider."""

    def get_provider_name(self) -> str:
        return "dummy"

    def build_params(self, *args, **kwargs):
        return {}, {}

    async def execute_chat(self, *args, **kwargs):
        raise NotImplementedError

    async def process_response(self, *args, **kwargs):
        raise NotImplementedError

    def create_image_message(self, *args, **kwargs):
        raise NotImplementedError


class RecordingCache:
    """In-memory ConversationCache implementation that records every call."""

    def __init__(self) -> None:
        self.storage: Dict[str, List[Dict[str, Any]]] = {}
        self.load_calls: List[str] = []
        self.store_calls: List[tuple] = []

    async def load(self, response_id: str) -> Optional[List[Dict[str, Any]]]:
        self.load_calls.append(response_id)
        return self.storage.get(response_id)

    async def store(
        self,
        response_id: str,
        messages: List[Dict[str, Any]],
        expire: Optional[int] = None,
    ) -> None:
        self.store_calls.append((response_id, messages, expire))
        self.storage[response_id] = messages


@pytest.mark.asyncio
async def test_recording_cache_satisfies_protocol() -> None:
    """A duck-typed class with the right coroutines is accepted as ConversationCache."""
    assert isinstance(RecordingCache(), ConversationCache)


@pytest.mark.asyncio
async def test_store_mirrors_to_supplemental_cache(tmp_path, monkeypatch) -> None:
    """persist_conversation_history writes pickle and also calls supplemental.store."""
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))

    provider = _DummyProvider()
    cache = RecordingCache()
    messages = [{"role": "user", "content": "hi"}]

    await provider.persist_conversation_history("rid-1", messages, cache)

    # Pickle file was written (default behavior).
    pickle_files = list(tmp_path.glob("*.pkl"))
    assert len(pickle_files) == 1

    # Supplemental cache also received the write.
    assert cache.store_calls == [("rid-1", messages, None)]


@pytest.mark.asyncio
async def test_load_prefers_supplemental_cache(tmp_path, monkeypatch) -> None:
    """If supplemental.load returns a hit, pickle is not consulted."""
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))

    provider = _DummyProvider()
    pickle_messages = [{"role": "user", "content": "from pickle"}]
    supplemental_messages = [{"role": "user", "content": "from supplemental"}]

    # Seed both stores with different content keyed by the same response id.
    await provider.persist_conversation_history("rid-2", pickle_messages, None)
    cache = RecordingCache()
    cache.storage["rid-2"] = supplemental_messages

    loaded = await provider._load_cached_conversation("rid-2", cache)

    assert loaded == supplemental_messages
    assert cache.load_calls == ["rid-2"]


@pytest.mark.asyncio
async def test_load_falls_back_to_pickle_on_supplemental_miss(
    tmp_path, monkeypatch
) -> None:
    """If supplemental.load returns None, pickle is used."""
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))

    provider = _DummyProvider()
    pickle_messages = [{"role": "user", "content": "from pickle"}]
    await provider.persist_conversation_history("rid-3", pickle_messages, None)

    cache = RecordingCache()  # empty
    loaded = await provider._load_cached_conversation("rid-3", cache)

    assert loaded == pickle_messages
    assert cache.load_calls == ["rid-3"]


@pytest.mark.asyncio
async def test_no_cache_preserves_pickle_only_behavior(tmp_path, monkeypatch) -> None:
    """With no supplemental cache, everything works exactly as before."""
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))

    provider = _DummyProvider()
    messages = [{"role": "user", "content": "plain"}]

    await provider.persist_conversation_history("rid-4", messages)
    loaded = await provider._load_cached_conversation("rid-4")

    assert loaded == messages


@pytest.mark.asyncio
async def test_supplemental_store_failure_does_not_prevent_pickle_write(
    tmp_path, monkeypatch
) -> None:
    """Errors in the supplemental store are logged and swallowed."""
    monkeypatch.setenv("LLM_CONVERSATION_CACHE_DIR", str(tmp_path))

    class BrokenCache:
        async def load(self, response_id):
            return None

        async def store(self, response_id, messages, expire=None):
            raise RuntimeError("backend is down")

    provider = _DummyProvider()
    messages = [{"role": "user", "content": "hi"}]

    # Must not raise.
    await provider.persist_conversation_history("rid-5", messages, BrokenCache())

    # Pickle write still succeeded.
    loaded = await provider._load_cached_conversation("rid-5")
    assert loaded == messages
