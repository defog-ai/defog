"""Disk-backed conversation cache for providers without stateful sessions."""

from copy import deepcopy
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from diskcache import Cache

from defog import config as defog_config

_CACHE: Optional[Cache] = None
_CACHE_LOCK = threading.Lock()


def _get_cache_directory() -> Path:
    configured_dir = defog_config.get("LLM_CONVERSATION_CACHE_DIR")
    if configured_dir:
        path = Path(configured_dir).expanduser()
    else:
        path = Path.home() / ".defog" / "cache" / "llm_conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache() -> Cache:
    """Return global disk cache instance."""
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = Cache(str(_get_cache_directory()))
    return _CACHE


def load_messages(response_id: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Load messages for a cached conversation."""
    if not response_id:
        return None
    data = get_cache().get(response_id)
    if not data:
        return None
    messages = data.get("messages") if isinstance(data, dict) else data
    if messages is None:
        return None
    return deepcopy(messages)


def store_messages(
    response_id: str, messages: List[Dict[str, Any]], expire: Optional[int] = None
) -> None:
    """Persist conversation messages under a response id."""
    if not response_id:
        return
    payload = {"messages": deepcopy(messages)}
    get_cache().set(response_id, payload, expire=expire)


def clear_cache() -> None:
    """Clear all cached conversations (primarily for tests)."""
    get_cache().clear()
