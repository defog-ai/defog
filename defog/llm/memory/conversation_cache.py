"""Disk-backed conversation cache for providers without stateful sessions."""

from copy import deepcopy
import threading
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

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


def _message_signature(message: Dict[str, Any]) -> Tuple[str, Hashable]:
    """Return a stable signature for deduplicating messages while expanding parents."""
    for key in ("id", "message_id", "response_id", "uuid"):
        value = message.get(key)
        if isinstance(value, str):
            return ("field", f"{key}:{value}")
    return ("object", id(message))


def _collect_message_with_parents(
    message: Dict[str, Any],
    seen: Set[Tuple[str, Hashable]],
    order: List[Dict[str, Any]],
    visiting: Set[Tuple[str, Hashable]],
) -> None:
    """Recursively add parent messages ahead of the provided message."""
    signature = _message_signature(message)
    if signature in seen or signature in visiting:
        return

    visiting.add(signature)

    parent = message.get("parent")
    if isinstance(parent, dict):
        _collect_message_with_parents(parent, seen, order, visiting)
    elif isinstance(parent, (list, tuple)):
        for parent_msg in parent:
            if isinstance(parent_msg, dict):
                _collect_message_with_parents(parent_msg, seen, order, visiting)

    visiting.remove(signature)

    if signature not in seen:
        seen.add(signature)
        order.append(message)


def _expand_messages_with_parents(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return messages including any recursively referenced parent messages."""
    expanded: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Hashable]] = set()
    visiting: Set[Tuple[str, Hashable]] = set()

    for message in messages:
        if isinstance(message, dict):
            _collect_message_with_parents(message, seen, expanded, visiting)
        else:
            expanded.append(message)

    return expanded


def store_messages(
    response_id: str, messages: List[Dict[str, Any]], expire: Optional[int] = None
) -> None:
    """Persist conversation messages under a response id."""
    if not response_id:
        return
    expanded_messages = _expand_messages_with_parents(messages)
    payload = {"messages": deepcopy(expanded_messages)}
    get_cache().set(response_id, payload, expire=expire)


def clear_cache() -> None:
    """Clear all cached conversations (primarily for tests)."""
    get_cache().clear()
