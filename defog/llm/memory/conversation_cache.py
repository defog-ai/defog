"""Async file-backed conversation cache for providers without stateful sessions."""

import os
import pickle
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Set, Tuple

import aiofiles

from defog import config as defog_config


def _get_cache_directory() -> Path:
    configured_dir = defog_config.get("LLM_CONVERSATION_CACHE_DIR")
    if configured_dir:
        path = Path(configured_dir).expanduser()
    else:
        path = Path.home() / ".defog" / "cache" / "llm_conversations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_filename(response_id: str) -> str:
    """Replace non-alphanumeric characters (except _ and -) for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", response_id)


def _get_pickle_path(response_id: str) -> Path:
    return _get_cache_directory() / f"{_sanitize_filename(response_id)}.pkl"


async def load_messages(
    response_id: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    """Load messages for a cached conversation."""
    if not response_id:
        return None
    path = _get_pickle_path(response_id)
    if not path.exists():
        return None
    try:
        async with aiofiles.open(path, "rb") as f:
            raw = await f.read()
        data = pickle.loads(raw)
        # Check TTL
        expire_at = data.get("expire_at") if isinstance(data, dict) else None
        if expire_at is not None and time.time() > expire_at:
            path.unlink(missing_ok=True)
            return None
        messages = data.get("messages") if isinstance(data, dict) else data
        if messages is None:
            return None
        return deepcopy(messages)
    except Exception:
        return None


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


async def store_messages(
    response_id: str, messages: List[Dict[str, Any]], expire: Optional[int] = None
) -> None:
    """Persist conversation messages under a response id."""
    if not response_id:
        return
    expanded_messages = _expand_messages_with_parents(messages)
    payload: Dict[str, Any] = {"messages": deepcopy(expanded_messages)}
    if expire is not None:
        payload["expire_at"] = time.time() + expire
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path = _get_pickle_path(response_id)
    tmp_path = path.with_suffix(".pkl.tmp")
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(raw)
    os.replace(str(tmp_path), str(path))


async def clear_cache() -> None:
    """Clear all cached conversations (primarily for tests)."""
    cache_dir = _get_cache_directory()
    for p in cache_dir.glob("*.pkl*"):
        p.unlink(missing_ok=True)
