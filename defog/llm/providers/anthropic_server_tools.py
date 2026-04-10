"""Helpers for building and normalizing Anthropic server-side tool specs.

Anthropic exposes several first-party "server-side" tools that the model can
use directly without us writing the implementation: web_search, web_fetch,
code_execution, and advisor. This module provides:

- Builders for each tool's spec dict
- A ``normalize_server_tools`` function that accepts the user's loose input
  shape (list of names, dict of name -> config, or raw dict list) and returns
  the canonical list of tool spec dicts plus the set of beta headers needed.

The defaults track the latest tool versions that support dynamic filtering
and programmatic tool calling. These versions are Claude API + Microsoft
Foundry only (no Bedrock/Vertex) and require Claude Opus 4.6 / Sonnet 4.6
or newer Mythos Preview models. Bedrock/Vertex callers must override the
version explicitly via the dict form of ``server_tools``.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


ANTHROPIC_SERVER_TOOL_NAMES = {
    "web_search",
    "web_fetch",
    "code_execution",
    "advisor",
}

# Default versions: latest with dynamic filtering / programmatic-calling support.
# NOTE: These versions are Claude API + Microsoft Foundry only (no Bedrock/Vertex)
# and require Claude Opus 4.6 / Sonnet 4.6 / Mythos Preview models. Callers on
# Bedrock or Vertex must override via the dict form of server_tools to use the
# stable versions (web_search_20250305, web_fetch_20250910, code_execution_20250825).
DEFAULT_VERSIONS: Dict[str, str] = {
    "web_search": "web_search_20260209",
    "web_fetch": "web_fetch_20260209",
    "code_execution": "code_execution_20260120",
    "advisor": "advisor_20260301",
}

# Beta headers needed by tool — only advisor today (the others are GA on the
# first-party Claude API).
SERVER_TOOL_BETA_HEADERS: Dict[str, str] = {
    "advisor": "advisor-tool-2026-03-01",
}

# Models known to support dynamic filtering / programmatic tool calling with
# the latest tool versions. Used to surface a clear error if the caller pairs
# the new versions with an unsupported model rather than letting the API
# return a vague 400.
DYNAMIC_FILTERING_SUPPORTED_MODELS: Tuple[str, ...] = (
    "opus-4-6",
    "sonnet-4-6",
)

# Versions that require dynamic-filtering-capable executor models
# (Opus 4.6 / Sonnet 4.6 or newer). web_search and web_fetch at these
# versions only run on those models; we validate up front to surface a
# clear error.
_DYNAMIC_FILTERING_VERSIONS = {
    "web_search_20260209",
    "web_fetch_20260209",
}

# Models that support the advisor tool as an executor. Advisor is a beta
# feature; the executor is one of these models and the advisor model
# (passed via the ``model`` field) is typically Opus 4.6.
_ADVISOR_EXECUTOR_SUPPORTED_MODELS: Tuple[str, ...] = (
    "haiku-4-5",
    "sonnet-4-6",
    "opus-4-6",
)

# Code execution version required for programmatic tool calling.
PROGRAMMATIC_CODE_EXEC_MIN_VERSION = "code_execution_20260120"


def _model_supports_dynamic_filtering(model: str) -> bool:
    return any(tag in model for tag in DYNAMIC_FILTERING_SUPPORTED_MODELS)


def _is_newer_or_equal_code_exec_version(version: str) -> bool:
    """Return True iff the given code_execution version is >= 20260120.

    Versions take the form ``code_execution_<YYYYMMDD>``. The string compare
    works because the date suffix is fixed-width.
    """
    if not version.startswith("code_execution_"):
        return False
    return version >= PROGRAMMATIC_CODE_EXEC_MIN_VERSION


def build_web_search_tool(
    version: Optional[str] = None,
    max_uses: Optional[int] = None,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
    user_location: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an Anthropic web_search tool spec."""
    spec: Dict[str, Any] = {
        "type": version or DEFAULT_VERSIONS["web_search"],
        "name": "web_search",
    }
    if max_uses is not None:
        spec["max_uses"] = max_uses
    if allowed_domains is not None:
        spec["allowed_domains"] = list(allowed_domains)
    if blocked_domains is not None:
        spec["blocked_domains"] = list(blocked_domains)
    if user_location is not None:
        spec["user_location"] = user_location
    return spec


def build_web_fetch_tool(
    version: Optional[str] = None,
    max_uses: Optional[int] = None,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
    citations: bool = False,
    max_content_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Build an Anthropic web_fetch tool spec."""
    spec: Dict[str, Any] = {
        "type": version or DEFAULT_VERSIONS["web_fetch"],
        "name": "web_fetch",
    }
    if max_uses is not None:
        spec["max_uses"] = max_uses
    if allowed_domains is not None:
        spec["allowed_domains"] = list(allowed_domains)
    if blocked_domains is not None:
        spec["blocked_domains"] = list(blocked_domains)
    if citations:
        spec["citations"] = {"enabled": True}
    if max_content_tokens is not None:
        spec["max_content_tokens"] = max_content_tokens
    return spec


def build_code_execution_tool(
    version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an Anthropic code_execution tool spec."""
    return {
        "type": version or DEFAULT_VERSIONS["code_execution"],
        "name": "code_execution",
    }


def build_advisor_tool(
    model: str,
    version: Optional[str] = None,
    max_uses: Optional[int] = None,
    caching: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an Anthropic advisor tool spec.

    ``model`` is the advisor model to consult (e.g. ``claude-opus-4-6``).
    """
    spec: Dict[str, Any] = {
        "type": version or DEFAULT_VERSIONS["advisor"],
        "name": "advisor",
        "model": model,
    }
    if max_uses is not None:
        spec["max_uses"] = max_uses
    if caching is not None:
        spec["caching"] = caching
    return spec


def _build_from_name_and_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a server tool spec from a name + config dict."""
    if name == "web_search":
        return build_web_search_tool(
            version=config.get("version"),
            max_uses=config.get("max_uses"),
            allowed_domains=config.get("allowed_domains"),
            blocked_domains=config.get("blocked_domains"),
            user_location=config.get("user_location"),
        )
    if name == "web_fetch":
        return build_web_fetch_tool(
            version=config.get("version"),
            max_uses=config.get("max_uses"),
            allowed_domains=config.get("allowed_domains"),
            blocked_domains=config.get("blocked_domains"),
            citations=config.get("citations", False),
            max_content_tokens=config.get("max_content_tokens"),
        )
    if name == "code_execution":
        return build_code_execution_tool(version=config.get("version"))
    if name == "advisor":
        if "model" not in config:
            raise ValueError(
                "advisor server tool requires a 'model' key in its config "
                "(e.g. server_tools={'advisor': {'model': 'claude-opus-4-6'}})"
            )
        return build_advisor_tool(
            model=config["model"],
            version=config.get("version"),
            max_uses=config.get("max_uses"),
            caching=config.get("caching"),
        )
    raise ValueError(
        f"Unknown Anthropic server tool name: {name!r}. Known names: "
        f"{sorted(ANTHROPIC_SERVER_TOOL_NAMES)}"
    )


def _name_for_spec(spec: Dict[str, Any]) -> Optional[str]:
    """Return the canonical short name for a raw spec dict (e.g. 'web_search').

    Falls back to ``spec.get('name')``; if absent, infers from ``spec['type']``
    by stripping the date suffix.
    """
    if "name" in spec and spec["name"] in ANTHROPIC_SERVER_TOOL_NAMES:
        return spec["name"]
    type_str = spec.get("type", "")
    for known in ANTHROPIC_SERVER_TOOL_NAMES:
        if type_str.startswith(known + "_"):
            return known
    return None


def normalize_server_tools(
    server_tools: Union[List[str], List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    *,
    model: str,
    programmatic_tool_calling: bool,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Normalize ``server_tools`` to a list of Anthropic tool spec dicts.

    Accepts:
    - List of names (e.g. ``["web_search", "code_execution"]``)
    - Dict of name -> per-tool config (e.g. ``{"web_search": {"max_uses": 5}}``)
    - Raw list of dicts (escape hatch for full control / new versions)

    Returns ``(specs, beta_headers_needed)``.

    Behavior:
    - Applies ``DEFAULT_VERSIONS`` to any name-form entry.
    - Auto-adds ``code_execution_20260120`` if a dynamic-filtering version of
      ``web_search`` or ``web_fetch`` is present (Anthropic requires code
      execution alongside dynamic filtering). Logs a debug message.
    - Validates that the model supports dynamic filtering when those versions
      are requested; raises ``ValueError`` with a clear suggestion if not.
    - When ``programmatic_tool_calling`` is on, ensures ``code_execution``
      exists and forces its version to ``code_execution_20260120`` (or accepts
      a newer version); raises if an older version was explicitly passed.
    """
    if server_tools is None:
        server_tools = []

    specs: List[Dict[str, Any]] = []

    # Step 1: build the initial spec list from whichever input shape we got.
    if isinstance(server_tools, dict):
        # Dict of name -> config
        for name, config in server_tools.items():
            if name not in ANTHROPIC_SERVER_TOOL_NAMES:
                raise ValueError(
                    f"Unknown Anthropic server tool name: {name!r}. Known names: "
                    f"{sorted(ANTHROPIC_SERVER_TOOL_NAMES)}"
                )
            specs.append(_build_from_name_and_config(name, config or {}))
    elif isinstance(server_tools, list):
        for entry in server_tools:
            if isinstance(entry, str):
                if entry not in ANTHROPIC_SERVER_TOOL_NAMES:
                    raise ValueError(
                        f"Unknown Anthropic server tool name: {entry!r}. Known names: "
                        f"{sorted(ANTHROPIC_SERVER_TOOL_NAMES)}"
                    )
                specs.append(_build_from_name_and_config(entry, {}))
            elif isinstance(entry, dict):
                # Raw dict passthrough — but normalize/validate the inferred name
                spec = deepcopy(entry)
                if "type" not in spec:
                    raise ValueError(
                        "Raw server tool dict must include a 'type' field "
                        f"(got {entry!r})"
                    )
                inferred = _name_for_spec(spec)
                if inferred is None:
                    raise ValueError(
                        f"Could not determine server tool name from spec {entry!r}. "
                        "Set 'name' explicitly or use a known type prefix."
                    )
                if "name" not in spec:
                    spec["name"] = inferred
                specs.append(spec)
            else:
                raise ValueError(
                    f"server_tools list entries must be strings or dicts, "
                    f"got {type(entry).__name__}"
                )
    else:
        raise ValueError(
            "server_tools must be a list (of names or dicts) or a dict "
            f"of name -> config; got {type(server_tools).__name__}"
        )

    # Step 2: collect canonical names actually present.
    name_to_spec: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        n = _name_for_spec(spec)
        if n is not None:
            name_to_spec[n] = spec

    # Step 3: dynamic filtering — auto-add code_execution if web_search or
    # web_fetch is present at a dynamic-filtering version and code_execution is
    # not already in the list.
    dynamic_filtering_active = False
    for n in ("web_search", "web_fetch"):
        if n in name_to_spec:
            if name_to_spec[n].get("type") in _DYNAMIC_FILTERING_VERSIONS:
                dynamic_filtering_active = True
                break

    if dynamic_filtering_active and "code_execution" not in name_to_spec:
        logger.debug(
            "Auto-adding code_execution server tool because a dynamic-filtering "
            "web_search/web_fetch version was requested. Anthropic requires "
            "code execution to be enabled alongside dynamic filtering."
        )
        ce_spec = build_code_execution_tool()
        specs.append(ce_spec)
        name_to_spec["code_execution"] = ce_spec

    # Step 4: validate model supports dynamic filtering if any
    # dynamic-filtering version is in play.
    requires_dynamic_model = any(
        spec.get("type") in _DYNAMIC_FILTERING_VERSIONS for spec in specs
    )
    if requires_dynamic_model and not _model_supports_dynamic_filtering(model):
        raise ValueError(
            f"Model {model!r} does not support the default Anthropic server "
            "tool versions (which require Opus 4.6 / Sonnet 4.6 or newer). "
            "Either switch to a supported model, or override the version "
            "explicitly via the dict form of server_tools, e.g.: "
            "server_tools={'web_search': {'version': 'web_search_20250305'}}."
        )

    # Step 5: programmatic tool calling constraints.
    if programmatic_tool_calling:
        if "code_execution" not in name_to_spec:
            raise ValueError(
                "programmatic_tool_calling=True requires 'code_execution' to "
                "be enabled in server_tools."
            )
        ce_spec = name_to_spec["code_execution"]
        ce_version = ce_spec.get("type", "")
        if not _is_newer_or_equal_code_exec_version(ce_version):
            raise ValueError(
                f"programmatic_tool_calling=True requires code_execution "
                f"version >= {PROGRAMMATIC_CODE_EXEC_MIN_VERSION}, got "
                f"{ce_version!r}. Remove the explicit 'version' override or "
                "set it to a newer version."
            )

    # Step 6: collect beta headers needed.
    beta_headers: Set[str] = set()
    for n in name_to_spec:
        if n in SERVER_TOOL_BETA_HEADERS:
            beta_headers.add(SERVER_TOOL_BETA_HEADERS[n])

    return specs, beta_headers
