"""Build provider request lines for a Batch job.

The request bodies come from the same ``build_params`` methods that
``chat_async`` uses, so a batched request asks the model the same thing as a
live call with the same messages, schema and reasoning effort.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from pydantic import BaseModel

from ..exceptions import ConfigurationError, LLMError
from ..llm_providers import LLMProvider
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_provider import OpenAIProvider
from ..utils import map_model_to_provider

# Published provider limits for one batch. A batch over a limit is refused
# with an error; it is never split behind the caller's back. Sizes use
# decimal megabytes, the smaller reading of "MB".
OPENAI_MAX_REQUESTS = 50_000
OPENAI_MAX_BYTES = 200 * 1_000_000
ANTHROPIC_MAX_REQUESTS = 100_000
ANTHROPIC_MAX_BYTES = 256 * 1_000_000

# Anthropic accepts 1-64 letters, digits, "_" and "-" as a custom_id.
ANTHROPIC_CUSTOM_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")

# OpenAIProvider.build_params builds Responses API requests.
OPENAI_ENDPOINT = "/v1/responses"

# build_params never uses the API key. A placeholder keeps the provider
# from looking one up while requests are built.
_BUILD_ONLY_KEY = "not-used-for-building-requests"


class BatchError(LLMError):
    """A batch cannot be submitted or collected."""


@dataclass
class BatchItem:
    """One request in a batch.

    ``id`` is returned with the result, so it must be unique in the batch.
    ``response_format`` is an optional pydantic model; when set, the model
    is asked for JSON that matches it, and ``collect`` can validate each
    answer against it.
    """

    id: str
    messages: List[Dict[str, Any]]
    response_format: Optional[type[BaseModel]] = None
    reasoning_effort: Optional[str] = None
    max_completion_tokens: Optional[int] = None


@dataclass
class BatchRequests:
    """Provider request lines, ready for ``submit``."""

    provider: str
    model: str
    lines: List[Dict[str, Any]]
    response_formats: Dict[str, type[BaseModel]] = field(default_factory=dict)

    @property
    def ids(self) -> List[str]:
        return [line["custom_id"] for line in self.lines]

    def to_jsonl(self) -> bytes:
        """The lines as a JSONL file (the OpenAI Batch input format)."""
        return "".join(json.dumps(line) + "\n" for line in self.lines).encode()


def _as_item(item: Union[BatchItem, Mapping[str, Any]]) -> BatchItem:
    if isinstance(item, BatchItem):
        return item
    if isinstance(item, Mapping):
        return BatchItem(**item)
    raise TypeError(
        f"Batch items must be BatchItem objects or mappings, got {type(item).__name__}"
    )


def _describe(ids: List[str]) -> str:
    shown = ", ".join(repr(i) for i in ids[:5])
    more = f" and {len(ids) - 5} more" if len(ids) > 5 else ""
    return shown + more


def validate_ids(ids: List[Any], provider: str) -> None:
    """Raise ValueError if any id is not usable. Ids are never changed."""
    not_strings = [i for i in ids if not isinstance(i, str) or not i]
    if not_strings:
        raise ValueError(
            f"Batch item ids must be non-empty strings: {_describe(not_strings)}"
        )
    seen, duplicates = set(), []
    for i in ids:
        if i in seen and i not in duplicates:
            duplicates.append(i)
        seen.add(i)
    if duplicates:
        raise ValueError(
            f"Batch item ids must be unique; repeated: {_describe(duplicates)}"
        )
    if provider == LLMProvider.ANTHROPIC.value:
        invalid = [i for i in ids if not ANTHROPIC_CUSTOM_ID_PATTERN.fullmatch(i)]
        if invalid:
            raise ValueError(
                "Anthropic batch ids must be 1-64 characters of letters, digits, "
                f"'_' or '-' (pattern ^[a-zA-Z0-9_-]{{1,64}}$). Invalid ids: "
                f"{_describe(invalid)}. Map them to valid ids before building "
                "the batch and keep your own map back to the original ids."
            )


def _check_limits(provider: str, count: int, size: int) -> None:
    if provider == LLMProvider.OPENAI.value:
        max_requests, max_bytes = OPENAI_MAX_REQUESTS, OPENAI_MAX_BYTES
    else:
        max_requests, max_bytes = ANTHROPIC_MAX_REQUESTS, ANTHROPIC_MAX_BYTES
    if count > max_requests:
        raise ValueError(
            f"{count} requests is more than the {provider} limit of "
            f"{max_requests} requests per batch. Split the items into "
            "several batches."
        )
    if size > max_bytes:
        raise ValueError(
            f"The batch is {size} bytes, more than the {provider} limit of "
            f"{max_bytes} bytes per batch. Split the items into several batches."
        )


def _openai_line(
    builder: OpenAIProvider, item: BatchItem, model: str
) -> Dict[str, Any]:
    params, _ = builder.build_params(
        messages=item.messages,
        model=model,
        max_completion_tokens=item.max_completion_tokens,
        reasoning_effort=item.reasoning_effort,
    )
    # timeout is a client setting, not part of the request body. None means
    # "not set", so leave those keys out of the JSON body.
    params.pop("timeout", None)
    body = {key: value for key, value in params.items() if value is not None}
    if item.response_format is not None:
        # The same conversion that client.responses.parse(text_format=...)
        # uses for a live call. It lives in a private openai module, so an
        # openai release may move it.
        try:
            from openai.lib._parsing._responses import type_to_text_format_param
        except ImportError as exc:
            import openai

            version = getattr(openai, "__version__", "unknown")
            raise ConfigurationError(
                "Structured output schemas for OpenAI batch requests need an "
                "openai version that provides openai.lib._parsing._responses."
                f"type_to_text_format_param. The installed openai version is "
                f"{version}. Install a compatible openai version."
            ) from exc

        body["text"] = {"format": type_to_text_format_param(item.response_format)}
    return {
        "custom_id": item.id,
        "method": "POST",
        "url": OPENAI_ENDPOINT,
        "body": body,
    }


def _anthropic_line(
    builder: AnthropicProvider, item: BatchItem, model: str
) -> Dict[str, Any]:
    # build_params adds the schema (output_config.format), the thinking
    # setting and the top-level cache_control, as for a live call.
    params, _ = builder.build_params(
        messages=item.messages,
        model=model,
        max_completion_tokens=item.max_completion_tokens,
        reasoning_effort=item.reasoning_effort,
        response_format=item.response_format,
    )
    params.pop("timeout", None)
    return {"custom_id": item.id, "params": params}


def build_requests(
    items: Iterable[Union[BatchItem, Mapping[str, Any]]], model: str
) -> BatchRequests:
    """Build the provider request lines for ``items`` sent to ``model``.

    The provider is picked from the model name (OpenAI and Anthropic models
    are supported). Raises ValueError for empty input, invalid or repeated
    ids, and batches over the provider's request-count or size limit, and
    ConfigurationError for a model of another provider or for an OpenAI
    schema when the installed openai version cannot convert it.
    """
    batch_items = [_as_item(item) for item in items]
    if not batch_items:
        raise ValueError("A batch needs at least one item.")

    provider = map_model_to_provider(model).value
    if provider not in (LLMProvider.OPENAI.value, LLMProvider.ANTHROPIC.value):
        raise ConfigurationError(
            f"Batch requests support OpenAI and Anthropic models; "
            f"'{model}' belongs to provider '{provider}'."
        )

    validate_ids([item.id for item in batch_items], provider)

    for item in batch_items:
        schema = item.response_format
        if schema is not None and not (
            isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            raise TypeError(
                f"Item {item.id!r}: response_format must be a pydantic model class."
            )

    if provider == LLMProvider.OPENAI.value:
        openai_builder = OpenAIProvider(api_key=_BUILD_ONLY_KEY)
        lines = [_openai_line(openai_builder, item, model) for item in batch_items]
        size = sum(len(json.dumps(line)) + 1 for line in lines)
    else:
        anthropic_builder = AnthropicProvider(api_key=_BUILD_ONLY_KEY)
        lines = [
            _anthropic_line(anthropic_builder, item, model) for item in batch_items
        ]
        size = len(json.dumps({"requests": lines}))

    _check_limits(provider, len(lines), size)

    return BatchRequests(
        provider=provider,
        model=model,
        lines=lines,
        response_formats={
            item.id: item.response_format
            for item in batch_items
            if item.response_format is not None
        },
    )
