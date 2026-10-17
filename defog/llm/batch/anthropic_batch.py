"""Anthropic Message Batches: requests go in the create call, results stream."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .requests import BatchError, BatchRequests
from .results import BatchJob, BatchResult, BatchStatus, BatchUsage, get, to_dict

# stop_reason values that mean the answer was cut off.
CUT_OFF_STOP_REASONS = ("max_tokens", "model_context_window_exceeded")


async def submit(client: Any, requests: BatchRequests) -> BatchJob:
    batch = await client.messages.batches.create(requests=requests.lines)
    return BatchJob(
        provider="anthropic",
        model=requests.model,
        batch_id=batch.id,
        ids=requests.ids,
    )


async def poll(client: Any, job: BatchJob) -> BatchStatus:
    batch = await client.messages.batches.retrieve(job.batch_id)
    counts = get(batch, "request_counts")
    return BatchStatus(
        batch_id=job.batch_id,
        provider="anthropic",
        status=batch.processing_status,
        done=batch.processing_status == "ended",
        counts={
            name: get(counts, name, 0) or 0
            for name in ("processing", "succeeded", "errored", "canceled", "expired")
        }
        if counts is not None
        else {},
    )


def _usage(usage: Optional[Dict[str, Any]]) -> Optional[BatchUsage]:
    if not usage:
        return None
    one_hour = (usage.get("cache_creation") or {}).get("ephemeral_1h_input_tokens")
    return BatchUsage(
        # Anthropic's input_tokens excludes cache reads and cache writes.
        input_tokens=usage.get("input_tokens") or 0,
        cached_input_tokens=usage.get("cache_read_input_tokens") or 0,
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens") or 0,
        cache_creation_1h_input_tokens=one_hour or 0,
        output_tokens=usage.get("output_tokens") or 0,
    )


def _error_text(error: Any) -> str:
    # An errored result holds an error response: {"type": "error", "error": {...}}
    inner = error.get("error") if isinstance(error, dict) else None
    if isinstance(inner, dict):
        error = inner
    if isinstance(error, dict):
        kind = error.get("type")
        message = error.get("message")
        return f"{kind}: {message}" if kind else str(message)
    return str(error)


def to_result(entry: Any, model: str) -> BatchResult:
    """Turn one line of the results stream into a BatchResult."""
    line = to_dict(entry)
    outcome = line.get("result") or {}
    kind = outcome.get("type")
    result = BatchResult(
        id=line.get("custom_id"),
        status="errored",
        provider="anthropic",
        model=model,
        raw=line,
    )
    if kind == "succeeded":
        message = outcome.get("message") or {}
        texts = [
            block.get("text") or ""
            for block in message.get("content") or []
            if block.get("type") == "text"
        ]
        result.text = "".join(texts) if texts else None
        result.usage = _usage(message.get("usage"))
        result.stop_reason = message.get("stop_reason")
        if result.stop_reason in CUT_OFF_STOP_REASONS:
            result.status = "incomplete"
        elif result.stop_reason == "refusal":
            result.status = "refused"
        else:
            result.status = "succeeded"
    elif kind == "expired":
        result.status = "expired"
    elif kind == "canceled":
        result.status = "canceled"
    else:
        result.error = _error_text(outcome.get("error") or kind)
    return result


async def collect(client: Any, job: BatchJob) -> List[BatchResult]:
    batch = await client.messages.batches.retrieve(job.batch_id)
    if batch.processing_status != "ended":
        raise BatchError(
            f"Anthropic batch {job.batch_id} has not ended "
            f"(status {batch.processing_status!r})."
        )
    stream = await client.messages.batches.results(job.batch_id)
    return [to_result(entry, job.model) async for entry in stream]
