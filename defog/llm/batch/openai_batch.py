"""OpenAI Batch API: upload a JSONL file, create the batch, read the files."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .requests import BatchError, BatchRequests, OPENAI_ENDPOINT
from .results import BatchJob, BatchResult, BatchStatus, BatchUsage, get

FINAL_STATUSES = ("completed", "failed", "expired", "cancelled")


async def submit(
    client: Any, requests: BatchRequests, metadata: Optional[Dict[str, str]] = None
) -> BatchJob:
    uploaded = await client.files.create(
        file=("batch.jsonl", requests.to_jsonl(), "application/jsonl"),
        purpose="batch",
    )
    create_args: Dict[str, Any] = {
        "input_file_id": uploaded.id,
        "endpoint": OPENAI_ENDPOINT,
        "completion_window": "24h",
    }
    if metadata:
        create_args["metadata"] = metadata
    batch = await client.batches.create(**create_args)
    return BatchJob(
        provider="openai",
        model=requests.model,
        batch_id=batch.id,
        ids=requests.ids,
        input_file_id=uploaded.id,
    )


def _errors(batch: Any) -> List[str]:
    errors = get(batch, "errors")
    data = (get(errors, "data") or []) if errors is not None else []
    messages = []
    for error in data:
        code = get(error, "code")
        message = get(error, "message")
        line = get(error, "line")
        where = f" (line {line})" if line is not None else ""
        messages.append(f"{code}: {message}{where}")
    return messages


async def poll(client: Any, job: BatchJob) -> BatchStatus:
    batch = await client.batches.retrieve(job.batch_id)
    counts = get(batch, "request_counts")
    return BatchStatus(
        batch_id=job.batch_id,
        provider="openai",
        status=batch.status,
        done=batch.status in FINAL_STATUSES,
        counts={
            "total": get(counts, "total", 0) or 0,
            "completed": get(counts, "completed", 0) or 0,
            "failed": get(counts, "failed", 0) or 0,
        }
        if counts is not None
        else {},
        errors=_errors(batch),
    )


async def _read_jsonl(client: Any, file_id: Optional[str]) -> List[Dict[str, Any]]:
    if not file_id:
        return []
    content = await client.files.content(file_id)
    text = content.text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _usage(usage: Optional[Dict[str, Any]]) -> Optional[BatchUsage]:
    if not usage:
        return None
    raw_input = usage.get("input_tokens") or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return BatchUsage(
        # OpenAI's input_tokens includes the cached tokens.
        input_tokens=max(raw_input - cached, 0),
        cached_input_tokens=cached,
        output_tokens=usage.get("output_tokens") or 0,
        reasoning_tokens=(usage.get("output_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
    )


def _text_and_refusal(body: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    texts, refusals = [], []
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text") or "")
            elif part.get("type") == "refusal":
                refusals.append(part.get("refusal") or "")
    return ("".join(texts) if texts else None), (
        "".join(refusals) if refusals else None
    )


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message")
        return f"{code}: {message}" if code else str(message)
    return str(error)


def to_result(line: Dict[str, Any], model: str) -> BatchResult:
    """Turn one line of the output or error file into a BatchResult."""
    custom_id = line.get("custom_id")
    response = line.get("response") or {}
    body = response.get("body") or {}
    status_code = response.get("status_code")
    error = line.get("error")
    result = BatchResult(
        id=custom_id, status="succeeded", provider="openai", model=model, raw=line
    )

    if error or (status_code is not None and status_code != 200):
        error = error or body.get("error") or {"message": f"HTTP {status_code}"}
        code = error.get("code") if isinstance(error, dict) else None
        if code == "batch_expired":
            result.status = "expired"
        elif code == "batch_cancelled":
            result.status = "canceled"
        else:
            result.status = "errored"
        result.error = _error_text(error)
        result.usage = _usage(body.get("usage"))
        return result

    result.usage = _usage(body.get("usage"))
    text, refusal = _text_and_refusal(body)
    result.text = text
    response_status = body.get("status")
    if response_status == "incomplete":
        result.status = "incomplete"
        result.stop_reason = (body.get("incomplete_details") or {}).get("reason")
    elif refusal is not None and text is None:
        result.status = "refused"
        result.error = refusal
    elif response_status not in (None, "completed"):
        result.status = "errored"
        result.error = _error_text(body.get("error") or response_status)
    return result


async def collect(client: Any, job: BatchJob) -> List[BatchResult]:
    batch = await client.batches.retrieve(job.batch_id)
    if batch.status not in FINAL_STATUSES:
        raise BatchError(
            f"OpenAI batch {job.batch_id} has not finished (status {batch.status!r})."
        )
    if batch.status == "failed":
        details = "; ".join(_errors(batch)) or "no details given"
        raise BatchError(f"OpenAI batch {job.batch_id} failed: {details}")
    lines = await _read_jsonl(client, get(batch, "output_file_id"))
    lines += await _read_jsonl(client, get(batch, "error_file_id"))
    return [to_result(line, job.model) for line in lines]
