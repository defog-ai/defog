"""Submit, poll and collect a batch for either provider."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel

from defog import config as defog_config

from . import anthropic_batch, openai_batch
from .requests import BatchRequests
from .results import (
    BatchJob,
    BatchResult,
    BatchStatus,
    apply_response_format,
    price,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _client(provider: str, client: Any) -> AsyncIterator[Any]:
    """Use the caller's client, or create one and close it afterwards."""
    if client is not None:
        yield client
        return
    if provider == "openai":
        from openai import AsyncOpenAI

        async with AsyncOpenAI(api_key=defog_config.get("OPENAI_API_KEY")) as owned:
            yield owned
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic

        async with AsyncAnthropic(
            api_key=defog_config.get("ANTHROPIC_API_KEY")
        ) as owned:
            yield owned
    else:
        raise ValueError(f"Unsupported batch provider: {provider!r}")


async def submit(
    requests: BatchRequests,
    *,
    client: Any = None,
    metadata: Optional[Dict[str, str]] = None,
) -> BatchJob:
    """Send the requests built by ``build_requests`` as one batch.

    ``client`` is an ``AsyncOpenAI`` or ``AsyncAnthropic`` client; without
    one, a client is created from OPENAI_API_KEY or ANTHROPIC_API_KEY.
    ``metadata`` is attached to an OpenAI batch (Anthropic batches take
    none). Store ``job.to_dict()`` to collect the batch from another process.
    """
    async with _client(requests.provider, client) as active:
        if requests.provider == "openai":
            return await openai_batch.submit(active, requests, metadata)
        if metadata:
            raise ValueError("Anthropic Message Batches do not accept metadata.")
        return await anthropic_batch.submit(active, requests)


async def poll(job: BatchJob, *, client: Any = None) -> BatchStatus:
    """Check a batch once. ``status.done`` is True when it has ended."""
    async with _client(job.provider, client) as active:
        if job.provider == "openai":
            return await openai_batch.poll(active, job)
        return await anthropic_batch.poll(active, job)


# Providers expire a batch that has not finished after 24 hours. The default
# wait is a little longer, so an expired batch is seen as ended, not as a timeout.
DEFAULT_WAIT_TIMEOUT = 25 * 60 * 60.0


async def wait(
    job: BatchJob,
    *,
    client: Any = None,
    poll_interval: float = 60.0,
    timeout: Optional[float] = DEFAULT_WAIT_TIMEOUT,
) -> BatchStatus:
    """Poll until the batch has ended and return its last status.

    Raises TimeoutError when ``timeout`` seconds pass first (default: 25
    hours, a little over the 24 hours after which providers expire a batch;
    pass ``None`` to wait without a limit). The batch keeps running at the
    provider; call ``wait`` or ``collect`` again later. Raises ValueError
    when ``poll_interval`` is not above zero.
    """
    if poll_interval <= 0:
        raise ValueError("poll_interval must be greater than 0.")
    started = time.monotonic()
    async with _client(job.provider, client) as active:
        while True:
            status = await poll(job, client=active)
            if status.done:
                return status
            if timeout is not None and time.monotonic() - started >= timeout:
                raise TimeoutError(
                    f"Batch {job.batch_id} is still {status.status!r} after "
                    f"{timeout} seconds."
                )
            # Never sleep past the timeout.
            delay = poll_interval
            if timeout is not None:
                delay = min(delay, timeout - (time.monotonic() - started))
            await asyncio.sleep(delay)


async def collect(
    job: BatchJob,
    *,
    client: Any = None,
    response_format: Union[type[BaseModel], Mapping[str, type[BaseModel]], None] = None,
) -> List[BatchResult]:
    """Download the results of an ended batch, in the order of ``job.ids``.

    Returns exactly one result per requested id, with one status (see
    ``RESULT_STATUSES``). Expired, canceled and errored requests are
    returned, not retried. A requested id with no result line gets the
    status ``missing``. When the provider returns more than one line for an
    id, the first line is kept. Lines for an id that was not requested, or
    with no id, are dropped; every dropped line is named in one warning on
    this module's logger. Each result with usage is priced at the Batch
    rates in ``cost_in_cents``.

    With ``response_format`` (one pydantic model, or a mapping from id to
    model, such as ``BatchRequests.response_formats``), each succeeded
    answer is validated and stored in ``parsed``; an answer that does not
    match is marked ``invalid`` with the validation error.

    Raises BatchError when the batch has not ended, or when an OpenAI batch
    failed as a whole (for example a malformed input file).
    """
    async with _client(job.provider, client) as active:
        if job.provider == "openai":
            results = await openai_batch.collect(active, job)
        else:
            results = await anthropic_batch.collect(active, job)

    requested = set(job.ids)
    by_id: Dict[str, BatchResult] = {}
    repeated: List[Optional[str]] = []
    unknown: List[Optional[str]] = []
    for result in results:
        if result.id not in requested:
            unknown.append(result.id)
        elif result.id in by_id:
            repeated.append(result.id)
        else:
            by_id[result.id] = result
    if repeated or unknown:
        logger.warning(
            "Batch %s: dropped %d extra result lines. Repeated ids: %s. Ids "
            "that were not requested (None for a line with no id): %s.",
            job.batch_id,
            len(repeated) + len(unknown),
            repeated,
            unknown,
        )
    ordered = [
        by_id.get(request_id)
        or BatchResult(
            id=request_id,
            status="missing",
            provider=job.provider,
            model=job.model,
            error="The batch ended without a result for this request.",
        )
        for request_id in job.ids
    ]

    for result in ordered:
        result.cost_in_cents = price(result)
    apply_response_format(ordered, response_format)
    return ordered
