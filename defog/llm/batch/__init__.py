"""Send many requests through the OpenAI Batch API or Anthropic Message Batches.

Typical use::

    requests = build_requests(items, model="claude-haiku-5-5")
    job = await submit(requests)
    await wait(job)
    results = await collect(job, response_format=requests.response_formats)

See docs/llm/batch-processing.md.
"""

from .api import collect, poll, submit, wait
from .requests import (
    ANTHROPIC_MAX_BYTES,
    ANTHROPIC_MAX_REQUESTS,
    OPENAI_MAX_BYTES,
    OPENAI_MAX_REQUESTS,
    BatchError,
    BatchItem,
    BatchRequests,
    build_requests,
    validate_ids,
)
from .results import (
    RESULT_STATUSES,
    BatchJob,
    BatchResult,
    BatchStatus,
    BatchUsage,
    parse,
    price,
)

__all__ = [
    "build_requests",
    "submit",
    "poll",
    "wait",
    "collect",
    "price",
    "parse",
    "validate_ids",
    "BatchItem",
    "BatchRequests",
    "BatchJob",
    "BatchStatus",
    "BatchResult",
    "BatchUsage",
    "BatchError",
    "RESULT_STATUSES",
    "OPENAI_MAX_REQUESTS",
    "OPENAI_MAX_BYTES",
    "ANTHROPIC_MAX_REQUESTS",
    "ANTHROPIC_MAX_BYTES",
]
