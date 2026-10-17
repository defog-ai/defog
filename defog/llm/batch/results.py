"""One result shape for OpenAI Batch and Anthropic Message Batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, ValidationError

from ..cost import CostCalculator

# Result statuses:
#   succeeded  - the model answered in full
#   incomplete - the answer was cut off (for example at the token limit)
#   refused    - the model refused to answer
#   invalid    - the answer did not match the requested schema
#   errored    - the provider returned an error for this request
#   expired    - the batch ended before this request ran
#   canceled   - the batch was canceled before this request ran
#   missing    - the batch ended and returned no line for this request
RESULT_STATUSES = (
    "succeeded",
    "incomplete",
    "refused",
    "invalid",
    "errored",
    "expired",
    "canceled",
    "missing",
)


@dataclass
class BatchJob:
    """A submitted batch. ``to_dict``/``from_dict`` let a caller store it."""

    provider: str
    model: str
    batch_id: str
    ids: List[str]
    input_file_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BatchJob":
        return cls(**data)


@dataclass
class BatchStatus:
    """The state of a batch at one poll."""

    batch_id: str
    provider: str
    status: str  # the provider's own status name
    done: bool
    counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchUsage:
    """Token counts for one request.

    ``input_tokens`` is the uncached input only, for both providers.
    ``reasoning_tokens`` is reported by OpenAI only; it is part of
    ``output_tokens``. Anthropic counts thinking inside ``output_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_creation_1h_input_tokens: int = 0
    reasoning_tokens: Optional[int] = None


@dataclass
class BatchResult:
    """The result of one request in a batch."""

    id: str
    status: str
    provider: str
    model: str
    text: Optional[str] = None
    usage: Optional[BatchUsage] = None
    cost_in_cents: Optional[float] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    parsed: Any = None
    raw: Optional[Dict[str, Any]] = None


def price(result: BatchResult) -> Optional[float]:
    """Cost of one result in cents at the provider's Batch prices.

    Returns None when the result has no usage (for example an expired
    request) or the model has no Batch price.
    """
    usage = result.usage
    if usage is None:
        return None
    return CostCalculator.calculate_cost(
        result.model,
        usage.input_tokens,
        usage.output_tokens,
        usage.cached_input_tokens,
        usage.cache_creation_input_tokens,
        batch=True,
        cache_creation_1h_input_tokens=usage.cache_creation_1h_input_tokens or None,
    )


def parse(result: BatchResult, response_format: type[BaseModel]) -> BaseModel:
    """Validate the answer text of one result against ``response_format``.

    Raises ValueError when the result has no complete answer and
    pydantic.ValidationError when the answer does not match the schema.
    Constraints such as minimum or maxLength are checked here even when the
    provider dropped them from the schema it enforced.
    """
    if result.status not in ("succeeded", "invalid") or result.text is None:
        raise ValueError(
            f"Result {result.id!r} has status {result.status!r} and no answer to parse."
        )
    return response_format.model_validate_json(result.text)


def apply_response_format(
    results: List[BatchResult],
    response_format: Union[type[BaseModel], Mapping[str, type[BaseModel]], None],
) -> None:
    """Set ``parsed`` on each succeeded result, or mark it ``invalid``."""
    if response_format is None:
        return
    for result in results:
        if result.status != "succeeded":
            continue
        if isinstance(response_format, Mapping):
            schema = response_format.get(result.id)
            if schema is None:
                continue
        else:
            schema = response_format
        try:
            result.parsed = parse(result, schema)
        except ValidationError as exc:
            result.status = "invalid"
            result.error = f"Answer does not match {schema.__name__}: {exc}"
        except ValueError as exc:
            # A succeeded result with no answer text.
            result.status = "invalid"
            result.error = str(exc)


def get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert an SDK object (pydantic model) or mapping to a plain dict."""
    if isinstance(obj, Mapping):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(f"Cannot read a batch result line of type {type(obj).__name__}")
