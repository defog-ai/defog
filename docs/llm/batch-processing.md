# Batch Processing

`defog.llm.batch` sends many independent requests to the
[OpenAI Batch API](https://platform.openai.com/docs/guides/batch) or to
[Anthropic Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing).
Both providers charge half the standard price and return the results
within 24 hours, usually much sooner. Use it when the answers are not
needed right away.

The module has four async steps and one helper for prices:

1. `build_requests(items, model)` turns your requests into the provider's
   batch format. It uses the same request settings as `chat_async`.
2. `submit(requests)` sends them as one batch and returns a `BatchJob`.
3. `poll(job)` checks the batch once; `wait(job)` polls until it ends.
4. `collect(job)` downloads one `BatchResult` per request.

## Example

```python
import asyncio

from pydantic import BaseModel

from defog.llm.batch import BatchItem, build_requests, collect, submit, wait


class Capital(BaseModel):
    country: str
    capital: str


async def main():
    countries = ["Japan", "Kenya", "Peru"]
    items = [
        BatchItem(
            id=f"req-{n:03d}",
            messages=[
                {"role": "system", "content": "Answer with the capital city."},
                {"role": "user", "content": f"What is the capital of {country}?"},
            ],
            response_format=Capital,
            reasoning_effort="low",
        )
        for n, country in enumerate(countries)
    ]
    requests = build_requests(items, "claude-haiku-5-5")

    job = await submit(requests)
    saved = job.to_dict()  # store this to collect from another process

    await wait(job, poll_interval=60)
    results = await collect(job, response_format=requests.response_formats)

    for result in results:
        if result.status == "succeeded":
            print(result.id, result.parsed.capital, result.cost_in_cents)
        else:
            print(result.id, result.status, result.error)


asyncio.run(main())
```

`submit`, `poll`, `wait` and `collect` create an `AsyncOpenAI` or
`AsyncAnthropic` client from `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Pass
`client=` to use your own client. To collect later, rebuild the job with
`BatchJob.from_dict(saved)`.

## Requests

Each item has:

| Field                   | Meaning                                                   |
| ----------------------- | --------------------------------------------------------- |
| `id`                    | Your id for the request. It comes back on its result.     |
| `messages`              | Chat messages, as for `chat_async`.                       |
| `response_format`       | Optional pydantic model for structured output.            |
| `reasoning_effort`      | Optional reasoning effort, as for `chat_async`.           |
| `max_completion_tokens` | Optional output limit.                                    |

Items can also be plain dicts with the same keys. The provider is chosen
from the model name. Only OpenAI and Anthropic models are supported;
other models raise `ConfigurationError`.

OpenAI requests go to the `/v1/responses` endpoint.

## Ids

Ids must be unique, non-empty strings. Anthropic accepts only ids that
match `^[a-zA-Z0-9_-]{1,64}$`. `build_requests` raises `ValueError` and
names the bad ids when one does not match. It never changes an id for you,
so the id on each result is always the id you sent.

## Size limits

One batch can hold at most:

| Provider  | Requests | Size                        |
| --------- | -------- | --------------------------- |
| OpenAI    | 50,000   | 200 MB (input file)         |
| Anthropic | 100,000  | 256 MB (request body)       |

`build_requests` raises `ValueError` when the requests are over a limit.
It does not split them into several batches. Split the items yourself and
build one batch from each part.

## Results

`collect` returns one `BatchResult` per requested id, in the order of the
ids. Each result has a `status`:

| Status       | Meaning                                                      |
| ------------ | ------------------------------------------------------------ |
| `succeeded`  | The model answered in full.                                  |
| `incomplete` | The answer was cut off, for example at the token limit.     |
| `refused`    | The model refused to answer.                                 |
| `invalid`    | The answer did not match `response_format`.                  |
| `errored`    | The provider returned an error for this request.             |
| `expired`    | The batch ended before this request ran.                     |
| `canceled`   | The batch was canceled before this request ran.              |
| `missing`    | The batch ended without a result line for this request.      |

A result also has `text`, `usage`, `cost_in_cents`, `stop_reason`,
`error`, `parsed` and `raw` (the provider's line). Lines with an id that
was not requested, or a repeated id, come after the requested ids.

Nothing is retried. Send the expired, errored or missing ids again in a
new batch if you need them.

`collect` raises `BatchError` when the batch has not ended yet, or when
an OpenAI batch failed as a whole (for example because of a malformed
input file).

### Structured output

Pass `response_format` to `collect`: one pydantic model for all results,
or a mapping from id to model, such as `requests.response_formats`. Each
succeeded answer is validated into `parsed`. To validate one result
yourself, call `parse(result, Model)`.

Validation in `collect` also checks constraints such as `maxLength`
that a provider may leave out of the schema it enforces.

## Prices

`collect` prices each result with `CostCalculator.calculate_cost(...,
batch=True)` and stores the amount in `cost_in_cents`. Results with no
usage, such as expired requests, have `cost_in_cents=None`. `price(result)`
does the same for one result. See [Cost Tracking](cost-tracking.md) for
the Batch rates.
