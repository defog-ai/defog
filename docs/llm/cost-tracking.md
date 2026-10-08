## Cost Tracking

All LLM operations include detailed cost tracking.

### Basic Cost Information

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages
)

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Cost: ${response.cost_in_cents / 100:.4f}")
```

### OpenAI Flex and Batch costs

OpenAI calls made with `flex_processing=True` report Flex token costs in
`response.cost_in_cents`. Each response uses the service tier returned by
OpenAI. A response marked `default` uses the existing standard prices even
when Flex was requested. If the response omits its tier, the requested tier
is used.

Costs add across all tool turns and any final structured-output call. A
paused response includes costs up to the pause. A resumed call reports only
its new costs; add the paused and resumed costs for the complete total.
Each response's token counts are included once, including when a tool's
budget reaches zero.

For a completed OpenAI Batch API job, calculate cost explicitly:

```python
from defog.llm.cost import CostCalculator

cost = CostCalculator.calculate_cost(
    model="gpt-5-mini",
    input_tokens=1000,          # Uncached input only
    cached_input_tokens=1000,  # Subtract these from OpenAI's total input first
    output_tokens=500,
    batch=True,
)
print(f"{cost:.5f} cents")  # 0.06375 cents
```

The same tokens cost `0.12750` cents with standard pricing. Use
`service_tier="flex"` for a manual Flex calculation. Batch takes precedence
if both options are supplied; the discounts do not multiply. Running several
ordinary requests concurrently does not make them Batch API requests. This
library does not submit Batch API jobs.

Flex and Batch rates come from the [official OpenAI pricing page](https://developers.openai.com/api/docs/pricing),
checked on 2026-09-15 (GPT-6 models on 2026-09-23). Rates include input, cached input, and output, with
the published long-context prices where available. For Batch models with
no separate cached-input price, all input uses the Batch input price.

### OpenAI long-context prices

One OpenAI request whose input tokens plus cached input tokens exceed
272,000 is billed at the model's published long-context rates. The
calculator applies those rates on the standard tier as well as on Flex
and Batch, for the models that publish them (gpt-6-astra, gpt-6-sol, gpt-6-luna,
gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5, gpt-5.4, gpt-5.4-pro, and their dated snapshots).
A model with no published long-context price, such as gpt-5.5-pro, keeps
its standard price. For gpt-5.6-luna with 300,000 uncached input tokens and
1,000 output tokens:

| Tier     | Cost in cents |
| -------- | ------------- |
| standard | 12.18         |
| flex     | 6.09          |
| batch    | 6.09          |

The threshold applies to each request separately. `chat_async` prices
every response OpenAI returns during one call (tool turns, the final
structured-output request) on its own token counts and then adds the
amounts, so two 200,000-token requests are two short-context requests.
When you call `calculate_cost` yourself, pass the token counts of one
request rather than a sum over several requests.
An unlisted OpenAI model/tier price returns `None`; a price is not inferred
from another model. Version 1.6.18 also corrects the standard gpt-5.6-sol
(and gpt-5.6) rates to the published $4 input / $0.40 cached input / $20
output per 1M tokens, and bills cached input at the regular input rate for
OpenAI models with no cached input discount (gpt-5.5-pro, gpt-5.4-pro),
which earlier versions left out of the standard price. Other providers'
prices are unchanged. These estimates do not include regional surcharges,
provider-hosted tool fees, or separate cache-write charges.

### Prompt-length prices

Some Anthropic and Gemini models charge more when the prompt of one
request is longer than a limit. The higher rates then apply to every token
of that request, output included.

| Model           | Limit (prompt tokens) | Input / cached input / output per 1M tokens, over the limit |
| --------------- | --------------------- | ------------------------------------------------------------ |
| claude-haiku-5-5 | 100,000              | $0.50 / $0.05 / $2.50 (cache writes $0.625 for 5 minutes, $1 for 1 hour) |
| gemini-3.1-pro  | 200,000               | $4 / $0.40 / $18                                              |
| gemini-2.5-pro  | 200,000               | $2.50 / none / $15                                            |

For Anthropic, the prompt length is the uncached input plus cache reads
plus cache writes. For Gemini, it is the input token count that Gemini
reports, which already includes the cached part.

```python
from defog.llm.cost import CostCalculator

CostCalculator.calculate_cost("claude-haiku-5-5", 100_000, 1_000)  # 1.05 cents
CostCalculator.calculate_cost("claude-haiku-5-5", 150_000, 1_000)  # 7.75 cents
```

As with the OpenAI long-context prices, `chat_async` prices each request
of a tool loop on its own prompt and adds the amounts. To price several
requests yourself, pass one dict of token counts per request:

```python
CostCalculator.calculate_requests_cost(
    "claude-haiku-5-5",
    [{"input_tokens": 60_000, "output_tokens": 1_000}] * 3,
)  # 1.95 cents; one 180,000-token request with 3,000 output tokens costs 9.75
```

`calculate_cost` takes an optional `prompt_tokens` argument. Pass it when
your input token count already includes cached tokens, as Gemini's does.
Batch discounts for these providers are not applied.

### Anthropic 1-hour cache writes

Anthropic charges more to write to the 1-hour prompt cache
(`cache_control: {"type": "ephemeral", "ttl": "1h"}`) than to the 5-minute
cache: 2x the base input price instead of 1.25x. Each response reports how its
cache writes split between the two in `usage.cache_creation`. `chat_async`
prices the 1-hour writes of each request at the 1-hour rate and the rest at the
5-minute rate. `response.cache_creation_input_tokens` still reports all cache
writes together. When a response has no split, all of its cache writes use the
5-minute rate, as before.

The 1-hour rates come from [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing),
checked on 2026-10-08. Older models that the page no longer lists
(claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku) have no
1-hour rate and keep the 5-minute rate.

To price one request yourself, pass the 1-hour part of the cache writes:

```python
from defog.llm.cost import CostCalculator

cost = CostCalculator.calculate_cost(
    "claude-haiku-5-5",
    input_tokens=1_000,                     # usage.input_tokens
    output_tokens=500,                      # usage.output_tokens
    cached_input_tokens=2_000,              # usage.cache_read_input_tokens
    cache_creation_input_tokens=30_000,     # usage.cache_creation_input_tokens
    cache_creation_1h_input_tokens=20_000,  # usage.cache_creation.ephemeral_1h_input_tokens
)  # 0.562 cents; 0.412 cents if all 30,000 writes were 5-minute writes
```

### OpenAI list price and served tier

An OpenAI response also reports the standard-tier price of the same usage
and the service tier OpenAI actually served:

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-5.6-luna",
    messages=messages,
    flex_processing=True,
)
response.cost_in_cents        # amount paid, at the served tier
response.list_cost_in_cents   # the same usage at standard prices
response.service_tier         # "flex", "default", ... or "mixed"
```

Both values are built the same way as `cost_in_cents`: each provider
response (tool turns, the final structured-output request, the requests
before a pause) is priced on its own token counts, then the amounts are
added. Two Flex requests of 200,000 uncached input and 1,000 output tokens
each on gpt-5.6-luna give `cost_in_cents == 4.12`,
`list_cost_in_cents == 8.24` and `service_tier == "flex"`. If OpenAI serves
the second request on the `default` tier instead, the amount paid is 6.18,
the list price stays 8.24 and `service_tier` is `"mixed"`. The tier in the
response has priority over the tier requested: a `flex` response to a call
made without `flex_processing` reports `service_tier == "flex"`. A response
without a tier reports the requested tier, or `"default"`.

An OpenAI model with no cached input discount (gpt-5.5-pro, gpt-5.4-pro)
bills cached input at its regular input rate on every tier, so 10,000
uncached input, 200,000 cached input and 5,000 output tokens on gpt-5.5-pro
cost 720 cents at standard prices and 360 cents on Flex, and
`list_cost_in_cents` is 720 either way.

`list_cost_in_cents` is `None` when any request's model has no standard
price, and `cost_in_cents` is `None` when any request's tier price is
unknown; the other value is still reported. Providers other than OpenAI
leave the two new fields, `list_cost_in_cents` and `service_tier`, as
`None`; their `cost_in_cents` is unchanged.

### Aggregate Cost Tracking

```python
from defog.llm.cost_tracker import CostTracker

# Initialize cost tracker
tracker = CostTracker()

# Track multiple operations
async def process_documents(docs):
    for doc in docs:
        response = await chat_async(...)
        tracker.add_cost(
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_cents=response.cost_in_cents
        )

# Get cost summary
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost_cents'] / 100:.2f}")
print(f"By provider: {summary['by_provider']}")
print(f"By model: {summary['by_model']}")
```
