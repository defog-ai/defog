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
checked on 2026-09-15. Rates include input, cached input, and output, with
the published long-context prices where available. For Batch models with
no separate cached-input price, all input uses the Batch input price.
An unlisted OpenAI model/tier price returns `None`; a price is not inferred
from another model. Existing standard prices and other providers' prices
are unchanged. These estimates do not include regional surcharges,
provider-hosted tool fees, or separate cache-write charges.

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
