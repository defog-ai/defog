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