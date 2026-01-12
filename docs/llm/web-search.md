## Web Search

Search the web for current information with AI-powered analysis.

### Basic Web Search

```python
from defog.llm.web_search import web_search_tool
from defog.llm.llm_providers import LLMProvider

result = await web_search_tool(
    question="What are the latest developments in AI?",
    model="claude-haiku-4-5",
    provider=LLMProvider.ANTHROPIC,
    max_tokens=2048
)

print(result["search_results"])   # Analyzed search results
print(result["websites_cited"])   # Source citations
print(result["usage"])            # Token usage statistics
```

### Supported Providers

Web search is supported on the following providers:

| Provider | Example Model | Notes |
|----------|--------------|-------|
| OpenAI | `gpt-4.1-mini`, `gpt-4.1` | Uses Responses API with web_search tool |
| Anthropic | `claude-haiku-4-5`, `claude-sonnet-4` | Uses web_search_20250305 tool |
| Gemini | `gemini-2.0-flash`, `gemini-3-flash-preview` | Uses google_search via Interactions API |

### Structured Output

Get search results as a structured Pydantic model:

```python
from pydantic import BaseModel, Field
from typing import List

class SearchAnswer(BaseModel):
    answer: str = Field(description="The answer to the search query")
    confidence: float = Field(description="Confidence score between 0 and 1")
    key_facts: List[str] = Field(description="Key facts found during the search")

result = await web_search_tool(
    question="What is the capital of France?",
    model="gpt-4.1-mini",
    provider=LLMProvider.OPENAI,
    max_tokens=2048,
    response_format=SearchAnswer
)

# search_results is now a SearchAnswer instance
print(result["search_results"].answer)
print(result["search_results"].key_facts)
```

### Reasoning Effort

For models that support extended reasoning, you can control the reasoning effort:

```python
# OpenAI (o-series, gpt-5 models)
result = await web_search_tool(
    question="Analyze the impact of AI on healthcare",
    model="o3",
    provider=LLMProvider.OPENAI,
    reasoning_effort="high"  # "low", "medium", "high"
)

# Gemini (gemini-3 models)
result = await web_search_tool(
    question="Analyze the impact of AI on healthcare",
    model="gemini-3-flash-preview",
    provider=LLMProvider.GEMINI,
    reasoning_effort="medium"  # "minimal", "low", "medium", "high"
)

# Anthropic (claude-3-7, claude-4 models)
result = await web_search_tool(
    question="Analyze the impact of AI on healthcare",
    model="claude-sonnet-4",
    provider=LLMProvider.ANTHROPIC,
    reasoning_effort="medium"  # "low", "medium", "high"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | str | required | The search query/question to answer |
| `model` | str | required | The model to use (e.g., "gpt-4.1", "gemini-3-flash-preview") |
| `provider` | LLMProvider | required | The LLM provider (OPENAI, ANTHROPIC, or GEMINI) |
| `max_tokens` | int | 8192 | Maximum tokens for the response |
| `verbose` | bool | True | Whether to log progress |
| `response_format` | Type[BaseModel] | None | Optional Pydantic model for structured output |
| `reasoning_effort` | str | None | Reasoning effort level (provider-specific values) |

### Return Value

The function returns a dictionary with the following keys:

- `usage`: Token usage statistics (`input_tokens`, `output_tokens`, etc.)
- `search_results`: The search results as a string, or a Pydantic model instance if `response_format` is provided
- `websites_cited`: List of cited sources with `url` and `title`/`source`
