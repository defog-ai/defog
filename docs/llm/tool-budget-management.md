## Tool Budget Management

Limit the number of times specific tools can be called during a conversation to control costs and prevent excessive usage.

### Basic Usage

```python
from pydantic import BaseModel, Field
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

def web_search(input: SearchInput) -> str:
    """Search the web for information"""
    return f"Results for '{input.query}'"

# Set usage limits for each tool
tool_budget = {
    "web_search": 3,    # Allow only 3 web searches
}

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Search for Python, Java, and JavaScript trends"}],
    tools=[web_search],
    tool_budget=tool_budget
)
```

When a tool's budget is exhausted, the LLM will be notified and will adapt its approach without access to that tool.