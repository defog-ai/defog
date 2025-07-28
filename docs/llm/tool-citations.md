## Tool Citations

Automatically add citations to LLM responses when tools are used, showing which tool outputs were used to generate the response. This feature is only available for OpenAI and Anthropic providers.

### Basic Usage

```python
from pydantic import BaseModel
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

class WeatherInput(BaseModel):
    location: str
    unit: str = "fahrenheit"

def get_weather(input: WeatherInput):
    """Get current weather for a location"""
    return {"temperature": 72, "condition": "sunny", "humidity": 65}

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=[get_weather],
    tool_choice="auto",
    insert_tool_citations=True  # Enable citations
)

# Response will include citations
print(response.content)    # "The weather in San Francisco is currently sunny with a temperature of 72°F [1]..."
print(response.citations)  # List of citation blocks with tool references
```

### How It Works

1. Tools are executed normally during the conversation
2. Tool outputs are converted to documents with metadata
3. The citations tool processes the response and adds inline citations
4. Both the cited content and citation details are returned in the response

### Response Structure

When `insert_tool_citations=True`, the response includes:
- `content`: The LLM's response with inline citations added
- `citations`: A list of citation blocks, each containing:
  - `type`: "text"
  - `text`: The text content
  - `citations`: List of references to tool outputs

### Limitations

- Only supported for OpenAI and Anthropic providers
- Raises `ValueError` if used with unsupported providers
- Requires tools to be used in the conversation
- No effect if no tools are called