## Function Calling

Define tools for LLMs to call with automatic schema generation. As of version 0.50.0, you can use regular Python functions directly as tools!

### Basic Function Definition

You can define tools in two ways:

#### Option 1: Regular Functions (NEW - Simpler!)

```python
from defog.llm.utils import chat_async
from typing import Optional

# Regular function with type annotations
def get_weather(location: str, units: str = "celsius") -> str:
    """
    Get current weather for a location.
    
    Args:
        location: City and country
        units: Temperature units (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

# Async functions are also supported
async def search_web(query: str, max_results: Optional[int] = 5) -> list[str]:
    """Search the web for information"""
    # Implementation here
    return [f"Result {i} for {query}" for i in range(max_results)]

# Use directly as tools - no Pydantic needed!
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather, search_web],
    tool_choice="auto"
)
```

#### Option 2: Pydantic-based Functions (Traditional)

```python
from pydantic import BaseModel, Field
from defog.llm.utils import chat_async

class WeatherInput(BaseModel):
    location: str = Field(description="City and country")
    units: str = Field(default="celsius", description="Temperature units")

def get_weather(input: WeatherInput) -> str:
    """Get current weather for a location"""
    return f"Weather in {input.location}: 22°{input.units[0].upper()}, sunny"

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    tool_choice="auto"
)
```

### When to Use Each Approach

**Use Regular Functions when:**
- You want simpler, more Pythonic code
- Your function parameters are straightforward types
- You're prototyping or building quickly
- You prefer type hints over Pydantic models

**Use Pydantic Functions when:**
- You need complex validation logic
- You want detailed field descriptions and constraints
- You're working with nested data structures
- You need custom validators or serializers

### Advanced Tool Configuration

```python
from typing import Literal, Optional
from defog.llm.utils import create_tool_from_function

class DatabaseQuery(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: Literal["prod", "staging", "dev"] = Field(default="dev")
    timeout: Optional[int] = Field(default=30, description="Query timeout")
    explain: bool = Field(default=False, description="Include query plan")

@create_tool_from_function
async def execute_database_query(params: DatabaseQuery) -> dict:
    """Execute a database query with safety controls"""
    # Implementation here
    return {"results": [...], "execution_time": 0.5}

# Use with advanced options
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[execute_database_query],
    tool_choice="required",           # Force tool use
    parallel_tool_calls=True,         # Allow parallel execution
    max_tool_iterations=3,            # Limit recursive calls
)
```

### Keep Tool Outputs Token-Efficient

Pass lightweight previews back to the model while still storing full tool results:

```python
def sample_query_output(function_name: str, tool_result: dict, **_):
    # Keep just a small slice of the data
    return {
        "rows": tool_result.get("rows", [])[:5],
        "row_count": len(tool_result.get("rows", [])),
    }

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o-mini",
    messages=messages,
    tools=[execute_database_query],
    tool_sample_functions={"execute_database_query": sample_query_output},
    tool_result_preview_max_tokens=2000,  # Trim what gets sent back to the LLM
)
```

The assistant receives the sampled/truncated preview, while `response.tool_outputs` always contains the full tool output for logging or follow-up use.
