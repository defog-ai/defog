# Core Chat Functions

The library provides a unified interface for working with multiple LLM providers.

Supported providers: OpenAI, Anthropic, Gemini, Grok (xAI), Together.

## Basic Usage

```python
from defog.llm.utils import chat_async, chat_async_legacy, LLMResponse
from defog.llm.llm_providers import LLMProvider

# Unified async interface with explicit provider specification
response: LLMResponse = await chat_async(
    provider=LLMProvider.OPENAI,  # or "openai", LLMProvider.ANTHROPIC, etc.
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_completion_tokens=1000,
    temperature=0.0
)

print(response.content)  # Response text
print(f"Cost: ${response.cost_in_cents/100:.4f}")

# Alternative: Legacy model-to-provider inference
response = await chat_async_legacy(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Advanced Parameters

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    
    # Core parameters
    max_completion_tokens=2000,
    temperature=0.7,
    
    # Advanced parameters
    reasoning_effort="high",          # For o1 models: low, medium, high
    response_format=MyPydanticModel,  # Structured output
    tools=[...],                      # Function calling
    tool_choice="auto",               # auto, none, required, or specific tool
    
    # Provider-specific options
    top_p=0.9,                        # Nucleus sampling
    frequency_penalty=0.0,            # Reduce repetition
    presence_penalty=0.0,             # Encourage new topics
    
    # Logging and debugging
    verbose=True,                     # Detailed logging
    return_usage=True,                # Include token usage
)
```

## Provider-Specific Examples

```python
# OpenAI
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[my_function],
    tool_choice="auto"
)

# Anthropic
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=messages,
    response_format=MyPydanticModel
)

# Gemini
response = await chat_async(
    provider=LLMProvider.GEMINI,
    model="gemini-2.0-flash",
    messages=messages
)

# Together AI
response = await chat_async(
    provider=LLMProvider.TOGETHER,
    model="mixtral-8x7b",
    messages=messages
)

# Grok (xAI)
response = await chat_async(
    provider=LLMProvider.GROK,
    model="grok-4",
    messages=messages,
    tools=[my_function],          # Function calling supported
)
```
