# Core Chat Functions

The library provides a unified interface for working with multiple LLM providers.

Supported providers: OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter.

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
    reasoning_effort="high",          # Provider-specific; see Claude Sonnet 5 below
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

## Claude Sonnet 5 thinking

In `chat_async`, `claude-sonnet-5` accepts `reasoning_effort` values `"low"`, `"medium"`, `"high"`,
`"xhigh"`, and `"max"`. Defog sends `thinking={"type": "adaptive"}` and places
the requested effort in `output_config.effort`, without `budget_tokens`.
Pydantic structured output stays in `output_config.format` in the same request:

```python
from pydantic import BaseModel
from defog.llm import chat_async

class Answer(BaseModel):
    text: str

response = await chat_async(
    provider="anthropic",
    model="claude-sonnet-5",
    messages=[{"role": "user", "content": "Say hello."}],
    reasoning_effort="low",
    response_format=Answer,
)
```

Omitting `reasoning_effort`, passing `None`, or passing `"none"` preserves defog's
existing thinking-off behavior: `thinking={"type": "disabled"}` and no effort
field. Anthropic's API defaults to adaptive thinking when the `thinking` field
is omitted; defog explicitly sends it. Defog omits `temperature` for Sonnet 5,
including caller-supplied values, because the model rejects non-default sampling
parameters. Legacy thinking budgets and other models' effort handling are unchanged.

This support is limited to the chat API. `web_search_tool` builds its requests
separately and still sends legacy thinking parameters for Sonnet 5 when an effort
is supplied; those requests remain unsupported.

See Anthropic's [effort reference](https://platform.claude.com/docs/en/build-with-claude/effort)
and [Sonnet 5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide).

## Custom Base URLs

You can point any provider at a custom endpoint (e.g. a proxy, self-hosted model, or Azure deployment) using the `base_url` parameter:

```python
# Use a custom endpoint for OpenAI-compatible APIs
response = await chat_async(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    base_url="https://my-openai-proxy.example.com/v1/",
)

# Use a custom endpoint for Anthropic
response = await chat_async(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}],
    base_url="https://my-anthropic-proxy.example.com",
)
```

Alternatively, set the corresponding environment variable and it will be picked up automatically:
- `OPENAI_BASE_URL`
- `ANTHROPIC_BASE_URL`
- `GEMINI_BASE_URL`
- `OPENROUTER_BASE_URL`
- `ZAI_BASE_URL`

Or use the `LLMConfig` object for multi-provider configuration:

```python
from defog.llm.config import LLMConfig

config = LLMConfig(base_urls={
    "openai": "https://my-proxy.example.com/v1/",
    "anthropic": "https://my-anthropic-proxy.example.com",
})

response = await chat_async(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    config=config,
)
```

## Provider-Specific Examples

```python
# OpenAI
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-5-mini",
    messages=messages,
    tools=[my_function],
    tool_choice="auto",
    flex_processing=True,  # Lower cost; requests may take longer
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
    model="gemini-2.5-flash",
    messages=messages
)

# DeepSeek (native, via api.deepseek.com)
response = await chat_async(
    provider=LLMProvider.DEEPSEEK,  # or "deepseek"
    model="deepseek-v4.1-flash",  # or the API name "deepseek-flash"
    messages=messages
)

# DeepSeek with structured output
response = await chat_async(
    provider="deepseek",
    model="deepseek-v4.1-flash",
    messages=messages,
    response_format=MyPydanticModel,
)

# ZAI (native, via api.z.ai)
response = await chat_async(
    provider=LLMProvider.ZAI,  # or "zai"
    model="glm-5.2",  # or "glm-5.1"
    messages=messages,
    tools=[my_function],
    tool_choice="auto",
)

# OpenRouter (access any model via a single API key)
response = await chat_async(
    provider=LLMProvider.OPENROUTER,
    model="anthropic/claude-sonnet-4.6",  # provider/model format
    messages=messages,
    tools=[my_function],
    tool_choice="auto"
)

# OpenRouter with an OpenAI model
response = await chat_async(
    provider="openrouter",
    model="openai/gpt-4.1-mini",
    messages=messages,
    response_format=MyPydanticModel
)

# Control reasoning for a model that supports it
response = await chat_async(
    provider="openrouter",
    model="openai/gpt-5-mini",
    messages=messages,
    reasoning_effort="high",  # sends OpenRouter reasoning.effort
)

# Restrict OpenRouter to specific upstream providers
response = await chat_async(
    provider="openrouter",
    model="openai/gpt-5-mini",
    messages=messages,
    providers=["azure"],  # sends OpenRouter provider.only
)

# Advanced OpenRouter routing object
response = await chat_async(
    provider="openrouter",
    model="openai/gpt-5-mini",
    messages=messages,
    providers={"order": ["azure"], "allow_fallbacks": False},
)

```

### Native DeepSeek

The DeepSeek provider (`provider="deepseek"`) talks directly to
`api.deepseek.com` using a `DEEPSEEK_API_KEY`. Model ids are the bare DeepSeek
names. For DeepSeek-V4.1-Flash, use `deepseek-flash` or Defog's
`deepseek-v4.1-flash` alias, which is sent to the API as `deepseek-flash`.
`response.model` retains the name you supplied. The legacy names
`deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` are still accepted by
DeepSeek and now use V4.1-Flash pricing. `deepseek-v4-pro` is also supported.

V4.1-Flash pricing in USD per million tokens, per the
[DeepSeek rate card](https://api-docs.deepseek.com/quick_start/pricing/)
(September 10, 2026):

| Token type | Peak | Off-peak |
| --- | ---: | ---: |
| Input (cache miss) | $0.30 | $0.15 |
| Input (cache hit) | $0.006 | $0.003 |
| Output | $1.20 | $0.60 |

Peak hours are 01:00–04:00 and 06:00–10:00 UTC, Monday through Friday.
All other hours, including weekends, use off-peak rates. Defog selects the
rate at cost calculation time and returns `response.cost_in_cents` in cents.

Pass `reasoning_effort="low"`, `"high"`, or `"max"` to control DeepSeek's
thinking effort, or `"none"` to disable thinking. Leaving it unset uses
DeepSeek's default (`"high"`, thinking enabled). The setting also applies
to tool follow-ups and structured-output repair calls. DeepSeek ignores
`temperature` while thinking is enabled. See the
[thinking-mode documentation](https://api-docs.deepseek.com/guides/thinking_mode/).

```python
response = await chat_async(
    provider="deepseek",
    model="deepseek-flash",
    messages=messages,
    reasoning_effort="low",
)
```

Defog retains the API's `reasoning_content` in assistant messages that call
tools and in cached replies, including replies without tool calls, so
conversations continued with `previous_response_id` can use tools. When
supplying your own assistant history, preserve `reasoning_content` from
DeepSeek responses.

**Structured output note:** DeepSeek does not support OpenAI's `json_schema`
response-format mode (it returns `400 invalid_request_error`). When you pass a
`response_format` Pydantic model to the native DeepSeek provider, it transparently
uses `{"type": "json_object"}` mode under the hood and injects the schema into
the system prompt to shape the output. You still get a parsed Pydantic instance
back in `response.content` — the json_object handling is internal.

### Native ZAI

The ZAI provider (`provider="zai"`) talks directly to
`api.z.ai/api/paas/v4` using a `ZAI_API_KEY`. Model ids are the bare ZAI names
(`glm-5.2`, `glm-5.1`). It supports regular chat, `json_object` structured
output, and function tools through the native chat completions endpoint.

## Mid-Conversation System Messages (Anthropic, Claude Opus 4.8)

System instructions normally live at the start of the conversation and are
hoisted into Anthropic's top-level `system` field. Editing that field — for
example, appending a new instruction partway through a long session —
invalidates the prompt cache for the entire conversation that follows it.

On **Claude Opus 4.8**, `chat_async` handles this automatically. A
`{"role": "system"}` message that appears *after* the conversation has started
is kept in place as a system turn (rather than being hoisted), which preserves
the cached prefix while still giving the instruction system-level priority from
that point onward. No flag is required.

```python
response = await chat_async(
    provider="anthropic",
    model="claude-opus-4-8",
    messages=[
        {"role": "system", "content": "You are a code review assistant."},
        {"role": "user", "content": "Review process() in utils.py."},
        {"role": "assistant", "content": "Looks fine for small inputs."},
        {"role": "user", "content": "Now review the calling code."},
        # Appended mid-session — kept in place, so the cached prefix above
        # (system prompt + earlier turns) still hits the cache.
        {"role": "system", "content": "From now on, require type annotations."},
    ],
)
```

This is a safe, transparent optimization with no behavioral surprises:

- The **leading** system prompt (before any user/assistant turn) is always
  hoisted into the top-level `system` field, as before.
- A mid-conversation system message is kept in place **only when its position is
  unambiguously legal** for Anthropic's API — it directly follows a user turn
  and is either the last message or directly precedes an assistant turn.
- In any other position — or on any other model or provider — every system
  message is hoisted into the top-level system prompt, exactly as it was before.
  No previously valid request changes behavior.
- Consecutive in-place system messages are merged into one (Anthropic rejects
  adjacent system turns).
