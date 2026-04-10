## Anthropic Server-Side Tools & Programmatic Tool Calling

`chat_async` exposes Anthropic's first-party server-side tools — `web_search`,
`web_fetch`, `code_execution`, and `advisor` — alongside user-defined tools and
MCP servers. It also supports **programmatic tool calling**, where Claude
writes Python in the code execution sandbox that calls your local tools as
`await my_tool(...)` — with intermediate results staying in the sandbox so
they never re-enter the model context.

### Quick start: web search + code execution + a local tool

```python
from defog.llm import chat_async

response = await chat_async(
    provider="anthropic",
    model="claude-opus-4-6",
    messages=[
        {"role": "user", "content": "What's the latest defog-python release on PyPI?"}
    ],
    server_tools=["web_search"],
)

print(response.content)
print(response.server_tool_outputs)   # raw web_search_tool_result blocks
print(response.server_tool_usage)     # {"web_search_requests": 1, ...}
```

### `server_tools` accepted shapes

```python
# 1. List of names — defaults applied
server_tools=["web_search", "web_fetch", "code_execution"]

# 2. Dict of name -> per-tool config
server_tools={
    "web_search":     {"max_uses": 5, "allowed_domains": ["arxiv.org"]},
    "web_fetch":      {"max_uses": 3, "citations": True},
    "code_execution": {},
    "advisor":        {"model": "claude-opus-4-6", "max_uses": 2},
}

# 3. Raw dict list (escape hatch for full control / new versions)
server_tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}]
```

You can combine `tools` (user callables), `server_tools`, and `mcp_servers`
in the same request. `server_tools` is Anthropic-only; passing it to any
other provider raises `ValueError`.

### Default versions

| Name | Default version | Notes |
|------|-----------------|-------|
| `web_search` | `web_search_20260209` | Dynamic-filtering version |
| `web_fetch` | `web_fetch_20260209` | Dynamic-filtering version |
| `code_execution` | `code_execution_20260120` | Required for programmatic tool calling |
| `advisor` | `advisor_20260301` | Adds beta header `advisor-tool-2026-03-01` |

The default versions are **Claude API + Microsoft Foundry only** and require
**Claude Opus 4.6 / Sonnet 4.6** or newer. On Bedrock or Vertex (or with
older models), override the version per call:

```python
server_tools={
    "web_search":     {"version": "web_search_20250305"},
    "web_fetch":      {"version": "web_fetch_20250910"},
    "code_execution": {"version": "code_execution_20250825"},
}
```

When the model supports it, `web_search` and `web_fetch` at the dynamic
filtering version automatically pull in `code_execution_20260120`. You'll
see a debug log when this happens.

### Programmatic tool calling

```python
from pydantic import BaseModel
from defog.llm import chat_async


class QueryArgs(BaseModel):
    sql: str


async def query_database(input: QueryArgs) -> list:
    """Run a SQL query and return rows as JSON."""
    # talk to your real DB here
    return [{"customer": "Acme", "revenue": 50_000}]


response = await chat_async(
    provider="anthropic",
    model="claude-opus-4-6",
    messages=[
        {"role": "user", "content": "Who is the top customer by revenue?"}
    ],
    tools=[query_database],
    server_tools=["code_execution"],
    programmatic_tool_calling=True,
)

print(response.content)        # final natural-language answer
print(response.tool_outputs)   # local query_database invocations
print(response.container_id)   # reuse this in a follow-up call
```

When `programmatic_tool_calling=True`:

- Each user-supplied tool spec gets `allowed_callers=["code_execution_20260120"]`,
  so Claude can call it from inside the sandbox.
- `code_execution` must be in `server_tools` (we auto-add it if you also pass
  `web_search` or `web_fetch` at a dynamic-filtering version).
- `strict_tools` is forced off and `parallel_tool_calls=False` is rejected
  (Anthropic disallows `disable_parallel_tool_use` with programmatic calling).
- Custom `tool_choice` forcing a specific tool is rejected.
- Structured outputs (`response_format=...`) are not supported.
- MCP tools merged in via `mcp_servers` are still permitted but only as
  direct-call tools — they cannot be called from the sandbox.

When responding to a sandbox-issued tool call, the user message we send back
contains **only `tool_result` blocks** (no text, no thinking, no images),
which is the shape Anthropic requires.

### Container reuse

Long-running code execution and programmatic tool calling sessions live
inside a container. The id is exposed on `LLMResponse.container_id`. To
continue the same session in a follow-up call, pass it through:

```python
followup = await chat_async(
    provider="anthropic",
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Run another query in the same session."}],
    tools=[query_database],
    server_tools=["code_execution"],
    programmatic_tool_calling=True,
    container_id=response.container_id,
)
```

The provider also handles `pause_turn` automatically. If a long-running
code execution pauses, we echo the assistant content verbatim and re-call
the API with the same container, transparent to the caller.

### Response shape additions

`LLMResponse` gains four optional fields when server tools are in play:

| Field | Type | Description |
|-------|------|-------------|
| `server_tool_outputs` | `Optional[List[Dict[str, Any]]]` | One entry per server-side tool result block (`web_search_tool_result`, `web_fetch_tool_result`, `code_execution_tool_result`, `bash_code_execution_tool_result`, `text_editor_code_execution_tool_result`, `advisor_tool_result`). Each entry has `type`, `tool_use_id`, and `result`. |
| `server_tool_usage` | `Optional[Dict[str, int]]` | Cumulative server-tool counters straight from `response.usage.server_tool_use` (e.g. `{"web_search_requests": 3, "code_execution_requests": 1}`). Use this to compute server-tool cost. |
| `container_id` | `Optional[str]` | Container id returned by code-execution / programmatic calls. |
| `container_expires_at` | `Optional[str]` | ISO 8601 expiry for the container. |

`cost_in_cents` does **not** include server-tool pricing — use
`server_tool_usage` to add it on if you need exact accounting.

### Constraints to be aware of

1. `server_tools` is Anthropic-only.
2. `programmatic_tool_calling=True` requires `code_execution` (or a
   web tool at a dynamic-filtering version that auto-adds it).
3. The default tool versions need Opus 4.6 / Sonnet 4.6 or newer; override
   the version explicitly for older models, Bedrock, or Vertex.
4. `web_fetch` requires the URL to already appear in the transcript —
   Anthropic enforces this server-side.
5. `programmatic_tool_calling=True` cannot be combined with `response_format`,
   `parallel_tool_calls=False`, or a forced `tool_choice`.
