# Human-in-the-Loop Tool Calls (Pause / Resume)

`chat_async` can **pause an agent loop from inside a tool**, hand control back
to your application, and **resume later** — even minutes later, on a different
worker, or after a process restart — delivering the answer as that tool's
`tool_result`.

This is the standard "ask the user a clarifying question" / approval /
confirmation pattern. The fast (sub-second) case already works by simply
`await`-ing inside the tool. This primitive is for the case where the answer
arrives out-of-band and you need to release the run and pick it up again.

Supported providers: **Anthropic** and **OpenAI**.

## The primitive

A tool raises `PauseToolExecution` to suspend the loop:

```python
from defog.llm import chat_async, PauseToolExecution

async def ask_user_question(question: str) -> dict:
    """Ask the end user a clarifying question when the request is ambiguous."""
    raise PauseToolExecution(payload={"questions": [question]})
```

`chat_async` catches it (even under `parallel_tool_calls=True`), stops the
loop, and returns an `LLMResponse` describing the pause:

```python
resp = await chat_async(
    provider="anthropic",
    model="claude-sonnet-4-6",
    messages=messages,
    tools=[ask_user_question],
)

resp.status            # "paused"
resp.pending_tool_use  # {"id": ..., "name": "ask_user_question", "input": {...}}
resp.pause_payload     # {"questions": [...]}  — whatever the tool passed
resp.messages          # full in-flight history incl. the assistant tool_use
                       # turn, with thinking blocks intact (Anthropic)
resp.response_id       # resume handle for OpenAI (see below)
```

`PauseToolExecution` subclasses `BaseException` (like `asyncio.CancelledError`)
so the `except Exception` handlers throughout the tool-execution path — and
`asyncio.gather(return_exceptions=True)` under parallel execution — cannot
accidentally swallow it.

## Resuming

Persist what the paused response gives you, collect the answer, then call
`chat_async` again with `resume_tool_results` — a dict mapping each pending
tool-call id to the result to deliver as that tool's output. defog appends the
matching `tool_result` turn and continues the loop. You never build provider
message blocks by hand.

### Anthropic

The captured assistant turn (thinking + `tool_use`) lives in `resp.messages`,
so resuming only needs the messages plus the answer:

```python
tool_use_id = resp.pending_tool_use["id"]

final = await chat_async(
    provider="anthropic",
    model="claude-sonnet-4-6",
    messages=resp.messages,                       # persisted from the pause
    tools=[ask_user_question],                    # same tools as the paused run
    resume_tool_results={tool_use_id: {"answer": "Use the prod database."}},
)

final.content  # the model's final answer, computed with the user's input
```

### OpenAI

OpenAI keeps the assistant turn (reasoning + function call) server-side,
referenced by a response id, so resuming additionally needs
`previous_response_id`:

```python
tool_use_id = resp.pending_tool_use["id"]

final = await chat_async(
    provider="openai",
    model="gpt-4.1",
    messages=resp.messages,                       # persisted from the pause
    tools=[ask_user_question],
    previous_response_id=resp.response_id,         # <- required for OpenAI
    resume_tool_results={tool_use_id: {"answer": "Use the prod database."}},
)
```

OpenAI retains the originating response for 30 days when `store=True` (the
default), which is what makes "resume on a different worker / after a restart"
work.

## Persisting across a restart

`resp.messages`, `resp.pending_tool_use`, `resp.pause_payload`, and (OpenAI)
`resp.response_id` are all plain JSON-serializable data. Persist them to your
database / queue, return control, and resume from a completely fresh process:

```python
import json

# At pause time:
record = {
    "messages": resp.messages,
    "pending_tool_use": resp.pending_tool_use,
    "pause_payload": resp.pause_payload,
    "response_id": resp.response_id,   # OpenAI
}
db.save(run_id, json.dumps(record))

# Later, anywhere:
record = json.loads(db.load(run_id))
answer = collect_answer_from_user(record["pause_payload"])
final = await chat_async(
    provider="openai",
    model="gpt-4.1",
    messages=record["messages"],
    tools=[ask_user_question],
    previous_response_id=record["response_id"],
    resume_tool_results={record["pending_tool_use"]["id"]: answer},
)
```

## Parallel tool calls

If the model issues several tool calls in one turn and one of them pauses, the
work of the sibling tools that already finished is **not** discarded — their
results are captured alongside the assistant turn (Anthropic: a partial
`tool_result` turn; OpenAI: stashed `function_call_output` items). On resume you
only need to supply results for the tool calls that were actually paused;
defog fills in the rest.

If more than one tool pauses in the same turn, only the first is surfaced as
`pending_tool_use`; provide results for every pending id via
`resume_tool_results` (defog raises a clear `ValueError` listing any it is still
missing).

## Notes & limitations

- Pass the **same `tools`** to the resume call that the paused run used, so the
  agent loop can continue.
- `resume_tool_results` is only supported for the `anthropic` and `openai`
  providers. Raising `PauseToolExecution` under any other provider raises a
  clear `ConfigurationError`.
- A paused `LLMResponse` has `content=None` and reports only best-effort
  partial token usage up to the pause point.
- This is for the suspend-and-resume-later flow. If the answer is available
  within the request, just `await` for it inside the tool — no pause needed.
