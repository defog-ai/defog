## Structured Output

Get validated, structured responses using Pydantic models across providers (OpenAI, Anthropic, Gemini, Grok, Together).

### Basic Structured Output

```python
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    sentiment: str
    key_points: List[str]
    confidence: float

response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Analyze this text..."}],
    response_format=Analysis
)

# Access structured data
analysis = response.parsed  # Type: Analysis
print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence}")
```

### Complex Structured Output

```python
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    name: str
    role: str
    email: Optional[str] = None

class MeetingNotes(BaseModel):
    date: datetime
    attendees: List[Person]
    agenda_items: List[str]
    decisions: List[str]
    action_items: List[dict]
    next_meeting: Optional[datetime] = None

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": meeting_transcript}],
    response_format=MeetingNotes
)

notes = response.parsed
print(f"Meeting on {notes.date}")
print(f"Attendees: {', '.join(p.name for p in notes.attendees)}")

### Repair behavior for loose JSON providers

DeepSeek and ZAI support JSON-object mode but do not enforce the complete
Pydantic schema at generation time. OpenRouter models can also occasionally
return loose JSON. For these providers, Defog repairs structured output in this
order:

1. Normalize JSON syntax and apply safe local schema fixes, such as converting
   `null` to `""` for string fields and removing keys forbidden by the model.
2. If semantic validation errors remain, request patches only for the invalid
   JSON-pointer paths. Valid fields and valid list items are not regenerated.
3. If the output cannot be parsed at all, make one bounded full-object repair
   call using the broken output and schema. The original source conversation is
   not replayed.

Repair calls are bounded to one attempt and fail closed: if a patch changes an
unlisted path, omits an invalid path, or the assembled object still fails the
original Pydantic validation, `response.content` contains the original raw
string.

When a repair was needed, `response.structured_output_repairs` contains audit
metadata:

```python
repair = response.structured_output_repairs
if repair:
    print(repair["strategy"])  # deterministic, field_patch, or full_object
    print(repair["attempts"])
    print(repair["deterministic_fields"])
    print(repair["model_patched_fields"])
    print(repair["input_tokens"], repair["output_tokens"])
    print(repair["cost_in_cents"], repair["success"])
```

Repair-call tokens and cost are included in the top-level `input_tokens`,
`output_tokens`, and `cost_in_cents` totals. The telemetry fields report the
repair portion separately.
