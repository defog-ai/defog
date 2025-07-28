## Structured Output

Get validated, structured responses using Pydantic models.

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
```