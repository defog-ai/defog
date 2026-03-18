## YouTube Transcription

Generate detailed summaries and transcripts from YouTube videos.

### Basic Usage

```python
from defog.llm.youtube import get_youtube_summary

# Get basic summary
summary = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    model="gemini-2.5-pro"
)

print(summary)
```

### Detailed Transcription

```python
# Get detailed transcript with timestamps
transcript = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=...",
    model="gemini-2.5-pro",
    verbose=True,
    
    # Detailed instructions
    system_instructions=[
        "Provide detailed transcript with timestamps (HH:MM:SS)",
        "Include speaker names if available",
        "Separate content into logical sections",
        "Include [MUSIC] and [APPLAUSE] markers",
        "Extract any text shown on screen",
        "Note visual demonstrations with [DEMO] tags",
        "Highlight key quotes with quotation marks"
    ],
    
)
```

### Structured Output

Use `response_format` with a Pydantic model to get structured data instead of plain text:

```python
from pydantic import BaseModel
from defog.llm.youtube import get_youtube_summary

class VideoSummary(BaseModel):
    title: str
    key_points: list[str]
    overall_sentiment: str

result = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=...",
    task_description="Summarize this video with a title, key points, and overall sentiment.",
    response_format=VideoSummary,
)

print(result.title)
print(result.key_points)
```

When `response_format` is provided, the Gemini model is constrained to return valid JSON matching the schema, and the result is returned as a validated Pydantic model instance.