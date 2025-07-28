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
    
    # Advanced options
    include_auto_chapters=True,       # Use YouTube's chapter markers
    language_preference="en",         # Preferred transcript language
    fallback_to_auto_captions=True,   # Use auto-generated if needed
    max_retries=3                     # Retry on failure
)
```