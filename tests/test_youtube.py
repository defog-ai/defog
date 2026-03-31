import asyncio
import os
import pytest
from pydantic import BaseModel
from typing import List
from defog.llm.youtube import get_youtube_summary, _parse_structured_response


@pytest.mark.asyncio
async def test_youtube_transcript_end_to_end():
    """End-to-end test for YouTube transcript generation."""
    # Skip test if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    # Use a short, public YouTube video for testing
    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"

    # Get transcript
    transcript = await get_youtube_summary(video_url)

    # Basic assertions
    assert transcript is not None
    assert isinstance(transcript, str)
    assert len(transcript) > 0

    # Check that transcript contains some expected content
    # (This will vary by video, but should contain some words)
    assert len(transcript.split()) > 10

    print(f"Generated transcript ({len(transcript)} characters):")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)


@pytest.mark.asyncio
async def test_youtube_transcript_end_to_end_with_system_instructions():
    """End-to-end test for YouTube transcript generation with system instructions."""
    # Skip test if GEMINI_API_KEY is not set
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"
    summary = await get_youtube_summary(
        video_url,
        system_instructions=[
            "Focus on the overall message of the video, not a step by step transcript."
        ],
        task_description="Please explain this video like I am a 5 year old.",
    )
    assert summary is not None
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary.split()) > 10
    print(f"Generated summary:\n{summary}")


class VideoSummary(BaseModel):
    title: str
    key_points: List[str]
    overall_sentiment: str


@pytest.mark.asyncio
async def test_youtube_transcript_with_response_format():
    """End-to-end test for YouTube transcript with structured output."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    video_url = "https://www.youtube.com/watch?v=EysJTNLQVZw"

    result = await get_youtube_summary(
        video_url,
        task_description="Summarize this video with a title, key points, and overall sentiment.",
        response_format=VideoSummary,
    )

    assert isinstance(result, VideoSummary)
    assert len(result.title) > 0
    assert len(result.key_points) > 0
    assert len(result.overall_sentiment) > 0
    print(f"Structured result: {result}")


def test_parse_structured_response_with_pydantic():
    """Test _parse_structured_response with a Pydantic model."""

    class SimpleModel(BaseModel):
        name: str
        value: int

    content = '{"name": "test", "value": 42}'
    result = _parse_structured_response(content, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "test"
    assert result.value == 42


def test_parse_structured_response_with_markdown_wrapping():
    """Test _parse_structured_response strips markdown code fences."""

    class SimpleModel(BaseModel):
        name: str

    content = '```json\n{"name": "wrapped"}\n```'
    result = _parse_structured_response(content, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "wrapped"


def test_parse_structured_response_without_pydantic():
    """Test _parse_structured_response returns dict when no Pydantic model."""
    content = '{"key": "value"}'
    result = _parse_structured_response(content, dict)
    assert result == {"key": "value"}


def test_parse_structured_response_extracts_json_from_text():
    """Test _parse_structured_response extracts JSON embedded in text."""

    class SimpleModel(BaseModel):
        name: str

    content = 'Here is the result: {"name": "extracted"} and some trailing text'
    result = _parse_structured_response(content, SimpleModel)
    assert isinstance(result, SimpleModel)
    assert result.name == "extracted"


def test_parse_structured_response_raises_on_invalid_json():
    """Test _parse_structured_response raises ValueError on unparseable content."""
    with pytest.raises(ValueError, match="Could not parse JSON"):
        _parse_structured_response("not json at all", dict)


if __name__ == "__main__":
    asyncio.run(test_youtube_transcript_end_to_end())
