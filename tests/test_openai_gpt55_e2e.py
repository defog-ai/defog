"""Live OpenAI GPT-5.5 smoke test.

Requires OPENAI_API_KEY in .env. This skips while gpt-5.5 API access is still
rolling out, but treats other provider failures as real test failures.
"""

import os

import pytest

from defog.llm import chat_async
from defog.llm.llm_providers import LLMProvider


pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


_UNAVAILABLE_MARKERS = (
    "model_not_found",
    "does not exist",
    "not found",
    "not currently available",
    "not available",
    "do not have access",
    "don't have access",
    "coming soon",
)


@pytest.mark.asyncio
async def test_gpt_5_5_chat_async_e2e():
    try:
        response = await chat_async(
            provider=LLMProvider.OPENAI,
            model="gpt-5.5",
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: gpt-5.5-ok",
                }
            ],
            max_completion_tokens=1024,
            reasoning_effort="low",
            max_retries=1,
        )
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _UNAVAILABLE_MARKERS):
            pytest.skip(f"gpt-5.5 API access is not available yet: {exc}")
        raise

    assert response.model == "gpt-5.5"
    assert "gpt-5.5-ok" in str(response.content).strip()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.cost_in_cents is not None
