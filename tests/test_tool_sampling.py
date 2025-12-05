import pytest

from defog.llm.llm_providers import LLMProvider
from defog.llm.utils import chat_async
from tests.conftest import skip_if_no_api_key


def heavy_tool():
    """Return a deliberately large payload to exercise sampling."""
    return list(range(1200))


def sample_first_five(tool_result, **_):
    return tool_result[:5]


@pytest.mark.asyncio(loop_scope="session")
@skip_if_no_api_key("openai")
async def test_chat_async_tool_sampling_reduces_input_tokens_end_to_end():
    """
    Integration test: run chat_async with a real provider and confirm that tool_sample_functions
    reduces the input_tokens reported by the provider.
    """
    model = "gpt-4.1"
    messages = [
        {
            "role": "user",
            "content": (
                "Call the heavy_tool to inspect a large list, then reply with a very short summary."
            ),
        }
    ]

    without_sampling = await chat_async(
        provider=LLMProvider.OPENAI,
        model=model,
        messages=messages,
        tools=[heavy_tool],
        temperature=0.0,
        max_completion_tokens=500,
        max_retries=1,
    )

    with_sampling = await chat_async(
        provider=LLMProvider.OPENAI,
        model=model,
        messages=messages,
        tools=[heavy_tool],
        tool_sample_functions={"heavy_tool": sample_first_five},
        temperature=0.0,
        max_completion_tokens=500,
        max_retries=1,
    )

    # Ensure tool outputs are present and sampling flag is set appropriately
    assert without_sampling.tool_outputs, "Expected tool outputs without sampling"
    assert with_sampling.tool_outputs, "Expected tool outputs with sampling"
    assert without_sampling.tool_outputs[0]["sampling_applied"] is False
    assert with_sampling.tool_outputs[0]["sampling_applied"] is True

    # The tool output returned to the model should be smaller with sampling applied
    assert len(with_sampling.tool_outputs[0]["result_for_llm"]) < len(
        without_sampling.tool_outputs[0]["result_for_llm"]
    )

    # Sampling should reduce overall input token usage across the provider calls
    assert without_sampling.input_tokens > with_sampling.input_tokens
