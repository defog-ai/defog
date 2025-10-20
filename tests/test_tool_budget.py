import pytest
import os
from pydantic import BaseModel
from defog.llm import chat_async
from defog.llm.llm_providers import LLMProvider
from defog.llm.config.settings import LLMConfig
from pprint import pprint


# Pydantic models for tool inputs
class AddInput(BaseModel):
    a: int
    b: int


class MultiplyInput(BaseModel):
    a: int
    b: int


class FibonacciInput(BaseModel):
    n: int


# Tool functions
def add(input: AddInput) -> int:
    """Add two numbers together."""
    return input.a + input.b


def multiply(input: MultiplyInput) -> int:
    """Multiply two numbers together."""
    return input.a * input.b


def fibonacci(input: FibonacciInput) -> int:
    """Calculate the nth Fibonacci number."""
    n = input.n
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
)
async def test_tool_budget_with_anthropic():
    """Test tool budget feature with Anthropic provider."""
    # Create config with parallel tool calls disabled
    config = LLMConfig(enable_parallel_tool_calls=False)

    # Define tool budget - add can be called twice, multiply once
    tool_budget = {"add": 2, "multiply": 1}

    messages = [
        {
            "role": "user",
            "content": """Please help me with these calculations:
            1. Add 5 + 3
            2. Multiply 4 * 6
            3. Add 10 + 7
            4. Calculate fibonacci of 8
            5. Try to add 2 + 2
            
            Call the tools one by one and tell me the results.""",
        }
    ]

    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-haiku-4-5",
        messages=messages,
        tools=[add, multiply, fibonacci],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    # Check that tools were called
    assert response.tool_outputs is not None
    assert len(response.tool_outputs) >= 4  # At least 4 successful tool calls

    # Verify tool outputs
    tool_results = {
        output["name"]: output["result"] for output in response.tool_outputs
    }

    # Check expected results
    assert "add" in tool_results
    assert "multiply" in tool_results
    assert "fibonacci" in tool_results

    # Count tool usage
    add_count = sum(1 for output in response.tool_outputs if output["name"] == "add")
    multiply_count = sum(
        1 for output in response.tool_outputs if output["name"] == "multiply"
    )

    # Verify budget was respected
    assert add_count == 2  # Exactly 2 add calls (budget limit)
    assert multiply_count == 1  # Exactly 1 multiply call (budget limit)

    # Check that the response mentions the budget limitation
    assert (
        "budget" in response.content.lower()
        or "limit" in response.content.lower()
        or "cannot" in response.content.lower()
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
async def test_tool_budget_with_openai():
    """Test tool budget feature with OpenAI provider."""
    # Create config with parallel tool calls disabled
    config = LLMConfig(enable_parallel_tool_calls=False)

    # Define tool budget
    tool_budget = {"fibonacci": 2}

    messages = [
        {
            "role": "user",
            "content": """Calculate the fibonacci numbers for: 5, 8, 10, and 13.
            Use the fibonacci tool for each calculation.""",
        }
    ]

    response = await chat_async(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1",
        messages=messages,
        tools=[fibonacci],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    # Check that tools were called
    assert response.tool_outputs is not None

    # Count fibonacci calls
    fib_count = sum(
        1 for output in response.tool_outputs if output["name"] == "fibonacci"
    )

    # Verify budget was respected
    assert fib_count == 2  # Exactly 2 fibonacci calls (budget limit)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Gemini API key not set")
async def test_tool_budget_with_gemini():
    """Test tool budget feature with Gemini provider.

    Note: Gemini doesn't support disabling parallel tool calls, so it may
    exceed budgets when making multiple tool calls in a single response.
    """
    # Create config with parallel tool calls disabled (not supported by Gemini)
    config = LLMConfig(enable_parallel_tool_calls=False)

    # Define tool budget - mixed limits
    tool_budget = {"add": 3, "multiply": 1}

    messages = [
        {
            "role": "user",
            "content": """Please perform these calculations:
            1. Add 2 + 3
            2. Add 5 + 7
            3. Multiply 4 * 6
            4. Add 10 + 15
            5. Try to multiply 8 * 9
            6. Try to add 20 + 30
            
            Use the tools for each calculation and report the results.""",
        }
    ]

    response = await chat_async(
        provider=LLMProvider.GEMINI,
        model="gemini-2.5-flash",
        messages=messages,
        tools=[add, multiply],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    # Check that tools were called
    assert response.tool_outputs is not None

    # Count tool usage
    add_count = sum(1 for output in response.tool_outputs if output["name"] == "add")
    multiply_count = sum(
        1 for output in response.tool_outputs if output["name"] == "multiply"
    )

    # Since Gemini doesn't support disabling parallel tool calls, it may exceed
    # the budget in a single response. We'll check that it at least made some calls
    # and that the response acknowledges the limitations.
    assert add_count >= 3  # At least 3 add calls (may exceed due to parallel calls)
    assert (
        multiply_count >= 1
    )  # At least 1 multiply call (may exceed due to parallel calls)

    # For stricter budget enforcement, Gemini would need to be called with
    # sequential tool calls, which it doesn't currently support

    # Note: We don't check if the response mentions budget limitations because
    # Gemini may have completed all calculations in parallel before budget limits
    # could be enforced


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
)
async def test_tool_budget_exhaustion():
    """Test that all tools become unavailable when budget is exhausted."""
    # Create config with parallel tool calls disabled
    config = LLMConfig(enable_parallel_tool_calls=False)

    # Very restrictive budget
    tool_budget = {"add": 1, "multiply": 1, "fibonacci": 1}

    messages = [
        {
            "role": "user",
            "content": """Please do these calculations in order:
            1. Add 5 + 5
            2. Multiply 3 * 3
            3. Fibonacci of 7
            4. Now try to add 2 + 2
            5. Try to multiply 4 * 4
            6. Try fibonacci of 10
            
            Tell me what happens when you try the last three calculations.""",
        }
    ]

    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-haiku-4-5",
        messages=messages,
        tools=[add, multiply, fibonacci],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    pprint(response, indent=4)

    # Each tool should be called exactly once
    assert response.tool_outputs is not None
    tool_counts = {}
    for output in response.tool_outputs:
        tool_counts[output["name"]] = tool_counts.get(output["name"], 0) + 1

    assert tool_counts.get("add", 0) == 1
    assert tool_counts.get("multiply", 0) == 1
    assert tool_counts.get("fibonacci", 0) == 1

    # Response should indicate inability to perform remaining calculations
    content_lower = response.content.lower()
    assert any(
        word in content_lower
        for word in ["cannot", "unable", "limit", "budget", "exhausted"]
    )


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
)
async def test_unlimited_tools_with_budget():
    """Test that tools not in budget can be called unlimited times."""
    # Create config with parallel tool calls disabled
    config = LLMConfig(enable_parallel_tool_calls=False)

    # Only limit add, leave others unlimited
    tool_budget = {"add": 2}

    messages = [
        {
            "role": "user",
            "content": """Please calculate:
            1. Add 1 + 1
            2. Multiply 2 * 2
            3. Fibonacci of 5
            4. Add 3 + 3
            5. Multiply 4 * 4
            6. Fibonacci of 8
            7. Multiply 5 * 5
            8. Fibonacci of 10
            
            Do all calculations using the tools.""",
        }
    ]

    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-haiku-4-5",
        messages=messages,
        tools=[add, multiply, fibonacci],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    # Count tool usage
    tool_counts = {}
    for output in response.tool_outputs:
        tool_counts[output["name"]] = tool_counts.get(output["name"], 0) + 1

    # Add should be limited to 2
    assert tool_counts.get("add", 0) == 2

    # Multiply and fibonacci should have more than 2 calls (unlimited)
    assert tool_counts.get("multiply", 0) >= 3
    assert tool_counts.get("fibonacci", 0) >= 3


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not set"
)
async def test_zero_budget():
    """Test that tools with zero budget cannot be called at all."""
    # Create config with parallel tool calls disabled
    config = LLMConfig(enable_parallel_tool_calls=False)

    tool_budget = {"add": 0, "multiply": 2}

    messages = [{"role": "user", "content": "Try to add 5 + 5, then multiply 3 * 4"}]

    response = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model="claude-haiku-4-5",
        messages=messages,
        tools=[add, multiply],
        tool_choice="auto",
        tool_budget=tool_budget,
        temperature=0.0,
        config=config,
    )

    # Check tool usage
    tool_names = (
        [output["name"] for output in response.tool_outputs]
        if response.tool_outputs
        else []
    )

    # Add should never be called
    assert "add" not in tool_names

    # Multiply should be called
    assert "multiply" in tool_names
