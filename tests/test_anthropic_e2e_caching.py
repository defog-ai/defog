"""
End-to-end test: Anthropic provider with tool calls, follow-up questions
(previous_response_id), and prompt caching verification.

Requires ANTHROPIC_API_KEY in .env. Run with:
    PYTHONPATH=. python -m pytest tests/test_anthropic_e2e_caching.py -v --envfile .env
"""

import pytest
from defog.llm import chat_async
from defog.llm.llm_providers import LLMProvider


async def calculator(a: float, b: float, operation: str) -> str:
    """
    A calculator tool that performs arithmetic on two numbers.
    Supports: add, subtract, multiply, divide, power.
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        result = a / b
    elif operation == "power":
        result = a**b
    else:
        return f"Unknown operation: {operation}"
    return str(result)


# Long system prompt to exceed Anthropic's 1024-token minimum for prompt caching
SYSTEM_PROMPT = """\
You are a precise mathematical assistant. You must always use the calculator \
tool for any arithmetic operations - never do mental math.

Rules:
1. For every arithmetic operation, you MUST call the calculator tool.
2. Show the intermediate results clearly.
3. When chaining operations, use the exact result from the previous calculator \
call as input to the next one.
4. Format large numbers with commas for readability.
5. Always verify your tool call results make sense before presenting them.

Additional context about precision:
- When working with very large numbers (greater than 10^15), floating point \
precision may be a concern.
- Python's arbitrary precision integers handle exact arithmetic for whole numbers.
- For division operations, results may be approximate due to floating point \
representation.
- When raising numbers to powers, the result can grow extremely quickly.
- Scientific notation may be appropriate for results exceeding 20 digits.

Mathematical background for reference:
- The fundamental theorem of arithmetic states that every integer greater than \
1 is either prime or can be represented uniquely as a product of prime numbers.
- Modular arithmetic is a system of arithmetic for integers where numbers wrap \
around after reaching a certain value called the modulus.
- The distributive property states that a(b + c) = ab + ac, which can be useful \
for mental verification of calculator results.
- Exponentiation is right-associative: a^b^c = a^(b^c), not (a^b)^c.
- The binomial theorem provides a formula for expanding (a + b)^n.
- Euler's identity e^(iπ) + 1 = 0 connects five fundamental mathematical constants.
- The golden ratio φ = (1 + √5) / 2 ≈ 1.618033988749895 appears frequently in \
nature and mathematics.
- Pi (π) ≈ 3.14159265358979323846 is the ratio of a circle's circumference to \
its diameter.
- The natural logarithm base e ≈ 2.71828182845904523536 is fundamental to \
calculus and analysis.
- Fermat's Last Theorem states that no three positive integers a, b, c satisfy \
a^n + b^n = c^n for any integer n > 2.
- The Riemann hypothesis concerns the distribution of prime numbers.
- The Collatz conjecture states that the sequence defined by: if n is even, \
divide by 2; if n is odd, multiply by 3 and add 1, always eventually reaches 1.
- Goldbach's conjecture posits that every even integer greater than 2 can be \
expressed as the sum of two primes.
- The twin prime conjecture suggests there are infinitely many pairs of primes \
that differ by 2.
- The P vs NP problem asks whether every problem whose solution can be quickly \
verified can also be quickly solved.
- The Navier-Stokes existence and smoothness problem concerns the mathematical \
properties of solutions to fluid dynamics equations.
- The Birch and Swinnerton-Dyer conjecture deals with elliptic curves.
- The Hodge conjecture relates algebraic geometry and topology.
- Yang-Mills existence and mass gap concerns quantum field theory foundations.

Remember: Always use the calculator tool. Never compute arithmetic yourself."""

MODEL = "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_tool_calls_and_follow_up_with_caching():
    """First request should trigger tool calls and create a prompt cache.
    Follow-up request via previous_response_id should read from that cache."""

    # --- Turn 1: multiply then add ---
    messages_1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Using the calculator tool, compute the following step by step:\n"
                "1. Multiply 987654321 by 123456789\n"
                "2. Then add 999999999999 to that result\n"
                "Show me the intermediate and final results."
            ),
        },
    ]

    response_1 = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model=MODEL,
        messages=messages_1,
        tools=[calculator],
        temperature=0.0,
        store=True,
        strict_tools=False,
    )

    # Turn 1 assertions
    assert response_1.tool_outputs, "First response should have tool outputs"
    assert len(response_1.tool_outputs) >= 2, (
        "Expected at least 2 tool calls (multiply + add)"
    )
    assert response_1.response_id, "First response should have a response_id"

    # Verify the multiply tool call happened with correct args
    multiply_call = next(
        (
            t
            for t in response_1.tool_outputs
            if t["args"].get("operation") == "multiply"
        ),
        None,
    )
    assert multiply_call is not None, "Should have a multiply tool call"
    assert multiply_call["args"]["a"] == 987654321
    assert multiply_call["args"]["b"] == 123456789

    # First request should create cache (system prompt is large enough)
    assert (
        response_1.cache_creation_input_tokens
        and response_1.cache_creation_input_tokens > 0
    ), (
        f"Expected cache creation tokens > 0, got {response_1.cache_creation_input_tokens}"
    )

    # --- Turn 2: follow-up using previous_response_id ---
    messages_2 = [
        {
            "role": "user",
            "content": (
                "Now take the final result from above and raise it to the power of 2 "
                "using the calculator tool."
            ),
        }
    ]

    response_2 = await chat_async(
        provider=LLMProvider.ANTHROPIC,
        model=MODEL,
        messages=messages_2,
        tools=[calculator],
        temperature=0.0,
        store=True,
        strict_tools=False,
        previous_response_id=response_1.response_id,
    )

    # Turn 2 assertions
    assert response_2.tool_outputs, "Second response should have tool outputs"
    assert len(response_2.tool_outputs) >= 1, "Expected at least 1 tool call (power)"

    power_call = next(
        (t for t in response_2.tool_outputs if t["args"].get("operation") == "power"),
        None,
    )
    assert power_call is not None, "Should have a power tool call"
    assert power_call["args"]["b"] == 2, "Should raise to the power of 2"

    # The follow-up should read from the prompt cache
    assert response_2.cached_input_tokens and response_2.cached_input_tokens > 0, (
        f"Expected cached input tokens > 0, got {response_2.cached_input_tokens}"
    )

    # Sanity check: cached tokens should be a significant portion of total input
    total_input = response_2.input_tokens + response_2.cached_input_tokens
    cache_ratio = response_2.cached_input_tokens / total_input
    assert cache_ratio > 0.5, (
        f"Expected cache hit rate > 50%, got {cache_ratio:.1%} "
        f"({response_2.cached_input_tokens} cached / {total_input} total)"
    )
