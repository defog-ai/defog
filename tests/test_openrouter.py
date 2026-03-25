"""Tests for the OpenRouter provider.

These tests require an OPENROUTER_API_KEY environment variable.
They exercise basic chat, structured output, tool calling, and
conversation continuation via the OpenRouter gateway.
"""

import unittest
import re

from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
from pydantic import BaseModel, Field
from tests.conftest import skip_if_no_api_key

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

messages_sql = [
    {
        "role": "system",
        "content": (
            "Your task is to generate SQL given a natural language question and "
            "schema of the user's database. Do not use aliases. "
            "Return only the SQL without ```."
        ),
    },
    {
        "role": "user",
        "content": """Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]

acceptable_sql = [
    "select count(*) from orders",
    "select count(order_id) from orders",
    "select count(*) as total_orders from orders",
    "select count(order_id) as total_orders from orders",
    "select count(*) as total_number_of_orders from orders",
]

messages_sql_structured = [
    {
        "role": "system",
        "content": (
            "Your task is to generate SQL given a natural language question and "
            "schema of the user's database. Do not use aliases."
        ),
    },
    {
        "role": "user",
        "content": """Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]


class ResponseFormat(BaseModel):
    reasoning: str
    sql: str


class Numbers(BaseModel):
    a: int = Field(default=0, description="First number")
    b: int = Field(default=0, description="Second number")


def numsum(input: Numbers):
    """This function returns the sum of two numbers."""
    return input.a + input.b


def numprod(input: Numbers):
    """This function returns the product of two numbers."""
    return input.a * input.b


async def multiply_numbers(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenRouterProvider(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the OpenRouter provider."""

    def _check_sql(self, sql: str):
        sql = sql.replace("```sql", "").replace("```", "").strip(";\n").lower()
        sql = re.sub(r"(\s+)", " ", sql)
        self.assertIn(sql, acceptable_sql)

    # -- Basic chat ----------------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_simple_chat(self):
        """Simple greeting via OpenRouter with an Anthropic model."""
        messages = [
            {"role": "user", "content": "Return a greeting in not more than 2 words\n"}
        ]

        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-sonnet-4.6",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)
        self.assertIsInstance(response.time, float)
        self.assertIsNotNone(response.response_id)

    @skip_if_no_api_key("openrouter")
    async def test_simple_chat_openai_model(self):
        """Simple greeting via OpenRouter with an OpenAI model."""
        messages = [
            {"role": "user", "content": "Return a greeting in not more than 2 words\n"}
        ]

        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="openai/gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)

    # -- SQL generation ------------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_sql_generation(self):
        """Generate SQL via OpenRouter."""
        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-sonnet-4.6",
            messages=messages_sql,
            temperature=0.0,
            max_retries=1,
        )
        self._check_sql(response.content)

    # -- Structured output ---------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_structured_output(self):
        """Structured output (json_schema) via OpenRouter."""
        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="openai/gpt-4.1-mini",
            messages=messages_sql_structured,
            temperature=0.0,
            response_format=ResponseFormat,
            max_retries=1,
        )
        self.assertIsInstance(response.content, ResponseFormat)
        self._check_sql(response.content.sql)
        self.assertIsInstance(response.content.reasoning, str)

    # -- Tool calling --------------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_tool_calls(self):
        """Tool calling via OpenRouter."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You have access to numsum and numprod tools. "
                    "Use them to answer math questions."
                ),
            },
            {
                "role": "user",
                "content": "What is 3 + 5?",
            },
        ]

        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-sonnet-4.6",
            messages=messages,
            tools=[numsum, numprod],
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsNotNone(response.tool_outputs)
        tool_names = [o["name"] for o in response.tool_outputs if o.get("tool_call_id")]
        self.assertIn("numsum", tool_names)
        # Find the numsum result
        for o in response.tool_outputs:
            if o.get("name") == "numsum":
                self.assertEqual(o["result"], 8)

    @skip_if_no_api_key("openrouter")
    async def test_tool_calls_openai_model(self):
        """Tool calling via OpenRouter with an OpenAI model."""
        messages = [
            {
                "role": "user",
                "content": "What is 12 * 7? Use the numprod tool.",
            },
        ]

        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="openai/gpt-4.1-mini",
            messages=messages,
            tools=[numsum, numprod],
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsNotNone(response.tool_outputs)
        tool_names = [o["name"] for o in response.tool_outputs if o.get("tool_call_id")]
        self.assertIn("numprod", tool_names)
        for o in response.tool_outputs:
            if o.get("name") == "numprod":
                self.assertEqual(o["result"], 84)

    # -- Conversation continuation -------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_conversation_continuation(self):
        """Multi-turn conversation via previous_response_id."""
        from defog.llm.memory import conversation_cache

        await conversation_cache.clear_cache()

        initial_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Use the multiply_numbers tool "
                    "whenever asked to multiply."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the multiply_numbers tool to multiply 6234 and 42. "
                    "Reply with just the product."
                ),
            },
        ]

        response1 = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-sonnet-4.6",
            messages=initial_messages,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsNotNone(response1.tool_outputs)
        tool_output = next(
            (o for o in response1.tool_outputs if o.get("name") == "multiply_numbers"),
            None,
        )
        self.assertIsNotNone(tool_output)
        self.assertIn(tool_output["result"], (261828, 261828.0))
        self.assertIsNotNone(response1.response_id)

        # Follow-up using previous_response_id
        follow_up = [
            {
                "role": "user",
                "content": (
                    "Now multiply the previous result by 84 using multiply_numbers. "
                    "Reply with just the final number."
                ),
            }
        ]

        response2 = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="anthropic/claude-sonnet-4.6",
            messages=follow_up,
            temperature=0.0,
            max_retries=1,
            previous_response_id=response1.response_id,
            tools=[multiply_numbers],
        )

        numbers = re.findall(r"-?\d+", str(response2.content))
        self.assertTrue(numbers, "expected numeric answer in follow-up response")
        self.assertEqual(int(numbers[-1]), 21993552)
        tool_output2 = next(
            (o for o in response2.tool_outputs if o.get("name") == "multiply_numbers"),
            None,
        )
        self.assertIsNotNone(tool_output2)
        self.assertIn(tool_output2["result"], (21993552, 21993552.0))

        await conversation_cache.clear_cache()

    # -- Provider as string --------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_provider_as_string(self):
        """Using 'openrouter' string instead of LLMProvider enum."""
        messages = [{"role": "user", "content": "Say hello in one word."}]

        response = await chat_async(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )
        self.assertIsInstance(response.content, str)
        self.assertTrue(len(response.content) > 0)

    # -- Token usage ---------------------------------------------------------

    @skip_if_no_api_key("openrouter")
    async def test_token_usage_reported(self):
        """Verify token usage is reported in the response."""
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        response = await chat_async(
            provider=LLMProvider.OPENROUTER,
            model="openai/gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )
        self.assertGreater(response.input_tokens, 0)
        self.assertGreater(response.output_tokens, 0)
