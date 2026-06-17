"""Tests for the ZAI provider.

The integration tests require a ``ZAI_API_KEY`` environment variable.

Run with:
    pytest tests/test_zai.py -v
"""

import re
import unittest

from pydantic import BaseModel, Field

from defog.llm.config import LLMConfig
from defog.llm.cost import CostCalculator
from defog.llm.llm_providers import LLMProvider
from defog.llm.providers import ZAIProvider
from defog.llm.providers.zai_provider import ZAI_BASE_URL, _normalize_completion_response
from defog.llm.utils import chat_async, get_provider_instance, map_model_to_provider
from tests.conftest import skip_if_no_api_key


messages_sql = [
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

acceptable_sql = [
    "select count(*) from orders",
    "select count(order_id) from orders",
    "select count(*) as total_orders from orders",
    "select count(order_id) as total_orders from orders",
    "select count(*) as total_number_of_orders from orders",
]


class SqlResponse(BaseModel):
    reasoning: str = Field(description="Your reasoning before writing the SQL.")
    sql: str = Field(description="The SQL query.")


def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class TestZAIProviderRegistration(unittest.TestCase):
    """Pure unit tests; no API key required."""

    def test_enum_member_exists(self):
        self.assertEqual(LLMProvider.ZAI.value, "zai")

    def test_get_provider_instance_uses_zai_class(self):
        config = LLMConfig(api_keys={"zai": "sk-test-not-real"})
        provider = get_provider_instance("zai", config)
        self.assertIsInstance(provider, ZAIProvider)
        self.assertEqual(provider.get_provider_name(), "zai")
        self.assertEqual(provider.base_url, ZAI_BASE_URL)
        self.assertEqual(provider.api_key, "sk-test-not-real")

    def test_glm_model_maps_to_zai(self):
        self.assertEqual(map_model_to_provider("glm-5.2"), LLMProvider.ZAI)
        self.assertEqual(map_model_to_provider("glm-5.1"), LLMProvider.ZAI)

    def test_response_format_uses_json_object_mode(self):
        provider = ZAIProvider(api_key="sk-test-not-real")
        self.assertEqual(provider._build_response_format(SqlResponse), {"type": "json_object"})

    def test_build_params_injects_schema_into_system_prompt(self):
        provider = ZAIProvider(api_key="sk-test-not-real")
        request_params, messages = provider.build_params(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            model="glm-5.2",
            response_format=SqlResponse,
        )
        self.assertEqual(request_params["response_format"], {"type": "json_object"})
        self.assertEqual(messages[0]["role"], "system")
        system_text = messages[0]["content"]
        self.assertIn("json", system_text.lower())
        self.assertIn("schema", system_text.lower())
        self.assertIn("reasoning", system_text)
        self.assertIn("sql", system_text)

    def test_build_params_omits_parallel_tool_calls(self):
        provider = ZAIProvider(api_key="sk-test-not-real")
        request_params, _ = provider.build_params(
            messages=[{"role": "user", "content": "Add 1 and 2"}],
            model="glm-5.2",
            tools=[add_numbers],
            parallel_tool_calls=True,
        )
        self.assertIn("tools", request_params)
        self.assertEqual(request_params["tool_choice"], "auto")
        self.assertNotIn("parallel_tool_calls", request_params)

    def test_normalizes_optional_response_fields(self):
        data = _normalize_completion_response(
            {
                "request_id": "req-1",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            }
        )
        self.assertEqual(data["id"], "req-1")
        self.assertEqual(data["usage"]["prompt_tokens"], 0)
        self.assertIsNone(data["choices"][0]["message"]["tool_calls"])

    def test_pricing_entries_resolve(self):
        cost = CostCalculator.calculate_cost(
            "glm-5.2",
            input_tokens=1000,
            output_tokens=1000,
            cached_input_tokens=1000,
        )
        self.assertGreater(cost, 0)


class TestZAIProviderIntegration(unittest.IsolatedAsyncioTestCase):
    """Live API tests; skipped without ZAI_API_KEY."""

    def _check_sql(self, sql: str):
        sql = sql.replace("```sql", "").replace("```", "").strip(";\n ").lower()
        sql = re.sub(r"(\s+)", " ", sql).strip()
        self.assertIn(sql, acceptable_sql)

    @skip_if_no_api_key("zai")
    async def test_simple_chat(self):
        response = await chat_async(
            provider=LLMProvider.ZAI,
            model="glm-5.2",
            messages=[
                {"role": "user", "content": "Return a greeting in not more than 2 words.\n"}
            ],
            temperature=0.0,
            max_completion_tokens=64,
            max_retries=1,
        )
        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)
        self.assertIsInstance(response.time, float)
        self.assertIsNotNone(response.response_id)

    @skip_if_no_api_key("zai")
    async def test_simple_chat_glm_5_1(self):
        response = await chat_async(
            provider=LLMProvider.ZAI,
            model="glm-5.1",
            messages=[{"role": "user", "content": "Return exactly: ok"}],
            temperature=0.0,
            max_completion_tokens=64,
            max_retries=1,
        )
        self.assertIn("ok", response.content.lower())

    @skip_if_no_api_key("zai")
    async def test_structured_output(self):
        response = await chat_async(
            provider=LLMProvider.ZAI,
            model="glm-5.2",
            messages=messages_sql,
            response_format=SqlResponse,
            temperature=0.0,
            max_completion_tokens=512,
            max_retries=1,
        )
        self.assertIsInstance(response.content, SqlResponse)
        self._check_sql(response.content.sql)
        self.assertIsInstance(response.content.reasoning, str)
        self.assertGreater(response.cost_in_cents, 0)

    @skip_if_no_api_key("zai")
    async def test_tool_call(self):
        response = await chat_async(
            provider=LLMProvider.ZAI,
            model="glm-5.2",
            messages=[
                {
                    "role": "user",
                    "content": "Use the available tool to add 17 and 25. Return only the final number.",
                }
            ],
            tools=[add_numbers],
            tool_choice="required",
            temperature=0.0,
            max_completion_tokens=256,
            max_retries=1,
        )
        self.assertIn("42", str(response.content))
        self.assertTrue(response.tool_outputs)
        self.assertEqual(response.tool_outputs[0]["name"], "add_numbers")
        self.assertEqual(response.tool_outputs[0]["result"], 42)


if __name__ == "__main__":
    unittest.main()
