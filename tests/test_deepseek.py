"""Tests for the DeepSeek provider.

The integration tests require a ``DEEPSEEK_API_KEY`` environment variable.
The registration and mocked request tests run without an API key.

Run with:
    pytest tests/test_deepseek.py -v
"""

import re
import unittest
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from defog.llm.config import LLMConfig
from defog.llm.cost import CostCalculator
from defog.llm.llm_providers import LLMProvider
from defog.llm.providers import DeepSeekProvider
from defog.llm.providers.deepseek_provider import DEEPSEEK_BASE_URL
from defog.llm.utils import chat_async, get_provider_instance
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


@pytest.mark.parametrize(
    ("model", "api_model"),
    [
        ("deepseek-v4.1-flash", "deepseek-flash"),
        ("deepseek-flash", "deepseek-flash"),
        ("deepseek-v4-flash", "deepseek-v4-flash"),
        ("deepseek-v4-flash-vision-exp", "deepseek-v4-flash-vision-exp"),
        ("deepseek-v4-pro", "deepseek-v4-pro"),
    ],
)
def test_deepseek_api_model_names(model, api_model):
    provider = DeepSeekProvider(api_key="sk-test-not-real")
    params, _ = provider.build_params(
        messages=[{"role": "user", "content": "Hello"}],
        model=model,
    )
    assert params["model"] == api_model


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["deepseek-v4.1-flash", "deepseek-flash"])
@pytest.mark.parametrize("mode", ["chat", "structured", "repair"])
async def test_flash_requests_and_cached_token_cost(monkeypatch, model, mode):
    """Exercise public chat, schema injection, and model-driven JSON repair."""
    structured_content = (
        '{"reasoning": "Count orders", "sql": "SELECT COUNT(*) FROM orders"}'
    )
    raw_content = {
        "chat": "Hello",
        "structured": structured_content,
        "repair": "not JSON",
    }[mode]

    def completion(content):
        return ChatCompletion.model_validate(
            {
                "id": "deepseek-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1500,
                    "prompt_tokens_details": {"cached_tokens": 500},
                    "completion_tokens": 200,
                    "total_tokens": 1700,
                },
            }
        )

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.chat.completions.create.side_effect = [
        completion(raw_content),
        completion(structured_content),
    ]
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: client)
    monkeypatch.setattr(DeepSeekProvider, "persist_conversation_history", AsyncMock())
    monkeypatch.setattr(
        "defog.llm.cost.calculator._is_deepseek_peak_time",
        lambda *args: True,
    )

    response = await chat_async(
        provider="deepseek",
        model=model,
        messages=[{"role": "user", "content": "Count the orders."}],
        response_format=SqlResponse if mode != "chat" else None,
        config=LLMConfig(api_keys={"deepseek": "sk-test-not-real"}),
        max_retries=1,
    )

    calls = client.chat.completions.create.await_args_list
    call_count = 2 if mode == "repair" else 1
    assert len(calls) == call_count
    assert all(call.kwargs["model"] == "deepseek-flash" for call in calls)
    assert response.model == model
    assert response.input_tokens == 1000 * call_count
    assert response.cached_input_tokens == 500 * call_count
    assert response.output_tokens == 200 * call_count
    assert response.cost_in_cents == pytest.approx(0.0543 * call_count)
    if mode == "chat":
        assert response.content == "Hello"
    else:
        assert response.content == SqlResponse.model_validate_json(structured_content)
        assert calls[0].kwargs["response_format"] == {"type": "json_object"}
        assert "JSON schema:" in calls[0].kwargs["messages"][0]["content"]


@pytest.mark.asyncio
@skip_if_no_api_key("deepseek")
@pytest.mark.parametrize("model", ["deepseek-v4.1-flash", "deepseek-flash"])
@pytest.mark.parametrize("structured", [False, True])
async def test_flash_live_chat(model, structured):
    response = await chat_async(
        provider="deepseek",
        model=model,
        messages=messages_sql,
        response_format=SqlResponse if structured else None,
        max_retries=1,
    )
    assert response.model == model
    assert response.content
    assert response.cost_in_cents > 0
    if structured:
        assert isinstance(response.content, SqlResponse)
        assert response.content.sql


class TestDeepSeekProviderRegistration(unittest.TestCase):
    """Pure unit tests — no API key required."""

    def test_enum_member_exists(self):
        self.assertEqual(LLMProvider.DEEPSEEK.value, "deepseek")

    def test_get_provider_instance_uses_deepseek_class(self):
        config = LLMConfig(
            api_keys={"deepseek": "sk-test-not-real"},
        )
        provider = get_provider_instance("deepseek", config)
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(provider.get_provider_name(), "deepseek")
        self.assertEqual(provider.base_url, DEEPSEEK_BASE_URL)
        self.assertEqual(provider.api_key, "sk-test-not-real")

    def test_response_format_uses_json_object_mode(self):
        """DeepSeek rejects ``json_schema``; provider must downgrade to
        ``json_object``."""
        provider = DeepSeekProvider(api_key="sk-test-not-real")
        rf = provider._build_response_format(SqlResponse)
        self.assertEqual(rf, {"type": "json_object"})

    def test_build_params_injects_schema_into_system_prompt(self):
        """json_object mode requires the literal word ``json`` in a message
        and (for shape control) a copy of the schema."""
        provider = DeepSeekProvider(api_key="sk-test-not-real")
        request_params, messages = provider.build_params(
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            model="deepseek-v4-pro",
            response_format=SqlResponse,
        )
        self.assertEqual(request_params["response_format"], {"type": "json_object"})
        self.assertEqual(messages[0]["role"], "system")
        system_text = messages[0]["content"]
        self.assertIn("json", system_text.lower())
        self.assertIn("schema", system_text.lower())
        self.assertIn("reasoning", system_text)
        self.assertIn("sql", system_text)

    def test_pricing_entries_resolve(self):
        """deepseek-v4-pro / -flash should resolve to non-zero costs."""
        pro = CostCalculator.calculate_cost(
            "deepseek-v4-pro",
            input_tokens=1000,
            output_tokens=1000,
            cached_input_tokens=0,
        )
        flash = CostCalculator.calculate_cost(
            "deepseek-v4-flash",
            input_tokens=1000,
            output_tokens=1000,
            cached_input_tokens=0,
        )
        self.assertGreater(pro, 0)
        self.assertGreater(flash, 0)
        # Flash is cheaper than Pro at every component.
        self.assertLess(flash, pro)


class TestDeepSeekProviderIntegration(unittest.IsolatedAsyncioTestCase):
    """Live API tests — skipped without DEEPSEEK_API_KEY."""

    def _check_sql(self, sql: str):
        sql = sql.replace("```sql", "").replace("```", "").strip(";\n ").lower()
        sql = re.sub(r"(\s+)", " ", sql).strip()
        self.assertIn(sql, acceptable_sql)

    @skip_if_no_api_key("deepseek")
    async def test_simple_chat(self):
        response = await chat_async(
            provider=LLMProvider.DEEPSEEK,
            model="deepseek-v4-pro",
            messages=[
                {
                    "role": "user",
                    "content": "Return a greeting in not more than 2 words.\n",
                }
            ],
            temperature=0.0,
            max_retries=1,
        )
        self.assertIsInstance(response.content, str)
        self.assertGreater(len(response.content), 0)
        self.assertIsInstance(response.time, float)
        self.assertIsNotNone(response.response_id)

    @skip_if_no_api_key("deepseek")
    async def test_structured_output(self):
        response = await chat_async(
            provider=LLMProvider.DEEPSEEK,
            model="deepseek-v4-pro",
            messages=messages_sql,
            response_format=SqlResponse,
            temperature=0.0,
            max_retries=1,
        )
        self.assertIsInstance(response.content, SqlResponse)
        self._check_sql(response.content.sql)
        self.assertIsInstance(response.content.reasoning, str)
        self.assertGreater(response.cost_in_cents, 0)


if __name__ == "__main__":
    unittest.main()
