import unittest
import pytest
import asyncio
from defog.llm.utils import (
    map_model_to_provider,
    chat_async,
)
from defog.llm.llm_providers import LLMProvider
import re

from pydantic import BaseModel
from tests.conftest import skip_if_no_api_key, skip_if_no_models, AVAILABLE_MODELS

messages_sql = [
    {
        "role": "system",
        "content": "Your task is to generate SQL given a natural language question and schema of the user's database. Do not use aliases. Return only the SQL without ```.",
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
]


class ResponseFormat(BaseModel):
    reasoning: str
    sql: str


messages_sql_structured = [
    {
        "role": "system",
        "content": "Your task is to generate SQL given a natural language question and schema of the user's database. Do not use aliases.",
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


class TestChatClients(unittest.IsolatedAsyncioTestCase):
    def check_sql(self, sql: str):
        sql = sql.replace("```sql", "").replace("```", "").strip(";\n").lower()
        sql = re.sub(r"(\s+)", " ", sql)
        self.assertIn(sql, acceptable_sql)

    def test_map_model_to_provider(self):
        self.assertEqual(
            map_model_to_provider("claude-3-5-sonnet-20241022"),
            LLMProvider.ANTHROPIC,
        )

        self.assertEqual(
            map_model_to_provider("gemini-1.5-flash-002"),
            LLMProvider.GEMINI,
        )

        self.assertEqual(map_model_to_provider("gpt-4o-mini"), LLMProvider.OPENAI)

        self.assertEqual(map_model_to_provider("deepseek-chat"), LLMProvider.DEEPSEEK)
        self.assertEqual(
            map_model_to_provider("deepseek-reasoner"), LLMProvider.DEEPSEEK
        )

        self.assertEqual(
            map_model_to_provider("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
            LLMProvider.TOGETHER,
        )

        self.assertEqual(
            map_model_to_provider("mistral-small-latest"), LLMProvider.MISTRAL
        )
        self.assertEqual(
            map_model_to_provider("mistral-medium-latest"), LLMProvider.MISTRAL
        )

        with self.assertRaises(Exception):
            map_model_to_provider("unknown-model")

    def test_deepseek_provider_capabilities(self):
        """Test DeepSeek provider instantiation and capabilities"""
        from defog.llm.utils import get_provider_instance
        from defog.llm.providers.deepseek_provider import DeepSeekProvider
        from defog.llm.config import LLMConfig

        # Test provider instantiation
        config = LLMConfig()
        provider = get_provider_instance("deepseek", config)
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(provider.get_provider_name(), "deepseek")

        # Test model capabilities
        # deepseek-chat supports tools, deepseek-reasoner does not
        self.assertTrue(provider.supports_tools("deepseek-chat"))
        self.assertFalse(provider.supports_tools("deepseek-reasoner"))

        # Both models support response_format
        self.assertTrue(provider.supports_response_format("deepseek-chat"))
        self.assertTrue(provider.supports_response_format("deepseek-reasoner"))

    def test_mistral_provider_capabilities(self):
        """Test Mistral provider instantiation and capabilities"""
        from defog.llm.utils import get_provider_instance
        from defog.llm.providers.mistral_provider import MistralProvider
        from defog.llm.config import LLMConfig

        # Test provider instantiation
        config = LLMConfig(api_keys={"mistral": "test-api-key"})
        provider = get_provider_instance("mistral", config)
        self.assertIsInstance(provider, MistralProvider)
        self.assertEqual(provider.get_provider_name(), "mistral")

        # Test model capabilities - all Mistral models support tools and structured output
        mistral_models = [
            "mistral-small-latest",
            "mistral-medium-latest",
            "mistral-large-latest",
        ]
        for model in mistral_models:
            self.assertTrue(provider.supports_tools(model))
            self.assertTrue(provider.supports_response_format(model))

    def test_deepseek_structured_output_build_params(self):
        """Test DeepSeek provider's structured output parameter building for both models"""
        from defog.llm.providers.deepseek_provider import DeepSeekProvider
        from defog.llm.config import LLMConfig

        config = LLMConfig()
        provider = DeepSeekProvider(config=config)

        # Test with Pydantic model for both DeepSeek models
        messages = [{"role": "user", "content": "Generate SQL for counting orders"}]
        deepseek_models = ["deepseek-chat", "deepseek-reasoner"]

        for model in deepseek_models:
            with self.subTest(model=model):
                # Test that Pydantic models get converted to JSON mode
                params, modified_messages = provider.build_params(
                    messages=messages, model=model, response_format=ResponseFormat
                )

                # Should set response_format to JSON mode
                self.assertEqual(params["response_format"], {"type": "json_object"})

                # Should modify the user message to include schema instructions
                self.assertIn("JSON schema", modified_messages[0]["content"])
                self.assertIn(
                    "reasoning", modified_messages[0]["content"]
                )  # From ResponseFormat schema
                self.assertIn(
                    "sql", modified_messages[0]["content"]
                )  # From ResponseFormat schema

                # Test temperature handling - deepseek-reasoner shouldn't have temperature
                if model == "deepseek-reasoner":
                    self.assertNotIn("temperature", params)
                else:
                    self.assertIn("temperature", params)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_api_key("deepseek")
    async def test_deepseek_structured_output_integration(self):
        """Test DeepSeek provider's structured output integration end-to-end for both models"""
        # This test would require an actual API key, so we'll just test the parameter building
        # In a real environment with DEEPSEEK_API_KEY, this would make an actual API call

        messages = [{"role": "user", "content": "Generate SQL to count orders"}]
        deepseek_models = ["deepseek-chat", "deepseek-reasoner"]

        for model in deepseek_models:
            with self.subTest(model=model):
                # Test that we can call chat_async with DeepSeek and structured output
                response = await chat_async(
                    provider=LLMProvider.DEEPSEEK,
                    model=model,
                    messages=messages,
                    response_format=ResponseFormat,
                    max_retries=1,
                )
                # If we get here, the API call succeeded
                self.assertIsInstance(response.content, ResponseFormat)

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_models()
    async def test_simple_chat_async(self):
        # Use a subset of available models for this test
        test_models = []
        if AVAILABLE_MODELS.get("anthropic"):
            test_models.append("claude-3-7-sonnet-latest")
        if AVAILABLE_MODELS.get("openai"):
            test_models.extend(["gpt-4.1-mini", "o4-mini", "o3"])
        if AVAILABLE_MODELS.get("gemini"):
            test_models.extend(["gemini-2.0-flash", "gemini-2.5-pro"])
        if AVAILABLE_MODELS.get("mistral"):
            test_models.append("mistral-small-latest")

        models = [m for m in test_models if m in sum(AVAILABLE_MODELS.values(), [])]
        messages = [
            {"role": "user", "content": "Return a greeting in not more than 2 words\n"}
        ]

        async def test_model(model):
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages,
                temperature=0.0,
                max_retries=1,
            )
            self.assertIsInstance(response.content, str)
            self.assertIsInstance(response.time, float)
            return model, response

        # Run all model tests in parallel
        results = await asyncio.gather(*[test_model(model) for model in models])

        # Verify all models completed successfully
        self.assertEqual(len(results), len(models))

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_models()
    async def test_sql_chat_async(self):
        # Use a subset of available models for SQL test
        test_models = []
        if AVAILABLE_MODELS.get("openai"):
            test_models.extend(
                ["gpt-4o-mini", "o3", "o4-mini", "gpt-4.1-mini", "gpt-4.1-nano"]
            )
        if AVAILABLE_MODELS.get("gemini"):
            test_models.extend(["gemini-2.0-flash", "gemini-2.5-pro"])
        if AVAILABLE_MODELS.get("mistral"):
            test_models.append("mistral-small-latest")

        models = [m for m in test_models if m in sum(AVAILABLE_MODELS.values(), [])]

        async def test_model(model):
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages_sql,
                temperature=0.0,
                max_retries=1,
            )
            self.check_sql(response.content)
            self.assertIsInstance(response.time, float)
            return model, response

        # Run all model tests in parallel
        results = await asyncio.gather(*[test_model(model) for model in models])

        # Verify all models completed successfully
        self.assertEqual(len(results), len(models))

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_models()
    async def test_sql_chat_structured_reasoning_effort_async(self):
        # Only test models that support reasoning effort
        test_models = []
        if AVAILABLE_MODELS.get("openai") and "o4-mini" in AVAILABLE_MODELS["openai"]:
            test_models.append("o4-mini")
        if (
            AVAILABLE_MODELS.get("anthropic")
            and "claude-3-7-sonnet-latest" in AVAILABLE_MODELS["anthropic"]
        ):
            test_models.append("claude-3-7-sonnet-latest")

        if not test_models:
            self.skipTest("No models with reasoning effort support available")

        reasoning_effort = ["low", "medium", "high", None]

        async def test_model_effort(model, effort):
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages_sql_structured,
                temperature=0.0,
                response_format=ResponseFormat,
                reasoning_effort=effort,
                max_retries=1,
            )
            self.check_sql(response.content.sql)
            self.assertIsInstance(response.content.reasoning, str)
            return (model, effort, response)

        # Create all test combinations
        test_tasks = []
        for effort in reasoning_effort:
            for model in test_models:
                test_tasks.append(test_model_effort(model, effort))

        # Run all tests in parallel
        results = await asyncio.gather(*test_tasks)

        # Verify all tests completed successfully
        self.assertEqual(len(results), len(reasoning_effort) * len(test_models))

    @pytest.mark.asyncio(loop_scope="session")
    @skip_if_no_models()
    async def test_sql_chat_structured_async(self):
        # Use a subset of available models for structured output test
        test_models = []
        if AVAILABLE_MODELS.get("openai"):
            test_models.extend(["gpt-4o", "o3", "o4-mini", "gpt-4.1", "gpt-4.1-nano"])
        if AVAILABLE_MODELS.get("gemini"):
            test_models.extend(["gemini-2.0-flash", "gemini-2.5-pro"])
        if AVAILABLE_MODELS.get("anthropic"):
            test_models.append("claude-3-7-sonnet-latest")
        if AVAILABLE_MODELS.get("mistral"):
            test_models.append("mistral-small-latest")

        models = [m for m in test_models if m in sum(AVAILABLE_MODELS.values(), [])]

        async def test_model(model):
            provider = map_model_to_provider(model)
            response = await chat_async(
                provider=provider,
                model=model,
                messages=messages_sql_structured,
                temperature=0.0,
                response_format=ResponseFormat,
                max_retries=1,
            )
            self.check_sql(response.content.sql)
            self.assertIsInstance(response.content.reasoning, str)
            return model, response

        # Run all model tests in parallel
        results = await asyncio.gather(*[test_model(model) for model in models])

        # Verify all models completed successfully
        self.assertEqual(len(results), len(models))


if __name__ == "__main__":
    unittest.main()
