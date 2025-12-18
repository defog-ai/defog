import unittest
import pytest
import asyncio
import sys
from copy import deepcopy
from types import SimpleNamespace
import uuid
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
            map_model_to_provider("claude-haiku-4-5"),
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
        self.assertEqual(
            map_model_to_provider("grok-4-fast-non-reasoning-latest"), LLMProvider.GROK
        )

        with self.assertRaises(Exception):
            map_model_to_provider("unknown-model")

    def test_deepseek_provider_capabilities(self):
        """Test DeepSeek provider instantiation and capabilities"""
        from defog.llm.utils import get_provider_instance
        from defog.llm.providers.deepseek_provider import DeepSeekProvider
        from defog.llm.config import LLMConfig

        # Test provider instantiation
        config = LLMConfig(api_keys={"deepseek": "test-api-key"})
        provider = get_provider_instance("deepseek", config)
        self.assertIsInstance(provider, DeepSeekProvider)
        self.assertEqual(provider.get_provider_name(), "deepseek")

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

    def test_grok_provider_capabilities(self):
        """Test Grok provider instantiation and capabilities"""
        from defog.llm.utils import get_provider_instance
        from defog.llm.providers.grok_provider import GrokProvider
        from defog.llm.config import LLMConfig

        config = LLMConfig(api_keys={"grok": "test-api-key"})
        provider = get_provider_instance("grok", config)
        self.assertIsInstance(provider, GrokProvider)
        self.assertEqual(provider.get_provider_name(), "grok")
        self.assertEqual(provider.api_key, "test-api-key")
        self.assertEqual(provider.base_url, "https://api.x.ai")

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
            test_models.append("claude-haiku-4-5")
        if AVAILABLE_MODELS.get("openai"):
            test_models.extend(["gpt-4.1-mini", "o4-mini", "o3"])
        if AVAILABLE_MODELS.get("gemini"):
            test_models.extend(
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3-flash-preview",
                    "gemini-3-pro-preview",
                ]
            )
        if AVAILABLE_MODELS.get("mistral"):
            test_models.append("mistral-small-latest")
        if AVAILABLE_MODELS.get("grok"):
            test_models.extend(AVAILABLE_MODELS["grok"])

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
    @skip_if_no_api_key("grok")
    async def test_grok_simple_chat_async(self):
        messages = [
            {
                "role": "user",
                "content": "Reply with a single-word greeting.",
            }
        ]

        response = await chat_async(
            provider=LLMProvider.GROK,
            model="grok-4-fast-non-reasoning-latest",
            messages=messages,
            temperature=0.0,
            max_retries=1,
        )

        self.assertIsInstance(response.content, str)
        self.assertLessEqual(len(response.content.split()), 3)

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
            test_models.extend(
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3-flash-preview",
                    "gemini-3-pro-preview",
                ]
            )
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
            and "claude-haiku-4-5" in AVAILABLE_MODELS["anthropic"]
        ):
            test_models.append("claude-haiku-4-5")

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
            test_models.extend(
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-3-flash-preview",
                    "gemini-3-pro-preview",
                ]
            )
        if AVAILABLE_MODELS.get("anthropic"):
            test_models.append("claude-haiku-4-5")
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

    @skip_if_no_api_key("anthropic")
    async def test_chat_async_conversation_follow_up_with_tools_real(self):
        from defog.llm.memory import conversation_cache

        conversation_cache.clear_cache()

        async def multiply_numbers(x: float, y: float) -> float:
            """Multiply two numbers via tool call."""
            return x * y

        model = (
            AVAILABLE_MODELS["anthropic"][0]
            if AVAILABLE_MODELS.get("anthropic")
            else "claude-haiku-4-5"
        )

        system_prompt = (
            "You are an assistant that must use available tools when asked. "
            "You have a tool named multiply_numbers for multiplying two numbers."
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Call the multiply_numbers tool exactly once to multiply 6234 and 42, "
                    "then respond with just the product and nothing else."
                ),
            },
        ]

        response1 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            messages=initial_messages,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
        )

        assert response1.tool_outputs, "expected multiply_numbers tool to be invoked"
        tool_output = response1.tool_outputs[0]
        assert tool_output["name"] == "multiply_numbers"
        assert tool_output["result"] in (261828, 261828.0)
        assert response1.response_id

        cached_first = conversation_cache.load_messages(response1.response_id)
        assert cached_first is not None
        assert cached_first[-1]["role"] == "assistant"

        follow_up_messages = [
            {
                "role": "user",
                "content": (
                    "Using the product you just computed, multiply the answer by 84."
                    "Reply with the final number only."
                ),
            }
        ]

        response2 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            messages=follow_up_messages,
            temperature=0.0,
            max_retries=1,
            previous_response_id=response1.response_id,
            tools=[multiply_numbers],
        )

        numbers = re.findall(r"-?\d+", str(response2.content))
        assert numbers, "expected numeric answer in follow-up response"
        assert int(numbers[-1]) == 21993552
        assert response2.tool_outputs, "expected multiply_numbers tool to be invoked"
        tool_output = response2.tool_outputs[0]
        assert tool_output["name"] == "multiply_numbers"
        assert tool_output["result"] in (21993552, 21993552.0)

        cached_second = conversation_cache.load_messages(response2.response_id)
        assert cached_second is not None
        assert cached_second[-1]["role"] == "assistant"
        assert cached_second[-1]["content"] == response2.content

        conversation_cache.clear_cache()

    @skip_if_no_api_key("gemini")
    async def test_chat_async_gemini_follow_up_with_tools_real(self):
        from defog.llm.memory import conversation_cache

        conversation_cache.clear_cache()

        async def multiply_numbers(x: float, y: float) -> float:
            """Multiply two numbers via tool call."""
            return x * y

        model = (
            "gemini-2.5-flash"
            if "gemini-2.5-flash" in AVAILABLE_MODELS.get("gemini", [])
            else AVAILABLE_MODELS["gemini"][0]
        )

        system_prompt = (
            "You are a helpful assistant. Use the multiply_numbers tool whenever a user asks "
            "you to multiply numbers."
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Invoke multiply_numbers exactly once to multiply 6234 and 42. "
                    "After the tool responds, respond with just the product and nothing else."
                ),
            },
        ]

        response1 = await chat_async(
            provider=LLMProvider.GEMINI,
            model=model,
            messages=initial_messages,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
        )

        assert response1.tool_outputs, "expected multiply_numbers tool to be invoked"
        tool_output = response1.tool_outputs[0]
        assert tool_output["name"] == "multiply_numbers"
        assert tool_output["result"] in (261828, 261828.0)
        assert response1.response_id

        # cached_first = conversation_cache.load_messages(response1.response_id)
        # assert cached_first is not None
        # assert cached_first[-1]["role"] == "assistant"

        follow_up_messages = [
            {
                "role": "user",
                "content": (
                    "Using the product you computed, multiply the answer by 84."
                    "Reply with just the final number and nothing else."
                ),
            }
        ]

        response2 = await chat_async(
            provider=LLMProvider.GEMINI,
            model=model,
            messages=follow_up_messages,
            temperature=0.0,
            max_retries=1,
            previous_response_id=response1.response_id,
            tools=[multiply_numbers],
        )

        numbers = re.findall(r"-?\d+", str(response2.content))
        assert numbers, "expected numeric answer in follow-up response"
        assert int(numbers[-1]) == 21993552
        assert response2.tool_outputs, "expected multiply_numbers tool to be invoked"
        tool_output = response2.tool_outputs[0]
        assert tool_output["name"] == "multiply_numbers"
        assert tool_output["result"] in (21993552, 21993552.0)

        # cached_second = conversation_cache.load_messages(response2.response_id)
        # assert cached_second is not None
        # assert cached_second[-1]["role"] == "assistant"
        # assert cached_second[-1]["content"] == response2.content

        conversation_cache.clear_cache()

    async def test_chat_async_conversation_follow_up_with_tools_real_grandparent_cache(
        self,
    ):
        from defog.llm.memory import conversation_cache

        conversation_cache.clear_cache()

        async def multiply_numbers(x: float, y: float) -> float:
            """Multiply two numbers via tool call."""
            return x * y

        model = (
            AVAILABLE_MODELS["anthropic"][0]
            if AVAILABLE_MODELS.get("anthropic")
            else "claude-haiku-4-5"
        )

        system_prompt = (
            "You are an assistant that must use available tools when asked. "
            "You have a tool named multiply_numbers for multiplying two numbers."
        )
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 6234 * 42?"},
        ]

        response1 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            messages=initial_messages,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
        )
        assert response1.response_id is not None
        original_response_id = response1.response_id

        second_messages = [{"role": "user", "content": "Multiply this number by 84"}]
        response2 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            messages=second_messages,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
            previous_response_id=original_response_id,
        )
        second_response_id = response2.response_id
        assert response2.response_id is not None
        assert response2.response_id != original_response_id
        assert response2.tool_outputs is not None
        assert response2.tool_outputs[0]["name"] == "multiply_numbers"
        assert response2.tool_outputs[0]["result"] in (21993552, 21993552.0)

        third_messages = [
            {
                "role": "user",
                "content": "Multiply the result of the *original* question by 7",
            }
        ]
        response3 = await chat_async(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            messages=third_messages,
            previous_response_id=second_response_id,
            tools=[multiply_numbers],
            temperature=0.0,
            max_retries=1,
        )
        print(response3)
        assert response3.response_id is not None
        assert response3.tool_outputs is not None
        assert response3.tool_outputs[0]["name"] == "multiply_numbers"
        assert response3.tool_outputs[0]["result"] in (1832796, 1832796.0)

        conversation_cache.clear_cache()


@pytest.mark.asyncio
async def test_anthropic_previous_response_uses_conversation_cache(monkeypatch):
    from defog.llm.providers.anthropic_provider import AnthropicProvider
    from defog.llm.memory import conversation_cache

    conversation_cache.clear_cache()
    provider = AnthropicProvider(api_key="api-key")

    captured_messages = []

    def fake_build_params(self, messages, model, **kwargs):
        captured_messages.append(deepcopy(messages))
        return ({"messages": messages, "model": model}, messages)

    monkeypatch.setattr(AnthropicProvider, "build_params", fake_build_params)

    call_index = {"value": 0}
    assistant_contents = ["First answer", "Second answer"]

    async def fake_process_response(
        self,
        client,
        response,
        request_params,
        tools,
        tool_dict,
        response_format=None,
        post_tool_function=None,
        post_response_hook=None,
        tool_handler=None,
        **kwargs,
    ):
        content = assistant_contents[call_index["value"]]
        call_index["value"] += 1
        return (content, [], 10, 5, 0, 0, None)

    monkeypatch.setattr(AnthropicProvider, "process_response", fake_process_response)

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            self.messages = self

        async def create(self, **kwargs):
            return SimpleNamespace(id=f"anth_{uuid.uuid4().hex}")

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )

    base_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]

    try:
        response1 = await provider.execute_chat(
            messages=base_messages,
            model="claude-3-sonnet",
        )
        assert response1.response_id is not None

        cached_first = conversation_cache.load_messages(response1.response_id)
        assert cached_first == base_messages + [
            {"role": "assistant", "content": "First answer"}
        ]
        assert captured_messages[0] == base_messages

        follow_up = [{"role": "user", "content": "Tell me more"}]

        response2 = await provider.execute_chat(
            messages=follow_up,
            model="claude-3-sonnet",
            previous_response_id=response1.response_id,
        )

        expected_request_messages = base_messages + [
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Tell me more"},
        ]
        assert captured_messages[1] == expected_request_messages

        cached_second = conversation_cache.load_messages(response2.response_id)
        assert cached_second == expected_request_messages + [
            {"role": "assistant", "content": "Second answer"}
        ]
    finally:
        conversation_cache.clear_cache()


@pytest.mark.asyncio
async def test_gemini_previous_response_uses_conversation_cache(monkeypatch):
    from defog.llm.providers.gemini_provider import GeminiProvider
    from defog.llm.memory import conversation_cache

    conversation_cache.clear_cache()
    provider = GeminiProvider(api_key="api-key")

    captured_messages = []

    def fake_build_params(self, messages, model, **kwargs):
        captured_messages.append(deepcopy(messages))
        return ({"temperature": 0.0}, [["placeholder"]])

    monkeypatch.setattr(GeminiProvider, "build_params", fake_build_params)

    call_index = {"value": 0}
    assistant_contents = ["Gemini first", "Gemini second"]

    async def fake_process_response(
        self,
        client,
        response,
        request_params,
        messages,
        tools,
        tool_dict,
        response_format=None,
        model: str = "",
        post_tool_function=None,
        post_response_hook=None,
        tool_handler=None,
        **kwargs,
    ):
        content = assistant_contents[call_index["value"]]
        call_index["value"] += 1
        return (content, [], 8, 4, None, None)

    monkeypatch.setattr(GeminiProvider, "process_response", fake_process_response)

    class FakeGeminiClient:
        def __init__(self, **kwargs):
            self.aio = SimpleNamespace(interactions=self)

        async def create(self, **kwargs):
            return SimpleNamespace(id=f"gem_{uuid.uuid4().hex}")

    monkeypatch.setattr(
        "defog.llm.providers.gemini_provider.genai.Client", FakeGeminiClient
    )
    monkeypatch.setattr(
        "defog.llm.providers.gemini_provider.types.GenerationConfig",
        lambda **kwargs: kwargs,
    )

    base_messages = [
        {"role": "system", "content": "You are Gemini."},
        {"role": "user", "content": "Hi"},
    ]

    try:
        response1 = await provider.execute_chat(
            messages=base_messages,
            model="gemini-1.5-flash",
        )
        assert response1.response_id is not None

        # cached_first = conversation_cache.load_messages(response1.response_id)
        # assert cached_first == base_messages + [
        #     {"role": "assistant", "content": "Gemini first"}
        # ]
        assert captured_messages[0] == base_messages

        follow_up = [{"role": "user", "content": "Any other info?"}]

        response2 = await provider.execute_chat(
            messages=follow_up,
            model="gemini-1.5-flash",
            previous_response_id=response1.response_id,
        )

        # Gemini provider with Interactions API only sends new messages
        expected_request_messages = follow_up
        assert captured_messages[1] == expected_request_messages

        # cached_second = conversation_cache.load_messages(response2.response_id)
        # assert cached_second == expected_request_messages + [
        #     {"role": "assistant", "content": "Gemini second"}
        # ]
    finally:
        conversation_cache.clear_cache()
