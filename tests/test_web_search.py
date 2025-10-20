import unittest
import pytest
import os
import asyncio
from defog.llm.web_search import web_search_tool
from defog.llm.llm_providers import LLMProvider
from tests.conftest import skip_if_no_api_key


class TestWebSearchTool(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_question = "What is the capital of France?"
        self.max_tokens = 1024

    def _validate_basic_structure(self, result):
        """Validate basic structure common to all providers"""
        self.assertIsInstance(result, dict)
        self.assertIn("usage", result)
        self.assertIn("search_results", result)
        self.assertIn("websites_cited", result)

    def _validate_usage_structure(self, usage, provider):
        """Validate usage structure with provider-specific fields"""
        self.assertIsInstance(usage, dict)
        self.assertIn("input_tokens", usage)
        self.assertIn("output_tokens", usage)
        self.assertIsInstance(usage["input_tokens"], int)
        self.assertIsInstance(usage["output_tokens"], int)
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)

        if provider == LLMProvider.GEMINI:
            self.assertIn("thinking_tokens", usage)
            self.assertIsInstance(usage["thinking_tokens"], int)

    def _validate_citations(self, citations, provider):
        """Validate citations structure with provider-specific fields"""
        self.assertIsInstance(citations, list)
        for citation in citations:
            self.assertIsInstance(citation, dict)
            if provider == LLMProvider.GEMINI:
                self.assertIn("source", citation)
                self.assertIn("url", citation)
                self.assertIsInstance(citation["source"], str)
            else:
                self.assertIn("url", citation)
                self.assertIn("title", citation)
                self.assertIsInstance(citation["title"], str)

            self.assertIsInstance(citation["url"], str)
            self.assertTrue(citation["url"].startswith(("http://", "https://")))

    def _validate_search_results(self, search_results, provider):
        """Validate search results structure with provider-specific types"""
        if provider == LLMProvider.ANTHROPIC:
            self.assertIsInstance(search_results, list)
            self.assertGreater(len(search_results), 0)
        else:
            self.assertIsInstance(search_results, str)
            self.assertGreater(len(search_results), 0)

    async def _test_provider_structure(self, provider, model, api_key_env):
        """Generic test for provider structure"""
        if not os.getenv(api_key_env):
            self.skipTest(f"{api_key_env} not set")

        result = await web_search_tool(
            question=self.test_question,
            model=model,
            provider=provider,
            max_tokens=self.max_tokens,
        )

        self._validate_basic_structure(result)
        self._validate_usage_structure(result["usage"], provider)
        self._validate_search_results(result["search_results"], provider)
        self._validate_citations(result["websites_cited"], provider)

        return result

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_web_search_openai_structure(self):
        await self._test_provider_structure(
            LLMProvider.OPENAI, "gpt-4.1-mini", "OPENAI_API_KEY"
        )

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_web_search_anthropic_structure(self):
        await self._test_provider_structure(
            LLMProvider.ANTHROPIC, "claude-haiku-4-5", "ANTHROPIC_API_KEY"
        )

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_web_search_gemini_structure(self):
        await self._test_provider_structure(
            LLMProvider.GEMINI, "gemini-2.0-flash", "GEMINI_API_KEY"
        )

    @pytest.mark.asyncio
    async def test_web_search_unsupported_provider(self):
        with self.assertRaises(ValueError) as context:
            await web_search_tool(
                question=self.test_question,
                model="test-model",
                provider=LLMProvider.GROK,
            )

        self.assertIn("Provider LLMProvider.GROK not supported", str(context.exception))

    @pytest.mark.asyncio
    async def test_web_search_different_questions(self):
        questions = [
            "What is machine learning?",
            "Current weather in Tokyo",
            "Latest news about artificial intelligence",
        ]

        providers_config = [
            (LLMProvider.OPENAI, "gpt-4.1-mini", "OPENAI_API_KEY"),
            (LLMProvider.ANTHROPIC, "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
            (LLMProvider.GEMINI, "gemini-2.0-flash", "GEMINI_API_KEY"),
        ]

        available_providers = [
            (provider, model)
            for provider, model, env_key in providers_config
            if os.getenv(env_key)
        ]

        if not available_providers:
            self.skipTest("No API keys set for testing")

        async def test_provider_question(provider, model, question):
            result = await web_search_tool(
                question=question,
                model=model,
                provider=provider,
                max_tokens=1024,
            )
            return provider, question, result

        # Create all test combinations
        test_tasks = []
        for provider, model in available_providers:
            for question in questions:
                test_tasks.append(test_provider_question(provider, model, question))

        # Run all tests in parallel
        results = await asyncio.gather(*test_tasks)

        # Validate all results
        for provider, question, result in results:
            with self.subTest(provider=provider.value, question=question):
                self._validate_basic_structure(result)
                self._validate_search_results(result["search_results"], provider)

    @pytest.mark.asyncio
    async def test_web_search_custom_max_tokens(self):
        providers_config = [
            (LLMProvider.OPENAI, "gpt-4.1-mini", "OPENAI_API_KEY"),
            (LLMProvider.ANTHROPIC, "claude-haiku-4-5", "ANTHROPIC_API_KEY"),
            (LLMProvider.GEMINI, "gemini-2.0-flash", "GEMINI_API_KEY"),
        ]

        available_providers = [
            (provider, model)
            for provider, model, env_key in providers_config
            if os.getenv(env_key)
        ]

        if not available_providers:
            self.skipTest("No API keys set for testing")

        async def test_provider(provider, model):
            result = await web_search_tool(
                question="Brief summary of Python programming language",
                model=model,
                provider=provider,
                max_tokens=1024,
            )
            return provider, result

        # Run all provider tests in parallel
        test_tasks = [
            test_provider(provider, model) for provider, model in available_providers
        ]
        results = await asyncio.gather(*test_tasks)

        # Validate all results
        for provider, result in results:
            with self.subTest(provider=provider.value):
                self._validate_basic_structure(result)


if __name__ == "__main__":
    unittest.main()
