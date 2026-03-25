"""Unit tests for the OpenRouter JSON repair pipeline.

These tests exercise _deterministic_json_repair, _llm_json_repair, and
_parse_with_repair without hitting any external API.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, Field

from defog.llm.providers.openrouter_provider import OpenRouterProvider


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------


class SimpleOutput(BaseModel):
    name: str
    value: int


class NestedOutput(BaseModel):
    items: list[str]
    metadata: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic repair
# ---------------------------------------------------------------------------


class TestDeterministicJsonRepair(unittest.TestCase):
    repair = staticmethod(OpenRouterProvider._deterministic_json_repair)

    def _roundtrip(self, broken: str) -> dict:
        """Repair, then json.loads — raise on failure."""
        return json.loads(self.repair(broken))

    # -- trailing commas ----------------------------------------------------

    def test_trailing_comma_object(self):
        result = self._roundtrip('{"a": 1, "b": 2,}')
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_trailing_comma_array(self):
        result = self._roundtrip('{"items": [1, 2, 3,]}')
        self.assertEqual(result, {"items": [1, 2, 3]})

    def test_nested_trailing_commas(self):
        result = self._roundtrip('{"a": {"b": 1,}, "c": [1,],}')
        self.assertEqual(result, {"a": {"b": 1}, "c": [1]})

    # -- Python/JS literals -------------------------------------------------

    def test_python_true_false_none(self):
        result = self._roundtrip('{"flag": True, "empty": None, "off": False}')
        self.assertEqual(result, {"flag": True, "empty": None, "off": False})

    def test_nan_infinity(self):
        result = self._roundtrip('{"a": NaN, "b": Infinity, "c": -Infinity}')
        self.assertEqual(result, {"a": None, "b": None, "c": None})

    def test_undefined(self):
        result = self._roundtrip('{"a": undefined}')
        self.assertEqual(result, {"a": None})

    # -- comments -----------------------------------------------------------

    def test_single_line_comment(self):
        broken = '{\n  "a": 1, // this is a comment\n  "b": 2\n}'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_leading_line_comment(self):
        broken = '// comment\n{"a": 1}'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"a": 1})

    def test_multiline_comment(self):
        broken = '{"a": /* some comment */ 1}'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"a": 1})

    # -- markdown fences ----------------------------------------------------

    def test_markdown_json_fence(self):
        broken = '```json\n{"a": 1}\n```'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"a": 1})

    def test_markdown_plain_fence(self):
        broken = '```\n{"a": 1}\n```'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"a": 1})

    # -- single quotes ------------------------------------------------------

    def test_single_quoted_json(self):
        broken = "{'name': 'hello', 'value': 42}"
        result = self._roundtrip(broken)
        self.assertEqual(result, {"name": "hello", "value": 42})

    # -- unbalanced brackets ------------------------------------------------

    def test_missing_closing_brace(self):
        broken = '{"name": "test", "value": 1'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"name": "test", "value": 1})

    def test_missing_closing_bracket(self):
        broken = '{"items": ["a", "b"'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"items": ["a", "b"]})

    def test_missing_multiple_closers(self):
        broken = '{"items": ["a", "b"'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"items": ["a", "b"]})

    # -- truncated content --------------------------------------------------

    def test_trailing_colon(self):
        broken = '{"name": "test", "value":'
        repaired = self.repair(broken)
        result = json.loads(repaired)
        self.assertEqual(result["name"], "test")
        self.assertIsNone(result["value"])

    def test_trailing_comma_incomplete(self):
        broken = '{"name": "test","value": 1,'
        repaired = self.repair(broken)
        result = json.loads(repaired)
        self.assertEqual(result, {"name": "test", "value": 1})

    # -- unclosed string ----------------------------------------------------

    def test_unclosed_string(self):
        broken = '{"name": "test'
        repaired = self.repair(broken)
        result = json.loads(repaired)
        self.assertEqual(result["name"], "test")

    # -- combined issues ----------------------------------------------------

    def test_combined_issues(self):
        broken = '```json\n{"flag": True, "val": NaN, "items": [1, 2,]}\n```'
        result = self._roundtrip(broken)
        self.assertEqual(result, {"flag": True, "val": None, "items": [1, 2]})

    # -- already valid JSON -------------------------------------------------

    def test_valid_json_passthrough(self):
        valid = '{"name": "test", "value": 42}'
        result = self._roundtrip(valid)
        self.assertEqual(result, {"name": "test", "value": 42})


# ---------------------------------------------------------------------------
# _parse_with_repair  (mocked LLM)
# ---------------------------------------------------------------------------


class TestParseWithRepair(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = OpenRouterProvider.__new__(OpenRouterProvider)
        # Minimal init for the base class methods we need
        self.provider.api_key = "test"
        self.provider.base_url = "http://test"
        self.provider.config = None
        self.provider.tool_handler = None

        self.mock_client = MagicMock()

    async def test_valid_json_no_repair(self):
        """Valid JSON should parse on the first try — no repair needed."""
        raw = '{"name": "alice", "value": 10}'
        result = await self.provider._parse_with_repair(
            raw, SimpleOutput, self.mock_client, "test-model"
        )
        self.assertIsInstance(result, SimpleOutput)
        self.assertEqual(result.name, "alice")
        self.assertEqual(result.value, 10)

    async def test_deterministic_repair_fixes_trailing_comma(self):
        """Trailing comma should be fixed deterministically."""
        raw = '{"name": "bob", "value": 5,}'
        result = await self.provider._parse_with_repair(
            raw, SimpleOutput, self.mock_client, "test-model"
        )
        self.assertIsInstance(result, SimpleOutput)
        self.assertEqual(result.name, "bob")
        self.assertEqual(result.value, 5)

    async def test_deterministic_repair_fixes_python_literals(self):
        """Python True/False/None should be fixed deterministically."""
        raw = '{"name": "charlie", "value": 7, "flag": True}'
        # True is not valid JSON — deterministic repair converts to true
        # SimpleOutput ignores extra fields, so this should parse fine
        result = await self.provider._parse_with_repair(
            raw, SimpleOutput, self.mock_client, "test-model"
        )
        self.assertIsInstance(result, SimpleOutput)
        self.assertEqual(result.name, "charlie")
        self.assertEqual(result.value, 7)

    async def test_llm_repair_called_when_deterministic_fails(self):
        """When deterministic repair can't fix it, the LLM fallback fires."""
        # Completely garbled content that deterministic repair can't fix
        raw = "The answer is name=alice and value=10, here you go!"

        # Mock the LLM repair response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"name": "alice", "value": 10}'))
        ]
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await self.provider._parse_with_repair(
            raw,
            SimpleOutput,
            self.mock_client,
            "test-model",
            {"messages": [{"role": "user", "content": "give me data"}]},
        )
        self.assertIsInstance(result, SimpleOutput)
        self.assertEqual(result.name, "alice")
        self.assertEqual(result.value, 10)

        # Verify the LLM was called
        self.mock_client.chat.completions.create.assert_called_once()

    async def test_llm_repair_uses_conversation_context(self):
        """The LLM repair call should include original messages as context."""
        raw = "totally broken json!!!"
        original_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Return structured data."},
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"name": "fixed", "value": 1}'))
        ]
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await self.provider._parse_with_repair(
            raw,
            SimpleOutput,
            self.mock_client,
            "test-model",
            {"messages": original_messages},
        )

        # Check that the repair call included the original messages +
        # the broken assistant turn + the user follow-up
        call_args = self.mock_client.chat.completions.create.call_args
        messages_sent = call_args.kwargs.get("messages", call_args[1].get("messages"))
        self.assertEqual(messages_sent[0]["role"], "system")
        self.assertEqual(messages_sent[1]["role"], "user")
        self.assertEqual(messages_sent[2]["role"], "assistant")
        self.assertEqual(messages_sent[2]["content"], raw)
        self.assertEqual(messages_sent[3]["role"], "user")
        self.assertIn("schema", messages_sent[3]["content"].lower())

    async def test_returns_raw_when_all_repair_fails(self):
        """When even LLM repair fails, raw content is returned."""
        raw = "completely unparseable garbage"

        # LLM also returns garbage
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="still broken"))]
        self.mock_client.chat = MagicMock()
        self.mock_client.chat.completions = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await self.provider._parse_with_repair(
            raw,
            SimpleOutput,
            self.mock_client,
            "test-model",
            {"messages": []},
        )
        # Should return the original raw content
        self.assertEqual(result, raw)

    async def test_no_repair_when_no_response_format(self):
        """With no response_format, raw content is returned as-is."""
        raw = "just a plain string"
        result = await self.provider._parse_with_repair(
            raw, None, self.mock_client, "test-model"
        )
        self.assertEqual(result, raw)


if __name__ == "__main__":
    unittest.main()
