"""Unit tests for shared schema-aware structured-output repair."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, ConfigDict, Field

from defog.llm.providers.deepseek_provider import DeepSeekProvider
from defog.llm.providers.openrouter_provider import OpenRouterProvider


class SimpleOutput(BaseModel):
    name: str
    value: int


class NestedOutput(BaseModel):
    items: list[str]
    metadata: dict[str, int] = Field(default_factory=dict)


class ExtractedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int
    optional_text: str
    required_text: str = Field(min_length=1)


class ExtractionOutput(BaseModel):
    items: list[ExtractedItem]
    summary: str


class TestDeterministicJsonRepair(unittest.TestCase):
    repair = staticmethod(OpenRouterProvider._deterministic_json_repair)

    def _roundtrip(self, broken: str) -> dict:
        return json.loads(self.repair(broken))

    def test_trailing_comma_object(self):
        self.assertEqual(self._roundtrip('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_trailing_comma_array(self):
        self.assertEqual(self._roundtrip('{"items": [1, 2, 3,]}'), {"items": [1, 2, 3]})

    def test_nested_trailing_commas(self):
        self.assertEqual(
            self._roundtrip('{"a": {"b": 1,}, "c": [1,],}'),
            {"a": {"b": 1}, "c": [1]},
        )

    def test_python_and_javascript_literals(self):
        self.assertEqual(
            self._roundtrip(
                '{"yes": True, "no": False, "none": None, "nan": NaN, '
                '"inf": Infinity, "neg_inf": -Infinity, "missing": undefined}'
            ),
            {
                "yes": True,
                "no": False,
                "none": None,
                "nan": None,
                "inf": None,
                "neg_inf": None,
                "missing": None,
            },
        )

    def test_comments(self):
        self.assertEqual(
            self._roundtrip('// leading\n{"a": /* inline */ 1, // trailing\n"b": 2}'),
            {"a": 1, "b": 2},
        )

    def test_markdown_fences(self):
        self.assertEqual(self._roundtrip('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(self._roundtrip('```\n{"a": 1}\n```'), {"a": 1})

    def test_single_quoted_json(self):
        self.assertEqual(
            self._roundtrip("{'name': 'hello', 'value': 42}"),
            {"name": "hello", "value": 42},
        )

    def test_unbalanced_and_truncated_json(self):
        self.assertEqual(self._roundtrip('{"items": ["a", "b"'), {"items": ["a", "b"]})
        self.assertEqual(
            self._roundtrip('{"name": "test", "value":'),
            {"name": "test", "value": None},
        )
        self.assertEqual(self._roundtrip('{"name": "test'), {"name": "test"})


class TestSchemaAwareRepair(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = DeepSeekProvider(api_key="test")
        self.mock_client = MagicMock()
        self.mock_client.chat.completions.create = AsyncMock()

    @staticmethod
    def _completion(
        content: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        response_id: str = "repair-id",
        cost: float | None = None,
    ) -> MagicMock:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.prompt_tokens_details = None
        usage.completion_tokens_details = None
        if cost is not None:
            usage.cost = cost
        else:
            del usage.cost
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=content))]
        response.usage = usage
        response.id = response_id
        return response

    async def _parse(
        self,
        raw: str,
        schema=ExtractionOutput,
        *,
        metadata=None,
        request_params=None,
    ):
        return await self.provider._parse_with_repair(
            raw,
            schema,
            self.mock_client,
            "deepseek-v4-flash",
            request_params or {"messages": []},
            repair_metadata=metadata,
        )

    async def test_valid_json_needs_no_repair(self):
        raw = '{"name": "alice", "value": 10}'
        result = await self._parse(raw, SimpleOutput)
        self.assertEqual(result, SimpleOutput(name="alice", value=10))
        self.mock_client.chat.completions.create.assert_not_awaited()

    async def test_null_string_and_forbidden_extra_are_repaired_locally(self):
        raw = json.dumps(
            {
                "items": [
                    {
                        "item_id": 1,
                        "optional_text": None,
                        "required_text": "already valid",
                        "review_span_ids": [7],
                    },
                    {
                        "item_id": 2,
                        "optional_text": "unchanged",
                        "required_text": "also valid",
                    },
                ],
                "summary": "keep me",
            }
        )
        metadata = {}

        result = await self._parse(raw, metadata=metadata)

        self.assertEqual(result.items[0].optional_text, "")
        self.assertEqual(result.items[0].required_text, "already valid")
        self.assertEqual(
            result.items[1].model_dump(),
            {
                "item_id": 2,
                "optional_text": "unchanged",
                "required_text": "also valid",
            },
        )
        self.assertEqual(result.summary, "keep me")
        self.assertEqual(metadata["strategy"], "deterministic")
        self.assertEqual(metadata["deterministic_fields"], 2)
        self.assertEqual(metadata["attempts"], 0)
        self.mock_client.chat.completions.create.assert_not_awaited()

    async def test_truncated_nested_json_is_recovered_then_schema_repaired(self):
        raw = (
            '{"items": [{"item_id": 1, "optional_text": null, '
            '"required_text": "valid"}], "summary": "complete"'
        )
        metadata = {}

        result = await self._parse(raw, metadata=metadata)

        self.assertIsInstance(result, ExtractionOutput)
        self.assertEqual(result.items[0].optional_text, "")
        self.assertTrue(metadata["syntax_repaired"])
        self.assertEqual(metadata["strategy"], "deterministic")
        self.mock_client.chat.completions.create.assert_not_awaited()

    async def test_field_patch_sends_only_invalid_records_and_preserves_valid_data(
        self,
    ):
        original = {
            "items": [
                {"item_id": 1, "optional_text": None, "required_text": ""},
                {
                    "item_id": 2,
                    "optional_text": "valid optional",
                    "required_text": "valid required",
                },
                {"item_id": 3, "optional_text": "valid", "required_text": ""},
            ],
            "summary": "immutable summary",
        }
        self.mock_client.chat.completions.create.return_value = self._completion(
            json.dumps(
                {
                    "repairs": [
                        {"path": "/items/0/required_text", "value": "first fixed"},
                        {"path": "/items/2/required_text", "value": "third fixed"},
                    ]
                }
            ),
            prompt_tokens=31,
            completion_tokens=9,
        )
        metadata = {}
        secret_source = "SOURCE TRANSCRIPT THAT MUST NOT BE REPLAYED"

        result = await self._parse(
            json.dumps(original),
            metadata=metadata,
            request_params={"messages": [{"role": "user", "content": secret_source}]},
        )

        self.assertIsInstance(result, ExtractionOutput)
        self.assertEqual(result.items[0].optional_text, "")
        self.assertEqual(result.items[0].required_text, "first fixed")
        self.assertEqual(result.items[1].model_dump(), original["items"][1])
        self.assertEqual(result.items[2].required_text, "third fixed")
        self.assertEqual(result.summary, original["summary"])

        call = self.mock_client.chat.completions.create.await_args.kwargs
        self.assertEqual(len(call["messages"]), 2)
        serialized_request = json.dumps(call["messages"])
        self.assertNotIn(secret_source, serialized_request)
        payload = json.loads(call["messages"][1]["content"])
        self.assertEqual(
            {field["path"] for field in payload["invalid_fields"]},
            {"/items/0/required_text", "/items/2/required_text"},
        )
        self.assertNotIn("immutable summary", serialized_request)
        self.assertNotIn('"item_id": 2', serialized_request)
        self.assertLess(call["max_tokens"], 4096)
        self.assertEqual(metadata["strategy"], "field_patch")
        self.assertEqual(metadata["attempts"], 1)
        self.assertEqual(metadata["deterministic_fields"], 1)
        self.assertEqual(metadata["model_patched_fields"], 2)
        self.assertEqual(metadata["input_tokens"], 31)
        self.assertEqual(metadata["output_tokens"], 9)
        self.assertTrue(metadata["success"])

    async def test_patch_for_non_error_path_rejects_entire_repair(self):
        raw = json.dumps(
            {
                "items": [
                    {"item_id": 1, "optional_text": "valid", "required_text": ""},
                    {"item_id": 2, "optional_text": "valid", "required_text": "valid"},
                ],
                "summary": "unchanged",
            }
        )
        self.mock_client.chat.completions.create.return_value = self._completion(
            '{"repairs": [{"path": "/items/1/required_text", "value": "tampered"}]}',
            prompt_tokens=5,
            completion_tokens=4,
        )
        metadata = {}

        result = await self._parse(raw, metadata=metadata)

        self.assertEqual(result, raw)
        self.assertFalse(metadata["success"])
        self.assertEqual(metadata["model_patched_fields"], 0)
        self.assertEqual(metadata["input_tokens"], 5)
        self.assertEqual(metadata["output_tokens"], 4)
        self.mock_client.chat.completions.create.assert_awaited_once()

    async def test_incomplete_patch_set_fails_closed(self):
        raw = json.dumps(
            {
                "items": [
                    {"item_id": 1, "optional_text": "ok", "required_text": ""},
                    {"item_id": 2, "optional_text": "ok", "required_text": ""},
                ],
                "summary": "unchanged",
            }
        )
        self.mock_client.chat.completions.create.return_value = self._completion(
            '{"repairs": [{"path": "/items/0/required_text", "value": "fixed"}]}'
        )

        result = await self._parse(raw)

        self.assertEqual(result, raw)
        self.mock_client.chat.completions.create.assert_awaited_once()

    async def test_unparseable_json_uses_isolated_full_object_fallback(self):
        raw = "unparseable output " * 900
        self.mock_client.chat.completions.create.return_value = self._completion(
            '{"name": "recovered", "value": 42}',
            prompt_tokens=41,
            completion_tokens=11,
        )
        metadata = {}
        secret_source = "PRIVATE ORIGINAL SOURCE"

        result = await self._parse(
            raw,
            SimpleOutput,
            metadata=metadata,
            request_params={"messages": [{"role": "user", "content": secret_source}]},
        )

        self.assertEqual(result, SimpleOutput(name="recovered", value=42))
        call = self.mock_client.chat.completions.create.await_args.kwargs
        serialized_request = json.dumps(call["messages"])
        self.assertNotIn(secret_source, serialized_request)
        self.assertIn(raw, serialized_request)
        self.assertGreater(call["max_tokens"], 4096)
        self.assertLessEqual(call["max_tokens"], 16384)
        self.assertEqual(metadata["strategy"], "full_object")
        self.assertEqual(metadata["attempts"], 1)
        self.assertEqual(metadata["input_tokens"], 41)
        self.assertEqual(metadata["output_tokens"], 11)
        self.assertTrue(metadata["success"])

    async def test_repair_call_failure_is_bounded_and_fails_closed(self):
        raw = json.dumps(
            {
                "items": [
                    {"item_id": 1, "optional_text": "valid", "required_text": ""}
                ],
                "summary": "unchanged",
            }
        )
        self.mock_client.chat.completions.create.side_effect = RuntimeError(
            "provider down"
        )
        metadata = {}

        result = await self._parse(raw, metadata=metadata)

        self.assertEqual(result, raw)
        self.assertEqual(metadata["attempts"], 1)
        self.assertFalse(metadata["success"])
        self.mock_client.chat.completions.create.assert_awaited_once()

    async def test_no_response_format_returns_raw_without_call(self):
        result = await self._parse("plain text", None)
        self.assertEqual(result, "plain text")
        self.mock_client.chat.completions.create.assert_not_awaited()

    async def test_process_response_aggregates_repair_usage_and_telemetry(self):
        raw = json.dumps(
            {
                "items": [
                    {"item_id": 1, "optional_text": "valid", "required_text": ""}
                ],
                "summary": "unchanged",
            }
        )
        initial = self._completion(
            raw, prompt_tokens=100, completion_tokens=20, response_id="initial-id"
        )
        self.mock_client.chat.completions.create.return_value = self._completion(
            '{"repairs": [{"path": "/items/0/required_text", "value": "fixed"}]}',
            prompt_tokens=7,
            completion_tokens=3,
        )

        processed = await self.provider.process_response(
            client=self.mock_client,
            response=initial,
            request_params={"messages": [{"role": "user", "content": "source"}]},
            tools=None,
            tool_dict={},
            response_format=ExtractionOutput,
            model="deepseek-v4-flash",
        )

        content, _, input_tokens, cached_tokens, output_tokens, _, _, telemetry = (
            processed
        )
        self.assertIsInstance(content, ExtractionOutput)
        self.assertEqual(input_tokens, 107)
        self.assertEqual(cached_tokens, 0)
        self.assertEqual(output_tokens, 23)
        self.assertEqual(telemetry["input_tokens"], 7)
        self.assertEqual(telemetry["output_tokens"], 3)
        self.assertGreater(telemetry["cost_in_cents"], 0)


if __name__ == "__main__":
    unittest.main()
