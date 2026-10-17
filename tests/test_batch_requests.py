"""Offline tests for building Batch request lines. No API call is made."""

import json
import sys

import pytest
from pydantic import BaseModel, Field

from defog.llm.batch import BatchItem, build_requests, requests as batch_requests
from defog.llm.exceptions import ConfigurationError
from defog.llm.providers.anthropic_provider import AnthropicProvider
from defog.llm.providers.openai_provider import OpenAIProvider


class Answer(BaseModel):
    city: str = Field(max_length=40)
    population: int = Field(ge=0)


MESSAGES = [
    {"role": "system", "content": "Answer in JSON."},
    {"role": "user", "content": "Which is the largest city in Japan?"},
]


def item(i="req-001", **kwargs):
    return BatchItem(id=i, messages=MESSAGES, **kwargs)


def test_openai_line_uses_responses_endpoint_and_build_params():
    requests = build_requests(
        [item(response_format=Answer, reasoning_effort="low")], "gpt-5-mini"
    )
    assert requests.provider == "openai"
    (line,) = requests.lines
    assert line["custom_id"] == "req-001"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"

    expected, _ = OpenAIProvider(api_key="x").build_params(
        messages=MESSAGES, model="gpt-5-mini", reasoning_effort="low"
    )
    expected.pop("timeout")
    body = dict(line["body"])
    text = body.pop("text")
    assert body == {k: v for k, v in expected.items() if v is not None}
    assert body["instructions"] == "Answer in JSON."
    assert body["reasoning"]["effort"] == "low"
    assert "timeout" not in body
    assert None not in body.values()
    fmt = text["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["name"] == "Answer"
    assert set(fmt["schema"]["properties"]) == {"city", "population"}


def test_anthropic_line_uses_build_params():
    requests = build_requests(
        [item(response_format=Answer, reasoning_effort="low")], "claude-haiku-5-5"
    )
    assert requests.provider == "anthropic"
    (line,) = requests.lines
    assert set(line) == {"custom_id", "params"}
    expected, _ = AnthropicProvider(api_key="x").build_params(
        messages=MESSAGES,
        model="claude-haiku-5-5",
        reasoning_effort="low",
        response_format=Answer,
    )
    expected.pop("timeout")
    params = line["params"]
    assert params == expected
    assert params["system"] == "Answer in JSON."
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"]["effort"] == "low"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["cache_control"] == {"type": "ephemeral"}
    # build_params sets a max_tokens default; Anthropic requires one.
    assert params["max_tokens"] == 64000
    json.dumps(params)  # the line must be JSON


def test_anthropic_thinking_off_without_effort_and_own_max_tokens():
    (line,) = build_requests(
        [item(max_completion_tokens=16_000)], "claude-haiku-5-5"
    ).lines
    assert line["params"]["thinking"] == {"type": "disabled"}
    assert line["params"]["max_tokens"] == 16_000
    assert "output_config" not in line["params"]


def test_items_may_be_mappings_and_keep_order():
    requests = build_requests(
        [{"id": f"req-{n:03d}", "messages": MESSAGES} for n in range(3)],
        "gpt-5-mini",
    )
    assert requests.ids == ["req-000", "req-001", "req-002"]
    lines = requests.to_jsonl().decode().splitlines()
    assert [json.loads(line)["custom_id"] for line in lines] == requests.ids


def test_response_formats_map_ids_to_schemas():
    requests = build_requests(
        [item("a", response_format=Answer), item("b")], "claude-haiku-5-5"
    )
    assert requests.response_formats == {"a": Answer}


@pytest.mark.parametrize("model", ["gemini-2.5-pro", "glm-4.6", "llama-3"])
def test_other_providers_are_refused(model):
    with pytest.raises(ConfigurationError):
        build_requests([item()], model)


def test_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one item"):
        build_requests([], "gpt-5-mini")


def test_schema_must_be_a_pydantic_model():
    with pytest.raises(TypeError, match="pydantic"):
        build_requests([item(response_format={"type": "object"})], "gpt-5-mini")


@pytest.mark.parametrize("model", ["gpt-5-mini", "claude-haiku-5-5"])
def test_repeated_ids_are_refused(model):
    with pytest.raises(ValueError, match="unique.*'req-001'"):
        build_requests([item(), item()], model)


@pytest.mark.parametrize("bad", ["", None, 7])
def test_ids_must_be_non_empty_strings(bad):
    with pytest.raises(ValueError, match="non-empty strings"):
        build_requests([item(bad)], "gpt-5-mini")


@pytest.mark.parametrize("bad", ["123:1", "req 1", "req.1", "é", "a" * 65])
def test_anthropic_refuses_ids_outside_its_pattern(bad):
    with pytest.raises(ValueError, match="Invalid ids") as excinfo:
        build_requests([item("ok-1"), item(bad)], "claude-haiku-5-5")
    assert repr(bad) in str(excinfo.value)
    assert "'ok-1'" not in str(excinfo.value)


def test_anthropic_accepts_ids_inside_its_pattern():
    ids = ["a", "A-b_9", "doc-2026-000123", "x" * 64]
    requests = build_requests([item(i) for i in ids], "claude-haiku-5-5")
    assert requests.ids == ids


def test_openai_ids_are_passed_unchanged():
    requests = build_requests([item("123:1")], "gpt-5-mini")
    assert requests.ids == ["123:1"]


def test_error_lists_at_most_five_ids():
    bad = [item(f"bad:{n}") for n in range(8)]
    with pytest.raises(ValueError, match="and 3 more"):
        build_requests(bad, "claude-haiku-5-5")


def test_too_many_requests_is_refused_not_split(monkeypatch):
    monkeypatch.setattr(batch_requests, "ANTHROPIC_MAX_REQUESTS", 2)
    with pytest.raises(
        ValueError, match="3 requests is more than the anthropic limit of 2"
    ):
        build_requests([item(f"r{n}") for n in range(3)], "claude-haiku-5-5")
    assert len(build_requests([item("r1"), item("r2")], "claude-haiku-5-5").lines) == 2

    monkeypatch.setattr(batch_requests, "OPENAI_MAX_REQUESTS", 1)
    with pytest.raises(ValueError, match="openai limit of 1 requests"):
        build_requests([item("r1"), item("r2")], "gpt-5-mini")


@pytest.mark.parametrize(
    "model,limit",
    [("gpt-5-mini", "OPENAI_MAX_BYTES"), ("claude-haiku-5-5", "ANTHROPIC_MAX_BYTES")],
)
def test_too_large_batch_is_refused(monkeypatch, model, limit):
    size = len(build_requests([item()], model).to_jsonl())
    monkeypatch.setattr(batch_requests, limit, size // 2)
    with pytest.raises(ValueError, match="bytes, more than the"):
        build_requests([item()], model)


def test_default_limits_match_the_published_values():
    assert batch_requests.OPENAI_MAX_REQUESTS == 50_000
    assert batch_requests.OPENAI_MAX_BYTES == 200_000_000
    assert batch_requests.ANTHROPIC_MAX_REQUESTS == 100_000
    assert batch_requests.ANTHROPIC_MAX_BYTES == 256_000_000


def test_openai_schema_needs_a_compatible_openai_version(monkeypatch):
    import openai

    # A None entry in sys.modules makes the import raise ImportError, as if
    # an openai release had moved the private helper.
    monkeypatch.setitem(sys.modules, "openai.lib._parsing._responses", None)
    with pytest.raises(ConfigurationError) as error:
        build_requests([item(response_format=Answer)], "gpt-5-mini")
    assert openai.__version__ in str(error.value)
    assert "compatible openai version" in str(error.value)
    assert isinstance(error.value.__cause__, ImportError)
