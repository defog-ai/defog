"""OpenAI Batch tests with a fake client. No API call is made."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from defog.llm.batch import (
    BatchError,
    BatchItem,
    BatchJob,
    build_requests,
    collect,
    parse,
    poll,
    submit,
)
from defog.llm.cost import CostCalculator

MODEL = "gpt-5-mini"


class Answer(BaseModel):
    city: str


def run(coro):
    return asyncio.run(coro)


def message_body(text, usage=None, status="completed"):
    return {
        "id": "resp_1",
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
        "usage": usage
        or {
            "input_tokens": 1_000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 50},
        },
    }


def ok_line(custom_id, text='{"city": "Tokyo"}', **kwargs):
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": message_body(text, **kwargs)},
        "error": None,
    }


def error_line(custom_id, code, message="request failed"):
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": None,
        "error": {"code": code, "message": message},
    }


class FakeFiles:
    def __init__(self, contents):
        self.contents = contents
        self.created = []

    async def create(self, file, purpose):
        self.created.append({"file": file, "purpose": purpose})
        return SimpleNamespace(id="file-in")

    async def content(self, file_id):
        lines = self.contents[file_id]
        return SimpleNamespace(text="\n".join(json.dumps(line) for line in lines))


class FakeBatches:
    def __init__(self, batch):
        self.batch = batch
        self.created = []
        self.retrieved = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="batch_1", status="validating")

    async def retrieve(self, batch_id):
        self.retrieved.append(batch_id)
        return self.batch


class FakeOpenAI:
    def __init__(self, batch=None, contents=None):
        self.files = FakeFiles(contents or {})
        self.batches = FakeBatches(batch)
        self.closed = False

    async def close(self):
        self.closed = True


def batch_object(status="completed", output=None, error=None, errors=None):
    return SimpleNamespace(
        id="batch_1",
        status=status,
        output_file_id=output,
        error_file_id=error,
        request_counts=SimpleNamespace(total=3, completed=2, failed=1),
        errors=SimpleNamespace(data=errors or []),
    )


def job(ids=("q1", "q2", "q3")):
    return BatchJob(provider="openai", model=MODEL, batch_id="batch_1", ids=list(ids))


def test_submit_uploads_jsonl_and_creates_responses_batch():
    requests = build_requests(
        [
            BatchItem(id="q1", messages=[{"role": "user", "content": "Hi"}]),
            BatchItem(id="q2", messages=[{"role": "user", "content": "Hello"}]),
        ],
        MODEL,
    )
    client = FakeOpenAI()
    result = run(submit(requests, client=client, metadata={"purpose": "test"}))

    (upload,) = client.files.created
    assert upload["purpose"] == "batch"
    name, data, content_type = upload["file"]
    assert name.endswith(".jsonl") and content_type == "application/jsonl"
    assert data == requests.to_jsonl()
    assert [json.loads(line)["custom_id"] for line in data.splitlines()] == [
        "q1",
        "q2",
    ]
    assert client.batches.created == [
        {
            "input_file_id": "file-in",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {"purpose": "test"},
        }
    ]
    assert result == BatchJob(
        provider="openai",
        model=MODEL,
        batch_id="batch_1",
        ids=["q1", "q2"],
        input_file_id="file-in",
    )
    assert BatchJob.from_dict(result.to_dict()) == result
    assert client.closed is False  # the caller's client stays open


def test_poll_reports_status_and_counts():
    client = FakeOpenAI(batch=batch_object(status="in_progress"))
    status = run(poll(job(), client=client))
    assert status.status == "in_progress"
    assert status.done is False
    assert status.counts == {"total": 3, "completed": 2, "failed": 1}

    client.batches.batch = batch_object(status="expired")
    assert run(poll(job(), client=client)).done is True


def test_collect_returns_every_id_in_order_with_its_status():
    output = [
        ok_line("q3", text='{"city": "Osaka"}'),
        ok_line("q1"),
    ]
    errors = [
        error_line(
            "q2",
            "batch_expired",
            "This request could not be executed before the completion window expired.",
        )
    ]
    client = FakeOpenAI(
        batch=batch_object(output="file-out", error="file-err"),
        contents={"file-out": output, "file-err": errors},
    )
    results = run(collect(job(), client=client, response_format=Answer))

    assert [r.id for r in results] == ["q1", "q2", "q3"]
    assert [r.status for r in results] == ["succeeded", "expired", "succeeded"]
    assert results[0].parsed == Answer(city="Tokyo")
    assert results[2].parsed == Answer(city="Osaka")
    assert results[0].usage.reasoning_tokens == 50
    assert "completion window" in results[1].error
    assert results[1].cost_in_cents is None
    assert results[1].parsed is None


def test_collect_prices_at_batch_rates_and_subtracts_cached_tokens():
    usage = {
        "input_tokens": 10_000,
        "input_tokens_details": {"cached_tokens": 4_000},
        "output_tokens": 500,
        "output_tokens_details": {"reasoning_tokens": 100},
    }
    client = FakeOpenAI(
        batch=batch_object(output="file-out"),
        contents={"file-out": [ok_line("q1", usage=usage)]},
    )
    (result,) = run(collect(job(["q1"]), client=client))
    assert result.usage.input_tokens == 6_000
    assert result.usage.cached_input_tokens == 4_000
    expected = CostCalculator.calculate_cost(MODEL, 6_000, 500, 4_000, batch=True)
    assert result.cost_in_cents == pytest.approx(expected)
    standard = CostCalculator.calculate_cost(MODEL, 6_000, 500, 4_000)
    assert result.cost_in_cents < standard


def test_missing_and_unexpected_lines_are_reported():
    client = FakeOpenAI(
        batch=batch_object(output="file-out"),
        contents={"file-out": [ok_line("q1"), ok_line("q1"), ok_line("zz")]},
    )
    results = run(collect(job(["q1", "q2"]), client=client))
    assert [(r.id, r.status) for r in results] == [
        ("q1", "succeeded"),
        ("q2", "missing"),
        ("q1", "succeeded"),
        ("zz", "succeeded"),
    ]
    assert "without a result" in results[1].error


def test_error_codes_map_to_statuses():
    lines = [
        error_line("q1", "batch_cancelled"),
        error_line("q2", "server_error", "The server had an error."),
        {
            "custom_id": "q3",
            "response": {
                "status_code": 400,
                "body": {"error": {"code": "invalid_prompt", "message": "Bad input."}},
            },
            "error": None,
        },
    ]
    client = FakeOpenAI(
        batch=batch_object(status="cancelled", error="file-err"),
        contents={"file-err": lines},
    )
    results = run(collect(job(), client=client))
    assert [r.status for r in results] == ["canceled", "errored", "errored"]
    assert results[1].error == "server_error: The server had an error."
    assert results[2].error == "invalid_prompt: Bad input."


def test_incomplete_refused_and_invalid_answers():
    incomplete = ok_line("q1", text='{"city": "To')
    incomplete["response"]["body"]["status"] = "incomplete"
    incomplete["response"]["body"]["incomplete_details"] = {
        "reason": "max_output_tokens"
    }
    refused = ok_line("q2")
    refused["response"]["body"]["output"] = [
        {"type": "message", "content": [{"type": "refusal", "refusal": "No."}]}
    ]
    client = FakeOpenAI(
        batch=batch_object(output="file-out"),
        contents={"file-out": [incomplete, refused, ok_line("q3", text="not json")]},
    )
    results = run(collect(job(), client=client, response_format={"q3": Answer}))
    assert [r.status for r in results] == ["incomplete", "refused", "invalid"]
    assert results[0].stop_reason == "max_output_tokens"
    assert results[0].cost_in_cents is not None  # cut-off answers are billed
    assert results[1].error == "No."
    assert "does not match Answer" in results[2].error


def test_collect_refuses_a_batch_that_has_not_finished():
    client = FakeOpenAI(batch=batch_object(status="in_progress"))
    with pytest.raises(BatchError, match="has not finished"):
        run(collect(job(), client=client))


def test_collect_raises_when_the_whole_batch_failed():
    failed = batch_object(
        status="failed",
        errors=[SimpleNamespace(code="invalid_json_line", message="Bad line.", line=2)],
    )
    client = FakeOpenAI(batch=failed)
    with pytest.raises(BatchError, match=r"invalid_json_line: Bad line\. \(line 2\)"):
        run(collect(job(), client=client))


def test_parse_reads_one_result():
    client = FakeOpenAI(
        batch=batch_object(output="file-out"),
        contents={"file-out": [ok_line("q1")]},
    )
    (result,) = run(collect(job(["q1"]), client=client))
    assert parse(result, Answer) == Answer(city="Tokyo")
    result.status = "expired"
    with pytest.raises(ValueError, match="no answer to parse"):
        parse(result, Answer)


def test_without_a_client_one_is_created_from_the_api_key_and_closed(monkeypatch):
    import openai

    from defog import config as defog_config

    created = []

    class OwnedClient(FakeOpenAI):
        def __init__(self, api_key):
            super().__init__(batch=batch_object(status="validating"))
            self.api_key = api_key
            self.exited = False
            created.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            self.exited = True

    monkeypatch.setattr(openai, "AsyncOpenAI", OwnedClient)
    monkeypatch.setattr(
        defog_config,
        "get",
        lambda name, default=None: "key-from-config"
        if name == "OPENAI_API_KEY"
        else default,
    )
    status = run(poll(job()))
    assert status.done is False
    (client,) = created
    assert client.api_key == "key-from-config"
    assert client.exited is True
