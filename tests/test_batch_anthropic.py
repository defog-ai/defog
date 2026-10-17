"""Anthropic Message Batches tests with a fake client. No API call is made."""

import asyncio
from types import SimpleNamespace

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse
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
    wait,
)
from defog.llm.cost import CostCalculator

MODEL = "claude-haiku-5-5"


class Answer(BaseModel):
    city: str


def run(coro):
    return asyncio.run(coro)


def usage(**overrides):
    values = {
        "input_tokens": 1_000,
        "output_tokens": 200,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    values.update(overrides)
    return values


def succeeded(custom_id, text='{"city": "Tokyo"}', stop_reason="end_turn", **u):
    return MessageBatchIndividualResponse.model_validate(
        {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "id": f"msg_{custom_id}",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [
                        {"type": "thinking", "thinking": "...", "signature": "s"},
                        {"type": "text", "text": text},
                    ],
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": usage(**u),
                },
            },
        }
    )


def other(custom_id, kind, error=None):
    result = {"type": kind}
    if error is not None:
        result["error"] = error
    return MessageBatchIndividualResponse.model_validate(
        {"custom_id": custom_id, "result": result}
    )


class AsyncLines:
    """What results() returns: an object read with ``async for``."""

    def __init__(self, lines):
        self.lines = list(lines)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for line in self.lines:
            yield line


class FakeBatches:
    def __init__(self, statuses=("ended",), lines=()):
        self.statuses = list(statuses)
        self.lines = lines
        self.created = []
        self.retrieve_calls = 0

    async def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_1", processing_status="in_progress")

    async def retrieve(self, batch_id):
        status = self.statuses[min(self.retrieve_calls, len(self.statuses) - 1)]
        self.retrieve_calls += 1
        return SimpleNamespace(
            id=batch_id,
            processing_status=status,
            request_counts=SimpleNamespace(
                processing=0, succeeded=2, errored=1, canceled=0, expired=1
            ),
        )

    async def results(self, batch_id):
        return AsyncLines(self.lines)


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = SimpleNamespace(batches=FakeBatches(**kwargs))


def job(ids=("q1", "q2", "q3", "q4")):
    return BatchJob(
        provider="anthropic", model=MODEL, batch_id="msgbatch_1", ids=list(ids)
    )


def test_submit_sends_the_built_lines():
    requests = build_requests(
        [BatchItem(id="q1", messages=[{"role": "user", "content": "Hi"}])], MODEL
    )
    client = FakeAnthropic()
    result = run(submit(requests, client=client))
    assert client.messages.batches.created == [requests.lines]
    assert result == BatchJob(
        provider="anthropic", model=MODEL, batch_id="msgbatch_1", ids=["q1"]
    )


def test_submit_refuses_metadata():
    requests = build_requests(
        [BatchItem(id="q1", messages=[{"role": "user", "content": "Hi"}])], MODEL
    )
    with pytest.raises(ValueError, match="metadata"):
        run(submit(requests, client=FakeAnthropic(), metadata={"a": "b"}))


def test_poll_reports_status_and_counts():
    client = FakeAnthropic(statuses=["in_progress"])
    status = run(poll(job(), client=client))
    assert (status.status, status.done) == ("in_progress", False)
    assert status.counts == {
        "processing": 0,
        "succeeded": 2,
        "errored": 1,
        "canceled": 0,
        "expired": 1,
    }


def test_collect_returns_every_id_with_its_status():
    lines = [
        succeeded("q4", text='{"city": "Osaka"}'),
        other(
            "q2",
            "errored",
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Bad input."},
            },
        ),
        other("q3", "expired"),
        succeeded("q1"),
    ]
    client = FakeAnthropic(lines=lines)
    results = run(collect(job(), client=client, response_format=Answer))

    assert [r.id for r in results] == ["q1", "q2", "q3", "q4"]
    assert [r.status for r in results] == [
        "succeeded",
        "errored",
        "expired",
        "succeeded",
    ]
    assert results[0].text == '{"city": "Tokyo"}'  # thinking blocks are skipped
    assert results[0].parsed == Answer(city="Tokyo")
    assert results[3].parsed == Answer(city="Osaka")
    assert results[1].error == "invalid_request_error: Bad input."
    assert results[2].cost_in_cents is None


def test_canceled_and_missing():
    client = FakeAnthropic(lines=[other("q1", "canceled")])
    results = run(collect(job(["q1", "q2"]), client=client))
    assert [r.status for r in results] == ["canceled", "missing"]


def test_cut_off_and_refused_answers():
    lines = [
        succeeded("q1", text='{"city": "To', stop_reason="max_tokens"),
        succeeded("q2", stop_reason="model_context_window_exceeded"),
        succeeded("q3", text="", stop_reason="refusal"),
        succeeded("q4", text="not json"),
    ]
    client = FakeAnthropic(lines=lines)
    results = run(collect(job(), client=client, response_format=Answer))
    assert [r.status for r in results] == [
        "incomplete",
        "incomplete",
        "refused",
        "invalid",
    ]
    assert results[0].stop_reason == "max_tokens"
    assert results[0].parsed is None


def test_collect_prices_at_batch_rates_with_cache_tokens():
    line = succeeded(
        "q1",
        input_tokens=2_000,
        output_tokens=500,
        cache_read_input_tokens=10_000,
        cache_creation_input_tokens=4_000,
        cache_creation={
            "ephemeral_5m_input_tokens": 1_000,
            "ephemeral_1h_input_tokens": 3_000,
        },
    )
    (result,) = run(collect(job(["q1"]), client=FakeAnthropic(lines=[line])))
    assert result.usage.input_tokens == 2_000
    assert result.usage.cached_input_tokens == 10_000
    assert result.usage.cache_creation_input_tokens == 4_000
    assert result.usage.cache_creation_1h_input_tokens == 3_000
    standard = CostCalculator.calculate_cost(
        MODEL, 2_000, 500, 10_000, 4_000, cache_creation_1h_input_tokens=3_000
    )
    assert result.cost_in_cents == pytest.approx(standard / 2)


def test_plain_dict_lines_are_read_too():
    line = succeeded("q1").model_dump(mode="json")
    (result,) = run(collect(job(["q1"]), client=FakeAnthropic(lines=[line])))
    assert result.status == "succeeded"
    assert parse(result, Answer) == Answer(city="Tokyo")


def test_collect_refuses_a_batch_that_has_not_ended():
    with pytest.raises(BatchError, match="has not ended"):
        run(collect(job(), client=FakeAnthropic(statuses=["in_progress"])))


def test_wait_polls_until_the_batch_ends():
    client = FakeAnthropic(statuses=["in_progress", "canceling", "ended"])
    status = run(wait(job(), client=client, poll_interval=0))
    assert status.done is True
    assert client.messages.batches.retrieve_calls == 3


def test_wait_raises_after_the_timeout():
    client = FakeAnthropic(statuses=["in_progress"])
    with pytest.raises(TimeoutError, match="still 'in_progress'"):
        run(wait(job(), client=client, poll_interval=0, timeout=0))


def test_without_a_client_one_is_created_from_the_api_key_and_closed(monkeypatch):
    import anthropic

    from defog import config as defog_config

    created = []

    class OwnedClient(FakeAnthropic):
        def __init__(self, api_key):
            super().__init__(statuses=["ended"])
            self.api_key = api_key
            self.exited = False
            created.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            self.exited = True

    monkeypatch.setattr(anthropic, "AsyncAnthropic", OwnedClient)
    monkeypatch.setattr(
        defog_config,
        "get",
        lambda name, default=None: "key-from-config"
        if name == "ANTHROPIC_API_KEY"
        else default,
    )
    status = run(poll(job()))
    assert status.done is True
    (client,) = created
    assert client.api_key == "key-from-config"
    assert client.exited is True
