"""Regression tests for deterministic OpenAI SDK client cleanup."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from defog.llm.exceptions import ProviderError
from defog.llm.pdf_processor import BasePDFProcessor, OpenAIPDFProcessor
from defog.llm.providers.deepseek_provider import DeepSeekProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_async_openai_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "AsyncOpenAI"
    return isinstance(node.func, ast.Attribute) and node.func.attr == "AsyncOpenAI"


def test_all_async_openai_clients_are_context_managed():
    """Prevent new SDK clients from relying on garbage collection for cleanup."""
    unmanaged_calls = []

    for path in (PROJECT_ROOT / "defog").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        managed_calls = {
            id(item.context_expr)
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncWith)
            for item in node.items
            if _is_async_openai_call(item.context_expr)
        }

        for node in ast.walk(tree):
            if _is_async_openai_call(node) and id(node) not in managed_calls:
                unmanaged_calls.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"
                )

    assert not unmanaged_calls, (
        "AsyncOpenAI clients must be created with `async with`: "
        + ", ".join(unmanaged_calls)
    )


class _FakeAsyncOpenAI:
    instances = []
    response = object()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.entered = False
        self.exited = False
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=self.response),
            )
        )
        self.instances.append(self)

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited = True


def _make_deepseek_provider(monkeypatch, process_response):
    import openai

    _FakeAsyncOpenAI.instances.clear()
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    provider = DeepSeekProvider(api_key="sk-test")
    provider.process_response = process_response
    provider.persist_conversation_history = AsyncMock()
    return provider


@pytest.mark.asyncio
async def test_deepseek_closes_client_after_success(monkeypatch):
    provider = _make_deepseek_provider(
        monkeypatch,
        AsyncMock(return_value=("ok", [], 1, 0, 1, None, "response-id", None)),
    )

    response = await provider.execute_chat(
        messages=[{"role": "user", "content": "hello"}],
        model="deepseek-v4-pro",
    )

    assert response.content == "ok"
    assert len(_FakeAsyncOpenAI.instances) == 1
    assert _FakeAsyncOpenAI.instances[0].entered
    assert _FakeAsyncOpenAI.instances[0].exited


@pytest.mark.asyncio
async def test_deepseek_closes_client_after_failure(monkeypatch):
    provider = _make_deepseek_provider(
        monkeypatch,
        AsyncMock(side_effect=RuntimeError("processing failed")),
    )

    with pytest.raises(ProviderError, match="processing failed"):
        await provider.execute_chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-v4-pro",
        )

    assert len(_FakeAsyncOpenAI.instances) == 1
    assert _FakeAsyncOpenAI.instances[0].entered
    assert _FakeAsyncOpenAI.instances[0].exited


@pytest.mark.asyncio
async def test_openai_pdf_processor_closes_client_after_analysis(monkeypatch):
    import openai

    expected = object()
    analyze_pdf = AsyncMock(return_value=expected)
    monkeypatch.setattr(BasePDFProcessor, "analyze_pdf", analyze_pdf)
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    _FakeAsyncOpenAI.instances.clear()

    processor = OpenAIPDFProcessor(api_key="sk-test")
    result = await processor.analyze_pdf("https://example.com/file.pdf", "summarize")

    assert result is expected
    analyze_pdf.assert_awaited_once_with(
        "https://example.com/file.pdf", "summarize", None
    )
    assert processor.client is None
    assert len(_FakeAsyncOpenAI.instances) == 1
    assert _FakeAsyncOpenAI.instances[0].entered
    assert _FakeAsyncOpenAI.instances[0].exited
