"""Mocked unit tests for Anthropic server-side tools and programmatic
tool calling support in chat_async / AnthropicProvider.

These tests do NOT hit the live Anthropic API. They patch
``client.messages.create`` with canned response objects and inspect the
parameters our provider builds and the messages it sends back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from defog.llm import chat_async
from defog.llm.providers.anthropic_provider import AnthropicProvider
from defog.llm.providers.anthropic_server_tools import (
    DEFAULT_VERSIONS,
    SERVER_TOOL_BETA_HEADERS,
    build_advisor_tool,
    build_code_execution_tool,
    build_web_fetch_tool,
    build_web_search_tool,
    normalize_server_tools,
)


# ---------------------------------------------------------------------------
# Helpers for fake Anthropic response objects
# ---------------------------------------------------------------------------


def make_block(type_: str, **kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **kwargs)


def make_usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    server_tool_use: Optional[Dict[str, int]] = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    if server_tool_use is not None:
        usage.server_tool_use = SimpleNamespace(**server_tool_use)
    return usage


def make_response(
    content: List[Any],
    stop_reason: str = "end_turn",
    usage: Optional[SimpleNamespace] = None,
    container: Optional[SimpleNamespace] = None,
    model: str = "claude-opus-4-6",
    response_id: str = "msg_01abc",
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=usage or make_usage(),
        container=container,
        model=model,
        id=response_id,
    )


# ---------------------------------------------------------------------------
# normalize_server_tools tests
# ---------------------------------------------------------------------------


class TestNormalizeServerTools:
    def test_normalize_server_tools_list_form(self):
        specs, headers = normalize_server_tools(
            ["web_search", "code_execution"],
            model="claude-opus-4-6",
            programmatic_tool_calling=False,
        )
        names = sorted(s["name"] for s in specs)
        assert names == ["code_execution", "web_search"]
        assert any(s["type"] == DEFAULT_VERSIONS["web_search"] for s in specs)
        assert headers == set()

    def test_normalize_server_tools_dict_form(self):
        specs, headers = normalize_server_tools(
            {
                "web_search": {"max_uses": 5, "allowed_domains": ["arxiv.org"]},
                "advisor": {"model": "claude-opus-4-6", "max_uses": 2},
            },
            model="claude-opus-4-6",
            programmatic_tool_calling=False,
        )
        web = next(s for s in specs if s["name"] == "web_search")
        assert web["max_uses"] == 5
        assert web["allowed_domains"] == ["arxiv.org"]
        adv = next(s for s in specs if s["name"] == "advisor")
        assert adv["model"] == "claude-opus-4-6"
        assert adv["max_uses"] == 2
        # Dynamic-filtering web_search auto-adds code_execution
        assert any(s["name"] == "code_execution" for s in specs)
        # Advisor uses a beta header
        assert SERVER_TOOL_BETA_HEADERS["advisor"] in headers

    def test_normalize_server_tools_raw_dict_passthrough(self):
        # Use the older stable web_search version to avoid the dynamic-filtering
        # auto-add and the model check.
        specs, _ = normalize_server_tools(
            [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            model="claude-haiku-4-5",
            programmatic_tool_calling=False,
        )
        assert len(specs) == 1
        assert specs[0]["type"] == "web_search_20250305"
        assert specs[0]["max_uses"] == 5

    def test_normalize_server_tools_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown Anthropic server tool"):
            normalize_server_tools(
                ["nonsense_tool"],
                model="claude-opus-4-6",
                programmatic_tool_calling=False,
            )

    def test_dynamic_filtering_auto_adds_code_execution(self):
        specs, _ = normalize_server_tools(
            ["web_search"],
            model="claude-opus-4-6",
            programmatic_tool_calling=False,
        )
        names = {s["name"] for s in specs}
        assert "code_execution" in names

    def test_dynamic_filtering_rejects_unsupported_model(self):
        with pytest.raises(ValueError, match="does not support the default"):
            normalize_server_tools(
                ["web_search"],
                model="claude-haiku-4-5",
                programmatic_tool_calling=False,
            )

    def test_programmatic_requires_code_execution(self):
        # Use the older, stable web_search version so the dynamic-filtering
        # auto-add path does not silently provide code_execution for us.
        with pytest.raises(ValueError, match="programmatic_tool_calling=True requires"):
            normalize_server_tools(
                [{"type": "web_search_20250305", "name": "web_search"}],
                model="claude-opus-4-6",
                programmatic_tool_calling=True,
            )

    def test_programmatic_with_web_search_auto_adds_code_execution(self):
        # Bare ["web_search"] + programmatic should be accepted because
        # dynamic filtering auto-adds code_execution at the right version.
        specs, _ = normalize_server_tools(
            ["web_search"],
            model="claude-opus-4-6",
            programmatic_tool_calling=True,
        )
        names = {s["name"] for s in specs}
        assert "code_execution" in names

    def test_programmatic_rejects_old_code_exec_version(self):
        with pytest.raises(ValueError, match="requires code_execution version"):
            normalize_server_tools(
                [{"type": "code_execution_20250522", "name": "code_execution"}],
                model="claude-opus-4-6",
                programmatic_tool_calling=True,
            )

    def test_advisor_requires_model(self):
        with pytest.raises(ValueError, match="advisor server tool requires"):
            normalize_server_tools(
                {"advisor": {}},
                model="claude-opus-4-6",
                programmatic_tool_calling=False,
            )

    def test_builders(self):
        ws = build_web_search_tool(max_uses=3, allowed_domains=["x.com"])
        assert ws["type"] == DEFAULT_VERSIONS["web_search"]
        assert ws["max_uses"] == 3
        wf = build_web_fetch_tool(citations=True)
        assert wf["citations"] == {"enabled": True}
        ce = build_code_execution_tool()
        assert ce["type"] == DEFAULT_VERSIONS["code_execution"]
        adv = build_advisor_tool("claude-opus-4-6", max_uses=1)
        assert adv["model"] == "claude-opus-4-6"
        assert adv["max_uses"] == 1


# ---------------------------------------------------------------------------
# build_params tests
# ---------------------------------------------------------------------------


class _Args(BaseModel):
    x: int


async def my_tool(input: _Args) -> str:  # noqa: D401
    """A simple test tool."""
    return f"got {input.x}"


class TestBuildParams:
    def test_build_params_appends_server_tools(self):
        provider = AnthropicProvider(api_key="sk-test")
        web = build_web_search_tool()
        ce = build_code_execution_tool()
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            tools=[my_tool],
            server_tools=[web, ce],
        )
        names = [t.get("name") for t in params["tools"]]
        # User tool first, then both server tools
        assert names == ["my_tool", "web_search", "code_execution"]

    def test_build_params_server_tools_no_disable_parallel(self):
        """Server tools must not set disable_parallel_tool_use (API rejects it)."""
        provider = AnthropicProvider(api_key="sk-test")
        web = build_web_search_tool()
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            tools=[my_tool],
            server_tools=[web],
            parallel_tool_calls=False,
        )
        tc = params.get("tool_choice", {})
        assert tc.get("disable_parallel_tool_use") is not True

    def test_build_params_server_tools_only_no_user_tools(self):
        provider = AnthropicProvider(api_key="sk-test")
        web = build_web_search_tool()
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            server_tools=[web],
        )
        assert params["tools"] == [web]
        assert params["tool_choice"] == {"type": "auto"}

    def test_build_params_advisor_adds_beta_header_via_normalize(self):
        # Beta headers are produced by normalize_server_tools, not build_params.
        # Verify that path produces the right header.
        _, headers = normalize_server_tools(
            {"advisor": {"model": "claude-opus-4-6"}},
            model="claude-opus-4-6",
            programmatic_tool_calling=False,
        )
        assert "advisor-tool-2026-03-01" in headers

    def test_build_params_programmatic_injects_allowed_callers(self):
        provider = AnthropicProvider(api_key="sk-test")
        ce = build_code_execution_tool()
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            tools=[my_tool],
            server_tools=[ce],
            programmatic_tool_calling=True,
            parallel_tool_calls=True,
        )
        user_tool_spec = next(t for t in params["tools"] if t.get("name") == "my_tool")
        assert user_tool_spec.get("allowed_callers") == ["code_execution_20260120"]

    def test_build_params_programmatic_forces_strict_off(self):
        provider = AnthropicProvider(api_key="sk-test")
        ce = build_code_execution_tool()
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            tools=[my_tool],
            server_tools=[ce],
            programmatic_tool_calling=True,
            parallel_tool_calls=True,
            strict_tools=True,  # should be ignored / forced off
        )
        user_tool_spec = next(t for t in params["tools"] if t.get("name") == "my_tool")
        assert (
            "strict" not in user_tool_spec or user_tool_spec.get("strict") is not True
        )
        # additionalProperties=False is added when strict=True; with strict
        # forced off it should not be present.
        assert user_tool_spec["input_schema"].get("additionalProperties") is not False

    def test_build_params_programmatic_rejects_parallel_disabled(self):
        provider = AnthropicProvider(api_key="sk-test")
        ce = build_code_execution_tool()
        with pytest.raises(ValueError, match="parallel_tool_calls=False"):
            provider.build_params(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-opus-4-6",
                tools=[my_tool],
                server_tools=[ce],
                programmatic_tool_calling=True,
                parallel_tool_calls=False,
            )

    def test_build_params_programmatic_rejects_forced_tool_choice(self):
        provider = AnthropicProvider(api_key="sk-test")
        ce = build_code_execution_tool()
        with pytest.raises(ValueError, match="tool_choice"):
            provider.build_params(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-opus-4-6",
                tools=[my_tool],
                tool_choice="my_tool",
                server_tools=[ce],
                programmatic_tool_calling=True,
                parallel_tool_calls=True,
            )

    def test_build_params_container_id_threaded_through(self):
        provider = AnthropicProvider(api_key="sk-test")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-6",
            server_tools=[build_code_execution_tool()],
            container_id="container_xyz",
        )
        assert params.get("container") == "container_xyz"


# ---------------------------------------------------------------------------
# process_response tests
# ---------------------------------------------------------------------------


class TestProcessResponse:
    @pytest.mark.asyncio
    async def test_collects_server_tool_outputs(self):
        provider = AnthropicProvider(api_key="sk-test")
        web_result = make_block(
            "web_search_tool_result",
            tool_use_id="srvtoolu_01",
            content=[{"title": "Hello", "url": "https://example.com"}],
        )
        text = make_block("text", text="Here is the answer.")
        usage = make_usage(server_tool_use={"web_search_requests": 2})
        response = make_response([text, web_result], usage=usage)

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

        result = await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": [], "model": "claude-opus-4-6"},
            tools=None,
            tool_dict={},
            server_tools=[build_web_search_tool()],
        )
        (
            content,
            tool_outputs,
            _input,
            _output,
            _cached,
            _cache_create,
            _details,
            server_tool_outputs,
            server_tool_usage,
            container_id,
            container_expires_at,
        ) = result
        assert "answer" in content
        assert (
            server_tool_outputs
            and server_tool_outputs[0]["type"] == "web_search_tool_result"
        )
        assert server_tool_outputs[0]["tool_use_id"] == "srvtoolu_01"
        assert server_tool_usage == {"web_search_requests": 2}
        # No local tool calls — tool_outputs should be empty
        assert tool_outputs == []
        assert container_id is None

    @pytest.mark.asyncio
    async def test_handles_pause_turn(self):
        provider = AnthropicProvider(api_key="sk-test")
        # First response: pause_turn with a code execution block
        srv_use = make_block(
            "server_tool_use",
            id="srvtoolu_01",
            name="code_execution",
            input={"code": "print('hello')"},
        )
        first = make_response(
            [srv_use],
            stop_reason="pause_turn",
            container=SimpleNamespace(
                id="container_abc", expires_at="2026-01-01T00:00:00Z"
            ),
        )
        # Second response (after resume): final answer
        text = make_block("text", text="done")
        ce_result = make_block(
            "code_execution_tool_result",
            tool_use_id="srvtoolu_01",
            content={"stdout": "hello\n"},
        )
        second = make_response(
            [text, ce_result],
            stop_reason="end_turn",
            container=SimpleNamespace(
                id="container_abc", expires_at="2026-01-01T00:00:00Z"
            ),
        )
        client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=second))
        )

        params: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "run code"}],
            "model": "claude-opus-4-6",
        }
        result = await provider.process_response(
            client=client,
            response=first,
            request_params=params,
            tools=None,
            tool_dict={},
            server_tools=[build_code_execution_tool()],
        )
        (
            content,
            tool_outputs,
            _input,
            _output,
            _cached,
            _cache_create,
            _details,
            server_tool_outputs,
            server_tool_usage,
            container_id,
            container_expires_at,
        ) = result
        assert content == "done"
        # Verify the loop iterated and used the same container
        assert client.messages.create.call_count == 1
        called_with = client.messages.create.call_args.kwargs
        assert called_with.get("container") == "container_abc"
        # Assistant echo for pause_turn should be present in messages
        assert any(m["role"] == "assistant" for m in params["messages"])
        # Container should be captured in the return tuple
        assert container_id == "container_abc"
        assert container_expires_at == "2026-01-01T00:00:00Z"
        # Code execution result should be in server_tool_outputs
        assert any(
            o["type"] == "code_execution_tool_result" for o in server_tool_outputs
        )

    @pytest.mark.asyncio
    async def test_non_programmatic_tool_path_passes_container_id(self):
        """When code_execution creates a container and the model then calls a
        regular user tool (non-programmatic path), the follow-up API call must
        include container_id in request_params.

        Regression test for https://github.com/defog-ai/defog/issues/270
        """
        provider = AnthropicProvider(api_key="sk-test")

        # First response: code_execution ran (server_tool_use + result),
        # AND the model calls a regular user tool.  Container is set.
        srv_code_use = make_block(
            "server_tool_use",
            id="srvtoolu_01",
            name="code_execution",
            input={"code": "print('hello')"},
        )
        ce_result = make_block(
            "code_execution_tool_result",
            tool_use_id="srvtoolu_01",
            content={"stdout": "hello\n"},
        )
        text_block = make_block("text", text="Let me call the tool")
        # Regular tool_use block (NOT from code_execution — no caller attr)
        regular_tool_use = make_block(
            "tool_use",
            id="toolu_regular_01",
            name="my_tool",
            input={"x": 7},
        )
        first = make_response(
            [srv_code_use, ce_result, text_block, regular_tool_use],
            stop_reason="tool_use",
            container=SimpleNamespace(
                id="container_xyz", expires_at="2026-06-01T00:00:00Z"
            ),
        )

        # Second response: final answer
        final_text = make_block("text", text="The result is got 7")
        second = make_response(
            [final_text],
            stop_reason="end_turn",
            container=SimpleNamespace(
                id="container_xyz", expires_at="2026-06-01T00:00:00Z"
            ),
        )

        mock_create = AsyncMock(return_value=second)
        client = SimpleNamespace(messages=SimpleNamespace(create=mock_create))

        params: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "run code and call tool"}],
            "model": "claude-opus-4-6",
            "tool_choice": {"type": "auto"},
        }
        tool_dict = {"my_tool": my_tool}
        result = await provider.process_response(
            client=client,
            response=first,
            request_params=params,
            tools=[my_tool],
            tool_dict=tool_dict,
            server_tools=[build_code_execution_tool()],
        )

        (
            content,
            tool_outputs,
            _input,
            _output,
            _cached,
            _cache_create,
            _details,
            server_tool_outputs,
            server_tool_usage,
            container_id,
            container_expires_at,
        ) = result

        # The follow-up API call must include the container
        assert mock_create.call_count == 1
        called_with = mock_create.call_args.kwargs
        assert called_with.get("container") == "container_xyz", (
            "Non-programmatic tool path must pass container_id in follow-up API call"
        )

        # Container should be captured in the return tuple
        assert container_id == "container_xyz"
        assert container_expires_at == "2026-06-01T00:00:00Z"

        # The tool was executed
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["name"] == "my_tool"

        # Final content captured
        assert "got 7" in content

    @pytest.mark.asyncio
    async def test_non_programmatic_tool_path_without_container(self):
        """When there is no container (no code_execution), the non-programmatic
        tool path should NOT add a container key to request_params."""
        provider = AnthropicProvider(api_key="sk-test")

        text_block = make_block("text", text="Calling tool")
        regular_tool_use = make_block(
            "tool_use",
            id="toolu_01",
            name="my_tool",
            input={"x": 5},
        )
        first = make_response(
            [text_block, regular_tool_use],
            stop_reason="tool_use",
            # No container
        )

        final_text = make_block("text", text="Done: got 5")
        second = make_response([final_text], stop_reason="end_turn")

        mock_create = AsyncMock(return_value=second)
        client = SimpleNamespace(messages=SimpleNamespace(create=mock_create))

        params: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "call tool"}],
            "model": "claude-opus-4-6",
            "tool_choice": {"type": "auto"},
        }
        tool_dict = {"my_tool": my_tool}
        result = await provider.process_response(
            client=client,
            response=first,
            request_params=params,
            tools=[my_tool],
            tool_dict=tool_dict,
        )

        content = result[0]
        assert "got 5" in content
        # No container should be set
        called_with = mock_create.call_args.kwargs
        assert "container" not in called_with

    @pytest.mark.asyncio
    async def test_container_id_across_pause_then_tool_use(self):
        """Multi-step scenario: code_execution triggers pause_turn (creating a
        container), then the resumed response has a regular tool_use. Both
        follow-up API calls must include the container_id.

        This tests the full sequence that triggers the bug in issue #270:
        pause_turn → resume → regular tool call.
        """
        provider = AnthropicProvider(api_key="sk-test")

        # Response 1: pause_turn with code_execution, container created
        srv_code_use = make_block(
            "server_tool_use",
            id="srvtoolu_01",
            name="code_execution",
            input={"code": "import time; time.sleep(5); print('done')"},
        )
        first = make_response(
            [srv_code_use],
            stop_reason="pause_turn",
            container=SimpleNamespace(
                id="container_multi", expires_at="2026-06-01T00:00:00Z"
            ),
        )

        # Response 2 (after pause_turn resume): code execution result +
        # regular tool_use
        ce_result = make_block(
            "code_execution_tool_result",
            tool_use_id="srvtoolu_01",
            content={"stdout": "done\n"},
        )
        text_block = make_block("text", text="Now calling tool")
        regular_tool = make_block(
            "tool_use",
            id="toolu_02",
            name="my_tool",
            input={"x": 99},
        )
        second = make_response(
            [ce_result, text_block, regular_tool],
            stop_reason="tool_use",
            container=SimpleNamespace(
                id="container_multi", expires_at="2026-06-01T00:00:00Z"
            ),
        )

        # Response 3: final answer
        final_text = make_block("text", text="Result: got 99")
        third = make_response(
            [final_text],
            stop_reason="end_turn",
            container=SimpleNamespace(
                id="container_multi", expires_at="2026-06-01T00:00:00Z"
            ),
        )

        mock_create = AsyncMock(side_effect=[second, third])
        client = SimpleNamespace(messages=SimpleNamespace(create=mock_create))

        params: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "run code then call tool"}],
            "model": "claude-opus-4-6",
            "tool_choice": {"type": "auto"},
        }
        tool_dict = {"my_tool": my_tool}
        result = await provider.process_response(
            client=client,
            response=first,
            request_params=params,
            tools=[my_tool],
            tool_dict=tool_dict,
            server_tools=[build_code_execution_tool()],
        )

        (
            content,
            tool_outputs,
            _input,
            _output,
            _cached,
            _cache_create,
            _details,
            server_tool_outputs,
            server_tool_usage,
            container_id,
            container_expires_at,
        ) = result

        assert mock_create.call_count == 2

        # First call (pause_turn resume) should have container
        first_call = mock_create.call_args_list[0].kwargs
        assert first_call.get("container") == "container_multi"

        # Second call (after regular tool execution) should also have container
        second_call = mock_create.call_args_list[1].kwargs
        assert second_call.get("container") == "container_multi", (
            "Non-programmatic tool path after pause_turn must pass container_id"
        )

        assert container_id == "container_multi"
        assert content == "Result: got 99"
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_post_tool_function_called_for_server_tool_results(self):
        """post_tool_function should be invoked for every server tool result
        block so callers can observe server-tool activity."""
        provider = AnthropicProvider(api_key="sk-test")
        srv_use = make_block(
            "server_tool_use",
            id="srvtoolu_ws01",
            name="web_search",
            input={"query": "test query"},
        )
        ws_result = make_block(
            "web_search_tool_result",
            tool_use_id="srvtoolu_ws01",
            content=[{"title": "Result", "url": "https://example.com"}],
        )
        text = make_block("text", text="Found it.")
        response = make_response([srv_use, ws_result, text])

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

        post_tool_calls = []

        async def mock_post_tool(function_name, input_args, tool_result, tool_id):
            post_tool_calls.append(
                {
                    "function_name": function_name,
                    "input_args": input_args,
                    "tool_result": tool_result,
                    "tool_id": tool_id,
                }
            )

        await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": [], "model": "claude-opus-4-6"},
            tools=None,
            tool_dict={},
            server_tools=[build_web_search_tool()],
            post_tool_function=mock_post_tool,
        )

        assert len(post_tool_calls) == 1
        call = post_tool_calls[0]
        assert call["function_name"] == "web_search"
        assert call["input_args"] == {"query": "test query"}
        assert call["tool_id"] == "srvtoolu_ws01"
        assert isinstance(call["tool_result"], list)

    @pytest.mark.asyncio
    async def test_post_tool_function_called_for_code_execution_results(self):
        """post_tool_function should fire for code_execution results when the
        server_tool_use and result are in the same response."""
        provider = AnthropicProvider(api_key="sk-test")
        srv_use = make_block(
            "server_tool_use",
            id="srvtoolu_ce01",
            name="code_execution",
            input={"code": "print('hi')"},
        )
        ce_result = make_block(
            "code_execution_tool_result",
            tool_use_id="srvtoolu_ce01",
            content={"stdout": "hi\n"},
        )
        text = make_block("text", text="done")
        response = make_response([srv_use, ce_result, text])

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

        post_tool_calls = []

        async def mock_post_tool(function_name, input_args, tool_result, tool_id):
            post_tool_calls.append(
                {
                    "function_name": function_name,
                    "input_args": input_args,
                    "tool_result": tool_result,
                    "tool_id": tool_id,
                }
            )

        await provider.process_response(
            client=client,
            response=response,
            request_params={
                "messages": [{"role": "user", "content": "run code"}],
                "model": "claude-opus-4-6",
            },
            tools=None,
            tool_dict={},
            server_tools=[build_code_execution_tool()],
            post_tool_function=mock_post_tool,
        )

        assert len(post_tool_calls) == 1
        call = post_tool_calls[0]
        assert call["function_name"] == "code_execution"
        assert call["input_args"] == {"code": "print('hi')"}
        assert call["tool_id"] == "srvtoolu_ce01"
        assert call["tool_result"] == {"stdout": "hi\n"}

    @pytest.mark.asyncio
    async def test_post_tool_function_called_across_pause_turn(self):
        """When pause_turn splits server_tool_use and its result across two
        responses, post_tool_function should still be called (with the block
        type as fallback name since the use block is in a prior response)."""
        provider = AnthropicProvider(api_key="sk-test")
        # First response: pause_turn with server_tool_use only
        srv_use = make_block(
            "server_tool_use",
            id="srvtoolu_ce01",
            name="code_execution",
            input={"code": "print('hi')"},
        )
        first = make_response(
            [srv_use],
            stop_reason="pause_turn",
            container=SimpleNamespace(
                id="container_pt", expires_at="2026-01-01T00:00:00Z"
            ),
        )
        # Second response: result block only (use block was in prior response)
        ce_result = make_block(
            "code_execution_tool_result",
            tool_use_id="srvtoolu_ce01",
            content={"stdout": "hi\n"},
        )
        text = make_block("text", text="done")
        second = make_response(
            [ce_result, text],
            stop_reason="end_turn",
            container=SimpleNamespace(
                id="container_pt", expires_at="2026-01-01T00:00:00Z"
            ),
        )

        client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=second))
        )

        post_tool_calls = []

        async def mock_post_tool(function_name, input_args, tool_result, tool_id):
            post_tool_calls.append(
                {
                    "function_name": function_name,
                    "tool_result": tool_result,
                    "tool_id": tool_id,
                }
            )

        await provider.process_response(
            client=client,
            response=first,
            request_params={
                "messages": [{"role": "user", "content": "run code"}],
                "model": "claude-opus-4-6",
            },
            tools=None,
            tool_dict={},
            server_tools=[build_code_execution_tool()],
            post_tool_function=mock_post_tool,
        )

        # post_tool_function should have been called for the result block
        assert len(post_tool_calls) == 1
        call = post_tool_calls[0]
        assert call["tool_id"] == "srvtoolu_ce01"
        assert call["tool_result"] == {"stdout": "hi\n"}

    @pytest.mark.asyncio
    async def test_post_tool_function_sync_works_for_server_tools(self):
        """A synchronous post_tool_function should also be called for server
        tool results."""
        provider = AnthropicProvider(api_key="sk-test")
        srv_use = make_block(
            "server_tool_use",
            id="srvtoolu_s01",
            name="web_search",
            input={"query": "sync test"},
        )
        ws_result = make_block(
            "web_search_tool_result",
            tool_use_id="srvtoolu_s01",
            content=[{"title": "Sync"}],
        )
        text = make_block("text", text="ok")
        response = make_response([srv_use, ws_result, text])

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

        post_tool_calls = []

        def sync_post_tool(function_name, input_args, tool_result, tool_id):
            post_tool_calls.append(function_name)

        await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": [], "model": "claude-opus-4-6"},
            tools=None,
            tool_dict={},
            server_tools=[build_web_search_tool()],
            post_tool_function=sync_post_tool,
        )

        assert post_tool_calls == ["web_search"]

    @pytest.mark.asyncio
    async def test_post_tool_function_fallback_when_no_server_tool_use_block(self):
        """When a server tool result block has no matching server_tool_use block
        (e.g. the use block was in a previous response), the fallback should
        use the result block type as function_name."""
        provider = AnthropicProvider(api_key="sk-test")
        # Only result block, no corresponding server_tool_use in this response
        ws_result = make_block(
            "web_search_tool_result",
            tool_use_id="srvtoolu_orphan",
            content=[{"title": "Orphan"}],
        )
        text = make_block("text", text="here")
        response = make_response([ws_result, text])

        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock()))

        post_tool_calls = []

        async def mock_post_tool(function_name, input_args, tool_result, tool_id):
            post_tool_calls.append(
                {"function_name": function_name, "input_args": input_args}
            )

        await provider.process_response(
            client=client,
            response=response,
            request_params={"messages": [], "model": "claude-opus-4-6"},
            tools=None,
            tool_dict={},
            server_tools=[build_web_search_tool()],
            post_tool_function=mock_post_tool,
        )

        assert len(post_tool_calls) == 1
        # Falls back to the block type name since no server_tool_use was found
        assert post_tool_calls[0]["function_name"] == "web_search_tool_result"
        assert post_tool_calls[0]["input_args"] == {}

    @pytest.mark.asyncio
    async def test_programmatic_tool_call_user_message_only_tool_results(self):
        """When responding to a code-execution-issued tool_use, the next user
        message in params['messages'] must contain ONLY tool_result blocks."""
        provider = AnthropicProvider(api_key="sk-test")
        # First response: assistant emits text, server_tool_use (code exec),
        # and a tool_use block tagged as called from code_execution.
        text_block = make_block("text", text="Let me query the db")
        srv_code_use = make_block(
            "server_tool_use",
            id="srvtoolu_01",
            name="code_execution",
            input={"code": "await query_database(sql='SELECT 1')"},
        )
        prog_tool_use = make_block(
            "tool_use",
            id="toolu_prog_01",
            name="my_tool",
            input={"x": 42},
            caller=SimpleNamespace(type="code_execution_20260120"),
        )
        first = make_response(
            [text_block, srv_code_use, prog_tool_use],
            stop_reason="tool_use",
        )
        # Second response: final answer.
        final_text = make_block("text", text="The answer is 42")
        second = make_response([final_text], stop_reason="end_turn")
        client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(return_value=second))
        )

        params: Dict[str, Any] = {
            "messages": [{"role": "user", "content": "compute"}],
            "model": "claude-opus-4-6",
            "tool_choice": {"type": "auto"},
        }
        tool_dict = {"my_tool": my_tool}
        result = await provider.process_response(
            client=client,
            response=first,
            request_params=params,
            tools=[my_tool],
            tool_dict=tool_dict,
            server_tools=[build_code_execution_tool()],
            programmatic_tool_calling=True,
            parallel_tool_calls=True,
        )

        # Locate the user message we appended in response to the tool call
        # (it's the last user message in params["messages"]).
        user_msgs = [m for m in params["messages"] if m["role"] == "user"]
        assert len(user_msgs) >= 2
        reply = user_msgs[-1]
        assert isinstance(reply["content"], list)
        # CRITICAL: every block in the user reply must be tool_result.
        for block in reply["content"]:
            assert block.get("type") == "tool_result", (
                f"Expected only tool_result blocks in programmatic reply, got {block}"
            )
        # And there should be exactly one (one tool_use)
        assert len(reply["content"]) == 1
        assert reply["content"][0]["tool_use_id"] == "toolu_prog_01"

        # Final answer captured
        content = result[0]
        assert content == "The answer is 42"


# ---------------------------------------------------------------------------
# chat_async-level rejection of server_tools on non-anthropic providers
# ---------------------------------------------------------------------------


class TestChatAsyncProviderGuard:
    @pytest.mark.asyncio
    async def test_chat_async_rejects_server_tools_for_non_anthropic(self):
        with pytest.raises(
            ValueError, match="only supported for the anthropic provider"
        ):
            await chat_async(
                provider="openai",
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                server_tools=["web_search"],
            )

    @pytest.mark.asyncio
    async def test_chat_async_rejects_programmatic_for_non_anthropic(self):
        with pytest.raises(
            ValueError, match="only supported for the anthropic provider"
        ):
            await chat_async(
                provider="openai",
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                programmatic_tool_calling=True,
            )

    @pytest.mark.asyncio
    async def test_chat_async_rejects_container_id_for_non_anthropic(self):
        with pytest.raises(
            ValueError, match="only supported for the anthropic provider"
        ):
            await chat_async(
                provider="openai",
                model="gpt-4.1",
                messages=[{"role": "user", "content": "hi"}],
                container_id="container_x",
            )


# ---------------------------------------------------------------------------
# Beta header threading via execute_chat
# ---------------------------------------------------------------------------


class TestBetaHeaderThreading:
    @pytest.mark.asyncio
    async def test_advisor_adds_beta_header_to_client(self):
        provider = AnthropicProvider(api_key="sk-test")
        text = make_block("text", text="advice")
        adv_result = make_block(
            "advisor_tool_result",
            tool_use_id="srvtoolu_adv",
            content=[{"type": "text", "text": "advice"}],
        )
        response = make_response([text, adv_result])

        captured = {}

        class FakeAsyncAnthropic:
            def __init__(self, **client_kwargs):
                captured["headers"] = client_kwargs.get("default_headers")
                self.messages = SimpleNamespace(create=AsyncMock(return_value=response))

        with patch(
            "defog.llm.providers.anthropic_provider.AsyncAnthropic", FakeAsyncAnthropic
        ):
            await provider.execute_chat(
                messages=[{"role": "user", "content": "advise"}],
                model="claude-opus-4-6",
                server_tools={"advisor": {"model": "claude-opus-4-6"}},
            )

        assert captured["headers"] is not None
        ab = captured["headers"]["anthropic-beta"]
        assert "advisor-tool-2026-03-01" in ab
        assert "interleaved-thinking-2025-05-14" in ab
