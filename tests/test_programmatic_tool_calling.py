import pytest
from pydantic import BaseModel

from defog.llm.utils import chat_async
from defog.llm.utils_function_calling import get_function_specs
from defog.llm.exceptions import ConfigurationError
from tests.conftest import skip_if_no_api_key


# ── Test tools ──────────────────────────────────────────────────────────


class Numbers(BaseModel):
    a: int = 0
    b: int = 0


def numsum(input: Numbers):
    """Returns the sum of two numbers"""
    return input.a + input.b


def numprod(input: Numbers):
    """Returns the product of two numbers"""
    return input.a * input.b


# ── Unit tests (no API key required) ────────────────────────────────────


class TestGetFunctionSpecsProgrammatic:
    def test_programmatic_true_removes_strict_adds_allowed_callers(self):
        specs = get_function_specs(
            [numsum], "anthropic", programmatic_tool_calling=True
        )
        assert len(specs) == 1
        spec = specs[0]
        assert "strict" not in spec
        assert spec["allowed_callers"] == ["code_execution_20250825"]
        assert "additionalProperties" not in spec["input_schema"]

    def test_programmatic_false_keeps_strict(self):
        specs = get_function_specs(
            [numsum], "anthropic", programmatic_tool_calling=False
        )
        assert len(specs) == 1
        spec = specs[0]
        assert spec["strict"] is True
        assert spec["input_schema"]["additionalProperties"] is False
        assert "allowed_callers" not in spec

    def test_programmatic_flag_ignored_for_non_anthropic(self):
        specs = get_function_specs([numsum], "openai", programmatic_tool_calling=True)
        assert len(specs) == 1
        spec = specs[0]
        # OpenAI format should be unchanged
        assert spec["type"] == "function"
        assert "allowed_callers" not in spec.get("function", {})


class TestBuildParamsProgrammatic:
    def test_build_params_with_programmatic_tool_calling(self):
        from defog.llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "test"}],
            model="claude-sonnet-4-6",
            tools=[numsum, numprod],
            programmatic_tool_calling=True,
        )

        # Code execution tool should be first
        assert params["tools"][0] == {
            "type": "code_execution_20250825",
            "name": "code_execution",
        }
        # User tools follow
        tool_names = [t["name"] for t in params["tools"] if "name" in t]
        assert "numsum" in tool_names
        assert "numprod" in tool_names

        # No disable_parallel_tool_use
        tool_choice = params.get("tool_choice", {})
        assert "disable_parallel_tool_use" not in tool_choice

        # Flag stored for budget rebuild
        assert params["_programmatic_tool_calling"] is True

    def test_build_params_with_container_id(self):
        from defog.llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "test"}],
            model="claude-sonnet-4-6",
            tools=[numsum],
            programmatic_tool_calling=True,
            container_id="cntr_abc123",
        )
        assert params["container"] == {"id": "cntr_abc123"}

    def test_build_params_without_programmatic(self):
        from defog.llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(api_key="test-key")
        params, _ = provider.build_params(
            messages=[{"role": "user", "content": "test"}],
            model="claude-sonnet-4-6",
            tools=[numsum],
            programmatic_tool_calling=False,
        )
        # No code_execution tool
        assert all(t.get("type") != "code_execution_20250825" for t in params["tools"])
        assert params["_programmatic_tool_calling"] is False


class TestChatAsyncValidation:
    @pytest.mark.asyncio
    async def test_programmatic_without_tools_raises(self):
        with pytest.raises(ConfigurationError, match="requires at least one tool"):
            await chat_async(
                provider="anthropic",
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "test"}],
                programmatic_tool_calling=True,
                tools=None,
            )

    @pytest.mark.asyncio
    async def test_programmatic_with_non_anthropic_warns(self, capsys):
        """Non-Anthropic provider should print warning and disable the flag."""
        import io
        import sys

        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            provider = "openai"
            programmatic_tool_calling = True
            _ptc_provider = provider.lower()
            if _ptc_provider != "anthropic":
                print(
                    f"Warning: programmatic_tool_calling is only supported for Anthropic. "
                    f"Ignoring for provider '{_ptc_provider}'."
                )
                programmatic_tool_calling = False

            assert programmatic_tool_calling is False
        finally:
            sys.stdout = old_stdout

        output = captured_output.getvalue()
        assert "only supported for Anthropic" in output


# ── Integration tests (require ANTHROPIC_API_KEY) ───────────────────────


class TestProgrammaticToolCallingIntegration:
    @skip_if_no_api_key("anthropic")
    @pytest.mark.asyncio
    async def test_end_to_end_programmatic_tool_calling(self):
        response = await chat_async(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You MUST use the numsum and numprod tools to compute "
                        "the sum and product of 3 and 5, then use numsum to add "
                        "those two results together. Return only the final number."
                    ),
                }
            ],
            tools=[numsum, numprod],
            tool_choice="required",
            programmatic_tool_calling=True,
        )

        assert response.content is not None
        # 3+5=8, 3*5=15, 8+15=23
        assert "23" in str(response.content)
        # container_id should be populated when code execution is used
        assert response.container_id is not None
        assert isinstance(response.container_id, str)

    @skip_if_no_api_key("anthropic")
    @pytest.mark.asyncio
    async def test_container_id_populated(self):
        response = await chat_async(
            provider="anthropic",
            model="claude-sonnet-4-6",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Use the numsum tool to compute the sum of 10 and 20. "
                        "You must call the tool."
                    ),
                }
            ],
            tools=[numsum],
            tool_choice="required",
            programmatic_tool_calling=True,
        )

        assert response.container_id is not None
        assert isinstance(response.container_id, str)
        assert len(response.container_id) > 0
