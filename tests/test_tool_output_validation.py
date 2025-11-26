"""Tests for tool output size validation functionality."""

import pytest
import json
from pydantic import BaseModel
from defog.llm.memory.token_counter import TokenCounter
from defog.llm.tools.handler import ToolHandler


class TestTokenCounter:
    """Test TokenCounter tool output methods."""

    def test_count_tool_output_tokens_string(self):
        counter = TokenCounter()

        # Test with a simple string
        output = "Hello, this is a test output"
        token_count = counter.count_tool_output_tokens(output)
        assert token_count > 0
        assert token_count < 20  # Short string should have few tokens

    def test_count_tool_output_tokens_dict(self):
        counter = TokenCounter()

        # Test with a dictionary
        output = {"key": "value", "number": 123, "nested": {"data": "test"}}
        token_count = counter.count_tool_output_tokens(output)
        assert token_count > 0

        # Dictionary should serialize to JSON
        json_str = json.dumps(output)
        expected_count = counter.count_openai_tokens(json_str, "gpt-4")
        assert token_count == expected_count

    def test_count_tool_output_tokens_list(self):
        counter = TokenCounter()

        # Test with a list
        output = ["item1", "item2", "item3", {"nested": "value"}]
        token_count = counter.count_tool_output_tokens(output)
        assert token_count > 0

    def test_count_tool_output_tokens_non_serializable(self):
        counter = TokenCounter()

        # Test with non-JSON-serializable object
        class CustomObject:
            def __str__(self):
                return "CustomObject representation"

        output = CustomObject()
        token_count = counter.count_tool_output_tokens(output)
        assert token_count > 0

        # Should use str() representation
        expected_count = counter.count_openai_tokens(str(output), "gpt-4")
        assert token_count == expected_count

    def test_truncate_tool_output(self):
        counter = TokenCounter()

        long_output = "word " * 500
        original_tokens = counter.count_tool_output_tokens(long_output)

        truncated = counter.truncate_tool_output(long_output, max_tokens=50)
        truncated_tokens = counter.count_tool_output_tokens(truncated)

        assert original_tokens > truncated_tokens
        assert "truncated" in truncated

    def test_validate_tool_output_size_valid(self):
        counter = TokenCounter()

        # Test with small output that should be valid
        output = "This is a small output"
        is_valid, token_count = counter.validate_tool_output_size(
            output, max_tokens=1000
        )

        assert is_valid is True
        assert token_count > 0
        assert token_count < 1000

    def test_validate_tool_output_size_invalid(self):
        counter = TokenCounter()

        # Create a large output that exceeds the limit
        large_output = "word " * 5000  # Create a very long string
        is_valid, token_count = counter.validate_tool_output_size(
            large_output, max_tokens=100
        )

        assert is_valid is False
        assert token_count > 100

    def test_validate_tool_output_size_custom_model(self):
        counter = TokenCounter()

        # Test with different model
        output = {"data": "test" * 100}
        is_valid, token_count = counter.validate_tool_output_size(
            output, max_tokens=500, model="gpt-3.5-turbo"
        )

        assert isinstance(is_valid, bool)
        assert isinstance(token_count, int)


class TestToolHandler:
    """Test ToolHandler output validation."""

    def test_check_tool_output_size_valid(self):
        # Test valid output
        output = "This is a reasonable tool output"
        is_valid, token_count = ToolHandler.check_tool_output_size(output)

        assert is_valid is True
        assert token_count > 0
        assert token_count < 10000  # Default max

    def test_check_tool_output_size_invalid(self):
        # Test output that exceeds limit
        large_output = "x" * 50000  # Very large string
        is_valid, token_count = ToolHandler.check_tool_output_size(
            large_output, max_tokens=1000
        )

        assert is_valid is False
        assert token_count > 1000

    def test_check_tool_output_size_custom_params(self):
        # Test with custom parameters
        output = {"result": ["data1", "data2", "data3"]}
        is_valid, token_count = ToolHandler.check_tool_output_size(
            output, max_tokens=5000, model="gpt-4.1"
        )

        assert isinstance(is_valid, bool)
        assert isinstance(token_count, int)

    @pytest.mark.asyncio
    async def test_execute_tool_call_with_validation(self):
        """Test that execute_tool_call validates output size."""
        handler = ToolHandler()

        # Define Pydantic model for tool input
        class LargeToolInput(BaseModel):
            size: int = 100000

        # Mock tool that returns large output
        def large_output_tool(input: LargeToolInput):
            return "x" * input.size  # Very large output

        tool_dict = {"large_tool": large_output_tool}

        # Execute tool call
        result = await handler.execute_tool_call(
            tool_name="large_tool", args={"size": 100000}, tool_dict=tool_dict
        )

        # Should return error message about size
        assert "too large" in result
        assert "tokens" in result
        assert "large_tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_call_normal_output(self):
        """Test that normal-sized outputs pass validation."""
        handler = ToolHandler()

        # Define Pydantic model for tool input
        class NormalToolInput(BaseModel):
            message: str = ""

        # Mock tool with normal output
        def normal_tool(input: NormalToolInput):
            return f"Processed: {input.message}"

        tool_dict = {"normal_tool": normal_tool}

        # Execute tool call
        result = await handler.execute_tool_call(
            tool_name="normal_tool", args={"message": "Hello"}, tool_dict=tool_dict
        )

        # Should return actual result, not error
        assert result == "Processed: Hello"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_batch_validation(self):
        """Test batch execution validates each output."""
        handler = ToolHandler()

        # Define Pydantic models
        class SmallToolInput(BaseModel):
            dummy: str = "test"

        class LargeToolInput(BaseModel):
            size: int = 100000

        # Mock tools
        def small_tool(input: SmallToolInput):
            return "Small output"

        def large_tool(input: LargeToolInput):
            return "x" * input.size  # Very large output

        tool_dict = {"small_tool": small_tool, "large_tool": large_tool}

        # Batch tool calls
        tool_calls = [
            {"name": "small_tool", "arguments": {"dummy": "test"}},
            {"name": "large_tool", "arguments": {"size": 100000}},
            {"name": "small_tool", "arguments": {"dummy": "test"}},
        ]

        results = await handler.execute_tool_calls_batch(
            tool_calls=tool_calls, tool_dict=tool_dict, parallel_tool_calls=True
        )

        assert len(results) == 3
        assert results[0] == "Small output"  # First call should succeed
        assert "too large" in results[1]  # Second call should fail validation
        assert results[2] == "Small output"  # Third call should succeed

    @pytest.mark.asyncio
    async def test_sample_tool_result_uses_sampler(self):
        handler = ToolHandler(
            tool_sample_functions={
                "demo_tool": lambda tool_result, **_: tool_result[:1]
            }
        )

        sampled = await handler.sample_tool_result(
            "demo_tool", [1, 2, 3], {"limit": 1}, tool_id="abc"
        )

        assert sampled == [1]

    def test_prepare_result_for_llm_uses_preview_limit(self):
        handler = ToolHandler(tool_result_preview_max_tokens=5)
        text, truncated, token_count = handler.prepare_result_for_llm(
            "token " * 50, model="gpt-4o-mini"
        )

        assert token_count > 5
        assert truncated is True
        assert "...[truncated" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
