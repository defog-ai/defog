import unittest
import pytest
from defog.llm.utils import chat_async
from defog.llm.utils_function_calling import (
    get_function_specs,
    is_pydantic_style_function,
    wrap_regular_function,
    convert_tool_choice,
)
from tests.conftest import skip_if_no_api_key

from pydantic import BaseModel, Field
import httpx
from io import StringIO
import json
from typing import Optional
import time

# ==================================================================================================
# Functions for function calling
# ==================================================================================================

IO_STREAM = StringIO()


def log_to_file(function_name, input_args, tool_result, tool_id):
    """
    Simple function to test logging to a StringIO object.
    Used in test_post_tool_calls_openai and test_post_tool_calls_anthropic
    """
    sorted_input_args = {k: input_args[k] for k in sorted(input_args)}
    print(tool_id)
    message = {
        "function_name": function_name,
        "args": sorted_input_args,
        "result": tool_result,
        "tool_id": tool_id,
    }
    message = json.dumps(message, indent=4)
    IO_STREAM.write(message + "\n")
    return IO_STREAM.getvalue()


class WeatherInput(BaseModel):
    latitude: float = Field(default=0.0, description="The latitude of the location")
    longitude: float = Field(default=0.0, description="The longitude of the location")


async def get_weather(input: WeatherInput):
    """
    This function returns the current temperature (in celsius) for the given latitude and longitude.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={input.latitude}&longitude={input.longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m",
        )
        return_object = r.json()
        return return_object["current"]["temperature_2m"]


class Numbers(BaseModel):
    a: int = 0
    b: int = 0


def numsum(input: Numbers):
    """
    This function returns the sum of two numbers
    """
    timestamp = time.time()
    result = input.a + input.b
    print(f"[{timestamp:.6f}] numsum({input.a}, {input.b}) = {result}")
    return result


def numprod(input: Numbers):
    """
    This function returns the product of two numbers
    """
    timestamp = time.time()
    result = input.a * input.b
    print(f"[{timestamp:.6f}] numprod({input.a}, {input.b}) = {result}")
    return result


# ==================================================================================================
# Tests
# ==================================================================================================
class TestGetFunctionSpecs(unittest.TestCase):
    def setUp(self):
        self.openai_model = "gpt-4.1"
        self.anthropic_model = "claude-haiku-4-5"
        self.grok_model = "grok-4-fast-non-reasoning-latest"
        self.tools = [get_weather, numsum, numprod]
        self.maxDiff = None
        self.openai_specs = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "This function returns the current temperature (in celsius) for the given latitude and longitude.",
                    "parameters": {
                        "properties": {
                            "latitude": {
                                "description": "The latitude of the location",
                                "type": "number",
                            },
                            "longitude": {
                                "description": "The longitude of the location",
                                "type": "number",
                            },
                        },
                        "type": "object",
                        "required": ["latitude", "longitude"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numsum",
                    "description": "This function returns the sum of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "type": "object",
                        "required": ["a", "b"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numprod",
                    "description": "This function returns the product of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "type": "object",
                        "required": ["a", "b"],
                    },
                },
            },
        ]
        self.anthropic_specs = [
            {
                "name": "get_weather",
                "description": "This function returns the current temperature (in celsius) for the given latitude and longitude.",
                "input_schema": {
                    "properties": {
                        "latitude": {
                            "description": "The latitude of the location",
                            "type": "number",
                        },
                        "longitude": {
                            "description": "The longitude of the location",
                            "type": "number",
                        },
                    },
                    "type": "object",
                    "required": ["latitude", "longitude"],
                },
            },
            {
                "name": "numsum",
                "description": "This function returns the sum of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "type": "object",
                    "required": ["a", "b"],
                },
            },
            {
                "name": "numprod",
                "description": "This function returns the product of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "type": "object",
                    "required": ["a", "b"],
                },
            },
        ]

    def test_get_function_specs(self):
        openai_specs = get_function_specs(self.tools, self.openai_model)
        anthropic_specs = get_function_specs(self.tools, self.anthropic_model)
        grok_specs = get_function_specs(self.tools, self.grok_model)

        self.assertEqual(openai_specs, self.openai_specs)
        self.assertEqual(anthropic_specs, self.anthropic_specs)
        self.assertEqual(grok_specs, self.anthropic_specs)


class TestToolUseFeatures(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = [get_weather, numsum, numprod]
        self.weather_qn = "What is the current temperature in Singapore? Return the answer as a number and nothing else."
        self.weather_qn_specific = "What is the current temperature in Singapore? Singapore's latitude is 1.3521 and longitude is 103.8198. Return the answer as a number and nothing else."
        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 890872? Always use the tools provided for all calculation, even simple calculations. Return only the final answer, nothing else."
        self.arithmetic_answer = "73561281"
        self.arithmetic_expected_tool_outputs = [
            {"name": "numprod", "args": {"a": 31283, "b": 2323}, "result": 72670409},
            {
                "name": "numsum",
                "args": {"a": 72670409, "b": 890872},
                "result": 73561281,
            },
        ]

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_use_arithmetic_async_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [
            output["name"]
            for output in result.tool_outputs
            if output["name"] != "reasoning"
        ]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_use_weather_async_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [
            output["name"]
            for output in result.tool_outputs
            if output["name"] != "reasoning"
        ]
        tool_outputs = [
            output for output in result.tool_outputs if output["name"] != "reasoning"
        ]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(tool_outputs[0]["result"]), 21)
        self.assertLessEqual(float(tool_outputs[0]["result"]), 38)

    # =============================
    # Grok structured outputs
    # =============================
    class GrokStructuredMath(BaseModel):
        reasoning: str
        total: int

    @pytest.mark.asyncio
    @skip_if_no_api_key("grok")
    async def test_grok_structured_output_simple(self):
        messages = [
            {
                "role": "user",
                "content": "Add 19 and 23. Return the result in JSON with fields 'reasoning' and integer 'total'.",
            }
        ]

        result = await chat_async(
            provider="grok",
            model="grok-4-fast-non-reasoning-latest",
            messages=messages,
            response_format=TestToolUseFeatures.GrokStructuredMath,
            max_retries=1,
        )

        # Should parse into the pydantic model
        self.assertIsInstance(result.content, TestToolUseFeatures.GrokStructuredMath)
        # Basic sanity check on computation
        self.assertEqual(result.content.total, 42)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_arithmetic_async_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})
        self.assertEqual(result.content, self.arithmetic_answer)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_weather_async_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("grok")
    async def test_tool_use_arithmetic_async_grok(self):
        result = await chat_async(
            provider="grok",
            model="grok-4-fast-non-reasoning-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("grok")
    async def test_tool_use_weather_async_grok(self):
        result = await chat_async(
            provider="grok",
            model="grok-4-fast-non-reasoning-latest",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn_specific,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_use_arithmetic_async_anthropic_reasoning_effort(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            reasoning_effort="low",
            max_retries=1,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        self.assertEqual(result.content, self.arithmetic_answer)

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_tool_use_arithmetic_async_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_tool_use_weather_async_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": self.weather_qn_specific,
                },
            ],
            tools=self.tools,
            max_retries=1,
        )
        print(result)
        tools_used = [output["name"] for output in result.tool_outputs]
        self.assertSetEqual(set(tools_used), {"get_weather"})
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")
        self.assertEqual(
            result.tool_outputs[0]["args"], {"latitude": 1.3521, "longitude": 103.8198}
        )
        self.assertGreaterEqual(float(result.content), 21)
        self.assertLessEqual(float(result.content), 38)

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_post_tool_calls_openai(self):
        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [
            output["name"]
            for output in result.tool_outputs
            if output["name"] != "reasoning"
        ]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_post_tool_calls_anthropic(self):
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [
            output["name"]
            for output in result.tool_outputs
            if output["name"] != "reasoning"
        ]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_post_tool_calls_gemini(self):
        result = await chat_async(
            provider="gemini",
            model="gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=self.tools,
            post_tool_function=log_to_file,
        )
        print(result)
        self.assertEqual(result.content, self.arithmetic_answer)
        tools_used = [
            output["name"]
            for output in result.tool_outputs
            if output["name"] != "reasoning"
        ]
        self.assertSetEqual(set(tools_used), {"numsum", "numprod"})


class TestParallelToolCalls(unittest.IsolatedAsyncioTestCase):
    """Test parallel tool calls functionality."""

    def setUp(self):
        from defog.llm.utils_function_calling import execute_tools_parallel
        from defog.llm.tools.handler import ToolHandler

        self.execute_tools_parallel = execute_tools_parallel
        self.handler = ToolHandler()
        self.tools = [numsum, numprod]
        self.tool_dict = self.handler.build_tool_dict(self.tools)

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_multiple_calls(self):
        """Test that multiple tool calls can be executed in parallel."""

        # Simulate tool calls that would benefit from parallel execution
        tool_calls = [
            {
                "function": {
                    "name": "numsum",
                    "arguments": {"a": 12312323434, "b": 89230482903480},
                }
            },
            {"function": {"name": "numprod", "arguments": {"a": 2134, "b": 9823}}},
            {"function": {"name": "numsum", "arguments": {"a": 983247, "b": 2348796}}},
        ]

        # Test sequential execution
        start_time = time.time()
        sequential_results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=False
        )
        sequential_time = time.time() - start_time

        # Test parallel execution
        start_time = time.time()
        parallel_results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=True
        )
        parallel_time = time.time() - start_time

        # Results should be the same
        self.assertEqual(sequential_results, parallel_results)
        self.assertEqual(sequential_results, [89242795226914, 20962282, 3332043])

        # For simple arithmetic, parallel may not be faster, but should not be significantly slower
        # This test mainly ensures functionality works correctly
        print(f"Sequential time: {sequential_time:.4f}s")
        print(f"Parallel time: {parallel_time:.4f}s")

    @pytest.mark.asyncio
    async def test_tool_handler_batch_execution(self):
        """Test the tool handler batch execution method."""
        tool_calls = [
            {
                "function": {
                    "name": "numsum",
                    "arguments": {"a": 12312323434, "b": 89230482903480},
                }
            },
            {"function": {"name": "numprod", "arguments": {"a": 2134, "b": 9823}}},
            {"function": {"name": "numsum", "arguments": {"a": 983247, "b": 2348796}}},
        ]

        # Test sequential batch execution
        results_sequential = await self.handler.execute_tool_calls_batch(
            tool_calls, self.tool_dict, parallel_tool_calls=False
        )

        # Test parallel batch execution
        results_parallel = await self.handler.execute_tool_calls_batch(
            tool_calls, self.tool_dict, parallel_tool_calls=True
        )

        # Results should be identical
        self.assertEqual(results_sequential, results_parallel)
        self.assertEqual(results_sequential, [89242795226914, 20962282, 3332043])

    @pytest.mark.asyncio
    async def test_error_handling_in_parallel_execution(self):
        """Test error handling when tools fail in parallel execution."""
        tool_calls = [
            {"function": {"name": "numsum", "arguments": {"a": 1, "b": 2}}},
            {"function": {"name": "nonexistent_tool", "arguments": {"a": 3, "b": 4}}},
        ]

        results = await self.execute_tools_parallel(
            tool_calls, self.tool_dict, enable_parallel=True
        )

        # First tool should succeed, second should return error
        self.assertEqual(results[0], 3)
        self.assertIn("Error: Function nonexistent_tool not found", results[1])


class TestParallelToolCallsEndToEnd(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for parallel tool calls with real API calls."""

    def setUp(self):
        self.tools = [numsum, numprod]
        # More complex message that requires tool usage
        self.messages = [
            {
                "role": "user",
                "content": """Calculate the following using the provided tools:
1. The sum of 387293472 and 2348293482
2. The product of 12376 and 23245

You MUST use the numsum and numprod tools for these calculations. Do not calculate manually.
IMPORTANT: Since these are independent calculations, please call both tools in parallel (at the same time) rather than sequentially.
Return only the final results.""",
            }
        ]

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_openai_parallel_vs_sequential_speed(self):
        """Test OpenAI parallel vs sequential execution speed."""

        # Test parallel execution
        start_time = time.time()
        result_parallel = await chat_async(
            provider="openai",
            model="gpt-4.1",
            messages=self.messages,
            tools=self.tools,
            parallel_tool_calls=True,
            temperature=0,
            max_retries=1,
        )
        parallel_time = time.time() - start_time

        # Test sequential execution
        start_time = time.time()
        result_sequential = await chat_async(
            provider="openai",
            model="gpt-4.1",
            messages=self.messages,
            tools=self.tools,
            parallel_tool_calls=False,
            temperature=0,
            max_retries=1,
        )
        sequential_time = time.time() - start_time

        # Verify both produce correct results
        self.assertEqual(len(result_parallel.tool_outputs), 2)
        self.assertEqual(len(result_sequential.tool_outputs), 2)

        # Check that sum and product were calculated
        outputs_parallel = {
            o["name"]: o["result"] for o in result_parallel.tool_outputs
        }
        outputs_sequential = {
            o["name"]: o["result"] for o in result_sequential.tool_outputs
        }

        self.assertEqual(outputs_parallel["numsum"], 2735586954)
        self.assertEqual(outputs_parallel["numprod"], 287680120)
        self.assertEqual(outputs_parallel, outputs_sequential)

        # Log timing results
        print("\nOpenAI Timing Results:")
        print(f"  Parallel execution: {parallel_time:.2f}s")
        print(f"  Sequential execution: {sequential_time:.2f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

        # Parallel should generally be faster or at least not significantly slower
        # We don't assert exact timing as it depends on API response times

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_parallel_tool_behavior(self):
        """Test Anthropic's parallel tool call behavior."""

        # Test with parallel enabled (should make one API call with multiple tools)
        start_time = time.time()
        result_parallel = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=self.messages,
            tools=self.tools,
            parallel_tool_calls=True,
            temperature=0,
            max_retries=1,
        )
        parallel_time = time.time() - start_time

        # Test with parallel disabled (may require multiple API calls)
        start_time = time.time()
        result_sequential = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=self.messages,
            tools=self.tools,
            parallel_tool_calls=False,
            temperature=0,
            max_retries=1,
        )
        sequential_time = time.time() - start_time

        # Verify results
        self.assertEqual(len(result_parallel.tool_outputs), 2)
        self.assertEqual(len(result_sequential.tool_outputs), 2)

        # Check results
        outputs_parallel = {
            o["name"]: o["result"] for o in result_parallel.tool_outputs
        }
        self.assertEqual(outputs_parallel["numsum"], 2735586954)
        self.assertEqual(outputs_parallel["numprod"], 287680120)

        print("\nAnthropic Timing Results:")
        print(f"  Parallel execution: {parallel_time:.2f}s")
        print(f"  Sequential execution: {sequential_time:.2f}s")
        print(f"  Speedup: {sequential_time / parallel_time:.2f}x")

    # we can't really test Gemini, as it always has parallel tool calls enabled


# ==================================================================================================
# Regular function tests
# ==================================================================================================


# Test regular functions
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


async def multiply_numbers(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Args:
        x: First number to multiply
        y: Second number to multiply
    """
    return x * y


def greet(name: str, greeting: str = "Hello") -> str:
    """
    Generate a greeting message.

    Parameters:
        name: The name of the person to greet
        greeting: The greeting word to use (default: Hello)
    """
    return f"{greeting}, {name}!"


def optional_params(required: str, optional: Optional[int] = None) -> str:
    """Test function with optional parameters."""
    if optional is not None:
        return f"{required} - {optional}"
    return required


class TestRegularFunctionTools(unittest.TestCase):
    def test_is_pydantic_style_function(self):
        """Test function style detection."""
        # Regular functions should return False
        self.assertFalse(is_pydantic_style_function(add_numbers))
        self.assertFalse(is_pydantic_style_function(multiply_numbers))
        self.assertFalse(is_pydantic_style_function(greet))

        # Wrapped function should return True
        wrapped = wrap_regular_function(add_numbers)
        self.assertTrue(is_pydantic_style_function(wrapped))

    def test_wrap_regular_function(self):
        """Test wrapping regular functions."""
        # Test sync function
        wrapped_add = wrap_regular_function(add_numbers)
        self.assertEqual(wrapped_add.__name__, "add_numbers")
        self.assertEqual(wrapped_add.__doc__, "Add two numbers together.")

        # Test the wrapped function can be called
        input_model = wrapped_add.__dict__["_input_model"]
        inputs = input_model(a=5, b=3)
        result = wrapped_add(inputs)
        self.assertEqual(result, 8)

        # Test async function
        wrapped_multiply = wrap_regular_function(multiply_numbers)
        self.assertEqual(wrapped_multiply.__name__, "multiply_numbers")

    def test_function_with_defaults(self):
        """Test functions with default parameters."""
        wrapped_greet = wrap_regular_function(greet)
        input_model = wrapped_greet.__dict__["_input_model"]

        # Test with default
        inputs = input_model(name="Alice")
        result = wrapped_greet(inputs)
        self.assertEqual(result, "Hello, Alice!")

        # Test with custom greeting
        inputs = input_model(name="Bob", greeting="Hi")
        result = wrapped_greet(inputs)
        self.assertEqual(result, "Hi, Bob!")

    def test_function_with_optional(self):
        """Test functions with optional parameters."""
        wrapped_optional = wrap_regular_function(optional_params)
        input_model = wrapped_optional.__dict__["_input_model"]

        # Test without optional
        inputs = input_model(required="test")
        result = wrapped_optional(inputs)
        self.assertEqual(result, "test")

        # Test with optional
        inputs = input_model(required="test", optional=42)
        result = wrapped_optional(inputs)
        self.assertEqual(result, "test - 42")

    def test_get_function_specs_with_regular_functions(self):
        """Test that get_function_specs works with regular functions."""
        functions = [add_numbers, greet]

        # Test OpenAI format
        specs = get_function_specs(functions, "gpt-4.1")
        self.assertEqual(len(specs), 2)

        # Check first function
        self.assertEqual(specs[0]["type"], "function")
        self.assertEqual(specs[0]["function"]["name"], "add_numbers")
        self.assertEqual(
            specs[0]["function"]["description"], "Add two numbers together."
        )
        self.assertIn("a", specs[0]["function"]["parameters"]["properties"])
        self.assertIn("b", specs[0]["function"]["parameters"]["properties"])

        # Check second function
        self.assertEqual(specs[1]["function"]["name"], "greet")
        self.assertIn("name", specs[1]["function"]["parameters"]["properties"])
        self.assertIn("greeting", specs[1]["function"]["parameters"]["properties"])

        # Test Anthropic format
        specs = get_function_specs(functions, "claude-haiku-4-5")
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["name"], "add_numbers")
        self.assertIn("a", specs[0]["input_schema"]["properties"])


@pytest.mark.asyncio
@skip_if_no_api_key("openai")
async def test_regular_functions_with_chat_async():
    """Test using regular functions with chat_async."""

    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle given its length and width."""
        return length * width

    messages = [
        {
            "role": "user",
            "content": "What is the area of a rectangle with length 5.5 and width 3.2?",
        }
    ]

    response = await chat_async(
        provider="openai",
        model="gpt-4.1-mini",
        messages=messages,
        tools=[calculate_area],
        tool_choice="auto",
        temperature=0,
    )

    # Check that the tool was called
    assert response.tool_outputs is not None
    assert len(response.tool_outputs) > 0
    assert response.tool_outputs[0]["name"] == "calculate_area"
    assert response.tool_outputs[0]["result"] == 5.5 * 3.2


# ==================================================================================================
# Structured output with tool calls tests
# ==================================================================================================


class WeatherReport(BaseModel):
    """Structured weather report model."""

    location: str = Field(description="The location name")
    temperature: float = Field(description="Current temperature in Celsius")
    conditions: str = Field(description="Weather conditions description")


class CalculationResult(BaseModel):
    """Structured calculation result model."""

    operation: str = Field(description="The mathematical operation performed")
    result: float = Field(description="The calculation result")
    explanation: str = Field(description="Explanation of the calculation")


def get_text_long():
    """Returns a very long text string (20k repetitions)."""
    return "The quick brown fox jumps over the lazy dog.\n" * 5000


class TestToolOutputMaxTokens(unittest.IsolatedAsyncioTestCase):
    """Test the configurable tool_output_max_tokens feature."""

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_output_default_limit(self):
        """Test that get_text_long exceeds the default 10000 token limit."""
        messages = [
            {
                "role": "user",
                "content": "Call the get_text_long function and tell me what it returns.",
            }
        ]

        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=messages,
            tools=[get_text_long],
            temperature=0,
            max_retries=1,
        )

        # remove reasoning from tool outputs
        result.tool_outputs = [
            i for i in result.tool_outputs if i["name"] != "reasoning"
        ]

        # Tool should be called but output should be truncated message
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_text_long")
        # Check if we got the truncation message
        truncation_message = result.tool_outputs[0]["result"]
        self.assertIn("too large", truncation_message)
        self.assertIn("tokens", truncation_message)

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_tool_output_limit_disabled(self):
        """Test with tool_output_max_tokens set to -1 (disabled)."""
        messages = [
            {
                "role": "user",
                "content": "Call the get_text_long function. Just tell me if it succeeded without showing the full output.",
            }
        ]

        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=messages,
            tools=[get_text_long],
            temperature=0,
            max_retries=1,
            tool_output_max_tokens=-1,  # Disable limit
        )

        # remove reasoning from tool outputs
        result.tool_outputs = [
            i for i in result.tool_outputs if i["name"] != "reasoning"
        ]

        # Tool should execute successfully without truncation
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_text_long")
        # Result should contain the full output (20k repetitions)
        output_lines = result.tool_outputs[0]["result"].strip().split("\n")
        self.assertEqual(len(output_lines), 5000)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_tool_output_limit_anthropic(self):
        """Test tool output limits with Anthropic provider."""
        messages = [
            {
                "role": "user",
                "content": "Call the get_text_long function.",
            }
        ]

        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=messages,
            tools=[get_text_long],
            temperature=0,
            max_retries=1,
            tool_output_max_tokens=1000,  # Low limit
        )

        # Should get truncation message
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_text_long")
        truncation_message = result.tool_outputs[0]["result"]
        self.assertIn("too large", truncation_message)


class TestStructuredOutputWithTools(unittest.IsolatedAsyncioTestCase):
    """Test structured outputs combined with tool calls."""

    def setUp(self):
        self.tools = [get_weather, numsum, numprod]
        self.weather_message = [
            {
                "role": "user",
                "content": "Get the weather for Singapore (latitude 1.3521, longitude 103.8198) and return it as a structured weather report with location, temperature, and conditions.",
            }
        ]
        self.calculation_message = [
            {
                "role": "user",
                "content": "Calculate the sum of 150 and 250, then multiply the result by 3. Return the final result as a structured calculation report. Recall that you MUST use the tools provided for all calculation, even simple calculations.",
            }
        ]

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_openai_structured_output_with_tools(self):
        """Test OpenAI with tools and structured output."""
        # Test weather report with structured output
        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=self.weather_message,
            tools=self.tools,
            response_format=WeatherReport,
            temperature=0,
            max_retries=1,
        )

        # remove reasoning from tool outputs
        result.tool_outputs = [
            i for i in result.tool_outputs if i["name"] != "reasoning"
        ]

        # Verify tool was called
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")

        # Verify structured output
        self.assertIsInstance(result.content, WeatherReport)
        self.assertEqual(result.content.location.lower(), "singapore")
        self.assertIsInstance(result.content.temperature, float)
        self.assertGreaterEqual(result.content.temperature, 20)
        self.assertLessEqual(result.content.temperature, 40)
        self.assertIsInstance(result.content.conditions, str)
        self.assertGreater(len(result.content.conditions), 0)

    @pytest.mark.asyncio
    @skip_if_no_api_key("openai")
    async def test_openai_calculation_structured_output(self):
        """Test OpenAI with multiple tool calls and structured output."""
        result = await chat_async(
            provider="openai",
            model="gpt-5-nano",
            messages=self.calculation_message,
            tools=self.tools,
            response_format=CalculationResult,
            temperature=0,
            max_retries=1,
        )

        # remove reasoning from tool outputs
        result.tool_outputs = [
            i for i in result.tool_outputs if i["name"] != "reasoning"
        ]

        # Verify tools were called
        self.assertGreaterEqual(len(result.tool_outputs), 2)
        tool_names = [output["name"] for output in result.tool_outputs]
        self.assertIn("numsum", tool_names)
        self.assertIn("numprod", tool_names)

        # Verify structured output
        self.assertIsInstance(result.content, CalculationResult)
        # Check that operation mentions multiplication or contains * symbol
        self.assertTrue(
            "multi" in result.content.operation.lower()
            or "*" in result.content.operation
            or "x" in result.content.operation.lower()
            or "×" in result.content.operation.lower(),
            f"Expected operation to contain 'multi' or '*', got: {result.content.operation}",
        )
        self.assertEqual(result.content.result, 1200)  # (150 + 250) * 3
        self.assertIsInstance(result.content.explanation, str)
        self.assertGreater(len(result.content.explanation), 0)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_structured_output_with_tools(self):
        """Test Anthropic with tools and structured output."""
        # Test weather report with structured output
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=self.weather_message,
            tools=self.tools,
            response_format=WeatherReport,
            temperature=0,
            max_retries=1,
        )

        # Verify tool was called
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")

        # Verify structured output
        self.assertIsInstance(result.content, WeatherReport)
        self.assertEqual(result.content.location.lower(), "singapore")
        self.assertIsInstance(result.content.temperature, float)
        self.assertGreaterEqual(result.content.temperature, 20)
        self.assertLessEqual(result.content.temperature, 40)
        self.assertIsInstance(result.content.conditions, str)

    @pytest.mark.asyncio
    @skip_if_no_api_key("anthropic")
    async def test_anthropic_calculation_structured_output(self):
        """Test Anthropic with multiple tool calls and structured output."""
        result = await chat_async(
            provider="anthropic",
            model="claude-haiku-4-5",
            messages=self.calculation_message,
            tools=self.tools,
            response_format=CalculationResult,
            temperature=0,
            max_retries=1,
        )

        # Verify tools were called
        self.assertGreaterEqual(len(result.tool_outputs), 2)
        tool_names = [output["name"] for output in result.tool_outputs]
        self.assertIn("numsum", tool_names)
        self.assertIn("numprod", tool_names)

        # Verify structured output
        self.assertIsInstance(result.content, CalculationResult)
        self.assertEqual(result.content.result, 1200)  # (150 + 250) * 3
        self.assertIsInstance(result.content.explanation, str)

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_gemini_structured_output_with_tools(self):
        """Test Gemini with tools and structured output."""
        # Test weather report with structured output
        result = await chat_async(
            provider="gemini",
            model="gemini-3-flash-preview",
            messages=self.weather_message,
            tools=self.tools,
            response_format=WeatherReport,
            temperature=0,
            max_retries=1,
        )

        # Verify tool was called
        self.assertGreater(len(result.tool_outputs), 0)
        self.assertEqual(result.tool_outputs[0]["name"], "get_weather")

        # Verify structured output
        self.assertIsInstance(result.content, WeatherReport)
        self.assertEqual(result.content.location.lower(), "singapore")
        self.assertIsInstance(result.content.temperature, float)
        self.assertGreaterEqual(result.content.temperature, 20)
        self.assertLessEqual(result.content.temperature, 40)
        self.assertIsInstance(result.content.conditions, str)

    @pytest.mark.asyncio
    @skip_if_no_api_key("gemini")
    async def test_gemini_calculation_structured_output(self):
        """Test Gemini with multiple tool calls and structured output."""
        result = await chat_async(
            provider="gemini",
            model="gemini-3-flash-preview",
            messages=self.calculation_message,
            tools=self.tools,
            response_format=CalculationResult,
            temperature=0,
            max_retries=1,
        )

        # Verify tools were called
        self.assertGreaterEqual(len(result.tool_outputs), 2)
        tool_names = [output["name"] for output in result.tool_outputs]
        self.assertIn("numsum", tool_names)
        self.assertIn("numprod", tool_names)

        # Verify structured output
        self.assertIsInstance(result.content, CalculationResult)
        self.assertEqual(result.content.result, 1200)  # (150 + 250) * 3
        self.assertIsInstance(result.content.explanation, str)
