import inspect
import json
import logging
from typing import Dict, List, Callable, Any, Optional
from ..exceptions import ToolError
from ..utils_function_calling import (
    execute_tool,
    execute_tool_async,
    execute_tools_parallel,
    verify_post_tool_function,
)

logger = logging.getLogger(__name__)


class ToolHandler:
    """Handles tool calling logic for LLM providers."""

    def __init__(
        self,
        max_consecutive_errors: int = 3,
        tool_budget: Optional[Dict[str, int]] = None,
        image_result_keys: Optional[List[str]] = None,
        tool_output_max_tokens: Optional[int] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
        tool_result_preview_max_tokens: Optional[int] = None,
    ):
        self.max_consecutive_errors = max_consecutive_errors
        self.tool_budget = tool_budget.copy() if tool_budget else None
        self.tool_usage = {}
        self.image_result_keys = image_result_keys
        self.tool_output_max_tokens = (
            tool_output_max_tokens if tool_output_max_tokens is not None else 10000
        )
        self.tool_sample_functions = tool_sample_functions
        self.tool_result_preview_max_tokens = tool_result_preview_max_tokens
        logger.debug(
            "ToolHandler initialized with budget: %s, max_tokens: %s, sampler provided: %s, preview_max_tokens: %s",
            self.tool_budget,
            self.tool_output_max_tokens,
            bool(self.tool_sample_functions),
            self.tool_result_preview_max_tokens,
        )

    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available based on its budget."""
        if self.tool_budget is None:
            return True

        if tool_name not in self.tool_budget:
            # Tools not in budget have unlimited calls
            logger.debug(f"Tool '{tool_name}' not in budget, unlimited calls allowed")
            return True

        used = self.tool_usage.get(tool_name, 0)
        budget = self.tool_budget[tool_name]
        available = used < budget
        logger.debug(
            f"Tool '{tool_name}' availability: used={used}, budget={budget}, available={available}"
        )
        return available

    def _update_tool_usage(self, tool_name: str) -> None:
        """Update tool usage count after successful execution."""
        if self.tool_budget is None:
            return

        if tool_name in self.tool_budget:
            self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
            logger.debug(
                f"Updated tool usage for '{tool_name}': {self.tool_usage[tool_name]}/{self.tool_budget[tool_name]}"
            )

    def get_available_tools(self, tools: List[Callable]) -> List[Callable]:
        """Filter tools based on remaining budget."""
        if self.tool_budget is None:
            return tools

        available_tools = []
        for tool in tools:
            if self.is_tool_available(tool.__name__):
                available_tools.append(tool)

        logger.debug(
            f"Available tools after budget filtering: {[t.__name__ for t in available_tools]}"
        )
        return available_tools

    @staticmethod
    def check_tool_output_size(
        output: Any, max_tokens: int = 10000, model: str = "gpt-4.1"
    ) -> bool:
        """Check if the output size of a tool call is within the token limit."""
        from ..memory.token_counter import TokenCounter

        token_counter = TokenCounter()
        is_valid, token_count = token_counter.validate_tool_output_size(
            output, max_tokens, model
        )
        return is_valid, token_count

    @staticmethod
    def _stringify_tool_result(output: Any) -> str:
        """Convert a tool result into a safe string representation for the model."""
        if isinstance(output, str):
            return output
        try:
            return json.dumps(output)
        except (TypeError, ValueError):
            return str(output)

    @staticmethod
    def _filter_callable_kwargs(
        function: Callable, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Only pass keyword arguments accepted by the callable."""
        try:
            sig = inspect.signature(function)
        except (TypeError, ValueError):
            return kwargs

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    async def sample_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        input_args: Optional[Dict[str, Any]] = None,
        tool_id: Optional[str] = None,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
    ) -> Any:
        """
        Apply a user-provided sampler to reduce tool output size before sending to the LLM.
        Falls back to the original result on errors or when no sampler is configured.
        """
        sampler_map = tool_sample_functions or self.tool_sample_functions
        if not sampler_map:
            return tool_result

        sample_fn = None
        if callable(sampler_map):
            sample_fn = sampler_map  # type: ignore[assignment]
        elif isinstance(sampler_map, dict):
            sample_fn = sampler_map.get(tool_name) or sampler_map.get("*")

        if not sample_fn:
            return tool_result

        kwargs = {
            "function_name": tool_name,
            "tool_result": tool_result,
            "input_args": input_args or {},
            "tool_id": tool_id,
        }
        safe_kwargs = ToolHandler._filter_callable_kwargs(sample_fn, kwargs)

        try:
            if inspect.iscoroutinefunction(sample_fn):
                return await sample_fn(**safe_kwargs)
            return sample_fn(**safe_kwargs)
        except Exception as exc:
            logger.warning("Error sampling tool output for %s: %s", tool_name, exc)
            return tool_result

    def is_sampler_configured(
        self,
        tool_name: str,
        tool_sample_functions: Optional[Dict[str, Callable]] = None,
    ) -> bool:
        """Check if a sampler is available for the given tool name."""
        sampler_map = tool_sample_functions or self.tool_sample_functions
        if not sampler_map:
            return False
        if callable(sampler_map):
            return True
        if isinstance(sampler_map, dict):
            return tool_name in sampler_map or "*" in sampler_map
        return False

    def prepare_result_for_llm(
        self,
        tool_result: Any,
        preview_max_tokens: Optional[int] = None,
        model: str = "gpt-4.1",
    ) -> tuple[str, bool, int]:
        """
        Convert a tool result to text for the LLM and optionally truncate to a token budget.

        Returns:
            (text_for_llm, was_truncated, token_count_before_truncation)
        """
        from ..memory.token_counter import TokenCounter

        text_result = ToolHandler._stringify_tool_result(tool_result)
        token_counter = TokenCounter()
        token_count = token_counter.count_tool_output_tokens(text_result, model)

        effective_max = (
            preview_max_tokens
            if preview_max_tokens is not None
            else self.tool_result_preview_max_tokens
        )

        was_truncated = False
        if effective_max and effective_max > 0 and token_count > effective_max:
            text_result = token_counter.truncate_tool_output(
                text_result, max_tokens=effective_max, model=model
            )
            was_truncated = True

        return text_result, was_truncated, token_count

    async def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_dict: Dict[str, Callable],
        post_tool_function: Optional[Callable] = None,
        tool_id: Optional[str] = None,
    ) -> Any:
        """Execute a single tool call."""
        logger.debug(f"Executing tool call: '{tool_name}' with args: {args}")
        logger.debug(f"Current tool usage: {self.tool_usage}")

        # Check if tool is available based on budget
        if not self.is_tool_available(tool_name):
            # Return a message instead of raising an error
            msg = f"Tool '{tool_name}' has exceeded its usage budget and is no longer available."
            logger.debug(msg)
            return msg

        tool_to_call = tool_dict.get(tool_name)
        if tool_to_call is None:
            raise ToolError(tool_name, "Tool not found")

        try:
            # Execute tool depending on whether it is async
            if inspect.iscoroutinefunction(tool_to_call):
                result = await execute_tool_async(tool_to_call, args)
            else:
                result = execute_tool(tool_to_call, args)

            logger.debug(f"Tool '{tool_name}' executed successfully, result: {result}")

            # Update usage count after successful execution
            self._update_tool_usage(tool_name)
        except Exception as e:
            raise ToolError(tool_name, f"Error executing tool: {e}", e)

        # Execute post-tool function if provided
        if post_tool_function:
            try:
                if inspect.iscoroutinefunction(post_tool_function):
                    await post_tool_function(
                        function_name=tool_name,
                        input_args=args,
                        tool_result=result,
                        tool_id=tool_id,
                    )
                else:
                    post_tool_function(
                        function_name=tool_name,
                        input_args=args,
                        tool_result=result,
                        tool_id=tool_id,
                    )
            except Exception as e:
                raise ToolError(
                    tool_name, f"Error executing post_tool_function: {e}", e
                )

        # Only check if tool_output_max_tokens is not -1 (disabled)
        if self.tool_output_max_tokens != -1:
            is_valid, token_count = ToolHandler.check_tool_output_size(
                result, self.tool_output_max_tokens
            )
            if not is_valid:
                return f"Tool output for {tool_name} is too large at {token_count} tokens. Please rephrase the question asked so that the output is within the token limit."

        return result

    async def execute_tool_calls_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_dict: Dict[str, Callable],
        parallel_tool_calls: bool = False,
        post_tool_function: Optional[Callable] = None,
    ) -> List[Any]:
        """Execute multiple tool calls either in parallel or sequentially."""
        # Don't pre-check availability - let individual calls handle it

        try:
            # Execute tools with budget tracking
            if not parallel_tool_calls:
                # Sequential execution with budget updates
                results = []
                for tool_call in tool_calls:
                    func_name = tool_call.get("function", {}).get(
                        "name"
                    ) or tool_call.get("name")
                    func_args = tool_call.get("function", {}).get(
                        "arguments"
                    ) or tool_call.get("arguments", {})
                    tool_id = (
                        tool_call.get("id")
                        or tool_call.get("tool_call_id")
                        or tool_call.get("call_id")
                        or tool_call.get("tool_use_id")
                    )

                    # Use execute_tool_call which handles budget tracking
                    result = await self.execute_tool_call(
                        func_name,
                        func_args,
                        tool_dict,
                        post_tool_function,
                        tool_id=tool_id,
                    )
                    results.append(result)
                return results
            else:
                # Parallel execution - execute tools then update budgets
                results = await execute_tools_parallel(
                    tool_calls, tool_dict, parallel_tool_calls
                )

                # Update budgets for successful executions
                for tool_call, result in zip(tool_calls, results):
                    func_name = tool_call.get("function", {}).get(
                        "name"
                    ) or tool_call.get("name")
                    # Only update if result is not an error string
                    if not (isinstance(result, str) and result.startswith("Error:")):
                        self._update_tool_usage(func_name)

            # Execute post-tool function for each result if provided
            if post_tool_function:
                for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                    func_name = tool_call.get("function", {}).get(
                        "name"
                    ) or tool_call.get("name")
                    func_args = tool_call.get("function", {}).get(
                        "arguments"
                    ) or tool_call.get("arguments", {})
                    tool_id = (
                        tool_call.get("id")
                        or tool_call.get("tool_call_id")
                        or tool_call.get("call_id")
                        or tool_call.get("tool_use_id")
                    )

                    try:
                        if inspect.iscoroutinefunction(post_tool_function):
                            await post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
                                tool_id=tool_id,
                            )
                        else:
                            post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
                                tool_id=tool_id,
                            )
                    except Exception as e:
                        # Don't fail the entire batch for post-tool function errors
                        print(
                            f"Warning: Error executing post_tool_function for {func_name}: {e}"
                        )

            # Only check if tool_output_max_tokens is not -1 (disabled)
            if self.tool_output_max_tokens != -1:
                for idx, result in enumerate(results):
                    is_valid, token_count = ToolHandler.check_tool_output_size(
                        result, self.tool_output_max_tokens
                    )
                    if not is_valid:
                        results[idx] = (
                            f"Tool output for {func_name} is too large at {token_count} tokens. Please rephrase the question asked so that the output is within the token limit."
                        )

            return results
        except Exception as e:
            raise ToolError("batch", f"Error executing tool batch: {e}", e)

    def build_tool_dict(self, tools: List[Callable]) -> Dict[str, Callable]:
        """Build a dictionary mapping tool names to functions."""
        return {tool.__name__: tool for tool in tools}

    def validate_post_tool_function(
        self, post_tool_function: Optional[Callable]
    ) -> None:
        """Validate the post-tool function signature."""
        if post_tool_function:
            verify_post_tool_function(post_tool_function)
