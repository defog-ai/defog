import inspect
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
    ):
        self.max_consecutive_errors = max_consecutive_errors
        self.tool_budget = tool_budget.copy() if tool_budget else None
        self.tool_usage = {}
        self.image_result_keys = image_result_keys
        self.tool_output_max_tokens = (
            tool_output_max_tokens if tool_output_max_tokens is not None else 10000
        )
        logger.debug(
            f"ToolHandler initialized with budget: {self.tool_budget}, max_tokens: {self.tool_output_max_tokens}"
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

    async def execute_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_dict: Dict[str, Callable],
        post_tool_function: Optional[Callable] = None,
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
                    )
                else:
                    post_tool_function(
                        function_name=tool_name,
                        input_args=args,
                        tool_result=result,
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

                    # Use execute_tool_call which handles budget tracking
                    result = await self.execute_tool_call(
                        func_name, func_args, tool_dict, post_tool_function
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

                    try:
                        if inspect.iscoroutinefunction(post_tool_function):
                            await post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
                            )
                        else:
                            post_tool_function(
                                function_name=func_name,
                                input_args=func_args,
                                tool_result=result,
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
