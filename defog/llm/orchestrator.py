"""Agent orchestration system for hierarchical task delegation."""

import asyncio
import logging
import time
import traceback
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect

logger = logging.getLogger(__name__)

from .utils import chat_async
from .providers.base import BaseLLMProvider, LLMResponse
from .utils_logging import orch_logger


class ExecutionMode(Enum):
    """Execution mode for subagent tasks."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


@dataclass
class SubAgentTask:
    """Task definition for a subagent."""

    agent_id: str
    task_description: str
    context: Optional[Dict[str, Any]] = None
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    dependencies: Optional[List[str]] = None  # IDs of tasks that must complete first
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_timeout: Optional[float] = None


@dataclass
class SubAgentPlan:
    """Plan for creating a dynamic subagent."""

    agent_id: str
    system_prompt: str
    task_description: str
    tools: List[str]  # Tool names from available tools
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    dependencies: Optional[List[str]] = None


@dataclass
class SubAgentResult:
    """Result from a subagent execution."""

    agent_id: str
    task_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    error_trace: Optional[str] = None
    partial_results: Optional[List[Any]] = field(default_factory=list)
    execution_time: Optional[float] = None


class Agent:
    """Base agent class with provider, model, and tools."""

    def __init__(
        self,
        agent_id: str,
        provider: Union[str, BaseLLMProvider],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        **kwargs,
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.kwargs = kwargs  # Additional params for chat_async

    async def process(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Process messages with optional context injection."""
        # Prepare messages with system prompt and context
        final_messages = messages.copy()

        # Check if first message is already a system message
        has_system_message = (
            len(final_messages) > 0 and final_messages[0].get("role") == "system"
        )

        if self.system_prompt:
            system_content = self.system_prompt
            if context:
                system_content += f"\n\nContext:\n{json.dumps(context, indent=2)}"

            if has_system_message:
                # Update existing system message
                final_messages[0]["content"] = (
                    system_content + "\n\n" + final_messages[0]["content"]
                )
            else:
                # Insert system message at the beginning
                final_messages.insert(0, {"role": "system", "content": system_content})

        # Merge kwargs
        call_kwargs = {**self.kwargs, **kwargs}

        response = await chat_async(
            provider=self.provider,
            model=self.model,
            messages=final_messages,
            tools=self.tools,
            **call_kwargs,
        )

        return response


class AgentOrchestrator:
    """Orchestrator for managing main agent and subagents with task delegation.

    The orchestrator enables hierarchical task execution where a main agent can:
    1. Delegate tasks to pre-registered subagents
    2. Dynamically create specialized subagents based on task requirements

    Dynamic subagent creation uses a separate LLM (subagent_designer_provider/model)
    to analyze complex tasks and design optimal subagent architectures.
    """

    def __init__(
        self,
        main_agent: Agent,
        max_parallel_tasks: int = 5,
        available_tools: Optional[List[Callable]] = None,
        subagent_provider: Optional[str] = None,
        subagent_model: Optional[str] = None,
        subagent_designer_provider: str = "anthropic",
        subagent_designer_model: str = "claude-opus-4-20250514",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        retry_timeout: Optional[float] = None,
        fallback_model: Optional[str] = None,
    ):
        """Initialize the AgentOrchestrator.

        Args:
            main_agent: The primary agent that handles user interactions and orchestration
            max_parallel_tasks: Maximum number of subagent tasks to run concurrently
            available_tools: List of tool functions that can be assigned to dynamically created subagents
            subagent_provider: Default LLM provider for subagents (defaults to main agent's provider)
            subagent_model: Default LLM model for subagents (defaults to main agent's model)
            subagent_designer_provider: LLM provider specifically for analyzing tasks and designing subagent architectures
            subagent_designer_model: LLM model for subagent design (can be more powerful than main agent)
            max_retries: Maximum retry attempts for failed tasks
            retry_delay: Initial delay between retries in seconds
            retry_backoff: Exponential backoff factor for retries
            retry_timeout: Maximum time in seconds for a task before timeout
            fallback_model: Alternative model to use if primary model fails
        """
        self.main_agent = main_agent
        self.subagents = {}
        self.max_parallel_tasks = max_parallel_tasks
        self.task_results: Dict[str, SubAgentResult] = {}

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.retry_timeout = retry_timeout
        self.fallback_model = fallback_model

        # For dynamic subagent creation
        self.available_tools = available_tools or []
        self.tool_registry = {
            self._get_tool_name(tool): tool for tool in self.available_tools
        }
        self.subagent_provider = subagent_provider or main_agent.provider
        self.subagent_model = subagent_model or main_agent.model
        self.subagent_designer_provider = subagent_designer_provider
        self.subagent_designer_model = subagent_designer_model

        # Circuit breaker state
        self.failure_count = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60

        # Add delegation tools to main agent
        self._add_delegation_tool()
        if self.available_tools:
            self._add_dynamic_planning_tool()

    def _add_delegation_tool(self):
        """Add delegation tool to main agent's toolset."""
        from pydantic import BaseModel, Field
        from typing import List as ListType

        class DelegationRequest(BaseModel):
            """Request to delegate tasks to subagents."""

            tasks: ListType[Dict[str, Any]] = Field(
                description="List of tasks to delegate. Each task should have: agent_id, task_description, context (optional), execution_mode (optional), dependencies (optional), max_retries (optional), retry_delay (optional), retry_backoff (optional), retry_timeout (optional)"
            )

        async def delegate_to_subagents(input: DelegationRequest) -> Dict[str, Any]:
            """Delegate tasks to subagents based on the request."""
            tasks = []
            for task_data in input.tasks:
                task = SubAgentTask(
                    agent_id=task_data["agent_id"],
                    task_description=task_data["task_description"],
                    context=task_data.get("context"),
                    execution_mode=ExecutionMode(
                        task_data.get("execution_mode", "parallel")
                    ),
                    dependencies=task_data.get("dependencies"),
                    max_retries=task_data.get("max_retries", self.max_retries),
                    retry_delay=task_data.get("retry_delay", self.retry_delay),
                    retry_backoff=task_data.get("retry_backoff", self.retry_backoff),
                    retry_timeout=task_data.get("retry_timeout", self.retry_timeout),
                )
                tasks.append(task)

            results = await self._execute_subagent_tasks(tasks)

            # Format results for main agent and calculate total costs
            formatted_results = {
                "task_results": {},
                "total_cost_breakdown": {
                    "subagent_costs_in_cents": 0,
                    "tool_costs_in_cents": 0,
                    "total_cost_in_cents": 0,
                },
            }

            for result in results:
                formatted_results["task_results"][result.task_id] = {
                    "agent_id": result.agent_id,
                    "success": result.success,
                    "result": result.result,
                    "error": result.error,
                    "metadata": result.metadata,
                }

                # Add up costs if available
                if result.metadata:
                    subagent_cost = result.metadata.get("cost_in_cents", 0) or 0
                    tool_cost = result.metadata.get("tool_costs_in_cents", 0) or 0

                    formatted_results["total_cost_breakdown"][
                        "subagent_costs_in_cents"
                    ] += subagent_cost
                    formatted_results["total_cost_breakdown"][
                        "tool_costs_in_cents"
                    ] += tool_cost
                    formatted_results["total_cost_breakdown"][
                        "total_cost_in_cents"
                    ] += subagent_cost + tool_cost

            return formatted_results

        # Add to main agent's tools
        self.main_agent.tools.append(delegate_to_subagents)

    def register_subagent(self, agent: Agent):
        """Register a new subagent."""
        self.subagents[agent.agent_id] = agent

    def unregister_subagent(self, agent_id: str):
        """Unregister a subagent."""
        if agent_id in self.subagents:
            del self.subagents[agent_id]

    async def _execute_subagent_tasks(
        self, tasks: List[SubAgentTask]
    ) -> List[SubAgentResult]:
        """Execute subagent tasks with dependency and parallelism management."""
        results = []
        completed_tasks = set()

        # Group tasks by execution order considering dependencies
        task_groups = self._group_tasks_by_dependencies(tasks)

        for group in task_groups:
            # Within each group, respect execution modes
            parallel_tasks = []
            sequential_tasks = []

            for task in group:
                if task.execution_mode == ExecutionMode.PARALLEL:
                    parallel_tasks.append(task)
                else:
                    sequential_tasks.append(task)

            # Execute parallel tasks
            if parallel_tasks:
                parallel_results = await self._execute_parallel_tasks(parallel_tasks)
                results.extend(parallel_results)
                for result in parallel_results:
                    completed_tasks.add(result.task_id)

            # Execute sequential tasks
            for task in sequential_tasks:
                result = await self._execute_single_task(task)
                results.append(result)
                completed_tasks.add(result.task_id)

        return results

    def _group_tasks_by_dependencies(
        self, tasks: List[SubAgentTask]
    ) -> List[List[SubAgentTask]]:
        """Group tasks by dependency order."""
        # Simple topological sort
        task_map = {f"task_{i}": task for i, task in enumerate(tasks)}
        agent_to_task_id = {}

        for task_id, task in task_map.items():
            task.task_id = task_id
            agent_to_task_id[task.agent_id] = task_id

        # Convert agent_id dependencies to task_id dependencies
        for task in tasks:
            if task.dependencies:
                task_dependencies = []
                for dep in task.dependencies:
                    if dep in agent_to_task_id:
                        task_dependencies.append(agent_to_task_id[dep])
                    else:
                        # If dependency is already a task_id, keep it
                        if dep.startswith("task_"):
                            task_dependencies.append(dep)
                        else:
                            logger.warning(
                                f"Unknown dependency '{dep}' for task {task.task_id}"
                            )
                task.dependencies = task_dependencies

        groups = []
        completed = set()

        while len(completed) < len(tasks):
            current_group = []
            for task_id, task in task_map.items():
                if task_id not in completed:
                    # Check if all dependencies are completed
                    deps = task.dependencies or []
                    if all(dep in completed for dep in deps):
                        current_group.append(task)

            if not current_group:
                # Circular dependency or invalid dependency
                raise ValueError("Invalid task dependencies detected")

            groups.append(current_group)
            for task in current_group:
                completed.add(task.task_id)

        return groups

    async def _execute_parallel_tasks(
        self, tasks: List[SubAgentTask]
    ) -> List[SubAgentResult]:
        """Execute multiple tasks in parallel with retry logic for failed tasks."""
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def execute_with_limit(task):
            async with semaphore:
                return await self._execute_single_task(task)

        # Initial execution
        results = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks], return_exceptions=True
        )

        # Process results and identify failures
        processed_results = []
        failed_tasks = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle gather exceptions
                failed_result = SubAgentResult(
                    agent_id=tasks[i].agent_id,
                    task_id=tasks[i].task_id,
                    success=False,
                    result=None,
                    error=str(result),
                    error_type=self._classify_error(result),
                    error_trace=traceback.format_exc(),
                )
                processed_results.append(failed_result)
                failed_tasks.append(tasks[i])
            elif not result.success:
                # Task completed but failed
                processed_results.append(result)
                if self._should_retry_error(result.error_type or "UNKNOWN_ERROR"):
                    failed_tasks.append(tasks[i])
            else:
                # Success
                processed_results.append(result)

        # Retry failed tasks with reduced concurrency if any failures occurred
        if failed_tasks and len(failed_tasks) < len(tasks):
            logger.info(
                f"Retrying {len(failed_tasks)} failed parallel tasks with reduced concurrency"
            )

            # Reduce concurrency for retries to avoid cascading failures
            reduced_concurrency = max(1, self.max_parallel_tasks // 2)
            retry_semaphore = asyncio.Semaphore(reduced_concurrency)

            async def retry_with_limit(task):
                async with retry_semaphore:
                    return await self._execute_single_task(task)

            # Wait a bit before retrying to let any temporary issues resolve
            await asyncio.sleep(self.retry_delay)

            retry_results = await asyncio.gather(
                *[retry_with_limit(task) for task in failed_tasks],
                return_exceptions=True,
            )

            # Replace failed results with retry results
            for i, retry_result in enumerate(retry_results):
                failed_task = failed_tasks[i]

                # Find the original result index
                for j, original_result in enumerate(processed_results):
                    if (
                        original_result.agent_id == failed_task.agent_id
                        and original_result.task_id == failed_task.task_id
                    ):
                        if isinstance(retry_result, Exception):
                            # Still failed after retry
                            processed_results[j] = SubAgentResult(
                                agent_id=failed_task.agent_id,
                                task_id=failed_task.task_id,
                                success=False,
                                result=None,
                                error=str(retry_result),
                                error_type=self._classify_error(retry_result),
                                retry_count=1,
                                error_trace=traceback.format_exc(),
                            )
                        else:
                            # Use retry result
                            processed_results[j] = retry_result
                        break

        return processed_results

    async def _execute_single_task(self, task: SubAgentTask) -> SubAgentResult:
        """Execute a single subagent task with retry logic and exponential backoff."""
        if task.agent_id not in self.subagents:
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=False,
                result=None,
                error=f"Subagent '{task.agent_id}' not found",
                error_type="AGENT_NOT_FOUND",
            )

        agent = self.subagents[task.agent_id]

        # Use task-specific retry config or fall back to orchestrator defaults
        max_retries = getattr(task, "max_retries", self.max_retries)
        retry_delay = getattr(task, "retry_delay", self.retry_delay)
        retry_backoff = getattr(task, "retry_backoff", self.retry_backoff)
        retry_timeout = getattr(task, "retry_timeout", self.retry_timeout)

        start_time = time.time()
        partial_results = []
        last_error = None
        last_error_trace = None
        retry_count = 0

        # Log task start
        orch_logger.log_task_execution_start(
            task.agent_id, task.task_id, task.execution_mode.value
        )

        for attempt in range(max_retries + 1):
            try:
                # Check timeout
                if retry_timeout and (time.time() - start_time) > retry_timeout:
                    error_msg = f"Task timeout exceeded: {retry_timeout}s"
                    logger.warning(
                        f"Task {task.task_id} timed out after {retry_timeout}s"
                    )
                    return SubAgentResult(
                        agent_id=task.agent_id,
                        task_id=task.task_id,
                        success=False,
                        result=None,
                        error=error_msg,
                        error_type="TIMEOUT",
                        retry_count=retry_count,
                        partial_results=partial_results,
                        execution_time=time.time() - start_time,
                    )

                # Log retry attempt
                if attempt > 0:
                    orch_logger.log_retry_attempt(
                        task.task_id,
                        attempt,
                        max_retries,
                        (
                            retry_delay * (retry_backoff ** (attempt - 1))
                            if attempt < max_retries
                            else None
                        ),
                    )

                # Prepare messages for subagent
                messages = [{"role": "user", "content": task.task_description}]

                # Execute task
                response = await agent.process(messages, context=task.context)

                # Success - reset circuit breaker
                self.failure_count = 0

                # Calculate total cost including tool costs
                total_cost = response.cost_in_cents or 0
                tool_costs = 0

                # Add cost from each tool if available
                if response.tool_outputs:
                    for tool_output in response.tool_outputs:
                        # Check if tool has associated cost
                        if (
                            isinstance(tool_output.get("result"), dict)
                            and "cost_in_cents" in tool_output["result"]
                        ):
                            tool_costs += tool_output["result"]["cost_in_cents"]

                total_cost += tool_costs

                result = SubAgentResult(
                    agent_id=task.agent_id,
                    task_id=task.task_id,
                    success=True,
                    result=response.content,
                    retry_count=retry_count,
                    partial_results=partial_results,
                    execution_time=time.time() - start_time,
                    metadata={
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "total_tokens": response.input_tokens + response.output_tokens,
                        "cost_in_cents": response.cost_in_cents,
                        "tool_costs_in_cents": tool_costs,
                        "total_cost_in_cents": total_cost,
                        "tool_outputs": response.tool_outputs,
                    },
                )

                # Log task completion with cost breakdown
                enhanced_metadata = result.metadata.copy() if result.metadata else {}
                if "tool_costs_in_cents" in enhanced_metadata:
                    enhanced_metadata["cost_breakdown"] = {
                        "llm_cost": enhanced_metadata.get("cost_in_cents", 0),
                        "tool_cost": enhanced_metadata.get("tool_costs_in_cents", 0),
                        "total_cost": enhanced_metadata.get("total_cost_in_cents", 0),
                    }

                orch_logger.log_task_execution_complete(
                    task.agent_id,
                    task.task_id,
                    True,
                    result.result,
                    None,
                    enhanced_metadata,
                )

                return result

            except Exception as e:
                retry_count = attempt
                last_error = str(e)
                last_error_trace = traceback.format_exc()
                error_type = self._classify_error(e)

                # Log error details
                orch_logger.log_error_detail(error_type, last_error)

                # Store partial results if any
                if hasattr(e, "partial_result"):
                    partial_results.append(e.partial_result)

                # Check if we should retry based on error type
                if not self._should_retry_error(error_type) or attempt >= max_retries:
                    break

                # Circuit breaker check
                self.failure_count += 1
                if self.failure_count >= self.circuit_breaker_threshold:
                    logger.error(
                        f"Circuit breaker triggered after {self.failure_count} failures"
                    )
                    break

                # Try model fallback on the last retry for model errors
                if (
                    error_type == "MODEL_ERROR"
                    and attempt == max_retries - 1
                    and self.fallback_model
                ):
                    logger.info(
                        f"Attempting model fallback to {self.fallback_model} for task {task.task_id}"
                    )
                    try:
                        result = await self._execute_with_fallback_model(task, agent)
                        if result.success:
                            return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model also failed: {fallback_error}")
                        last_error = str(fallback_error)
                        last_error_trace = traceback.format_exc()

                # Wait before retry with exponential backoff
                if attempt < max_retries:
                    wait_time = retry_delay * (retry_backoff**attempt)
                    logger.info(f"Waiting {wait_time:.2f}s before retry")
                    await asyncio.sleep(wait_time)

        # All retries exhausted - try task decomposition as final fallback
        if (
            self.available_tools and len(task.task_description) > 100
        ):  # Only for complex tasks
            logger.info(
                f"Attempting task decomposition as final fallback for {task.task_id}"
            )
            try:
                decomp_result = await self._execute_with_decomposition(task)
                if decomp_result.success:
                    decomp_result.retry_count = retry_count
                    decomp_result.execution_time = time.time() - start_time
                    return decomp_result
            except Exception as decomp_error:
                logger.warning(f"Task decomposition failed: {decomp_error}")

        # All fallbacks exhausted
        final_result = SubAgentResult(
            agent_id=task.agent_id,
            task_id=task.task_id,
            success=False,
            result=None,
            error=last_error,
            error_type=(
                self._classify_error(Exception(last_error)) if last_error else "UNKNOWN"
            ),
            retry_count=retry_count,
            error_trace=last_error_trace,
            partial_results=partial_results,
            execution_time=time.time() - start_time,
        )

        # Log task failure
        orch_logger.log_task_execution_complete(
            task.agent_id,
            task.task_id,
            False,
            None,
            last_error,
            {"retry_count": retry_count, "execution_time": time.time() - start_time},
        )

        return final_result

    async def process(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Process messages through the main agent with subagent delegation capability."""
        return await self.main_agent.process(messages, **kwargs)

    def clear_all_memory(self):
        """Clear memory for all agents."""
        self.main_agent.clear_memory()
        for agent in self.subagents.values():
            agent.clear_memory()

    def _get_tool_name(self, tool: Callable) -> str:
        """Extract tool name from function."""
        return tool.__name__

    def _get_tool_description(self, tool: Callable) -> str:
        """Extract tool description from function docstring."""
        return inspect.getdoc(tool) or "No description available"

    def _add_dynamic_planning_tool(self):
        """Add tool for dynamically planning and creating subagents.

        This tool uses a dedicated LLM (subagent_designer_provider/model) to:
        1. Analyze complex user requests
        2. Design specialized subagent architectures
        3. Assign appropriate tools to each subagent
        4. Determine execution order and dependencies
        5. Create and execute the subagents automatically

        The subagent designer LLM can be different from the main agent's model,
        allowing you to use a more powerful model for complex task decomposition
        while keeping the main agent lightweight.
        """
        from pydantic import BaseModel, Field

        class DynamicPlanningRequest(BaseModel):
            """Request to dynamically create subagents and plan tasks."""

            user_request: str = Field(
                description="The user's original request that needs to be broken down"
            )
            analysis: str = Field(
                description="Your analysis of what needs to be done and why certain subagents are needed"
            )

        # Define the response format for planning
        class SubAgentPlanResponse(BaseModel):
            agent_id: str
            system_prompt: str
            task_description: str
            tools: List[str]
            execution_mode: str = "parallel"
            dependencies: List[str] = []

        class PlanningResponse(BaseModel):
            subagent_plans: List[SubAgentPlanResponse]
            reasoning: str

        async def plan_and_create_subagents(
            input: DynamicPlanningRequest,
        ) -> Dict[str, Any]:
            """Dynamically create subagents based on the user's request."""
            # Log the start of the request
            orch_logger.log_request_start(input.user_request)
            orch_logger.log_planning_analysis(input.analysis)

            # Use the subagent designer LLM to analyze the request and create subagent plans
            subagent_designer_messages = [
                {
                    "role": "system",
                    "content": f"""You are a planning agent that designs specialized subagents for complex tasks.
                        
Available tools that can be assigned to subagents:
{self._format_available_tools()}

Your task:
1. Analyze the user's request
2. Design specialized subagents with specific roles
3. Assign appropriate tools to each subagent
4. Create tasks for each subagent
5. Determine execution order (parallel or sequential)

Return a JSON object with this structure:
{{
    "subagent_plans": [
        {{
            "agent_id": "unique_id",
            "system_prompt": "Detailed prompt explaining the agent's role and expertise",
            "task_description": "Specific task for this agent",
            "tools": ["tool_name1", "tool_name2"],
            "execution_mode": "parallel" or "sequential",
            "dependencies": []  // List of agent_ids this depends on
        }}
    ],
    "reasoning": "Explanation of why these subagents and this structure"
}}""",
                },
                {
                    "role": "user",
                    "content": f"User request: {input.user_request}\n\nAnalysis: {input.analysis}",
                },
            ]

            orch_logger.log_llm_call(
                self.subagent_designer_provider,
                self.subagent_designer_model,
                "Planning",
            )

            try:
                planning_response = await chat_async(
                    provider=self.subagent_designer_provider,
                    model=self.subagent_designer_model,
                    messages=subagent_designer_messages,
                    temperature=0.2,
                    response_format=PlanningResponse,
                )

                # The response is already a structured Pydantic object
                plan_data = planning_response.content

                # Log the subagent plans
                orch_logger.log_subagent_plans(
                    [plan.model_dump() for plan in plan_data.subagent_plans],
                    plan_data.reasoning,
                )

                # Create subagents based on the plan
                created_agents = []
                tasks = []

                for plan in plan_data.subagent_plans:
                    agent_id = plan.agent_id

                    # Get tools for this subagent
                    agent_tools = []
                    for tool_name in plan.tools:
                        if tool_name in self.tool_registry:
                            agent_tools.append(self.tool_registry[tool_name])

                    # Filter out planning tools to prevent recursive agent creation
                    agent_tools = [
                        tool
                        for tool in agent_tools
                        if tool.__name__
                        not in ["plan_and_create_subagents", "delegate_to_subagents"]
                    ]

                    # Create the subagent
                    subagent = Agent(
                        agent_id=agent_id,
                        provider=self.subagent_provider,
                        model=self.subagent_model,
                        system_prompt=plan.system_prompt,
                        tools=agent_tools,
                    )

                    # Register the subagent
                    self.register_subagent(subagent)
                    created_agents.append(agent_id)

                    # Create task for this subagent
                    task = SubAgentTask(
                        agent_id=agent_id,
                        task_description=plan.task_description,
                        execution_mode=ExecutionMode(plan.execution_mode),
                        dependencies=plan.dependencies,
                    )
                    tasks.append(task)

                # Execute the tasks
                results = await self._execute_subagent_tasks(tasks)

                # Format results and calculate total costs
                formatted_results = {
                    "created_agents": created_agents,
                    "reasoning": plan_data.reasoning,
                    "task_results": {},
                    "total_cost_breakdown": {
                        "planning_cost_in_cents": planning_response.cost_in_cents or 0,
                        "subagent_costs_in_cents": 0,
                        "tool_costs_in_cents": 0,
                        "total_cost_in_cents": planning_response.cost_in_cents or 0,
                    },
                }

                # Aggregate costs from all subagents
                for result in results:
                    formatted_results["task_results"][result.task_id] = {
                        "agent_id": result.agent_id,
                        "success": result.success,
                        "result": result.result,
                        "error": result.error,
                        "metadata": result.metadata,
                    }

                    # Add up costs if available
                    if result.metadata:
                        subagent_cost = result.metadata.get("cost_in_cents", 0) or 0
                        tool_cost = result.metadata.get("tool_costs_in_cents", 0) or 0

                        formatted_results["total_cost_breakdown"][
                            "subagent_costs_in_cents"
                        ] += subagent_cost
                        formatted_results["total_cost_breakdown"][
                            "tool_costs_in_cents"
                        ] += tool_cost
                        formatted_results["total_cost_breakdown"][
                            "total_cost_in_cents"
                        ] += subagent_cost + tool_cost

                # Clean up created subagents after execution
                for agent_id in created_agents:
                    self.unregister_subagent(agent_id)

                # Log orchestration completion
                orch_logger.log_orchestration_complete(formatted_results)

                return formatted_results

            except Exception as e:
                logger.error(f"Error in planning LLM call: {e}", exc_info=True)
                return {"error": f"Planning LLM call failed: {str(e)}"}

        # Add to main agent's tools
        self.main_agent.tools.append(plan_and_create_subagents)

    def _format_available_tools(self) -> str:
        """Format available tools for the planning prompt."""
        tool_descriptions = []
        for name, tool in self.tool_registry.items():
            desc = self._get_tool_description(tool)
            tool_descriptions.append(f"- {name}: {desc}")
        return "\n".join(tool_descriptions)

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for retry decisions."""
        error_str = str(error).lower()

        # Network/timeout errors - retryable
        if any(
            term in error_str for term in ["timeout", "connection", "network", "socket"]
        ):
            return "NETWORK_ERROR"

        # Rate limiting - retryable with longer delay
        if any(term in error_str for term in ["rate limit", "quota", "throttle"]):
            return "RATE_LIMIT"

        # Authentication - usually not retryable
        if any(
            term in error_str
            for term in ["auth", "permission", "unauthorized", "forbidden"]
        ):
            return "AUTH_ERROR"

        # Model-specific errors - might be retryable with fallback
        if any(term in error_str for term in ["model", "overload", "capacity"]):
            return "MODEL_ERROR"

        # Input validation - not retryable
        if any(term in error_str for term in ["invalid", "malformed", "validation"]):
            return "VALIDATION_ERROR"

        # Generic server errors - retryable
        if any(
            term in error_str for term in ["server error", "500", "502", "503", "504"]
        ):
            return "SERVER_ERROR"

        return "UNKNOWN_ERROR"

    def _should_retry_error(self, error_type: str) -> bool:
        """Determine if an error type should be retried."""
        retryable_errors = {
            "NETWORK_ERROR",
            "RATE_LIMIT",
            "SERVER_ERROR",
            "MODEL_ERROR",
            "UNKNOWN_ERROR",
        }
        return error_type in retryable_errors

    async def _execute_with_fallback_model(
        self, task: SubAgentTask, agent: Agent
    ) -> SubAgentResult:
        """Execute task with fallback model."""
        original_model = agent.model
        agent.model = self.fallback_model

        try:
            messages = [{"role": "user", "content": task.task_description}]
            response = await agent.process(messages, context=task.context)

            # Calculate total cost including tool costs
            total_cost = response.cost_in_cents or 0
            tool_costs = 0

            # Add cost from each tool if available
            if response.tool_outputs:
                for tool_output in response.tool_outputs:
                    # Check if tool has associated cost
                    if (
                        isinstance(tool_output.get("result"), dict)
                        and "cost_in_cents" in tool_output["result"]
                    ):
                        tool_costs += tool_output["result"]["cost_in_cents"]

            total_cost += tool_costs

            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=True,
                result=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens,
                    "cost_in_cents": response.cost_in_cents,
                    "tool_costs_in_cents": tool_costs,
                    "total_cost_in_cents": total_cost,
                    "tool_outputs": response.tool_outputs,
                    "fallback_model_used": self.fallback_model,
                },
            )
        finally:
            # Restore original model
            agent.model = original_model

    async def _decompose_task(self, task: SubAgentTask) -> List[SubAgentTask]:
        """Decompose a complex task into simpler subtasks."""
        if not self.available_tools:
            return [task]  # Can't decompose without planning capability

        decomposition_messages = [
            {
                "role": "system",
                "content": """You are a task decomposition agent. Break down complex tasks into simpler, more manageable subtasks.
                
Guidelines:
1. Create 2-4 simpler subtasks from the original task
2. Each subtask should be self-contained and specific
3. Subtasks should be easier to execute than the original
4. Maintain logical dependencies between subtasks

Return a JSON array of subtask descriptions.""",
            },
            {
                "role": "user",
                "content": f"Decompose this task: {task.task_description}",
            },
        ]

        try:
            from pydantic import BaseModel
            from typing import List as ListType

            class TaskDecomposition(BaseModel):
                subtasks: ListType[str]
                reasoning: str

            response = await chat_async(
                provider=self.subagent_designer_provider,
                model=self.subagent_designer_model,
                messages=decomposition_messages,
                temperature=0.2,
                response_format=TaskDecomposition,
            )

            decomposition = response.content
            subtasks = []

            for i, subtask_desc in enumerate(decomposition.subtasks):
                subtask = SubAgentTask(
                    agent_id=task.agent_id,
                    task_description=subtask_desc,
                    context=task.context,
                    execution_mode=ExecutionMode.PARALLEL,
                    dependencies=[f"{task.task_id}_subtask_{i - 1}"] if i > 0 else None,
                    max_retries=max(
                        1, task.max_retries // 2
                    ),  # Reduce retries for subtasks
                )
                subtask.task_id = f"{task.task_id}_subtask_{i}"
                subtasks.append(subtask)

            logger.info(f"Task decomposed into {len(subtasks)} subtasks")

            return subtasks

        except Exception as e:
            logger.warning(f"Task decomposition failed: {e}")
            return [task]  # Return original task if decomposition fails

    async def _execute_with_decomposition(self, task: SubAgentTask) -> SubAgentResult:
        """Execute task by decomposing it into subtasks."""
        subtasks = await self._decompose_task(task)

        if len(subtasks) == 1:
            # Decomposition didn't help, return failure
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=False,
                result=None,
                error="Task decomposition did not simplify the task",
                error_type="DECOMPOSITION_FAILED",
            )

        # Execute subtasks sequentially
        subtask_results = []
        combined_result = []

        for subtask in subtasks:
            result = await self._execute_single_task(subtask)
            subtask_results.append(result)

            if result.success:
                combined_result.append(result.result)
            else:
                # If any subtask fails, the whole task fails
                return SubAgentResult(
                    agent_id=task.agent_id,
                    task_id=task.task_id,
                    success=False,
                    result=None,
                    error=f"Subtask {subtask.task_id} failed: {result.error}",
                    error_type="SUBTASK_FAILED",
                    partial_results=[r.result for r in subtask_results if r.success],
                )

        # All subtasks succeeded
        return SubAgentResult(
            agent_id=task.agent_id,
            task_id=task.task_id,
            success=True,
            result="\n".join(str(r) for r in combined_result),
            metadata={
                "decomposed": True,
                "subtask_count": len(subtasks),
                "subtask_results": [r.metadata for r in subtask_results if r.metadata],
            },
        )


# Backward compatibility alias - AgentOrchestrator is the class name
