# Agent Orchestration

This document covers the agent orchestration capabilities in defog-python, including multi-agent coordination, task delegation, and visualization of execution flows.

## Basic Agent Orchestrator

The `AgentOrchestrator` provides multi-agent coordination and task delegation:

```python
from defog.llm import Agent, AgentOrchestrator, ExecutionMode

# Create main orchestrator agent
main_agent = Agent(
    agent_id="orchestrator",
    provider="anthropic",
    model="claude-3-5-sonnet",
    system_prompt="You are an orchestrator that coordinates tasks between specialized agents."
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    main_agent=main_agent,
    max_parallel_tasks=5,
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0
)
```

## Creating and Registering Subagents

You can create specialized subagents and register them with the orchestrator:

```python
# Create specialized subagents
calculator_agent = Agent(
    agent_id="calculator",
    provider="openai",
    model="gpt-4o",
    system_prompt="You are a calculator assistant. Perform mathematical calculations.",
    tools=[calculator_tool]
)

text_processor_agent = Agent(
    agent_id="text_processor",
    provider="anthropic",
    model="claude-3-5-haiku",
    system_prompt="You are a text processing assistant.",
    tools=[text_processor_tool]
)

# Register subagents
orchestrator.register_subagent(calculator_agent)
orchestrator.register_subagent(text_processor_agent)
```

## Dynamic Agent Creation

The orchestrator can dynamically create subagents based on task requirements:

```python
# Create orchestrator with available tools
orchestrator = AgentOrchestrator(
    main_agent=main_agent,
    available_tools=[calculator_tool, text_processor_tool, data_formatter_tool],
    subagent_provider="anthropic",
    subagent_model="claude-3-5-haiku",
    subagent_designer_provider="anthropic",  # LLM provider for designing subagents
    subagent_designer_model="claude-3-5-sonnet"  # Model for analyzing tasks and creating subagents
)

# The orchestrator will automatically create subagents as needed
response = await orchestrator.process([
    {
        "role": "user",
        "content": "Calculate 25 * 4 and format the result as JSON"
    }
])
```

## Task Delegation

The main agent can delegate tasks to subagents:

```python
# The main agent uses the delegate_to_subagents tool
messages = [
    {
        "role": "user",
        "content": "Please calculate 15 * 7 and count the words in 'Hello world'"
    }
]

response = await orchestrator.process(messages)
print(response.content)
```

## Execution Modes

Tasks can be executed in parallel or sequential mode:

```python
from defog.llm import SubAgentTask, ExecutionMode

# Define tasks
tasks = [
    SubAgentTask(
        agent_id="calculator",
        task_description="Calculate 100 * 50",
        execution_mode=ExecutionMode.PARALLEL
    ),
    SubAgentTask(
        agent_id="text_processor",
        task_description="Count words in 'The quick brown fox'",
        execution_mode=ExecutionMode.PARALLEL
    ),
    SubAgentTask(
        agent_id="formatter",
        task_description="Format previous results as JSON",
        execution_mode=ExecutionMode.SEQUENTIAL,
        dependencies=["calculator", "text_processor"]
    )
]
```

## Error Handling and Retries

The orchestrator includes robust error handling:

```python
# Configure retry behavior
orchestrator = AgentOrchestrator(
    main_agent=main_agent,
    max_retries=3,              # Maximum retry attempts
    retry_delay=1.0,            # Initial delay between retries
    retry_backoff=2.0,          # Exponential backoff factor
    retry_timeout=30.0,         # Timeout for individual tasks
    fallback_model="gpt-4o"     # Fallback model for errors
)
```

## Complete Example

Here's a complete example showing dynamic agent creation and task execution:

```python
import asyncio
from defog.llm import Agent, AgentOrchestrator

# Define tools
async def web_search_tool(query: str) -> str:
    # Implementation
    return f"Search results for: {query}"

async def summarize_tool(text: str) -> str:
    # Implementation
    return f"Summary: {text[:100]}..."

async def main():
    # Create main orchestrator
    main_agent = Agent(
        agent_id="main_orchestrator",
        provider="anthropic",
        model="claude-3-5-sonnet",
        system_prompt="""You are a research orchestrator. 
        Analyze user requests and create specialized agents to handle different aspects.
        Use the plan_and_create_subagents tool for complex tasks."""
    )
    
    # Create orchestrator with tools
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search_tool, summarize_tool],
        subagent_provider="anthropic",
        subagent_model="claude-3-5-haiku",
        subagent_designer_provider="anthropic",
        subagent_designer_model="claude-3-5-sonnet"
    )
    
    # Process complex request
    response = await orchestrator.process([
        {
            "role": "user",
            "content": "Research the latest developments in quantum computing and provide a summary"
        }
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Cost: ${response.cost_in_cents / 100}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Visualizing Orchestrator Execution

The defog-python library includes utilities to visualize the execution flow of the orchestrator and its subagents:

### ASCII Flowchart Generation

```python
from defog.llm.utils_orchestrator_viz import generate_orchestrator_flowchart, generate_detailed_tool_trace

# After running the orchestrator
response = await orchestrator.process(messages)

# Generate ASCII flowchart
flowchart = generate_orchestrator_flowchart(response.tool_outputs)
print(flowchart)
```

This will produce output like:
```
┌─────────────────────────┐
│   Agent Orchestrator    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  plan_and_create_       │
│      subagents          │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌───────────────────────┐  ┌───────────────────────┐
│ Team Batting Analysis │  │ Individual Batting    │
│        Agent          │  │ Performance Agent     │
├───────────────────────┤  ├───────────────────────┤
│ Tools:                │  │ Tools:                │
│  • cricket_sql_query  │  │  • cricket_sql_query  │
│  • cricket_sql_query  │  │  • cricket_sql_query  │
│  • cricket_sql_query  │  └───────────────────────┘
└───────────────────────┘

Summary:
────────────────────────────────────────
Total Subagents Created: 2
Total Tool Calls: 6
Execution Mode(s): parallel
```

### Detailed Tool Trace

```python
# Generate detailed trace of all tool calls
trace = generate_detailed_tool_trace(response.tool_outputs)
print(trace)
```

This will show a hierarchical trace of all tool calls:
```
Orchestrator Execution Trace
==================================================

1. plan_and_create_subagents
   └─ team_batting_analysis_agent [✓]
       ├─ cricket_sql_query (question: List each team's total runs...) [✓]
       ├─ cricket_sql_query (question: Calculate average runs per...) [✓]
       └─ Tokens: 2556 ($0.01)
   └─ individual_batting_performance_agent [✓]
       ├─ cricket_sql_query (question: List top 5 highest scores...) [✗]
       ├─ cricket_sql_query (question: List top 5 batsmen by total...) [✓]
       └─ Tokens: 3588 ($0.02)
```

### Integration Example

```python
# Complete example with visualization
async def run_with_visualization():
    # Create and configure orchestrator
    orchestrator = AgentOrchestrator(
        main_agent=main_agent,
        available_tools=[web_search, execute_python, cricket_sql_query],
        subagent_provider="openai",
        subagent_model="gpt-4o"
    )
    
    # Process request
    response = await orchestrator.process(messages)
    
    # Display results with visualization
    print(f"Response: {response.content}")
    
    print("\n=== Execution Flowchart ===")
    print(generate_orchestrator_flowchart(response.tool_outputs))
    
    print("\n=== Tool Trace ===")
    print(generate_detailed_tool_trace(response.tool_outputs))
```

## Cost Tracking

The orchestrator now tracks costs across all components including:
- Planning LLM costs (for dynamic agent creation)
- Subagent LLM costs
- Individual tool execution costs

### Cost Breakdown Structure

When using the orchestrator tools (`plan_and_create_subagents` or `delegate_to_subagents`), the response includes a `total_cost_breakdown`:

```python
# Example cost breakdown from orchestrator response
{
    "total_cost_breakdown": {
        "planning_cost_in_cents": 0.5,      # Cost of planning LLM call
        "subagent_costs_in_cents": 2.3,     # Total LLM costs from all subagents
        "tool_costs_in_cents": 0.35,        # Total costs from tool executions
        "total_cost_in_cents": 3.15         # Combined total cost
    }
}
```

### Implementing Tool Costs

To track tool costs, your tools should return a `cost_in_cents` field in their response:

```python
async def expensive_api_tool(input: ToolInput) -> Dict[str, Any]:
    """Tool that calls an expensive API."""
    result = await call_expensive_api(input.query)
    
    # Calculate cost based on API usage
    api_cost = calculate_api_cost(result)
    
    return {
        "result": result.data,
        "cost_in_cents": api_cost  # Tool cost will be aggregated
    }
```

### Accessing Cost Information

```python
# After orchestrator execution
response = await orchestrator.process(messages)

# Check tool outputs for cost breakdown
for tool_output in response.tool_outputs:
    if tool_output.get("name") == "plan_and_create_subagents":
        result = tool_output.get("result", {})
        costs = result.get("total_cost_breakdown", {})
        
        print(f"Planning: ${costs['planning_cost_in_cents']/100:.2f}")
        print(f"Subagents: ${costs['subagent_costs_in_cents']/100:.2f}")
        print(f"Tools: ${costs['tool_costs_in_cents']/100:.2f}")
        print(f"Total: ${costs['total_cost_in_cents']/100:.2f}")
```

## Best Practices

1. **Agent Specialization**: Create agents with specific, focused roles
2. **Tool Selection**: Only provide agents with the tools they need
3. **Error Handling**: Always configure appropriate retry settings
4. **Conversation Management**: Manage message history to stay within context limits
5. **Cost Management**: Monitor token usage and costs, especially with parallel execution
6. **Task Dependencies**: Use dependencies to ensure proper execution order