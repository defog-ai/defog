## MCP Integration

The library supports Model Context Protocol (MCP) servers, allowing you to dynamically extend LLM capabilities with custom tools from any MCP-compatible server.

### Basic MCP Usage

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# Connect to MCP servers and use their tools
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4.1",
    mcp_servers=["http://localhost:8000/mcp"],
    messages=[
        {"role": "user", "content": "Query the database for user statistics"}
    ]
)

print(response.content)
```

### How It Works

1. **Automatic Tool Discovery**: When you provide `mcp_servers`, the library automatically:
   - Initializes connection with each MCP server
   - Discovers available tools via the MCP protocol
   - Converts MCP tools into Python functions
   - Makes them available to the LLM as callable tools

2. **Local and Remote Support**: Unlike traditional implementations that only work with remote servers, defog's implementation:
   - Handles the full MCP lifecycle locally
   - Supports both local (`localhost`) and remote MCP servers
   - Converts HTTP-based MCP tools into regular Python functions

3. **Dynamic Function Generation**: Each MCP tool is converted to a Python function with:
   - Proper function signatures based on the tool's schema
   - Automatic parameter validation
   - Async execution support
   - Proper error handling

### Multiple MCP Servers

```python
# Connect to multiple MCP servers simultaneously
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    mcp_servers=[
        "http://localhost:8000/mcp",      # Local database MCP server
        "https://api.example.com/mcp",    # Remote API MCP server
        "http://localhost:3000/mcp"       # Another local service
    ],
    messages=[
        {"role": "user", "content": "Analyze sales data and create a report"}
    ]
)
```

### MCP with Other Features

```python
# Combine MCP servers with other chat_async features
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4.1",
    mcp_servers=["http://localhost:8000/mcp"],
    messages=messages,
    
    # Use with structured output
    response_format=AnalysisReport,
    
    # Use with function calling
    tools=[my_custom_function],  # Your tools work alongside MCP tools
    
    # Use with tool budgets
    tool_budget={
        "mcp__database__query": 5,  # Limit MCP tool usage
        "my_custom_function": 3
    },
    
    # Other parameters
    temperature=0.0,
    max_completion_tokens=2000
)
```

### MCP Server Requirements

For an MCP server to work with defog:

1. Must implement the MCP protocol (version 2025-03-26)
2. Must support the following endpoints:
   - `/initialize` - Connection initialization
   - `/notifications/initialized` - Initialization confirmation
   - `/tools/list` - Tool discovery
   - `/tools/call` - Tool execution
3. Must return proper SSE (Server-Sent Events) responses
4. Should include proper tool schemas in the `inputSchema` field

### Example: Using a Database MCP Server

```python
# Assuming you have a database MCP server running locally
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    mcp_servers=["http://localhost:8000/mcp"],
    messages=[
        {
            "role": "user", 
            "content": "Find the top 10 customers by revenue this quarter"
        }
    ]
)

# The MCP server's SQL tools are automatically available
# The LLM can now query your database directly
print(response.content)
```

### Error Handling

```python
try:
    response = await chat_async(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1",
        mcp_servers=["http://localhost:8000/mcp"],
        messages=messages
    )
except httpx.ConnectError:
    print("Failed to connect to MCP server")
except RuntimeError as e:
    print(f"MCP error: {e}")
```

### Advanced: Tool Naming

MCP tools are automatically namespaced to avoid conflicts:

```python
# If your MCP server is named "database" and has a tool "query"
# It becomes available as "mcp__database__query"

# This allows multiple MCP servers without naming conflicts
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4.1",
    mcp_servers=[
        "http://localhost:8000/mcp",  # Has tool "query"
        "http://localhost:9000/mcp"   # Also has tool "query"
    ],
    messages=messages,
    tool_budget={
        "mcp__server1__query": 5,
        "mcp__server2__query": 3
    }
)
```