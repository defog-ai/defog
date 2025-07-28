# LLM Utilities and Tools

Comprehensive documentation for all LLM-related functionality in the defog library.

## Table of Contents

- [Core Chat Functions](#core-chat-functions)
- [Multimodal Support](#multimodal-support)
- [Function Calling](#function-calling)
- [Tool Budget Management](#tool-budget-management)
- [Structured Output](#structured-output)
- [Memory Management](#memory-management)
- [Code Interpreter](#code-interpreter)
- [Web Search](#web-search)
- [YouTube Transcription](#youtube-transcription)
- [Citations Tool](#citations-tool)
- [MCP Integration](#mcp-integration)
- [Cost Tracking](#cost-tracking)
- [Best Practices](#best-practices)

## Core Chat Functions

The library provides a unified interface for working with multiple LLM providers.

### Basic Usage

```python
from defog.llm.utils import chat_async, chat_async_legacy, LLMResponse
from defog.llm.llm_providers import LLMProvider

# Unified async interface with explicit provider specification
response: LLMResponse = await chat_async(
    provider=LLMProvider.OPENAI,  # or "openai", LLMProvider.ANTHROPIC, etc.
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_completion_tokens=1000,
    temperature=0.0
)

print(response.content)  # Response text
print(f"Cost: ${response.cost_in_cents/100:.4f}")

# Alternative: Legacy model-to-provider inference
response = await chat_async_legacy(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Advanced Parameters

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    
    # Core parameters
    max_completion_tokens=2000,
    temperature=0.7,
    
    # Advanced parameters
    reasoning_effort="high",          # For o1 models: low, medium, high
    response_format=MyPydanticModel,  # Structured output
    tools=[...],                      # Function calling
    tool_choice="auto",               # auto, none, required, or specific tool
    
    # Provider-specific options
    top_p=0.9,                        # Nucleus sampling
    frequency_penalty=0.0,            # Reduce repetition
    presence_penalty=0.0,             # Encourage new topics
    
    # Logging and debugging
    verbose=True,                     # Detailed logging
    return_usage=True,                # Include token usage
)
```

### Provider-Specific Examples

```python
# OpenAI
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[my_function],
    tool_choice="auto"
)

# Anthropic
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=messages,
    response_format=MyPydanticModel
)

# Gemini
response = await chat_async(
    provider=LLMProvider.GEMINI,
    model="gemini-2.0-flash",
    messages=messages
)

# Together AI
response = await chat_async(
    provider=LLMProvider.TOGETHER,
    model="mixtral-8x7b",
    messages=messages
)
```

## Multimodal Support

The library supports image inputs across all major providers with automatic format conversion.

### Image Input Examples

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider
import base64

# Using base64-encoded images
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

# Using image URLs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.jpg"
                }
            }
        ]
    }
]

# Works with all providers - automatic format conversion
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,  # or OPENAI, GEMINI
    model="claude-sonnet-4-20250514",
    messages=messages
)
```

### Multiple Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two charts"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart1_base64}"}
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{chart2_base64}"}
            }
        ]
    }
]
```

## Function Calling

Define tools for LLMs to call with automatic schema generation. As of version 0.50.0, you can use regular Python functions directly as tools!

### Basic Function Definition

You can define tools in two ways:

#### Option 1: Regular Functions (NEW - Simpler!)

```python
from defog.llm.utils import chat_async
from typing import Optional

# Regular function with type annotations
def get_weather(location: str, units: str = "celsius") -> str:
    """
    Get current weather for a location.
    
    Args:
        location: City and country
        units: Temperature units (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

# Async functions are also supported
async def search_web(query: str, max_results: Optional[int] = 5) -> list[str]:
    """Search the web for information"""
    # Implementation here
    return [f"Result {i} for {query}" for i in range(max_results)]

# Use directly as tools - no Pydantic needed!
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather, search_web],
    tool_choice="auto"
)
```

#### Option 2: Pydantic-based Functions (Traditional)

```python
from pydantic import BaseModel, Field
from defog.llm.utils import chat_async

class WeatherInput(BaseModel):
    location: str = Field(description="City and country")
    units: str = Field(default="celsius", description="Temperature units")

def get_weather(input: WeatherInput) -> str:
    """Get current weather for a location"""
    return f"Weather in {input.location}: 22°{input.units[0].upper()}, sunny"

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
    tool_choice="auto"
)
```

### When to Use Each Approach

**Use Regular Functions when:**
- You want simpler, more Pythonic code
- Your function parameters are straightforward types
- You're prototyping or building quickly
- You prefer type hints over Pydantic models

**Use Pydantic Functions when:**
- You need complex validation logic
- You want detailed field descriptions and constraints
- You're working with nested data structures
- You need custom validators or serializers

### Advanced Tool Configuration

```python
from typing import Literal, Optional
from defog.llm.utils import create_tool_from_function

class DatabaseQuery(BaseModel):
    query: str = Field(description="SQL query to execute")
    database: Literal["prod", "staging", "dev"] = Field(default="dev")
    timeout: Optional[int] = Field(default=30, description="Query timeout")
    explain: bool = Field(default=False, description="Include query plan")

@create_tool_from_function
async def execute_database_query(params: DatabaseQuery) -> dict:
    """Execute a database query with safety controls"""
    # Implementation here
    return {"results": [...], "execution_time": 0.5}

# Use with advanced options
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages,
    tools=[execute_database_query],
    tool_choice="required",           # Force tool use
    parallel_tool_calls=True,         # Allow parallel execution
    max_tool_iterations=3,            # Limit recursive calls
)
```

## Tool Budget Management

Limit the number of times specific tools can be called during a conversation to control costs and prevent excessive usage.

### Basic Usage

```python
from pydantic import BaseModel, Field
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

class SearchInput(BaseModel):
    query: str = Field(description="Search query")

def web_search(input: SearchInput) -> str:
    """Search the web for information"""
    return f"Results for '{input.query}'"

# Set usage limits for each tool
tool_budget = {
    "web_search": 3,    # Allow only 3 web searches
}

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "Search for Python, Java, and JavaScript trends"}],
    tools=[web_search],
    tool_budget=tool_budget
)
```

When a tool's budget is exhausted, the LLM will be notified and will adapt its approach without access to that tool.

## Tool Citations

Automatically add citations to LLM responses when tools are used, showing which tool outputs were used to generate the response. This feature is only available for OpenAI and Anthropic providers.

### Basic Usage

```python
from pydantic import BaseModel
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

class WeatherInput(BaseModel):
    location: str
    unit: str = "fahrenheit"

def get_weather(input: WeatherInput):
    """Get current weather for a location"""
    return {"temperature": 72, "condition": "sunny", "humidity": 65}

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=[get_weather],
    tool_choice="auto",
    insert_tool_citations=True  # Enable citations
)

# Response will include citations
print(response.content)    # "The weather in San Francisco is currently sunny with a temperature of 72°F [1]..."
print(response.citations)  # List of citation blocks with tool references
```

### How It Works

1. Tools are executed normally during the conversation
2. Tool outputs are converted to documents with metadata
3. The citations tool processes the response and adds inline citations
4. Both the cited content and citation details are returned in the response

### Response Structure

When `insert_tool_citations=True`, the response includes:
- `content`: The LLM's response with inline citations added
- `citations`: A list of citation blocks, each containing:
  - `type`: "text"
  - `text`: The text content
  - `citations`: List of references to tool outputs

### Limitations

- Only supported for OpenAI and Anthropic providers
- Raises `ValueError` if used with unsupported providers
- Requires tools to be used in the conversation
- No effect if no tools are called

## Structured Output

Get validated, structured responses using Pydantic models.

### Basic Structured Output

```python
from pydantic import BaseModel
from typing import List

class Analysis(BaseModel):
    sentiment: str
    key_points: List[str]
    confidence: float

response = await chat_async(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Analyze this text..."}],
    response_format=Analysis
)

# Access structured data
analysis = response.parsed  # Type: Analysis
print(f"Sentiment: {analysis.sentiment}")
print(f"Confidence: {analysis.confidence}")
```

### Complex Structured Output

```python
from typing import List, Optional
from datetime import datetime

class Person(BaseModel):
    name: str
    role: str
    email: Optional[str] = None

class MeetingNotes(BaseModel):
    date: datetime
    attendees: List[Person]
    agenda_items: List[str]
    decisions: List[str]
    action_items: List[dict]
    next_meeting: Optional[datetime] = None

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=[{"role": "user", "content": meeting_transcript}],
    response_format=MeetingNotes
)

notes = response.parsed
print(f"Meeting on {notes.date}")
print(f"Attendees: {', '.join(p.name for p in notes.attendees)}")
```

## Memory Management

Automatically manage long conversations by intelligently summarizing older messages.

### Basic Memory Usage

```python
from defog.llm import chat_async_with_memory, create_memory_manager, MemoryConfig

# Create a memory manager
memory_manager = create_memory_manager(
    token_threshold=50000,      # Compactify when reaching 50k tokens
    preserve_last_n_messages=10, # Keep last 10 messages intact
    summary_max_tokens=2000,    # Max tokens for summary
    enabled=True
)

# System messages are automatically preserved
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Tell me about Python"}
    ],
    memory_manager=memory_manager
)

# Continue conversation - memory is automatically managed
response2 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What about its use in data science?"}],
    memory_manager=memory_manager
)

# Check memory stats
stats = memory_manager.get_stats()
print(f"Total messages: {len(memory_manager.get_current_messages())}")
print(f"Compactifications: {stats['compactification_count']}")
```

### Memory Configuration Options

```python
# Use memory configuration without explicit manager
response = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=messages,
    memory_config=MemoryConfig(
        enabled=True,
        token_threshold=100000,       # 100k tokens before compactification
        preserve_last_n_messages=10,
        summary_max_tokens=4000,
        preserve_system_messages=True, # Always preserve system messages
        preserve_tool_calls=True,      # Keep function calls in memory
        compression_ratio=0.3          # Target 30% compression
    )
)
```

### Advanced Memory Management

```python
from defog.llm.memory import EnhancedMemoryManager

# Enhanced memory with cross-agent support
memory_manager = EnhancedMemoryManager(
    token_threshold=50000,
    preserve_last_n_messages=10,
    summary_max_tokens=2000,
    
    # Advanced options
    enable_cross_agent_memory=True,    # Share memory across agents
    memory_tags=["research", "analysis"], # Tag memories
    preserve_images=False,             # Exclude images from memory
    
    # Custom summarization
    summary_model="gpt-4o-mini",       # Use cheaper model for summaries
    summary_provider="openai",
    summary_instructions="Focus on key decisions and findings"
)

# Agent 1 with memory
response1 = await chat_async_with_memory(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Research topic X"}],
    memory_manager=memory_manager,
    agent_id="researcher"
)

# Agent 2 can access Agent 1's memory
response2 = await chat_async_with_memory(
    provider="anthropic",
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Summarize the research findings"}],
    memory_manager=memory_manager,
    agent_id="summarizer",
    include_cross_agent_memory=True
)
```

## Code Interpreter

Execute Python code in sandboxed environments with AI assistance.

### Basic Code Execution

```python
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider

result = await code_interpreter_tool(
    question="Analyze this CSV data and create a visualization",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string="name,age,score\nAlice,25,95\nBob,30,87\nCarol,28,92",
    instructions="Create a bar chart showing scores by name"
)

print(result["code"])    # Generated Python code
print(result["output"])  # Execution results
# Images are returned as base64 if generated
```

### Advanced Code Interpreter Options

```python
result = await code_interpreter_tool(
    question="Perform statistical analysis",
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    csv_string=data,
    
    # Advanced options
    allowed_libraries=["pandas", "matplotlib", "seaborn", "numpy", "scipy"],
    memory_limit_mb=512,              # Memory limit
    timeout_seconds=30,               # Execution timeout
    
    # Custom environment setup
    pre_code="""
    import warnings
    warnings.filterwarnings('ignore')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')
    """,
    
    # Output preferences
    return_images=True,               # Return generated images
    image_format="png",               # png, svg, jpg
    figure_dpi=150,                   # Image quality
    
    # Security
    allow_file_access=False,          # Restrict file system access
    allow_network_access=False        # Restrict network access
)

# Access results
if result.get("images"):
    for idx, img_base64 in enumerate(result["images"]):
        with open(f"output_{idx}.png", "wb") as f:
            f.write(base64.b64decode(img_base64))
```

## Web Search

Search the web for current information with AI-powered analysis.

### Basic Web Search

```python
from defog.llm.web_search import web_search_tool
from defog.llm.llm_providers import LLMProvider

result = await web_search_tool(
    question="What are the latest developments in AI?",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    max_tokens=2048
)

print(result["search_results"])   # Analyzed search results
print(result["websites_cited"])   # Source citations
```

### Advanced Search Configuration

```python
result = await web_search_tool(
    question="Latest quantum computing breakthroughs",
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    
    # Advanced options
    search_depth="comprehensive",      # quick, standard, comprehensive
    max_results=20,                   # Number of search results
    include_snippets=True,            # Include search snippets
    follow_links=True,                # Fetch full page content
    
    # Domain filtering
    preferred_domains=["arxiv.org", "nature.com", "science.org"],
    blocked_domains=["blogspot.com"],
    
    # Content preferences
    prefer_recent=True,               # Prioritize recent content
    date_range="6m",                  # Last 6 months only
    content_type="academic",          # academic, news, general
    
    # Output formatting
    include_timestamps=True,
    group_by_domain=True,
    summarize_per_domain=True
)
```

## YouTube Transcription

Generate detailed summaries and transcripts from YouTube videos.

### Basic Usage

```python
from defog.llm.youtube import get_youtube_summary

# Get basic summary
summary = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    model="gemini-2.5-pro"
)

print(summary)
```

### Detailed Transcription

```python
# Get detailed transcript with timestamps
transcript = await get_youtube_summary(
    video_url="https://www.youtube.com/watch?v=...",
    model="gemini-2.5-pro",
    verbose=True,
    
    # Detailed instructions
    system_instructions=[
        "Provide detailed transcript with timestamps (HH:MM:SS)",
        "Include speaker names if available",
        "Separate content into logical sections",
        "Include [MUSIC] and [APPLAUSE] markers",
        "Extract any text shown on screen",
        "Note visual demonstrations with [DEMO] tags",
        "Highlight key quotes with quotation marks"
    ],
    
    # Advanced options
    include_auto_chapters=True,       # Use YouTube's chapter markers
    language_preference="en",         # Preferred transcript language
    fallback_to_auto_captions=True,   # Use auto-generated if needed
    max_retries=3                     # Retry on failure
)
```

## Citations Tool

Generate well-cited answers from document collections.

### Basic Citations

```python
from defog.llm.citations import citations_tool
from defog.llm.llm_providers import LLMProvider

# Prepare documents
documents = [
    {"document_name": "research_paper.pdf", "document_content": "..."},
    {"document_name": "article.txt", "document_content": "..."}
]

# Get cited answer
result = await citations_tool(
    question="What are the main findings?",
    instructions="Provide detailed analysis with citations",
    documents=documents,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    max_tokens=16000
)

print(result["response"])         # Answer with citations
print(result["citations_used"])   # List of citations
```

### Advanced Citation Options

```python
result = await citations_tool(
    question="Compare the methodologies",
    instructions="Provide academic-style analysis",
    documents=documents,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    
    # Advanced options
    citation_style="academic",        # academic, inline, footnote
    min_citations_per_claim=2,        # Require multiple sources
    include_confidence_scores=True,   # Rate citation confidence
    extract_quotes=True,              # Include exact quotes
    max_quote_length=200,             # Limit quote size
    
    # Chunk handling
    chunk_size=2000,                  # Tokens per chunk
    chunk_overlap=200,                # Overlap between chunks
    relevance_threshold=0.7           # Minimum relevance score
)

# Access detailed citation information
for citation in result["citation_details"]:
    print(f"Source: {citation['source']}")
    print(f"Quote: {citation['quote']}")
    print(f"Confidence: {citation['confidence']}")
```

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

## Cost Tracking

All LLM operations include detailed cost tracking.

### Basic Cost Information

```python
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4o",
    messages=messages
)

print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"Cost: ${response.cost_in_cents / 100:.4f}")
```

### Aggregate Cost Tracking

```python
from defog.llm.cost_tracker import CostTracker

# Initialize cost tracker
tracker = CostTracker()

# Track multiple operations
async def process_documents(docs):
    for doc in docs:
        response = await chat_async(...)
        tracker.add_cost(
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_cents=response.cost_in_cents
        )

# Get cost summary
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost_cents'] / 100:.2f}")
print(f"By provider: {summary['by_provider']}")
print(f"By model: {summary['by_model']}")
```

## Best Practices

### 1. Provider Selection

- **OpenAI**: Best for function calling, structured output, general purpose
- **Anthropic**: Best for long context, complex reasoning, document analysis
- **Gemini**: Best for cost-effectiveness, multimodal tasks, large documents
- **Together**: Best for open-source models, specific use cases

### 2. Error Handling

```python
from defog.llm.utils import chat_async
from defog.llm.exceptions import LLMError, RateLimitError, ContextLengthError

try:
    response = await chat_async(...)
except RateLimitError:
    # Wait and retry
    await asyncio.sleep(60)
    response = await chat_async(...)
except ContextLengthError:
    # Use memory management or chunk content
    response = await chat_async_with_memory(...)
except LLMError as e:
    print(f"LLM error: {e}")
```

### 3. Performance Optimization

- Use appropriate models for tasks (don't use GPT-4 for simple tasks)
- Enable caching for repeated operations (especially PDFs)
- Use structured output for consistent parsing
- Batch operations when possible
- Configure memory management for long conversations

### 4. Cost Optimization

- Use cheaper models for summarization (e.g., gpt-4o-mini)
- Enable input caching for repeated content
- Set max_tokens appropriately
- Use token counting before sending requests
- Monitor costs with tracking utilities

### 5. Security Considerations

- Validate all tool inputs with Pydantic models
- Restrict code interpreter capabilities appropriately
- Use domain filtering for web searches
- Implement rate limiting for production use
- Sanitize any user-provided content

## See Also

- [Data Extraction](data-extraction.md) - PDF, Image, and HTML extraction
- [Database Operations](database-operations.md) - SQL generation and execution
- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [API Reference](api-reference.md) - Complete API documentation