# defog

A comprehensive Python toolkit for AI-powered data operations - from natural language SQL queries to multi-agent orchestration.

## Features

- 🤖 **Cross-provider LLM operations** - Unified interface for OpenAI, Anthropic, Gemini, Grok (xAI), and Together AI
- 📊 **SQL Agent** - Convert natural language to SQL with automatic table filtering for large databases
- 🔍 **Data extraction** - Extract structured data from PDFs, images, HTML, text documents, and even images embedded in HTML
- 🛠️ **Advanced AI tools** - Code interpreter, web search, YouTube transcription, document citations
- 🎭 **Agent orchestration** - Hierarchical task delegation and multi-agent coordination
- 💾 **Memory management** - Automatic conversation compactification for long contexts

## Installation

```bash
pip install --upgrade defog
```

## Quick Start

### 1. LLM Chat (Cross-Provider)

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# Works with any provider
response = await chat_async(
    provider=LLMProvider.ANTHROPIC,  # or OPENAI, GEMINI
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content)
```

#### OpenAI GPT‑5: Responses API controls

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-5.1",
    messages=[
        {"role": "system", "content": "You are concise and helpful."},
        {"role": "user", "content": "Summarize the benefits of unit tests."},
    ],
    # Optional Responses API controls for GPT‑5.1
    reasoning_effort="none",   # none | low | medium | high
    verbosity="low",              # low | medium | high
)
print(response.content)
```

### 2. Natural Language to SQL

```python
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

# Ask questions in natural language
result = await sql_answer_tool(
    question="What are the top 10 customers by total sales?",
    db_type="postgres",
    db_creds={
        "host": "localhost",
        "database": "mydb",
        "user": "postgres",
        "password": "password",
        "port": 5432
    },
    model="claude-sonnet-4-20250514",
    provider=LLMProvider.ANTHROPIC
)

print(f"SQL: {result['query']}")
print(f"Results: {result['results']}")
```

### 3. Extract Data from PDFs

```python
from defog.llm import extract_pdf_data

# Extract structured data from any PDF
data = await extract_pdf_data(
    pdf_url="https://example.com/financial_report.pdf",
    focus_areas=["revenue", "financial metrics"]
)

for datapoint_name, extracted_data in data["data"].items():
    print(f"{datapoint_name}: {extracted_data}")
```

### 4. Code Interpreter

```python
from defog.llm.code_interp import code_interpreter_tool
from defog.llm.llm_providers import LLMProvider

# Execute Python code with AI assistance
result = await code_interpreter_tool(
    question="Analyze this data and create a visualization",
    csv_string="name,sales\nAlice,100\nBob,150",
    model="gpt-4o",
    provider=LLMProvider.OPENAI
)

print(result["code"])    # Generated Python code
print(result["output"])  # Execution results
```

### 5. Using MCP Servers with chat_async

```python
from defog.llm.utils import chat_async
from defog.llm.llm_providers import LLMProvider

# Use MCP servers for dynamic tool integration
# Works with both local and remote MCP servers
response = await chat_async(
    provider=LLMProvider.OPENAI,
    model="gpt-4.1",
    mcp_servers=["http://localhost:8000/mcp"],  # Can be local or remote
    messages=[
        {"role": "user", "content": "How many users are in the first table?"}
    ]
)

# MCP tools are automatically converted to Python functions
# and made available to the LLM
print(response.content)
```

## Documentation

📚 **[Full Documentation](docs/README.md)** - Comprehensive guides and API reference

### Quick Links

- **[LLM Utilities](docs/llm/README.md)** - Chat, function calling, structured output, memory management
- **[Database Operations](docs/database/database-operations.md)** - SQL generation, query execution, schema documentation
- **[Data Extraction](docs/data-extraction/data-extraction.md)** - PDF, image, and HTML data extraction tools
- **[Agent Orchestration](docs/advanced/agent-orchestration.md)** - Multi-agent coordination and task delegation
- **[API Reference](docs/api-reference.md)** - Complete API documentation

## Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export TOGETHER_API_KEY="your-together-key"
export XAI_API_KEY="your-grok-xai-key"   # or GROK_API_KEY
```

## Advanced Use Cases

For advanced features like:
- Memory compactification for long conversations
- YouTube video transcription and summarization
- Multi-agent orchestration with shared context
- Database schema auto-documentation
- Model Context Protocol (MCP) support

See the [full documentation](docs/README.md).

## Development

### Testing and formatting
1. Run tests: `python -m pytest tests`
2. Format code: `ruff format`
3. Update documentation when adding features

## Using our MCP Server

1. Run `defog serve` once to complete your setup, and `defog db` to update your database credentials
2. Add to your MCP Client
    - Claude Code: `claude mcp add defog -- python3 -m defog.mcp_server`. 
    Or if you do not want to install the defog package globally or set up environment variables, run `claude mcp add dfg -- uv run --directory FULL_PATH_TO_VENV_DIRECTORY --env-file .env -m defog.mcp_server`
    - Claude Desktop: add the config below
    ```json
    {
        "mcpServers": {
            "defog": {
                "command": "python3",
                "args": ["-m", "defog.mcp_server"],
                "env": {
                    "OPENAI_API_KEY": "YOUR_OPENAI_KEY",
                    "ANTHROPIC_API_KEY": "YOUR_ANTHROPIC_KEY",
                    "GEMINI_API_KEY": "YOUR_GEMINI_KEY",
                    "DB_TYPE": "YOUR_DB_TYPE",
                    "DB_HOST": "YOUR_DB_HOST",
                    "DB_PORT": "YOUR_DB_PORT",
                    "DB_USER": "YOUR_DB_USER",
                    "DB_PASSWORD": "YOUR_DB_PASSWORD",
                    "DB_NAME": "YOUR_DB_NAME"
                }
            }
        }
        }
    ```

### Available MCP Tools and Resources

The Defog MCP server provides the following capabilities:

**Tools** (actions the AI can perform):
- `text_to_sql_tool` - Execute natural language queries against your database
- `list_database_schema` - List all tables and their schemas
- `youtube_video_summary` - Get transcript/summary of YouTube videos (requires Gemini API key)
- `extract_pdf_data` - Extract structured data from PDFs
- `extract_html_data` - Extract structured data from HTML pages
- `extract_text_data` - Extract structured data from text files

**Resources** (read-only data the AI can access):
- `schema://tables` - Get list of all tables in the database
- `schema://table/{table_name}` - Get detailed schema for a specific table
- `stats://table/{table_name}` - Get statistics and metadata for a table (row count, column statistics)
- `sample://table/{table_name}` - Get sample data (10 rows) from a table

## License

MIT License - see LICENSE file for details.
