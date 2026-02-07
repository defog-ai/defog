# Defog Documentation Hub

Welcome to the comprehensive documentation for the defog Python library. This hub provides detailed guides, API references, and examples for all features.

## 📚 Documentation Overview

### Core Features

#### [LLM Utilities and Tools](llm/)
Comprehensive guide to LLM operations including:
- **[Chat Functions](llm/core-chat-functions.md)** - Unified interface for OpenAI, Anthropic, Gemini, and Together AI
- **[Multimodal Support](llm/multimodal-support.md)** - Image inputs across all providers
- **[Function Calling](llm/function-calling.md)** - Define tools for LLMs with automatic schema generation
- **[Structured Output](llm/structured-output.md)** - Get validated responses using Pydantic models
- **Specialized Tools** - [Code interpreter](llm/code-interpreter.md), [web search](llm/web-search.md), [YouTube transcription](llm/youtube-transcription.md), [citations](llm/citations-tool.md)

#### [Database Operations](database/)
Everything related to databases and SQL:
- **[SQL Agent](database/database-operations.md)** - Natural language to SQL conversion with automatic table filtering
- **Query Execution** - Direct SQL execution across 11 database types
- **Schema Documentation** - AI-powered automatic schema documentation
- **[Metadata Management](database/metadata-management.md)** - Extraction and caching for performance
- **Local SQL Generation** - Use LLMs directly without API calls
- **Health Checks** - Validate configurations and data quality

#### [Data Extraction Tools](data-extraction/)
Extract structured data from various sources:
- **[PDF Analysis](data-extraction/data-extraction.md)** - Extract tables, charts, and data from PDFs with caching
- **Image Analysis** - Extract data from charts, graphs, and infographics
- **HTML Parsing** - Extract structured data from web content
- **[Text Data Extraction](data-extraction/text_data_extractor.md)** - Extract structured data from text files
- **Cost Optimization** - Strategies for efficient extraction

### Advanced Features

#### [Agent Orchestration](advanced/agent-orchestration.md)
Multi-agent coordination and task delegation:
- **Hierarchical Orchestration** - Coordinate multiple AI agents
- **Task Dependencies** - Define complex workflows with dependencies
- **Shared Context** - Cross-agent memory and context sharing
- **Thinking Agents** - Extended reasoning capabilities

#### [Advanced Configuration](advanced/advanced-configuration.md)
Fine-tune the library for your needs:
- **Environment Variables** - Configure API keys and endpoints
- **Custom Providers** - Add support for new LLM providers
- **Performance Tuning** - Optimize for speed and cost
- **Security Settings** - Configure authentication and access controls

### Developer Resources

#### [API Reference](api-reference.md)
Complete API documentation:
- **Function Signatures** - Detailed parameter descriptions
- **Return Types** - Response formats and structures
- **Error Handling** - Exception types and handling
- **Code Examples** - Practical usage examples

#### [CLI Reference](cli-reference.md)
Command-line interface documentation:
- **Available Commands** - List of all CLI commands
- **Usage Examples** - Common CLI workflows
- **Configuration** - CLI-specific settings

## 🚀 Quick Navigation

### By Use Case

**"I want to..."**

- **Chat with LLMs** → [LLM Utilities](llm-utilities.md#core-chat-functions)
- **Convert questions to SQL** → [SQL Agent](database-operations.md#sql-agent-tools)
- **Extract data from PDFs** → [PDF Extraction](data-extraction.md#pdf-data-extraction)
- **Analyze images/charts** → [Image Extraction](data-extraction.md#image-data-extraction)
- **Document my database** → [Schema Documentation](database-operations.md#schema-documentation)
- **Coordinate multiple agents** → [Agent Orchestration](agent-orchestration.md)
- **Search the web with AI** → [Web Search](llm-utilities.md#web-search)
- **Execute Python code** → [Code Interpreter](llm-utilities.md#code-interpreter)

### By Database Type

- **PostgreSQL/MySQL** → [Database Operations](database-operations.md#supported-databases)
- **BigQuery/Snowflake** → [Cloud Databases](database-operations.md#database-specific-examples)
- **SQLite/DuckDB** → [Local Databases](database-operations.md#supported-databases)

### By Provider

- **OpenAI (GPT-4)** → [Provider Examples](llm-utilities.md#provider-specific-examples)
- **Anthropic (Claude)** → [Provider Examples](llm-utilities.md#provider-specific-examples)
- **Google (Gemini)** → [Provider Examples](llm-utilities.md#provider-specific-examples)
- **Together AI** → [Provider Examples](llm-utilities.md#provider-specific-examples)

## 📖 Getting Started

New to defog? Start here:

1. **[Installation](../README.md#installation)** - Install the library
2. **[Quick Start](../README.md#quick-start)** - Basic examples to get you going
3. **[Environment Setup](advanced-configuration.md#environment-variables)** - Configure API keys
4. **[First Query](database-operations.md#basic-sql-generation-and-execution)** - Run your first SQL query

## 💡 Common Workflows

### Data Analysis Pipeline
1. [Extract data from PDFs](data-extraction.md#pdf-data-extraction)
2. [Store in database](database-operations.md#query-execution)
3. [Query with natural language](database-operations.md#sql-agent-tools)
4. [Visualize with code interpreter](llm-utilities.md#code-interpreter)

### Document Processing
1. [Analyze PDFs with AI](data-extraction.md#pdf-analysis-tool)
2. [Extract specific tables](data-extraction.md#filtering-specific-datapoints)
3. [Generate citations](llm-utilities.md#citations-tool)

### Database Documentation
1. [Extract metadata](database-operations.md#extract-metadata)
2. [Generate documentation](database-operations.md#schema-documentation)
3. [Export to markdown](database-operations.md#export-documentation)

## 🔧 Advanced Topics

- **[Cost Optimization](llm-utilities.md#cost-tracking)** - Track and reduce API costs
- **[Error Handling](api-reference.md#error-handling)** - Handle exceptions gracefully
- **[Performance Tuning](database-operations.md#performance-optimization)** - Optimize for speed
- **[Security Best Practices](database-operations.md#security-considerations)** - Secure your deployment

## 📚 Additional Resources

- **[Examples Directory](../examples/)** - Complete working examples
- **[Test Cases](../tests/)** - Learn from test implementations
- **[GitHub Issues](https://github.com/defog-ai/defog-python/issues)** - Report bugs or request features

## 🤝 Contributing

Want to contribute? Check out:
- [Contributing Guidelines](../CONTRIBUTING.md)
- [Development Setup](advanced-configuration.md#development-setup)
- [Testing Guide](../tests/README.md)

---

**Need help?** Can't find what you're looking for? [Open an issue](https://github.com/defog-ai/defog-python/issues) and we'll help you out!