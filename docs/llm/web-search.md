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