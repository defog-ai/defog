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