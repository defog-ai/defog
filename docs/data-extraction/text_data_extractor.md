# Text Data Extractor

The Text Data Extractor is an intelligent agent orchestrator that analyzes plain text documents (transcripts, speeches, reports, etc.) to identify and extract structured data suitable for SQL databases.

## Features

- **Automatic Data Point Discovery**: Identifies extractable data patterns in text
- **Parallel Extraction**: Processes multiple data points concurrently
- **Smart Schema Generation**: Creates appropriate Pydantic schemas for each data type
- **Multiple Data Types Support**:
  - Q&A pairs (interviews, press conferences)
  - Key-value pairs (statistics, metrics)
  - Lists (topics, recommendations)
  - Tables (structured data within text)
  - Statements (policy decisions, quotes)
- **Cost-Effective**: Uses efficient prompting and caching

## Installation

The Text Data Extractor is included in the defog package:

```bash
pip install defog
```

## Basic Usage

```python
import asyncio
from defog.llm import TextDataExtractor

async def main():
    # Initialize the extractor
    extractor = TextDataExtractor()
    
    # Your text content
    text_content = """
    Transcript of Press Conference
    
    REPORTER: What is the current inflation rate?
    SPEAKER: The inflation rate is currently at 2.3%, slightly above our 2% target.
    
    Key Economic Indicators:
    - GDP Growth: 2.5%
    - Unemployment: 4.2%
    """
    
    # Extract all data
    result = await extractor.extract_as_dict(text_content)
    
    # Access extracted data
    print(result['data'])

asyncio.run(main())
```

## Advanced Usage

### Focusing on Specific Areas

```python
# Focus extraction on specific types of data
result = await extractor.extract_as_dict(
    text_content,
    focus_areas=["Q&A exchanges", "economic indicators", "policy decisions"]
)
```

### Filtering Specific Datapoints

```python
# First, analyze to see what's available
analysis, _ = await extractor.analyze_text_structure(text_content)
print("Available datapoints:", [dp.name for dp in analysis.identified_datapoints])

# Then extract only specific ones
result = await extractor.extract_all_data(
    text_content,
    datapoint_filter=["qa_exchanges", "economic_metrics"]
)
```

### Custom Configuration

```python
extractor = TextDataExtractor(
    analysis_provider="anthropic",
    analysis_model="claude-sonnet-4-20250514",
    extraction_provider="openai",
    extraction_model="gpt-4.1",
    max_parallel_extractions=10,
    temperature=0.0,
    enable_caching=True
)
```

## Data Types and Schemas

### Q&A Pairs
Extracts question-answer exchanges with speaker identification:
```python
{
    "exchanges": [
        {
            "questioner": "REPORTER",
            "question": "What is the inflation outlook?",
            "answer": "We expect inflation to moderate over the next year..."
        }
    ],
    "total_exchanges": 5
}
```

### Key-Value Pairs
Extracts metrics, statistics, and labeled values:
```python
{
    "gdp_growth": 2.5,
    "unemployment_rate": 4.2,
    "inflation_rate": 2.3,
    "interest_rate": "4.25-4.5%"
}
```

### Lists
Extracts enumerated items, topics, or points:
```python
{
    "items": [
        "Implement new regulatory framework",
        "Review monetary policy stance",
        "Monitor inflation indicators"
    ],
    "item_count": 3
}
```

### Tables
Extracts tabular data in columnar format:
```python
{
    "columns": ["quarter", "revenue", "growth"],
    "data": [
        ["Q1 2024", 1500000, 5.2],
        ["Q2 2024", 1650000, 10.0]
    ],
    "row_count": 2
}
```

## Example: Processing a Policy Speech Transcript

```python
import asyncio
from defog.llm import extract_text_data

async def analyze_speech():
    # Read transcript
    with open("fed_speech.txt", "r") as f:
        transcript = f.read()
    
    # Extract with focus areas
    result = await extract_text_data(
        transcript,
        focus_areas=[
            "monetary policy decisions",
            "economic forecasts",
            "Q&A exchanges",
            "key statistics"
        ]
    )
    
    # Process results
    print(f"Document type: {result['metadata']['document_type']}")
    print(f"Extracted {len(result['data'])} datapoints")
    
    # Save to JSON
    import json
    with open("speech_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

asyncio.run(analyze_speech())
```

## Performance Optimization

1. **Enable Caching**: Reuses analysis results for repeated documents
   ```python
   extractor = TextDataExtractor(enable_caching=True)
   ```

2. **Adjust Parallelism**: Control concurrent extractions
   ```python
   extractor = TextDataExtractor(max_parallel_extractions=10)
   ```

3. **Use Focused Extraction**: Specify areas to reduce analysis scope
   ```python
   result = await extractor.extract_as_dict(
       text_content,
       focus_areas=["Q&A only"]
   )
   ```

## Cost Management

The extractor tracks costs for transparency:

```python
result = await extractor.extract_all_data(text_content)
print(f"Total cost: ${result.total_cost_cents / 100:.4f}")
print(f"Analysis cost: ${result.metadata['analysis_cost_cents'] / 100:.4f}")
print(f"Extraction cost: ${result.metadata['extraction_cost_cents'] / 100:.4f}")
```

## Integration with Other Extractors

The Text Data Extractor follows the same pattern as other Defog extractors:

```python
from defog.llm import (
    TextDataExtractor,
    PDFDataExtractor,
    HTMLDataExtractor,
    ImageDataExtractor
)

# All extractors share similar interfaces
text_extractor = TextDataExtractor()
pdf_extractor = PDFDataExtractor()
html_extractor = HTMLDataExtractor()
image_extractor = ImageDataExtractor()

# Extract from different sources
text_data = await text_extractor.extract_as_dict(text_content)
pdf_data = await pdf_extractor.extract_as_dict(pdf_url)
html_data = await html_extractor.extract_as_dict(html_content)
image_data = await image_extractor.extract_as_dict(image_url)
```

## Error Handling

```python
try:
    result = await extractor.extract_all_data(text_content)
    
    # Check individual extraction results
    for extraction in result.extraction_results:
        if not extraction.success:
            print(f"Failed to extract {extraction.datapoint_name}: {extraction.error}")
            
except ValueError as e:
    print(f"Text too large: {e}")
except Exception as e:
    print(f"Extraction failed: {e}")
```

## Best Practices

1. **Preprocess Text**: The extractor handles basic preprocessing, but you may want to clean your text first
2. **Use Focus Areas**: Specify what you're looking for to improve accuracy and reduce costs
3. **Monitor Costs**: Track extraction costs, especially for large documents
4. **Cache Results**: Enable caching for documents you'll process multiple times
5. **Validate Output**: Always validate extracted data matches your expectations

## Limitations

- Maximum text size: 5MB by default (configurable)
- Best suited for structured text with clear patterns
- May struggle with highly unstructured or creative text
- Costs scale with document size and complexity