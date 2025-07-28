# Data Extraction Tools

This document covers the data extraction capabilities of the defog library, including PDF, image, HTML, and text data extraction using AI.

## Table of Contents

- [PDF Data Extraction](#pdf-data-extraction)
- [PDF Analysis Tool](#pdf-analysis-tool)
- [Image Data Extraction](#image-data-extraction)
- [HTML Data Extraction](#html-data-extraction)
- [Text Data Extraction](#text-data-extraction)
- [Common Patterns](#common-patterns)

## PDF Data Extraction

Extract structured data from PDFs using intelligent AI analysis. The tool automatically identifies datapoints (tables, charts, key metrics) and extracts them into structured formats.

### Basic Usage

```python
from defog.llm import PDFDataExtractor, extract_pdf_data

# Quick extraction with convenience function
data = await extract_pdf_data(
    pdf_url="https://example.com/financial_report.pdf",
    focus_areas=["revenue", "financial metrics", "tables"]
)

for datapoint_name, extracted_data in data["data"].items():
    print(f"\n{datapoint_name}:")
    print(extracted_data)
```

### Advanced Usage with PDFDataExtractor

```python
from defog.llm import PDFDataExtractor

# Initialize extractor with specific models
extractor = PDFDataExtractor(
    analysis_provider="anthropic",
    analysis_model="claude-sonnet-4-20250514",
    extraction_provider="anthropic",
    extraction_model="claude-sonnet-4-20250514"
)

# Extract all identified datapoints
result = await extractor.extract_all_data(
    pdf_url="https://example.com/financial_report.pdf",
    focus_areas=["revenue", "financial metrics", "quarterly results"]
)

print(f"PDF Pages: {result.pdf_pages}")
print(f"Identified {result.total_datapoints_identified} datapoints")
print(f"Successfully extracted: {result.successful_extractions}")
print(f"Failed extractions: {result.failed_extractions}")
print(f"Total cost: ${result.total_cost_cents / 100:.4f}")

# Access extracted data
for extraction in result.extractions:
    if extraction.extraction_status == "success":
        print(f"\n{extraction.datapoint_name}:")
        print(f"Type: {extraction.data_type}")
        if extraction.columns and extraction.data:
            print(f"Columns: {extraction.columns}")
            print(f"Rows: {len(extraction.data)}")
```

### Filtering Specific Datapoints

```python
# Extract only specific datapoints
result = await extractor.extract_all_data(
    pdf_url="https://example.com/report.pdf",
    datapoint_filter=["revenue_table", "profit_margins", "quarterly_results"]
)

# Get as dictionary for easier access
data_dict = await extractor.extract_as_dict(
    pdf_url="https://example.com/report.pdf",
    focus_areas=["financial data"]
)

# Access specific tables
if "revenue_table" in data_dict["data"]:
    revenue_data = data_dict["data"]["revenue_table"]
    print(f"Revenue columns: {revenue_data['columns']}")
    print(f"Revenue data: {revenue_data['data']}")
```

## PDF Analysis Tool

Analyze PDFs from URLs with Claude's advanced capabilities, including input caching and smart chunking.

### Basic Usage

```python
from defog.llm.pdf_processor import analyze_pdf, PDFAnalysisInput

# Simple analysis
pdf_input = PDFAnalysisInput(
    url="https://arxiv.org/pdf/2301.07041.pdf",
    task="Summarize this research paper"
)

result = await analyze_pdf(pdf_input)
print(result["result"])
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class ResearchPaperSummary(BaseModel):
    title: str = Field(description="Paper title")
    authors: List[str] = Field(description="List of authors")
    abstract_summary: str = Field(description="One paragraph summary of abstract")
    main_contributions: List[str] = Field(description="Key contributions")
    methodology: str = Field(description="Research methodology used")
    results: List[str] = Field(description="Main results and findings")
    limitations: List[str] = Field(description="Stated limitations")

# Analyze with structured response
pdf_input = PDFAnalysisInput(
    url="https://arxiv.org/pdf/2301.07041.pdf",
    task="Extract detailed information about this research paper",
    response_format=ResearchPaperSummary
)

result = await analyze_pdf(pdf_input)
if result["success"]:
    summary: ResearchPaperSummary = result["result"]
    print(f"Title: {summary.title}")
    print(f"Authors: {', '.join(summary.authors)}")
    print(f"\nMain Contributions:")
    for contrib in summary.main_contributions:
        print(f"- {contrib}")
```

### Cost Optimization with Caching

The PDF analysis tool uses Anthropic's input caching for 5-minute cache windows:

```python
# First analysis - full cost
result1 = await analyze_pdf(PDFAnalysisInput(
    url="https://example.com/large_report.pdf",
    task="Summarize the executive summary"
))
print(f"Cost: ${result1['metadata']['total_cost_in_cents'] / 100:.4f}")
print(f"Cached tokens: {result1['metadata']['cached_tokens']}")  # 0 on first run

# Second analysis within 5 minutes - dramatically reduced cost
result2 = await analyze_pdf(PDFAnalysisInput(
    url="https://example.com/large_report.pdf",  # Same PDF
    task="Extract all financial tables"  # Different task
))
print(f"Cost: ${result2['metadata']['total_cost_in_cents'] / 100:.4f}")  # Much lower!
print(f"Cached tokens: {result2['metadata']['cached_tokens']}")  # Most tokens cached
```

### Handling Large PDFs

PDFs are automatically chunked if they exceed size limits:

```python
# Large PDFs (>80 pages or >24MB) are automatically split
result = await analyze_pdf(PDFAnalysisInput(
    url="https://example.com/1000_page_report.pdf",
    task="Summarize each major section of this report"
))

# The tool handles chunking transparently
print(f"Success: {result['success']}")
print(f"Result: {result['result']}")  # Combined analysis from all chunks
```

## Image Data Extraction

Extract structured data from images including charts, graphs, tables, and infographics.

### Basic Usage

```python
from defog.llm import ImageDataExtractor, extract_image_data

# Quick extraction
data = await extract_image_data(
    image_url="https://example.com/sales_chart.png",
    focus_areas=["sales data", "trend analysis"]
)

for datapoint_name, content in data["data"].items():
    print(f"{datapoint_name}: {content}")
```

### Advanced Image Analysis

```python
from defog.llm import ImageDataExtractor

# Initialize with specific providers
extractor = ImageDataExtractor(
    analysis_provider="anthropic",
    analysis_model="claude-sonnet-4-20250514",
    extraction_provider="openai",
    extraction_model="gpt-4o"
)

# Analyze complex visualizations
result = await extractor.extract_all_data(
    image_url="https://example.com/complex_dashboard.png",
    focus_areas=["KPIs", "time series data", "geographical distribution"]
)

print(f"Image type: {result.image_type}")  # e.g., "dashboard", "chart", "infographic"
print(f"Identified {result.total_datapoints_identified} datapoints")

# Process each extracted datapoint
for extraction in result.extractions:
    if extraction.extraction_status == "success":
        print(f"\n{extraction.datapoint_name}:")
        print(f"Type: {extraction.data_type}")
        
        if extraction.data_type == "table":
            print(f"Columns: {extraction.columns}")
            print(f"Data rows: {len(extraction.data)}")
            # Example: [['North', 150000], ['South', 120000], ...]
        
        elif extraction.data_type == "time_series":
            print(f"Time points: {len(extraction.data)}")
            # Example: [['2023-01', 45000], ['2023-02', 52000], ...]
```

### Handling Different Chart Types

```python
# Bar charts
bar_chart_data = await extract_image_data(
    image_url="https://example.com/bar_chart.png",
    focus_areas=["categories", "values", "axis labels"]
)

# Line graphs
line_graph_data = await extract_image_data(
    image_url="https://example.com/line_graph.png",
    focus_areas=["trend data", "time series", "multiple series if present"]
)

# Pie charts
pie_chart_data = await extract_image_data(
    image_url="https://example.com/pie_chart.png",
    focus_areas=["segments", "percentages", "labels"]
)

# Complex dashboards
dashboard_data = await extract_image_data(
    image_url="https://example.com/dashboard.png",
    focus_areas=["KPI values", "mini charts", "tables", "metrics"]
)
```

## HTML Data Extraction

Extract structured data from HTML content including tables, lists, product information, and even data from images embedded in the HTML.

### Basic Usage

```python
from defog.llm import HTMLDataExtractor, extract_html_data

# Extract from HTML string
html_content = """
<table class="financial-data">
    <tr><th>Quarter</th><th>Revenue</th><th>Profit</th></tr>
    <tr><td>Q1 2024</td><td>$1.2M</td><td>$300K</td></tr>
    <tr><td>Q2 2024</td><td>$1.5M</td><td>$400K</td></tr>
</table>
"""

data = await extract_html_data(
    html_content,
    focus_areas=["financial data", "quarterly results"]
)

for name, content in data["data"].items():
    if "columns" in content and "data" in content:
        print(f"{name}:")
        print(f"Columns: {content['columns']}")
        print(f"Data: {content['data']}")
```

### Advanced HTML Extraction

```python
from defog.llm import HTMLDataExtractor

# Use different providers for analysis vs extraction
extractor = HTMLDataExtractor(
    analysis_provider="openai",
    analysis_model="gpt-4",
    extraction_provider="gemini",
    extraction_model="gemini-2.5-pro",
    enable_image_extraction=True  # Enable extraction from images in HTML
)

# Extract from complex e-commerce page
html_content = """
<div class="products">
    <div class="product-card">
        <h3>Widget Pro</h3>
        <span class="price">$49.99</span>
        <span class="stock">In Stock: 150</span>
        <ul class="features">
            <li>Feature A</li>
            <li>Feature B</li>
        </ul>
    </div>
    <!-- More products... -->
</div>
"""

result = await extractor.extract_all_data(
    html_content=html_content,
    focus_areas=["product catalog", "pricing", "inventory", "features"]
)

# Filter specific datapoints
filtered_result = await extractor.extract_all_data(
    html_content=html_content,
    datapoint_filter=["product_catalog_table", "price_list"]
)
```

### Extracting from Different HTML Structures

```python
# Tables
table_html = "<table>...</table>"
table_data = await extract_html_data(table_html, focus_areas=["table data"])

# Lists
list_html = """
<ul class="specifications">
    <li>Processor: Intel i7</li>
    <li>RAM: 16GB</li>
    <li>Storage: 512GB SSD</li>
</ul>
"""
specs_data = await extract_html_data(list_html, focus_areas=["specifications"])

# Product cards
product_html = """
<div class="product-grid">
    <div class="product">
        <h3>Product Name</h3>
        <p class="price">$99.99</p>
        <p class="description">...</p>
    </div>
</div>
"""
products_data = await extract_html_data(product_html, focus_areas=["products", "prices"])

# Forms and structured data
form_html = """
<form class="survey-results">
    <div class="result">
        <span class="question">Satisfaction</span>
        <span class="percentage">85%</span>
    </div>
</form>
"""
survey_data = await extract_html_data(form_html, focus_areas=["survey results"])

# HTML with embedded images (charts, graphs, tables as images)
html_with_images = """
<div class="report">
    <h2>Sales Analysis</h2>
    <img src="https://example.com/sales-chart.png" 
         alt="Bar chart showing monthly sales data">
    <p>Sales increased by 25% this quarter.</p>
    
    <h2>Regional Distribution</h2>
    <img src="https://example.com/regional-pie-chart.png"
         alt="Pie chart showing sales by region">
</div>
"""

# Extract data from both HTML content and embedded images
data_with_images = await extract_html_data(
    html_with_images, 
    focus_areas=["sales data", "regional distribution"],
    enable_image_extraction=True  # This is True by default
)

# The extractor will identify images as potential data sources
# and extract structured data from charts, graphs, and image-based tables

# Handle relative image URLs with base_url parameter
html_with_relative_images = """
<div class="dashboard">
    <img src="/charts/sales-2024.png" alt="Sales chart">
    <img src="./images/revenue.png" alt="Revenue graph">
    <img src="../data/metrics.png" alt="Key metrics">
</div>
"""

data_with_base_url = await extract_html_data(
    html_with_relative_images,
    base_url="https://example.com/reports/2024/",  # Base URL for resolving relative paths
    focus_areas=["charts", "metrics"]
)

# The extractor will resolve:
# - "/charts/sales-2024.png" → "https://example.com/charts/sales-2024.png"
# - "./images/revenue.png" → "https://example.com/reports/2024/images/revenue.png"
# - "../data/metrics.png" → "https://example.com/reports/data/metrics.png"
```

## Text Data Extraction

Extract structured data from plain text documents (transcripts, speeches, reports) using intelligent AI analysis. The tool automatically identifies patterns like Q&A exchanges, key-value pairs, and structured information.

### Basic Usage

```python
from defog.llm import TextDataExtractor, extract_text_data

# Quick extraction with convenience function
text_content = """
Transcript of Press Conference

REPORTER: What is the current inflation rate?
SPEAKER: The inflation rate is 2.3%, slightly above our 2% target.

Key Economic Indicators:
- GDP Growth: 2.5%
- Unemployment: 4.2%
"""

data = await extract_text_data(
    text_content,
    focus_areas=["Q&A exchanges", "economic indicators"]
)

for datapoint_name, extracted_data in data["data"].items():
    print(f"\n{datapoint_name}:")
    print(extracted_data)
```

### Advanced Usage with TextDataExtractor

```python
from defog.llm import TextDataExtractor

# Initialize extractor with specific models
extractor = TextDataExtractor(
    analysis_provider="anthropic",
    analysis_model="claude-sonnet-4-20250514",
    extraction_provider="openai",
    extraction_model="gpt-4.1"
)

# Extract all identified datapoints
result = await extractor.extract_all_data(
    text_content,
    focus_areas=["Q&A exchanges", "policy decisions", "statistics"]
)

print(f"Document type: {result.document_type}")
print(f"Identified {result.total_datapoints_identified} datapoints")
print(f"Successfully extracted: {result.successful_extractions}")
print(f"Total cost: ${result.total_cost_cents / 100:.4f}")

# Access extracted data
for extraction in result.extraction_results:
    if extraction.success:
        print(f"\n{extraction.datapoint_name}:")
        print(extraction.extracted_data)
```

### Processing Large Transcripts

```python
# Read transcript from file
with open("fed_speech_transcript.txt", "r") as f:
    transcript = f.read()

# Extract with specific focus
result = await extractor.extract_as_dict(
    transcript,
    focus_areas=[
        "monetary policy decisions",
        "economic forecasts",
        "Q&A exchanges",
        "key statistics"
    ]
)

# Q&A exchanges will be structured as:
qa_data = result["data"]["qa_exchanges"]
for exchange in qa_data["exchanges"]:
    print(f"\nQ ({exchange['questioner']}): {exchange['question']}")
    print(f"A: {exchange['answer']}")

# Economic indicators as key-value pairs
if "economic_indicators" in result["data"]:
    indicators = result["data"]["economic_indicators"]
    print(f"\nGDP Growth: {indicators.get('gdp_growth')}%")
    print(f"Inflation: {indicators.get('inflation_rate')}%")
```

### Supported Data Types

- **Q&A Pairs**: Extracts question-answer exchanges with speaker identification
- **Key-Value Pairs**: Metrics, statistics, and labeled values
- **Lists**: Enumerated items, topics, or recommendations
- **Tables**: Structured data in columnar format
- **Statements**: Policy decisions, quotes with attribution

## Common Patterns

### Error Handling

All extractors provide detailed error information:

```python
result = await extractor.extract_all_data(url_or_content)

if result.failed_extractions > 0:
    for extraction in result.extractions:
        if extraction.extraction_status == "failed":
            print(f"Failed to extract {extraction.datapoint_name}")
            print(f"Error: {extraction.error_message}")
```

### Cost Tracking

Track costs across all extraction operations:

```python
# All extractors return cost information
result = await extractor.extract_all_data(...)
print(f"Analysis cost: ${result.analysis_cost_cents / 100:.4f}")
print(f"Extraction cost: ${result.extraction_cost_cents / 100:.4f}")
print(f"Total cost: ${result.total_cost_cents / 100:.4f}")
```

### Working with Extracted Data

```python
# Convert to pandas DataFrame
import pandas as pd

data = await extract_pdf_data(pdf_url)
for name, content in data["data"].items():
    if "columns" in content and "data" in content:
        df = pd.DataFrame(content["data"], columns=content["columns"])
        print(f"\n{name}:")
        print(df.head())

# Export to JSON
import json

with open("extracted_data.json", "w") as f:
    json.dump(data["data"], f, indent=2)

# Process numeric data
for name, content in data["data"].items():
    if content.get("data_type") == "numeric_table":
        values = [row[1] for row in content["data"]]  # Assuming second column has values
        print(f"{name} - Average: {sum(values) / len(values)}")
```

### Best Practices

1. **Focus Areas**: Be specific with focus areas to improve extraction accuracy
   ```python
   # Good
   focus_areas=["Q3 2024 revenue", "operating expenses table", "year-over-year growth"]
   
   # Too vague
   focus_areas=["numbers", "data"]
   ```

2. **Provider Selection**: Choose providers based on your needs
   - Anthropic: Best for complex documents, PDFs with caching
   - OpenAI: Good general purpose, strong with structured data
   - Gemini: Excellent for large documents, cost-effective

3. **Datapoint Filtering**: Use filters to reduce costs when you need specific data
   ```python
   # Only extract what you need
   result = await extractor.extract_all_data(
       content,
       datapoint_filter=["revenue_table", "expense_summary"]
   )
   ```

4. **Batch Processing**: Process multiple documents efficiently
   ```python
   async def process_documents(urls):
       extractor = PDFDataExtractor()
       results = []
       
       for url in urls:
           result = await extractor.extract_all_data(url)
           results.append(result)
       
       return results
   ```

## See Also

- [LLM Utilities](llm-utilities.md) - For the underlying LLM functionality
- [API Reference](api-reference.md) - For detailed API documentation
- [Examples](../examples/) - For complete working examples