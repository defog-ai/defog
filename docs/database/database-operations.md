# Database Operations

Comprehensive documentation for all database-related functionality in the defog library.

## Table of Contents

- [Supported Databases](#supported-databases)
- [SQL Agent Tools](#sql-agent-tools)
- [Query Execution](#query-execution)
- [Schema Documentation](#schema-documentation)
- [Metadata Management](#metadata-management)
- [Local LLM SQL Generation](#local-llm-sql-generation)
- [Health Check Utilities](#health-check-utilities)
- [Database Utilities](#database-utilities)
- [Best Practices](#best-practices)

## Supported Databases

The library supports 11 database types:

| Database | Type String | Connection Requirements |
|----------|------------|------------------------|
| PostgreSQL | `postgres` | host, port, database, user, password |
| MySQL | `mysql` | host, port, database, user, password |
| BigQuery | `bigquery` | project_id, dataset_id, credentials_json |
| Snowflake | `snowflake` | account, warehouse, database, schema, user, password |
| Databricks | `databricks` | server_hostname, http_path, access_token |
| SQL Server | `sqlserver` | server, database, user, password |
| Redshift | `redshift` | host, port, database, user, password |
| SQLite | `sqlite` | database (file path) |
| DuckDB | `duckdb` | database (file path) |

## SQL Agent Tools

Convert natural language questions to SQL queries and execute them on local databases.

### Basic SQL Generation and Execution

```python
from defog.llm.sql import sql_answer_tool
from defog.llm.llm_providers import LLMProvider

# Database connection
db_creds = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "postgres",
    "password": "password"
}

# Ask questions in natural language
result = await sql_answer_tool(
    question="What are the top 10 customers by total sales?",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-sonnet-4-20250514",
    provider=LLMProvider.ANTHROPIC
)

if result["success"]:
    print(f"SQL Query: {result['query']}")
    print(f"Results: {result['results']}")
    print(f"Columns: {result['columns']}")
else:
    print(f"Error: {result['error']}")
```

### Advanced SQL Agent Features

```python
# With business context and conversation history
result = await sql_answer_tool(
    question="Show me the trend for the same customers",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-sonnet-4-20250514",
    provider=LLMProvider.ANTHROPIC,
    
    # Business context
    glossary="""
    Total Sales: Sum of all order amounts for a customer
    Active Customer: Customer with purchase in last 90 days
    CLV: Customer Lifetime Value - total spend since first purchase
    """,
    
    # Conversation history for follow-up questions
    previous_context=[
        {
            "question": "What are the top 10 customers by total sales?",
            "sql_query": "SELECT customer_id, SUM(amount) as total_sales...",
            "results": "[(1, 50000), (2, 45000), ...]"
        }
    ],
    
    # Advanced options
    whitelist_tables=["customers", "orders", "order_items"],
    domain_filter="ecommerce",
    verbose=True,  # Show detailed logging
    is_delegated_tool_call=False  # For agent orchestration
)
```

### Automatic Table Filtering for Large Databases

For databases with many tables (>5) or columns (>1000), the tool automatically filters relevant tables:

```python
from defog.llm.sql import identify_relevant_tables_tool

# First, identify relevant tables for your question
relevance_result = await identify_relevant_tables_tool(
    question="Show me customer orders from last month",
    db_type="postgres",
    db_creds=db_creds,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    max_tables=10,  # Return top 10 most relevant tables
    glossary="Monthly orders: Orders placed within a calendar month"
)

print(f"Relevant tables: {relevance_result['tables']}")
print(f"Reasoning: {relevance_result['reasoning']}")

# Then use those tables for SQL generation
result = await sql_answer_tool(
    question="Show me customer orders from last month",
    db_type="postgres",
    db_creds=db_creds,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    whitelist_tables=relevance_result['tables']  # Use filtered tables
)
```

### Conversational SQL

Maintain context across multiple questions:

```python
# Initialize conversation context
conversation_context = []

# First question
result1 = await sql_answer_tool(
    question="Show me top 5 products by revenue",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    previous_context=conversation_context
)

# Update context
if result1["success"]:
    conversation_context.append({
        "question": "Show me top 5 products by revenue",
        "sql_query": result1["query"],
        "results": str(result1["results"][:5])  # Sample results
    })

# Follow-up question
result2 = await sql_answer_tool(
    question="What's the profit margin for these products?",
    db_type="postgres",
    db_creds=db_creds,
    model="claude-3-5-sonnet",
    provider=LLMProvider.ANTHROPIC,
    previous_context=conversation_context  # Include history
)
```

## Query Execution

Execute SQL queries directly without LLM generation.

### Basic Query Execution

```python
from defog.query import execute_query, async_execute_query

# Synchronous execution
colnames, results = execute_query(
    query="SELECT * FROM customers LIMIT 10",
    db_type="postgres",
    db_creds={
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "user": "postgres",
        "password": "password"
    }
)

print(f"Columns: {colnames}")
print(f"Results: {results}")

# Asynchronous execution (recommended)
colnames, results = await async_execute_query(
    query="SELECT COUNT(*) FROM orders WHERE status = 'completed'",
    db_type="postgres",
    db_creds=db_creds
)
```

### Database-Specific Examples

```python
# BigQuery
bq_creds = {
    "project_id": "my-project",
    "dataset_id": "my_dataset",
    "credentials_json": json.dumps(service_account_key)
}

colnames, results = await async_execute_query(
    query="SELECT * FROM `my-project.my_dataset.customers` LIMIT 10",
    db_type="bigquery",
    db_creds=bq_creds
)

# Snowflake
snow_creds = {
    "account": "my-account",
    "warehouse": "my_warehouse",
    "database": "my_database",
    "schema": "public",
    "user": "my_user",
    "password": "my_password"
}

colnames, results = await async_execute_query(
    query="SELECT * FROM customers LIMIT 10",
    db_type="snowflake",
    db_creds=snow_creds
)

## Schema Documentation

Automatically generate intelligent documentation for your database schemas using AI.

### Basic Schema Documentation

```python
from defog.llm.schema_documenter import SchemaDocumenter

# Initialize documenter
documenter = SchemaDocumenter(
    model="claude-3-5-sonnet",
    provider="anthropic",
    db_type="postgres",
    db_creds=db_creds
)

# Document all tables
all_docs = await documenter.document_all_tables()

# Document specific tables
documented_tables = await documenter.document_tables(
    tables=["customers", "orders", "products"],
    include_existing_comments=True  # Preserve existing DB comments
)

# Display documentation
for table_doc in documented_tables:
    print(f"\nTable: {table_doc['table_name']}")
    print(f"Description: {table_doc['table_description']}")
    print(f"Confidence: {table_doc['confidence']}")
    
    for col in table_doc['columns']:
        print(f"  - {col['column_name']} ({col['data_type']})")
        print(f"    {col['description']} [confidence: {col['confidence']}]")
```

### Advanced Documentation Features

```python
# With batch processing and custom settings
documented_tables = await documenter.document_tables(
    tables=["customers", "orders", "products", "inventory", "suppliers"],
    include_existing_comments=True,
    batch_size=3,  # Process 3 tables at a time
    temperature=0.1,  # Lower temperature for consistency
    max_retries=3  # Retry failed documentations
)

# Schema documentation includes:
# - Automatic pattern detection (emails, URLs, UUIDs, phone numbers)
# - Categorical column identification
# - Foreign key relationship detection
# - Business logic inference
# - Confidence scoring for each description
```

### Export Documentation

```python
# Export to markdown
def export_to_markdown(documented_tables, output_file="schema_docs.md"):
    with open(output_file, "w") as f:
        f.write("# Database Schema Documentation\n\n")
        
        for table in documented_tables:
            f.write(f"## {table['table_name']}\n\n")
            f.write(f"{table['table_description']}\n\n")
            f.write("| Column | Type | Description |\n")
            f.write("|--------|------|-------------|\n")
            
            for col in table['columns']:
                f.write(f"| {col['column_name']} | {col['data_type']} | {col['description']} |\n")
            
            f.write("\n")

# Export documentation
export_to_markdown(documented_tables)
```

## Metadata Management

### Extract Metadata

```python
from defog.local_metadata_extractor import (
    extract_metadata_from_db,
    extract_metadata_from_db_async
)

# Synchronous extraction
metadata = extract_metadata_from_db(
    db_type="postgres",
    db_creds=db_creds
)

# Asynchronous extraction (recommended)
metadata = await extract_metadata_from_db_async(
    db_type="postgres",
    db_creds=db_creds
)

# Metadata structure
print(f"Tables: {len(metadata)}")
for table in metadata[:5]:
    print(f"\nTable: {table['table_name']}")
    print(f"Schema: {table.get('table_schema', 'public')}")
    print(f"Columns: {len(table['columns'])}")
    print(f"Column names: {[col['column_name'] for col in table['columns'][:5]]}")
```

### Metadata Caching

Improve performance with intelligent metadata caching:

```python
from defog.metadata_cache import MetadataCache

# Initialize cache with TTL
cache = MetadataCache(ttl_seconds=3600)  # 1 hour TTL

# Generate cache key
cache_key = cache.generate_cache_key(db_type, db_creds)

# Cache metadata
cache.set(cache_key, metadata)

# Retrieve cached metadata
cached_metadata = cache.get(cache_key)
if cached_metadata:
    print("Using cached metadata")
else:
    print("Cache miss - fetching metadata")
    metadata = await extract_metadata_from_db_async(db_type, db_creds)
    cache.set(cache_key, metadata)

# Cache management
cache.invalidate(cache_key)  # Invalidate specific cache
cache.clear_all()  # Clear all cache
```

## Local LLM SQL Generation

Generate SQL queries using local LLMs without Defog API:

```python
from defog import Defog

# No API key needed for local generation
df = Defog()

# Use local LLM for SQL generation
result = await df.run_query(
    question="Show me top customers by total sales",
    db_type="postgres",
    db_creds=db_creds,
    use_llm_directly=True,  # Use local LLM instead of Defog API
    llm_provider="openai",
    llm_model="gpt-4o",
    table_metadata=metadata  # Optional: provide schema directly
)

if result["success"]:
    print(f"Generated SQL: {result['query_generated']}")
    print(f"Results: {result['data']}")
    print(f"Columns: {result['columns']}")
```

### Advanced Local SQL Generation

```python
# With custom prompts and settings
result = await df.run_query(
    question="Calculate monthly revenue growth rate",
    db_type="postgres",
    db_creds=db_creds,
    use_llm_directly=True,
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet",
    
    # Advanced options
    table_metadata=metadata,
    glossary="Revenue: Sum of order_amount from orders table",
    custom_prompt="Generate efficient SQL with CTEs when appropriate",
    temperature=0.1,
    max_tokens=2000,
    
    # Query optimization hints
    prefer_cte=True,  # Use CTEs for complex queries
    avoid_subqueries=True,  # Prefer JOINs over subqueries
    include_comments=True  # Add SQL comments
)
```

## Health Check Utilities

Validate your configuration and data quality:

### Golden Query Coverage

Check how well your golden queries cover your schema:

```python
# Check golden query coverage
coverage = df.check_golden_queries_coverage()

print(f"Table coverage: {coverage['table_coverage']}%")
print(f"Column coverage: {coverage['column_coverage']}%")
print(f"Tables not covered: {coverage['uncovered_tables']}")
print(f"Columns not covered: {coverage['uncovered_columns']}")

# Detailed coverage report
for table, details in coverage['table_details'].items():
    print(f"\n{table}:")
    print(f"  Columns covered: {details['covered_columns']}")
    print(f"  Columns missing: {details['missing_columns']}")
```

### Metadata Validation

```python
# Validate metadata format
is_valid = df.check_md_valid()
if not is_valid:
    print("Metadata validation failed")
    validation_errors = df.get_metadata_validation_errors()
    for error in validation_errors:
        print(f"Error: {error}")
```

### Glossary Consistency

```python
# Check glossary consistency
consistency = df.check_glossary_consistency()

print(f"Consistent terms: {consistency['consistent_count']}")
print(f"Inconsistent terms: {consistency['inconsistent_count']}")

# Show inconsistencies
for term, issues in consistency['inconsistencies'].items():
    print(f"\nTerm: {term}")
    print(f"Issues: {issues}")
```

## Database Utilities

### Categorical Column Detection

```python
from defog.util import identify_categorical_columns
import pandas as pd

# Load your data
df_data = pd.read_sql("SELECT * FROM products", connection)

# Identify categorical columns
categorical_cols = identify_categorical_columns(
    df=df_data,
    threshold=0.1  # Columns with <10% unique values
)

print(f"Categorical columns: {categorical_cols}")

# Get unique values for categorical columns
for col in categorical_cols:
    unique_values = df_data[col].unique()
    print(f"{col}: {unique_values[:10]}")  # First 10 values
```

### DDL Generation

```python
from defog.admin_methods import create_table_ddl, create_ddl_from_metadata

# Generate DDL from column definitions
ddl = create_table_ddl(
    table_name="new_customers",
    column_names=["id", "name", "email", "created_at"],
    column_types=["SERIAL PRIMARY KEY", "VARCHAR(255)", "VARCHAR(255)", "TIMESTAMP"]
)
print(ddl)

# Generate DDL from metadata
all_ddls = create_ddl_from_metadata(metadata)
for table_name, ddl in all_ddls.items():
    print(f"\n-- {table_name}")
    print(ddl)
```

### Schema Analysis

```python
# Analyze schema complexity
def analyze_schema(metadata):
    total_tables = len(metadata)
    total_columns = sum(len(table['columns']) for table in metadata)
    
    # Find largest tables
    tables_by_columns = sorted(
        metadata, 
        key=lambda t: len(t['columns']), 
        reverse=True
    )
    
    # Find tables with most relationships
    tables_with_fks = [
        t for t in metadata 
        if any(col.get('foreign_key') for col in t['columns'])
    ]
    
    return {
        "total_tables": total_tables,
        "total_columns": total_columns,
        "avg_columns_per_table": total_columns / total_tables,
        "largest_tables": [(t['table_name'], len(t['columns'])) 
                          for t in tables_by_columns[:5]],
        "tables_with_foreign_keys": len(tables_with_fks)
    }

analysis = analyze_schema(metadata)
print(f"Schema Analysis: {json.dumps(analysis, indent=2)}")
```

## Best Practices

### 1. Connection Management

```python
# Use connection pooling for production
from defog.db_utils import create_connection_pool

pool = create_connection_pool(
    db_type="postgres",
    db_creds=db_creds,
    min_connections=2,
    max_connections=10
)

# Use context managers
async with pool.acquire() as conn:
    results = await conn.fetch("SELECT * FROM customers LIMIT 10")
```

### 2. Error Handling

```python
from defog.exceptions import DatabaseError, QueryExecutionError

try:
    result = await sql_answer_tool(
        question="Show revenue",
        db_type="postgres",
        db_creds=db_creds,
        model="gpt-4o",
        provider=LLMProvider.OPENAI
    )
except DatabaseError as e:
    print(f"Database connection error: {e}")
except QueryExecutionError as e:
    print(f"Query execution failed: {e}")
    # Log the failed query for debugging
    print(f"Failed query: {e.query}")
```

### 3. Performance Optimization

- **Enable metadata caching** for repeated queries
- **Use table whitelisting** for large databases (>1000 columns)
- **Provide business glossaries** to improve query accuracy
- **Use asynchronous methods** for better concurrency
- **Batch operations** when documenting multiple tables

### 4. Security Considerations

```python
# Validate and sanitize inputs
def validate_db_creds(db_creds):
    required_fields = ["host", "database", "user", "password"]
    for field in required_fields:
        if field not in db_creds:
            raise ValueError(f"Missing required field: {field}")
    
    # Don't allow certain characters in database names
    if any(char in db_creds["database"] for char in [";", "--", "/*"]):
        raise ValueError("Invalid database name")

# Use read-only connections when possible
read_only_creds = {
    **db_creds,
    "options": "-c default_transaction_read_only=on"
}

# Limit query execution time
result = await sql_answer_tool(
    question="Complex analysis query",
    db_type="postgres",
    db_creds=read_only_creds,
    model="gpt-4o",
    provider=LLMProvider.OPENAI,
    query_timeout=30  # 30 second timeout
)
```

### 5. Monitoring and Logging

```python
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log queries and performance
async def monitored_sql_query(question, db_type, db_creds, **kwargs):
    start_time = datetime.now()
    
    try:
        result = await sql_answer_tool(
            question=question,
            db_type=db_type,
            db_creds=db_creds,
            **kwargs
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Query successful: {duration:.2f}s - {question[:50]}...")
        
        # Log slow queries
        if duration > 10:
            logger.warning(f"Slow query ({duration:.2f}s): {result['query']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Query failed: {question} - {str(e)}")
        raise
```

## See Also

- [LLM Utilities](llm-utilities.md) - For the underlying LLM functionality
- [Data Extraction](data-extraction.md) - For extracting data from documents
- [Agent Orchestration](agent-orchestration.md) - For multi-agent SQL workflows
- [API Reference](api-reference.md) - Complete API documentation