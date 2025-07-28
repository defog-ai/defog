# Advanced Configuration

This document covers advanced configuration options, hidden parameters, and optimization techniques for the defog-python library.

## Query Configuration

### Advanced Query Parameters

The `run_query` method supports many undocumented parameters:

```python
from defog import Defog

df = Defog(api_key="your-api-key")

result = await df.run_query(
    question="Complex business question",
    
    # Database options
    db_type="postgres",
    db_creds=db_creds,
    schema_whitelist=["public", "sales"],     # Only use these schemas
    schema_blacklist=["temp", "staging"],     # Exclude these schemas
    
    # LLM options
    use_llm_directly=True,                    # Use local LLM instead of API
    llm_provider="openai",                    # LLM provider
    llm_model="gpt-4o",                       # Specific model
    temperature=0.1,                          # Lower = more deterministic
    
    # Metadata options
    table_metadata=custom_metadata,           # Provide metadata directly
    cache_metadata=True,                      # Cache extracted metadata
    ignore_cache=False,                       # Use cached metadata if available
    
    # Pruning configuration
    subtable_pruning=True,                    # Enable table filtering
    glossary_pruning=True,                    # Enable glossary filtering
    prune_max_tokens=5000,                    # Max tokens for schema
    prune_bm25_num_columns=30,                # BM25 column limit
    prune_glossary_max_tokens=2000,           # Glossary token limit
    prune_glossary_num_cos_sim_units=5,      # Cosine similarity units
    prune_glossary_bm25_units=10,             # BM25 glossary units
    
    # Execution options
    profile=True,                             # Return timing information
    debug=True,                               # Enable debug logging
    dev=False,                                # Use dev environment
    temp=False,                               # Temporary query flag
    return_error=True,                        # Return errors vs raising
    
    # Advanced options
    hard_filters="",                          # SQL WHERE clause filters
    return_postgres_query=False,              # Convert to PostgreSQL
    
    # Reflection options (when query fails)
    reflection=True,                          # Enable query reflection
    reflection_llm_model="gpt-4o",            # Model for reflection
    reflection_llm_provider="openai",         # Provider for reflection
    
    # Previous context for conversational SQL
    previous_context=[
        {"question": "Show customers", "query": "SELECT * FROM customers"},
        {"question": "Filter by status", "query": "... WHERE status = 'active'"}
    ],
    
    # Domain-specific options
    glossary="Revenue: Total sales amount",   # Business glossary
    domain_filter="retail",                   # Domain context
)
```

### Performance Optimization

```python
# Optimize for large databases
optimization_config = {
    # Schema optimization
    "subtable_pruning": True,
    "prune_max_tokens": 3000,              # Reduce schema size
    "prune_bm25_num_columns": 20,          # Keep only top columns
    
    # Query optimization  
    "cache_metadata": True,                 # Cache schemas
    "use_llm_directly": True,              # Skip API overhead
    
    # Parallel execution
    "parallel_reflection": True,            # Parallel error handling
    "batch_size": 5,                       # Batch similar queries
}

result = await df.run_query(
    question="Show metrics",
    **optimization_config
)
```

## Glossary Configuration

### Advanced Glossary Features

```python
# Update glossary with advanced options
df.update_glossary(
    glossary="Standard business terms...",
    
    # Customized glossary (user-specific)
    customized_glossary="Department-specific terms...",
    
    # Compulsory terms (always included)
    glossary_compulsory=[
        "Revenue: Total sales including tax",
        "Margin: Gross profit percentage"
    ],
    
    # Prunable units (can be filtered)
    glossary_prunable_units=[
        {"term": "Legacy Metric", "definition": "Old calculation method"},
        {"term": "Test Field", "definition": "Used for testing only"}
    ],
    
    # Metadata
    user_type="analyst",                    # User role context
    domain="finance"                        # Domain context
)
```

### Glossary Modes

```python
# Get different glossary views
standard_glossary = df.get_glossary(mode="standard")
customized_glossary = df.get_glossary(mode="customized")
full_glossary = df.get_glossary(mode="full")

# Delete user-specific glossary
df.delete_glossary(user_type="analyst")
```

## Golden Query Configuration

### Advanced Golden Query Management

```python
# Update golden queries with validation
df.update_golden_queries(
    golden_queries=[
        {
            "question": "Monthly revenue trend",
            "query": "SELECT DATE_TRUNC('month', date) as month, SUM(amount) as revenue FROM sales GROUP BY 1",
            "metadata": {
                "complexity": "medium",
                "tables_used": ["sales"],
                "filters_available": ["date", "product", "region"]
            }
        }
    ],
    
    # Validation options
    validate_queries=True,                  # Test queries before saving
    scrub=True,                            # Clean/format queries
    
    # Metadata enrichment
    auto_detect_tables=True,               # Extract table references
    add_execution_stats=True               # Add performance metrics
)
```

## Client Configuration

### Defog Client Options

```python
from defog import Defog, AsyncDefog

# Synchronous client with all options
df = Defog(
    api_key="your-api-key",
    
    # API configuration
    base_url="https://custom-api.example.com",
    generate_query_url="https://custom-api.example.com/generate",
    
    # Logging
    verbose=True,                          # Enable verbose logging
    log_level="DEBUG",                     # Set log level
    
    # Retry configuration
    max_retries=3,
    retry_delay=1.0,
    retry_backoff=2.0,
    
    # Timeout configuration
    timeout_seconds=30,
    query_timeout_seconds=120,
    
    # Rate limiting
    rate_limit_per_minute=60,
    concurrent_requests=5
)

# Async client with connection pooling
async_df = AsyncDefog(
    api_key="your-api-key",
    
    # Connection pool
    connection_pool_size=10,
    max_connections_per_host=5,
    
    # Performance
    enable_http2=True,
    compress_requests=True
)
```

### Environment-Based Configuration

```python
import os
from defog import Defog

# Configuration hierarchy (in order of precedence):
# 1. Explicit parameters
# 2. Environment variables
# 3. Config files
# 4. Defaults

# Set up environment
os.environ["DEFOG_API_KEY"] = "your-api-key"
os.environ["DEFOG_BASE_URL"] = "https://api.defog.ai"
os.environ["DEFOG_VERBOSE"] = "true"
os.environ["DEFOG_CACHE_TTL"] = "7200"

# Client uses environment automatically
df = Defog()  # Uses environment variables
```

## Database Configuration

### Connection Pool Configuration

```python
# PostgreSQL with connection pooling
pg_creds = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "user": "postgres",
    "password": "password",
    
    # Connection pool options
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    
    # SSL options
    "sslmode": "require",
    "sslcert": "/path/to/client-cert.pem",
    "sslkey": "/path/to/client-key.pem"
}

# MySQL with advanced options
mysql_creds = {
    "host": "localhost",
    "port": 3306,
    "database": "mydb",
    "user": "root",
    "password": "password",
    
    # Character encoding
    "charset": "utf8mb4",
    "use_unicode": True,
    
    # Performance
    "autocommit": True,
    "connect_timeout": 10
}
```

### BigQuery Configuration

```python
# BigQuery with all options
bigquery_creds = {
    "project_id": "my-project",
    "dataset_id": "my_dataset",
    
    # Authentication options
    "credentials_path": "/path/to/service-account.json",
    # OR
    "credentials_json": {"type": "service_account", ...},
    
    # Performance
    "location": "US",
    "use_query_cache": True,
    "use_legacy_sql": False,
    
    # Limits
    "maximum_bytes_billed": 1000000000,  # 1GB limit
    "timeout_ms": 30000
}
```

## Logging Configuration

### Advanced Logging

```python
import logging
from defog import Defog

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('defog.log'),
        logging.StreamHandler()
    ]
)

# Enable specific loggers
logging.getLogger('defog.query').setLevel(logging.DEBUG)
logging.getLogger('defog.llm').setLevel(logging.INFO)
logging.getLogger('defog.cache').setLevel(logging.WARNING)

# Create client with custom logger
df = Defog(
    api_key="your-api-key",
    logger=logging.getLogger('my_app.defog')
)
```

### Query Logging

```python
# Enable query logging to file
from defog.util import write_logs

# Log queries automatically
def logged_query(question, **kwargs):
    result = df.run_query(question, **kwargs)
    
    write_logs(
        api_key=df.api_key,
        log_data={
            "question": question,
            "query": result.get("query"),
            "execution_time": result.get("execution_time"),
            "status": "success" if result.get("columns") else "error"
        }
    )
    
    return result
```

## Error Handling Configuration

### Retry Configuration

```python
from defog import Defog
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure client with custom retry logic
df = Defog(
    api_key="your-api-key",
    retry_config={
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=1, min=4, max=60),
        "retry_on_exceptions": [ConnectionError, TimeoutError],
        "before_sleep": lambda retry_state: print(f"Retrying... attempt {retry_state.attempt_number}")
    }
)
```

### Error Callbacks

```python
# Custom error handling
def handle_query_error(error, context):
    """Custom error handler"""
    if isinstance(error, ValueError):
        # Handle validation errors
        return {"error": "Invalid input", "suggestion": "Check your question"}
    elif isinstance(error, ConnectionError):
        # Handle connection errors
        return {"error": "Database unreachable", "retry": True}
    else:
        # Default handling
        raise error

df.error_handler = handle_query_error
```

## Performance Monitoring

### Metrics Collection

```python
# Enable performance metrics
from defog.metrics import MetricsCollector

metrics = MetricsCollector()

# Wrap client with metrics
df = Defog(
    api_key="your-api-key",
    metrics_collector=metrics
)

# Query with metrics
result = df.run_query("Show metrics", profile=True)

# Get performance data
perf_data = metrics.get_metrics()
print(f"Average query time: {perf_data['avg_query_time']}s")
print(f"Cache hit rate: {perf_data['cache_hit_rate']:.2%}")
```

## Security Configuration

### API Key Management

```python
# Secure API key handling
from defog.security import SecureConfig

# Load from secure storage
config = SecureConfig.from_keyring("defog_credentials")

# Or from encrypted file
config = SecureConfig.from_encrypted_file(
    "config.enc",
    password="your-password"
)

df = Defog(
    api_key=config.api_key,
    db_creds=config.db_creds
)
```

### Query Sanitization

```python
# Enable query sanitization
df = Defog(
    api_key="your-api-key",
    security_config={
        "sanitize_queries": True,
        "allowed_statements": ["SELECT"],
        "forbidden_keywords": ["DROP", "DELETE", "TRUNCATE"],
        "max_query_length": 10000,
        "parameter_validation": True
    }
)
```

## Best Practices

1. **Use environment variables** for sensitive configuration
2. **Enable caching** for better performance
3. **Configure appropriate timeouts** based on query complexity
4. **Set up proper logging** for debugging
5. **Use connection pooling** for database connections
6. **Implement retry logic** for resilience
7. **Monitor performance metrics** to optimize
8. **Sanitize queries** in production environments
9. **Use async clients** for concurrent operations
10. **Configure pruning** for large databases