# Metadata Management

This document covers all metadata-related functionality, including caching, validation, and documentation features.

## Metadata Caching

The library includes an intelligent caching system for database metadata to improve performance:

### Basic Usage

```python
from defog.metadata_cache import MetadataCache

# Initialize cache with custom TTL
cache = MetadataCache(ttl_seconds=7200)  # 2 hour TTL

# Generate cache key from credentials
cache_key = cache.generate_cache_key(
    db_type="postgres",
    db_creds={
        "host": "localhost",
        "database": "mydb",
        "user": "postgres"
    }
)

# Store metadata
cache.set(cache_key, metadata)

# Retrieve metadata
cached_metadata = cache.get(cache_key)
if cached_metadata:
    print("Using cached metadata")
else:
    print("Cache miss - need to fetch metadata")

# Check if cache is valid
if cache.is_valid(cache_key):
    metadata = cache.get(cache_key)

# Invalidate specific cache
cache.invalidate(cache_key)

# Clear all cache
cache.clear_all()
```

### Cache Location

Metadata cache is stored in:
- **Directory**: `~/.defog/cache/`
- **Format**: JSON files with timestamp
- **Naming**: SHA256 hash of connection details

### Automatic Caching in Queries

Enable automatic metadata caching:

```python
result = await df.run_query(
    question="Show me all customers",
    db_type="postgres",
    db_creds=db_creds,
    cache_metadata=True  # Enable caching
)
```

## Local Metadata Extraction

Extract metadata directly from databases without API calls:

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
    db_type="mysql",
    db_creds=db_creds
)

# Metadata structure
print(metadata)
# {
#     "table_metadata": [
#         {
#             "table_name": "customers",
#             "table_schema": "public",
#             "column_metadata": [
#                 {
#                     "column_name": "id",
#                     "data_type": "integer",
#                     "is_nullable": "NO",
#                     "column_default": "nextval(...)"
#                 }
#             ]
#         }
#     ]
# }
```

## Metadata Validation

### Schema Validation

```python
# Check if metadata is valid
is_valid = df.check_md_valid()

if not is_valid:
    print("Metadata validation failed")
    # Get detailed validation errors
    errors = df.get_metadata_validation_errors()
    for error in errors:
        print(f"Error: {error}")
```

### Health Checks

Comprehensive health checks for your setup:

```python
# Check golden query coverage
coverage = df.check_golden_queries_coverage()
print(f"Tables covered: {coverage['tables_covered']}/{coverage['total_tables']}")
print(f"Columns covered: {coverage['columns_covered']}/{coverage['total_columns']}")
print(f"Coverage percentage: {coverage['coverage_percentage']}%")

# Get uncovered tables/columns
for table in coverage['uncovered_tables']:
    print(f"Uncovered table: {table}")

for col in coverage['uncovered_columns']:
    print(f"Uncovered column: {col['table']}.{col['column']}")

# Validate golden queries
valid_queries = df.check_gold_queries_valid()
print(f"Valid queries: {valid_queries['valid_count']}/{valid_queries['total_count']}")

# Check glossary validity
glossary_valid = df.check_glossary_valid()
if not glossary_valid['is_valid']:
    for error in glossary_valid['errors']:
        print(f"Glossary error: {error}")

# Check glossary consistency
consistency = df.check_glossary_consistency()
if not consistency['is_consistent']:
    for issue in consistency['issues']:
        print(f"Consistency issue: {issue}")
```

## Metadata Documentation

### Vetting Long Descriptions

For columns with verbose descriptions (> 100 characters), you can use the metadata vetting functionality to generate concise descriptions programmatically.

## Admin Metadata Methods

### Create Empty Tables

Create empty tables based on metadata:

```python
# Create tables from metadata
df.create_empty_tables(
    table_names=["customers", "orders"],
    schemas=["public"],  # Optional: specify schemas
    if_not_exists=True   # Don't error if tables exist
)
```

### DDL Generation

Generate DDL statements from metadata:

```python
from defog.admin_methods import create_ddl_from_metadata

# Generate CREATE TABLE statements
ddl_statements = create_ddl_from_metadata(metadata)

for statement in ddl_statements:
    print(statement)
    # CREATE TABLE public.customers (
    #     id INTEGER PRIMARY KEY,
    #     name VARCHAR(255),
    #     email VARCHAR(255)
    # );
```

### Update Metadata

Update metadata with custom descriptions:

```python
# Update table and column descriptions
updated_metadata = df.update_metadata(
    metadata_updates={
        "customers": {
            "table_description": "Customer information",
            "columns": {
                "id": "Unique customer identifier",
                "name": "Customer full name",
                "email": "Customer email address"
            }
        }
    }
)
```

## Metadata Storage

### Configuration File

Metadata and settings are stored in:
- **Location**: `~/.defog/connection.json`
- **Contents**: API keys, endpoints, default settings

### Log Files

Query and error logs are stored in:
- **Location**: `~/.defog/logs/`
- **Format**: JSON lines with timestamps
- **Rotation**: Automatic rotation based on size

## Performance Optimization

### Metadata Pruning

For large databases, use pruning to improve performance:

```python
# Enable intelligent pruning
result = await df.run_query(
    question="Show customer orders",
    db_type="postgres",
    db_creds=db_creds,
    
    # Pruning options
    subtable_pruning=True,  # Prune irrelevant tables
    prune_max_tokens=5000,  # Token limit for metadata
    
    # Column pruning
    prune_bm25_num_columns=50,  # Keep top 50 relevant columns
    
    # Schema filtering
    schema_whitelist=["public", "sales"],  # Only these schemas
    schema_blacklist=["temp", "staging"]   # Exclude these
)
```

### Batch Operations

Process multiple metadata operations efficiently:

```python
# Batch metadata updates
batch_updates = [
    {"table": "customers", "updates": {...}},
    {"table": "orders", "updates": {...}},
    {"table": "products", "updates": {...}}
]

for update in batch_updates:
    df.update_table_metadata(
        table_name=update["table"],
        metadata=update["updates"]
    )
```

## Best Practices

1. **Enable caching** for frequently accessed databases
2. **Set appropriate TTL** based on schema change frequency
3. **Use local extraction** when possible to avoid API calls
4. **Validate metadata** regularly with health checks
5. **Document schemas** with meaningful descriptions
6. **Prune large schemas** to improve query performance
7. **Monitor cache size** and clear periodically if needed
8. **Use async methods** for better performance
9. **Batch updates** when modifying multiple tables
10. **Version control** your metadata definitions