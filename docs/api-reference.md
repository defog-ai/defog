# API Reference

This document provides a complete API reference for all classes, methods, and functions in the defog-python library.

## Core Classes

### Defog

Main client class for interacting with Defog API.

```python
class Defog:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.defog.ai",
        generate_query_url: Optional[str] = None,
        verbose: bool = False
    )
```

#### Methods

##### run_query
```python
def run_query(
    self,
    question: str,
    db_type: Optional[str] = None,
    db_creds: Optional[dict] = None,
    schema_whitelist: Optional[List[str]] = None,
    schema_blacklist: Optional[List[str]] = None,
    table_metadata: Optional[dict] = None,
    cache_metadata: bool = True,
    use_llm_directly: bool = False,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.0,
    subtable_pruning: bool = False,
    glossary_pruning: bool = False,
    prune_max_tokens: int = 5000,
    prune_bm25_num_columns: int = 30,
    prune_glossary_max_tokens: int = 2000,
    prune_glossary_num_cos_sim_units: int = 5,
    prune_glossary_bm25_units: int = 10,
    profile: bool = False,
    debug: bool = False,
    dev: bool = False,
    temp: bool = False,
    return_error: bool = False,
    ignore_cache: bool = False,
    hard_filters: str = "",
    return_postgres_query: bool = False,
    reflection: bool = True,
    reflection_llm_model: Optional[str] = None,
    reflection_llm_provider: Optional[str] = None,
    previous_context: Optional[List[dict]] = None,
    glossary: Optional[str] = None,
    domain_filter: Optional[str] = None
) -> dict
```

##### get_query
```python
def get_query(
    self,
    question: str,
    db_type: Optional[str] = None,
    schema_whitelist: Optional[List[str]] = None,
    schema_blacklist: Optional[List[str]] = None,
    table_metadata: Optional[dict] = None,
    hard_filters: str = "",
    return_postgres_query: bool = False,
    debug: bool = False,
    dev: bool = False,
    use_llm_directly: bool = False,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    # ... similar pruning parameters as run_query
) -> dict
```

##### generate_postgres_schema
```python
def generate_postgres_schema(
    self,
    db_type: Optional[str] = None,
    db_creds: Optional[dict] = None,
    scan_type: str = "async_bfs",
    schema_whitelist: Optional[List[str]] = None,
    schema_blacklist: Optional[List[str]] = None
) -> dict
```

##### update_glossary
```python
def update_glossary(
    self,
    glossary: Optional[str] = None,
    customized_glossary: Optional[str] = None,
    glossary_compulsory: Optional[List[str]] = None,
    glossary_prunable_units: Optional[List[dict]] = None,
    user_type: str = "default",
    domain: Optional[str] = None
) -> dict
```

##### get_glossary
```python
def get_glossary(
    self,
    mode: str = "standard",  # standard, customized, full
    user_type: str = "default"
) -> str
```

##### delete_glossary
```python
def delete_glossary(
    self,
    user_type: str = "default"
) -> dict
```

##### update_golden_queries
```python
def update_golden_queries(
    self,
    golden_queries: List[dict],
    validate_queries: bool = True,
    scrub: bool = False,
    auto_detect_tables: bool = True,
    add_execution_stats: bool = False
) -> dict
```

##### get_feedback
```python
def get_feedback(
    self,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> List[dict]
```

##### create_empty_tables
```python
def create_empty_tables(
    self,
    table_names: List[str],
    schemas: Optional[List[str]] = None,
    if_not_exists: bool = True
) -> dict
```

##### update_metadata
```python
def update_metadata(
    self,
    metadata_updates: dict
) -> dict
```

#### Health Check Methods

##### check_golden_queries_coverage
```python
def check_golden_queries_coverage(self) -> dict
# Returns:
# {
#     "tables_covered": 10,
#     "total_tables": 15,
#     "columns_covered": 45,
#     "total_columns": 60,
#     "coverage_percentage": 75.0,
#     "uncovered_tables": ["table1", "table2"],
#     "uncovered_columns": [{"table": "t1", "column": "c1"}]
# }
```

##### check_md_valid
```python
def check_md_valid(self) -> bool
```

##### check_gold_queries_valid
```python
def check_gold_queries_valid(self) -> dict
```

##### check_glossary_valid
```python
def check_glossary_valid(self) -> dict
```

##### check_glossary_consistency
```python
def check_glossary_consistency(self) -> dict
```

### AsyncDefog

Async version of the Defog client.

```python
class AsyncDefog(Defog):
    # All methods are async versions of Defog methods
    async def run_query(...) -> dict
    async def get_query(...) -> dict
    async def generate_postgres_schema(...) -> dict
    # etc.
```

## LLM Module

### chat_async

Main function for LLM interactions.

```python
async def chat_async(
    provider: Union[LLMProvider, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_completion_tokens: Optional[int] = None,
    temperature: float = 0.0,
    response_format=None,
    store: bool = True,
    metadata: Optional[Dict[str, str]] = None,
    timeout: int = 600,
    backup_model: Optional[str] = None,
    backup_provider: Optional[Union[LLMProvider, str]] = None,
    prediction: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    tools: Optional[List[Callable]] = None,
    tool_choice: Optional[str] = None,
    max_retries: Optional[int] = None,
    post_tool_function: Optional[Callable] = None,
    config: Optional[LLMConfig] = None,
    mcp_servers: Optional[List[str]] = None,
    image_result_keys: Optional[List[str]] = None,
    tool_budget: Optional[Dict[str, int]] = None,
    insert_tool_citations: bool = False,
    parallel_tool_calls: bool = False,
    tool_output_max_tokens: int = 10000,
    tool_result_preview_max_tokens: Optional[int] = None,
    tool_sample_functions: Optional[Dict[str, Callable]] = None,
    previous_response_id: Optional[str] = None,
) -> LLMResponse
```

- `tool_result_preview_max_tokens`: Optional token budget for the tool output preview that is sent back to the LLM. Full tool outputs are still stored on the response object.
- `tool_sample_functions`: Optional mapping of tool name to a callable (or a single callable) that returns a sampled version of the tool output before it is passed back to the LLM.

### LLMResponse

Response object from LLM calls.

```python
@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: Optional[int] = None
    output_tokens_details: Optional[Dict[str, int]] = None
    cost_in_cents: Optional[float] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[Dict[str, Any]]] = None
```

Each entry in `tool_outputs` contains the full tool `result`, and when sampling/truncation is enabled it also includes a `result_for_llm` preview plus `result_truncated_for_llm` and `sampling_applied` flags.

### Tool Functions

#### code_interpreter_tool
```python
async def code_interpreter_tool(
    question: str,
    model: str,
    provider: Union[str, LLMProvider],
    csv_string: Optional[str] = None,
    instructions: Optional[str] = None,
    allowed_libraries: Optional[List[str]] = None,
    memory_limit_mb: int = 512,
    timeout_seconds: int = 30,
    pre_code: Optional[str] = None,
    return_images: bool = True,
    image_format: str = "png",
    figure_dpi: int = 100,
    allow_file_access: bool = False,
    allow_network_access: bool = False
) -> dict
```

#### web_search_tool
```python
async def web_search_tool(
    question: str,
    model: str,
    provider: Union[str, LLMProvider],
    max_tokens: int = 2048,
    search_depth: str = "standard",  # quick, standard, comprehensive
    max_results: int = 10,
    include_snippets: bool = True,
    follow_links: bool = False,
    preferred_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
    prefer_recent: bool = True,
    date_range: Optional[str] = None,  # e.g., "1y", "6m", "30d"
    content_type: str = "general",  # academic, news, general
    include_timestamps: bool = False,
    group_by_domain: bool = False,
    summarize_per_domain: bool = False
) -> dict
```

#### get_youtube_summary
```python
async def get_youtube_summary(
    video_url: str,
    model: str,
    verbose: bool = False,
    system_instructions: Optional[List[str]] = None,
    include_auto_chapters: bool = True,
    language_preference: str = "en",
    fallback_to_auto_captions: bool = True,
    max_retries: int = 3
) -> str
```

#### citations_tool
```python
async def citations_tool(
    question: str,
    instructions: str,
    documents: List[dict],
    model: str,
    provider: Union[str, LLMProvider],
    max_tokens: int = 16000,
    citation_style: str = "inline",  # academic, inline, footnote
    min_citations_per_claim: int = 1,
    include_confidence_scores: bool = False,
    extract_quotes: bool = False,
    max_quote_length: int = 200,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    relevance_threshold: float = 0.5
) -> dict
```

#### sql_answer_tool
```python
async def sql_answer_tool(
    question: str,
    db_type: str,
    db_creds: dict,
    model: str,
    provider: Union[str, LLMProvider],
    glossary: Optional[str] = None,
    previous_context: Optional[List[dict]] = None,
    whitelist_tables: Optional[List[str]] = None,
    domain_filter: Optional[str] = None,
    is_delegated_tool_call: bool = False,
    verbose: bool = False
) -> dict
```

#### analyze_pdf
```python
async def analyze_pdf(
    pdf_input: PDFAnalysisInput
) -> dict

class PDFAnalysisInput(BaseModel):
    url: str
    task: str
    response_format: Optional[Type[BaseModel]] = None
    max_pages_per_chunk: int = 40
    enable_caching: bool = True
    cache_duration_minutes: int = 5
    parallel_chunks: bool = True
    include_metadata: bool = False
```

## Memory Management

### MemoryManager

```python
class MemoryManager:
    def __init__(
        self,
        token_threshold: int = 100000,
        preserve_last_n_messages: int = 10,
        summary_max_tokens: int = 2000,
        enabled: bool = True
    )
    
    def add_messages(self, messages: List[dict]) -> None
    def get_current_messages(self) -> List[dict]
    def get_stats(self) -> dict
    def needs_compactification(self) -> bool
    async def compactify(self, provider: str, model: str) -> None
```

### EnhancedMemoryManager

```python
class EnhancedMemoryManager(MemoryManager):
    def __init__(
        self,
        token_threshold: int = 100000,
        preserve_last_n_messages: int = 10,
        summary_max_tokens: int = 2000,
        enabled: bool = True,
        enable_cross_agent_memory: bool = True,
        memory_tags: Optional[List[str]] = None,
        compression_ratio: float = 0.3,
        preserve_tool_calls: bool = True,
        preserve_images: bool = False,
        summary_model: Optional[str] = None,
        summary_provider: Optional[str] = None,
        summary_instructions: Optional[str] = None
    )
```

## Agent Orchestration

### EnhancedAgentOrchestrator

```python
class EnhancedAgentOrchestrator:
    def __init__(
        self,
        config: Optional[EnhancedOrchestratorConfig] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        enable_thinking_agents: bool = False,
        enable_shared_context: bool = False,
        enable_alternative_paths: bool = False,
        enable_cross_agent_memory: bool = False,
        max_parallel_agents: int = 5,
        agent_timeout_seconds: int = 300
    )
    
    async def execute_tasks(
        self,
        tasks: List[SubAgentTask],
        agents: Optional[dict] = None,
        continue_on_failure: bool = True,
        collect_thinking_traces: bool = False,
        generate_summary: bool = False
    ) -> dict
    
    async def execute_workflow(
        self,
        tasks: List[SubAgentTask],
        **kwargs
    ) -> dict
```

### ThinkingAgent

```python
class ThinkingAgent:
    def __init__(
        self,
        agent_id: str,
        config: Optional[ThinkingAgentConfig] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    )
    
    async def execute_task(
        self,
        task: str,
        thinking_directives: Optional[List[str]] = None,
        available_agents: Optional[List[str]] = None,
        shared_artifacts: Optional[List[str]] = None
    ) -> dict
```

### SubAgentTask

```python
@dataclass
class SubAgentTask:
    agent_id: str
    task_description: str
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    dependencies: Optional[List[str]] = None
    priority: str = "medium"  # low, medium, high
    estimated_duration_seconds: Optional[int] = None
    input_data: Optional[dict] = None
    require_all_dependencies: bool = True
    confidence_threshold: float = 0.0
    alternative_paths_allowed: bool = False
    condition: Optional[Callable] = None
    fallback_agent: Optional[str] = None
```

## Database Utilities

### MetadataCache

```python
class MetadataCache:
    def __init__(self, ttl_seconds: int = 3600)
    
    def generate_cache_key(
        self,
        db_type: str,
        db_creds: dict
    ) -> str
    
    def get(self, cache_key: str) -> Optional[dict]
    def set(self, cache_key: str, metadata: dict) -> None
    def invalidate(self, cache_key: str) -> None
    def clear_all(self) -> None
    def is_valid(self, cache_key: str) -> bool
```

### SchemaDocumenter

```python
class SchemaDocumenter:
    def __init__(
        self,
        model: str,
        provider: str,
        db_type: str,
        db_creds: dict
    )
    
    async def document_all_tables(self) -> List[dict]
    
    async def document_tables(
        self,
        tables: List[str],
        include_existing_comments: bool = True,
        batch_size: int = 5
    ) -> List[dict]
    
    async def apply_documentation_to_db(
        self,
        documented_tables: List[dict]
    ) -> dict
```

### PDFDataExtractor

```python
class PDFDataExtractor:
    def __init__(
        self,
        analysis_provider: Union[str, LLMProvider] = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: Union[str, LLMProvider] = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0
    )
    
    async def analyze_pdf_structure(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None
    ) -> tuple[PDFAnalysisResponse, Dict[str, Any]]
    
    async def extract_all_data(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> PDFDataExtractionResult
    
    async def extract_as_dict(
        self,
        pdf_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]
```

### ImageDataExtractor

```python
class ImageDataExtractor:
    def __init__(
        self,
        analysis_provider: Union[str, LLMProvider] = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: Union[str, LLMProvider] = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0
    )
    
    async def analyze_image_structure(
        self,
        image_url: str,
        focus_areas: Optional[List[str]] = None
    ) -> tuple[ImageAnalysisResponse, Dict[str, Any]]
    
    async def extract_all_data(
        self,
        image_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> ImageDataExtractionResult
    
    async def extract_as_dict(
        self,
        image_url: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]
```

### HTMLDataExtractor

```python
class HTMLDataExtractor:
    def __init__(
        self,
        analysis_provider: Union[str, LLMProvider] = "anthropic",
        analysis_model: str = "claude-sonnet-4-20250514",
        extraction_provider: Union[str, LLMProvider] = "anthropic",
        extraction_model: str = "claude-sonnet-4-20250514",
        max_parallel_extractions: int = 5,
        temperature: float = 0.0
    )
    
    async def analyze_html_structure(
        self,
        html_content: str,
        focus_areas: Optional[List[str]] = None
    ) -> tuple[HTMLAnalysisResponse, Dict[str, Any]]
    
    async def extract_all_data(
        self,
        html_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> HTMLDataExtractionResult
    
    async def extract_as_dict(
        self,
        html_content: str,
        focus_areas: Optional[List[str]] = None,
        datapoint_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]
```

### Convenience Functions

```python
# PDF extraction
async def extract_pdf_data(
    pdf_url: str,
    focus_areas: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]

# Image extraction
async def extract_image_data(
    image_url: str,
    focus_areas: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]

# HTML extraction
async def extract_html_data(
    html_content: str,
    focus_areas: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]
```

## Utility Functions

### Query Execution

```python
def execute_query(
    query: str,
    db_type: str,
    db_creds: dict
) -> Tuple[List[str], List[List[Any]]]

async def async_execute_query(
    query: str,
    db_type: str,
    db_creds: dict
) -> Tuple[List[str], List[List[Any]]]
```

### Metadata Extraction

```python
def extract_metadata_from_db(
    db_type: str,
    db_creds: dict
) -> dict

async def extract_metadata_from_db_async(
    db_type: str,
    db_creds: dict
) -> dict
```

### Categorical Column Detection

```python
def identify_categorical_columns(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> List[str]

async def async_identify_categorical_columns(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> List[str]
```

### Logging

```python
def write_logs(
    api_key: str,
    log_data: dict
) -> None
```

### DDL Generation

```python
def create_table_ddl(
    table_name: str,
    column_names: List[str],
    column_types: List[str]
) -> str

def create_ddl_from_metadata(
    metadata: dict
) -> List[str]
```

## Exceptions

```python
class DefogError(Exception):
    """Base exception for Defog errors"""

class DefogAPIError(DefogError):
    """API-related errors"""

class DefogConnectionError(DefogError):
    """Database connection errors"""

class DefogValidationError(DefogError):
    """Input validation errors"""

class DefogLLMError(DefogError):
    """LLM-related errors"""
```

## Type Definitions

```python
from typing import TypedDict, Literal, Union
from enum import Enum

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    TOGETHER = "together"

DBType = Literal[
    "postgres", "mysql", "bigquery", "snowflake",
    "databricks", "sqlserver", "redshift", "sqlite",
    "duckdb"
]

class DBCredentials(TypedDict):
    host: str
    port: int
    database: str
    user: str
    password: str
    # Additional optional fields vary by database type
```
