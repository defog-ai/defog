from defog import (
    generate_schema,
    async_generate_schema,
    query_methods,
    async_query_methods,
    admin_methods,
    async_admin_methods,
    health_methods,
    async_health_methods,
)
from typing import Optional, Union
from defog.llm.llm_providers import LLMProvider
import warnings

from defog.__version__ import __version__

SUPPORTED_DB_TYPES = [
    "postgres",
    "redshift",
    "mysql",
    "bigquery",
    "snowflake",
    "databricks",
    "sqlserver",
    "sqlite",
    "duckdb",
]


class BaseDefog:
    """
    The base class for Defog and AsyncDefog
    """

    api_key: Optional[str]
    db_type: str
    db_creds: dict
    base_url: str
    generate_query_url: str

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_type: str = "",
        db_creds: dict = {},
        base_url: str = "https://api.defog.ai",
        generate_query_url: str = "https://api.defog.ai/generate_query_chat",
        verbose: bool = False,
    ):
        if api_key is not None:
            warnings.warn(
                "The 'api_key' parameter is deprecated and will be removed in a future version. "
                "Defog now focuses on local generation and no longer requires an API key.",
                DeprecationWarning,
                stacklevel=3,
            )
        self.check_db_creds(db_type, db_creds)
        self.api_key = api_key
        self.db_type = db_type
        self.db_creds = db_creds
        self.base_url = base_url
        self.generate_query_url = generate_query_url

    @staticmethod
    def check_db_creds(db_type: str, db_creds: dict):
        # print(db_creds)
        if db_creds == {}:
            # special case for empty db_creds. Some customers just want these to be empty so they can just get the query and run it without giving the defog library any credentials
            return
        if db_type == "postgres" or db_type == "redshift":
            if "host" not in db_creds:
                raise KeyError("db_creds must contain a 'host' key.")
            if "port" not in db_creds:
                raise KeyError("db_creds must contain a 'port' key.")
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "mysql":
            if "host" not in db_creds:
                raise KeyError("db_creds must contain a 'host' key.")
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "snowflake":
            if "account" not in db_creds:
                raise KeyError("db_creds must contain a 'account' key.")
            if "warehouse" not in db_creds:
                raise KeyError("db_creds must contain a 'warehouse' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "databricks":
            if "server_hostname" not in db_creds:
                raise KeyError("db_creds must contain a 'server_hostname' key.")
            if "access_token" not in db_creds:
                raise KeyError("db_creds must contain a 'access_token' key.")
            if "http_path" not in db_creds:
                raise KeyError("db_creds must contain a 'http_path' key.")
        elif db_type == "sqlserver":
            if "server" not in db_creds:
                raise KeyError("db_creds must contain a 'server' key.")
            if "user" not in db_creds:
                raise KeyError("db_creds must contain a 'user' key.")
            if "password" not in db_creds:
                raise KeyError("db_creds must contain a 'password' key.")
        elif db_type == "bigquery":
            if "json_key_path" not in db_creds:
                raise KeyError("db_creds must contain a 'json_key_path' key.")
        elif db_type == "sqlite":
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
        elif db_type == "duckdb":
            if "database" not in db_creds:
                raise KeyError("db_creds must contain a 'database' key.")
        else:
            raise ValueError(
                f"Database `{db_type}` is not supported right now. db_type must be one of {', '.join(SUPPORTED_DB_TYPES)}"
            )


class Defog(BaseDefog):
    """
    The main class for Defog (Synchronous)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_type: str = "",
        db_creds: dict = {},
        base_url: str = "https://api.defog.ai",
        generate_query_url: str = "https://api.defog.ai/generate_query_chat",
        verbose: bool = False,
    ):
        """Initializes the synchronous version of the Defog class"""
        super().__init__(
            api_key=api_key,
            db_type=db_type,
            db_creds=db_creds,
            base_url=base_url,
            generate_query_url=generate_query_url,
            verbose=verbose,
        )

    def generate_db_schema(
        self,
        tables: list,
        scan: bool = True,
        upload: bool = True,
        return_tables_only: bool = False,
        return_format: str = "csv",
    ) -> str: ...

    def run_query(
        self,
        question: str,
        hard_filters: str = "",
        previous_context: list = [],
        glossary: str = "",
        query: dict = None,
        retries: int = 3,
        dev: bool = False,
        temp: bool = False,
        profile: bool = False,
        ignore_cache: bool = False,
        model: str = "",
        use_golden_queries: bool = True,
        subtable_pruning: bool = False,
        glossary_pruning: bool = False,
        prune_max_tokens: int = 2000,
        prune_bm25_num_columns: int = 10,
        prune_glossary_max_tokens: int = 1000,
        prune_glossary_num_cos_sim_units: int = 10,
        prune_glossary_bm25_units: int = 10,
        use_llm_directly: bool = False,
        llm_provider: Optional[Union[LLMProvider, str]] = None,
        llm_model: Optional[str] = None,
        table_metadata: Optional[dict] = None,
        cache_metadata: bool = False,
    ): ...


class AsyncDefog(BaseDefog):
    """
    The main class for Defog (Asynchronous)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_type: str = "",
        db_creds: dict = {},
        base_url: str = "https://api.defog.ai",
        generate_query_url: str = "https://api.defog.ai/generate_query_chat",
        verbose: bool = False,
    ):
        """Initializes the asynchronous version of the Defog class"""
        super().__init__(
            api_key=api_key,
            db_type=db_type,
            db_creds=db_creds,
            base_url=base_url,
            generate_query_url=generate_query_url,
            verbose=verbose,
        )

    async def generate_db_schema(
        self,
        tables: list,
        scan: bool = True,
        upload: bool = True,
        return_tables_only: bool = False,
        return_format: str = "csv",
    ) -> str: ...

    async def run_query(
        self,
        question: str,
        hard_filters: str = "",
        previous_context: list = [],
        glossary: str = "",
        query: dict = None,
        retries: int = 3,
        dev: bool = False,
        temp: bool = False,
        profile: bool = False,
        ignore_cache: bool = False,
        model: str = "",
        use_golden_queries: bool = True,
        subtable_pruning: bool = False,
        glossary_pruning: bool = False,
        prune_max_tokens: int = 2000,
        prune_bm25_num_columns: int = 10,
        prune_glossary_max_tokens: int = 1000,
        prune_glossary_num_cos_sim_units: int = 10,
        prune_glossary_bm25_units: int = 10,
        use_llm_directly: bool = False,
        llm_provider: Optional[Union[LLMProvider, str]] = None,
        llm_model: Optional[str] = None,
        table_metadata: Optional[dict] = None,
        cache_metadata: bool = False,
    ): ...


# Add all methods from generate_schema to Defog
for name in dir(generate_schema):
    attr = getattr(generate_schema, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from async_generate_schema to AsyncDefog
for name in dir(async_generate_schema):
    attr = getattr(async_generate_schema, name)
    if callable(attr):
        # Add the method to AsyncDefog
        setattr(AsyncDefog, name, attr)

# Add all methods from query_methods to Defog
for name in dir(query_methods):
    attr = getattr(query_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from async_query_methods to AsyncDefog
for name in dir(async_query_methods):
    attr = getattr(async_query_methods, name)
    if callable(attr):
        # Add the method to AsyncDefog
        setattr(AsyncDefog, name, attr)

# Add all methods from admin_methods to Defog
for name in dir(admin_methods):
    attr = getattr(admin_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from async_admin_methods to AsyncDefog
for name in dir(async_admin_methods):
    attr = getattr(async_admin_methods, name)
    if callable(attr):
        # Add the method to AsyncDefog
        setattr(AsyncDefog, name, attr)

# Add all methods from health_methods to Defog
for name in dir(health_methods):
    attr = getattr(health_methods, name)
    if callable(attr):
        # Add the method to Defog
        setattr(Defog, name, attr)

# Add all methods from async_health_methods to AsyncDefog
for name in dir(async_health_methods):
    attr = getattr(async_health_methods, name)
    if callable(attr):
        # Add the method to AsyncDefog
        setattr(AsyncDefog, name, attr)

# Export LLMProvider for external use
__all__ = ["Defog", "AsyncDefog", "LLMProvider", "__version__"]
