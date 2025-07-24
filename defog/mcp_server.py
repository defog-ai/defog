#!/usr/bin/env python3
"""
MCP Server for Defog Python
Provides tools for SQL queries, code interpretation, web search, and YouTube transcription
"""

from typing import List, Dict, Any
import logging
import json
import os
import httpx
import aiofiles

# we use fastmcp 2.0 provider instead of the fastmcp provided by mcp
# this is because this version makes it easier to change multiple variables, like the port
from fastmcp import FastMCP
from defog.llm.youtube import get_youtube_summary
from defog.llm.sql import sql_answer_tool
from defog.llm.pdf_data_extractor import extract_pdf_data as extract_pdf_data_tool
from defog.llm.html_data_extractor import HTMLDataExtractor
from defog.llm.text_data_extractor import TextDataExtractor
from defog import config
from defog.local_metadata_extractor import extract_metadata_from_db_async
from defog.query import async_execute_query_once

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Defog MCP Server")


def validate_database_credentials() -> Dict[str, Any]:
    """
    Validate database credentials based on DB_TYPE environment variable.
    Returns a dictionary with validation results.
    """
    db_type = config.get("DB_TYPE")

    if not db_type:
        return {"valid": False, "error": "DB_TYPE environment variable not set"}

    # Define required credentials for each database type
    required_creds = {
        "postgres": ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "mysql": ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "bigquery": ["DB_KEY_PATH"],
        "snowflake": [
            "DB_USER",
            "DB_PASSWORD",
            "DB_ACCOUNT",
            "DB_WAREHOUSE",
            "DB_NAME",
        ],
        "databricks": ["DB_HOST", "DB_PATH", "DB_TOKEN"],
        "sqlserver": ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"],
        "sqlite": ["DATABASE_PATH"],
        "duckdb": ["DATABASE_PATH"],
    }

    if db_type not in required_creds:
        return {"valid": False, "error": f"Unsupported database type: {db_type}"}

    # Check for missing credentials
    missing_creds = []
    for cred in required_creds[db_type]:
        if not config.get(cred):
            missing_creds.append(cred)

    if missing_creds:
        return {
            "valid": False,
            "error": f"Missing required credentials for {db_type}: {', '.join(missing_creds)}",
        }

    return {
        "valid": True,
        "db_type": db_type,
        "message": f"Successfully validated {db_type} database credentials",
    }


def get_db_creds(db_type: str) -> Dict[str, Any]:
    """
    Get database credentials based on DB_TYPE.
    Returns a dictionary with the appropriate credentials for the database type.
    """
    creds = {}
    if db_type == "postgres":
        creds = {
            "host": config.get("DB_HOST"),
            "port": config.get("DB_PORT"),
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "database": config.get("DB_NAME"),
        }
    elif db_type == "mysql":
        creds = {
            "host": config.get("DB_HOST"),
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "database": config.get("DB_NAME"),
        }
    elif db_type == "bigquery":
        creds = {
            "json_key_path": config.get("DB_KEY_PATH"),
        }
    elif db_type == "snowflake":
        creds = {
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "account": config.get("DB_ACCOUNT"),
            "warehouse": config.get("DB_WAREHOUSE"),
            "database": config.get("DB_NAME"),
        }
    elif db_type == "databricks":
        creds = {
            "server_hostname": config.get("DB_HOST"),
            "http_path": config.get("DB_PATH"),
            "access_token": config.get("DB_TOKEN"),
        }
    elif db_type == "sqlserver":
        creds = {
            "server": config.get("DB_HOST"),
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "database": config.get("DB_NAME"),
        }
    elif db_type == "sqlite":
        creds = {
            "database": config.get("DATABASE_PATH"),
        }
    elif db_type == "duckdb":
        creds = {
            "database": config.get("DATABASE_PATH"),
        }
    elif db_type == "redshift":
        creds = {
            "host": config.get("DB_HOST"),
            "port": config.get("DB_PORT"),
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "database": config.get("DB_NAME"),
        }
    return creds


# Only register text_to_sql_tool if database credentials are valid
validation_result = validate_database_credentials()

# Register resources if database credentials are valid
if validation_result["valid"]:
    # Schema Resources
    @mcp.resource("schema://tables")
    async def get_all_tables():
        """Get a list of all tables in the database"""
        db_type = config.get("DB_TYPE")
        db_creds = get_db_creds(db_type)

        try:
            # Extract metadata for all tables
            schema_result = await extract_metadata_from_db_async(
                db_type=db_type,
                db_creds=db_creds,
                tables=[],  # Empty list means all tables
            )

            # Return list of table names
            tables = list(schema_result.keys())
            return {"database_type": db_type, "tables": tables, "count": len(tables)}
        except Exception as e:
            logger.error(f"Error in get_all_tables resource: {e}")
            return {"error": str(e), "status": "error"}

    @mcp.resource("schema://table/{table_name}")
    async def get_table_schema(table_name: str):
        """Get detailed schema for a specific table"""
        db_type = config.get("DB_TYPE")
        db_creds = get_db_creds(db_type)

        try:
            # Extract metadata for specific table
            schema_result = await extract_metadata_from_db_async(
                db_type=db_type,
                db_creds=db_creds,
                tables=[table_name],
            )

            if table_name not in schema_result:
                return {"error": f"Table '{table_name}' not found", "status": "error"}

            table_info = schema_result[table_name]

            # Handle both new and old format
            if isinstance(table_info, dict) and "columns" in table_info:
                return {
                    "database_type": db_type,
                    "table_name": table_name,
                    "description": table_info.get("table_description", ""),
                    "columns": table_info["columns"],
                    "column_count": len(table_info["columns"]),
                }
            else:
                # Old format (list of columns)
                return {
                    "database_type": db_type,
                    "table_name": table_name,
                    "description": "",
                    "columns": table_info,
                    "column_count": len(table_info),
                }

        except Exception as e:
            logger.error(f"Error in get_table_schema resource: {e}")
            return {"error": str(e), "status": "error"}

    # Metadata Resources
    @mcp.resource("stats://table/{table_name}")
    async def get_table_stats(table_name: str):
        """Get statistics and metadata for a specific table"""
        db_type = config.get("DB_TYPE")
        db_creds = get_db_creds(db_type)

        try:
            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            colnames, rows = await async_execute_query_once(
                db_type, db_creds, count_query
            )
            row_count = rows[0][0] if rows else 0

            # Get table schema first
            schema_result = await extract_metadata_from_db_async(
                db_type=db_type,
                db_creds=db_creds,
                tables=[table_name],
            )

            if table_name not in schema_result:
                return {"error": f"Table '{table_name}' not found", "status": "error"}

            table_info = schema_result[table_name]
            columns = (
                table_info.get("columns", table_info)
                if isinstance(table_info, dict)
                else table_info
            )

            # Get column statistics for numeric columns
            column_stats = {}
            for col_info in columns:
                col_name = col_info["column_name"]
                col_type = col_info["data_type"].lower()

                # Check if column is numeric
                if any(
                    numeric_type in col_type
                    for numeric_type in [
                        "int",
                        "float",
                        "decimal",
                        "numeric",
                        "double",
                        "real",
                    ]
                ):
                    try:
                        stats_query = f"""
                        SELECT 
                            MIN({col_name}) as min_value,
                            MAX({col_name}) as max_value,
                            AVG({col_name}) as avg_value,
                            COUNT(DISTINCT {col_name}) as distinct_count,
                            COUNT({col_name}) as non_null_count
                        FROM {table_name}
                        """
                        col_colnames, col_rows = await async_execute_query_once(
                            db_type, db_creds, stats_query
                        )

                        if col_rows and col_rows[0]:
                            column_stats[col_name] = {
                                "type": "numeric",
                                "min": col_rows[0][0],
                                "max": col_rows[0][1],
                                "avg": col_rows[0][2],
                                "distinct_count": col_rows[0][3],
                                "non_null_count": col_rows[0][4],
                                "null_count": row_count - col_rows[0][4],
                            }
                    except Exception as e:
                        logger.warning(
                            f"Could not get stats for column {col_name}: {e}"
                        )

                # For non-numeric columns, just get distinct count
                else:
                    try:
                        distinct_query = f"""
                        SELECT 
                            COUNT(DISTINCT {col_name}) as distinct_count,
                            COUNT({col_name}) as non_null_count
                        FROM {table_name}
                        """
                        col_colnames, col_rows = await async_execute_query_once(
                            db_type, db_creds, distinct_query
                        )

                        if col_rows and col_rows[0]:
                            column_stats[col_name] = {
                                "type": "non_numeric",
                                "distinct_count": col_rows[0][0],
                                "non_null_count": col_rows[0][1],
                                "null_count": row_count - col_rows[0][1],
                            }
                    except Exception as e:
                        logger.warning(
                            f"Could not get stats for column {col_name}: {e}"
                        )

            return {
                "database_type": db_type,
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(columns),
                "columns": columns,
                "column_statistics": column_stats,
            }

        except Exception as e:
            logger.error(f"Error in get_table_stats resource: {e}")
            return {"error": str(e), "status": "error"}

    # Sample Data Resources
    @mcp.resource("sample://table/{table_name}")
    async def get_table_sample(table_name: str):
        """Get sample data from a specific table"""
        db_type = config.get("DB_TYPE")
        db_creds = get_db_creds(db_type)

        try:
            # Different databases have different LIMIT syntax
            if db_type in ["sqlserver"]:
                sample_query = f"SELECT TOP 10 * FROM {table_name}"
            else:
                sample_query = f"SELECT * FROM {table_name} LIMIT 10"

            colnames, rows = await async_execute_query_once(
                db_type, db_creds, sample_query
            )

            # Convert rows to list of dictionaries for better readability
            sample_data = []
            for row in rows:
                row_dict = {}
                for i, col_name in enumerate(colnames):
                    # Convert to string for JSON serialization
                    value = row[i]
                    if value is None:
                        row_dict[col_name] = None
                    else:
                        row_dict[col_name] = str(value)
                sample_data.append(row_dict)

            return {
                "database_type": db_type,
                "table_name": table_name,
                "columns": colnames,
                "sample_size": len(sample_data),
                "data": sample_data,
            }

        except Exception as e:
            logger.error(f"Error in get_table_sample resource: {e}")
            return {"error": str(e), "status": "error"}

    @mcp.tool(
        description="Execute a natural language query against a SQL database. Supports multiple database types."
    )
    async def text_to_sql_tool(
        question: str,
    ) -> str:
        """
        Execute a natural language query against a SQL database. This works best for questions involving structured data.

        Args:
            question: Natural language question to query the database

        Returns:
            JSON string containing query results or error message
        """
        db_type = config.get("DB_TYPE")
        creds = get_db_creds(db_type)

        if not creds:
            raise ValueError(f"Unsupported database type: {db_type}")

        # select model/provider based on what API keys are available
        # we prefer to use openai/o4-mini, gemini/gemini-pro-2.5, anthropic/claude-sonnet-4 - in that order
        if config.get("OPENAI_API_KEY"):
            model = "o4-mini"
            provider = "openai"
        elif config.get("GEMINI_API_KEY"):
            model = "gemini-pro-2.5"
            provider = "gemini"
        elif config.get("ANTHROPIC_API_KEY"):
            model = "claude-sonnet-4-20250514"
            provider = "anthropic"
        else:
            raise ValueError("No API keys found")

        # Parse table metadata if provided
        result = await sql_answer_tool(
            question=question,
            db_type=db_type,
            db_creds=creds,
            model=model,
            provider=provider,
        )

        return json.dumps(result, indent=2, default=str)

    @mcp.tool(
        description="List all tables and their schemas in the configured database. Returns table names, column names, and data types."
    )
    async def list_database_schema() -> str:
        """
        List all tables and their schemas in the configured database.

        Returns:
            JSON string containing database schema information
        """
        db_type = config.get("DB_TYPE")
        creds = get_db_creds(db_type)

        if not creds:
            return json.dumps(
                {"error": f"Unsupported database type: {db_type}"}, indent=2
            )

        try:
            # Extract metadata from database
            schema_result = await extract_metadata_from_db_async(
                db_type=db_type,
                db_creds=creds,
                tables=[],  # Empty list means all tables
            )

            # Format the result for better readability
            formatted_result = {
                "database_type": db_type,
                "tables": {},
            }

            for table_name, table_info in schema_result.items():
                if isinstance(table_info, dict) and "columns" in table_info:
                    # Handle new format with table_description
                    formatted_result["tables"][table_name] = {
                        "description": table_info.get("table_description", ""),
                        "columns": table_info["columns"],
                    }
                else:
                    # Handle old format (list of columns)
                    formatted_result["tables"][table_name] = {
                        "description": "",
                        "columns": table_info,
                    }

            # Add summary statistics
            formatted_result["summary"] = {
                "total_tables": len(formatted_result["tables"]),
                "total_columns": sum(
                    len(table_data["columns"])
                    for table_data in formatted_result["tables"].values()
                ),
            }

            return json.dumps(formatted_result, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error in list_database_schema: {e}")
            return json.dumps({"error": str(e), "status": "error"}, indent=2)


if config.get("GEMINI_API_KEY"):

    @mcp.tool(
        description="Get a detailed transcript/summary of a YouTube video. Returns formatted transcript with timestamps."
    )
    async def youtube_video_summary(
        video_url: str,
        task_description: str,
        system_instructions: List[str] = None,
    ) -> str:
        """
        Get a summary of a YouTube video.

        Args:
            video_url: YouTube video URL
            task_description: Description of the task to accomplish
            system_instructions: System instructions to use for the task

        Returns:
            JSON string containing summary
        """
        try:
            result = await get_youtube_summary(
                video_url=video_url,
                verbose=False,
                task_description=task_description,
                system_instructions=system_instructions,
            )

            return json.dumps({"summary": result}, indent=2)

        except Exception as e:
            logger.error(f"Error in youtube_transcript: {e}")
            return json.dumps({"error": str(e), "status": "error"})
else:
    logger.warning("No API keys found for Gemini, YouTube tool will not be available")


if config.get("ANTHROPIC_API_KEY") or config.get("OPENAI_API_KEY"):

    @mcp.tool(
        description="Extract structured data from a PDF document. Requires a URL to the PDF."
    )
    async def extract_pdf_data(
        pdf_url: str,
    ) -> str:
        """
        Extract structured data from a PDF document.

        Args:
            pdf_url: URL to the PDF document

        Returns:
            JSON string containing extracted data or error message
        """
        if config.get("OPENAI_API_KEY"):
            provider = "openai"
            model = "o3"
        else:
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"

        try:
            result = await extract_pdf_data_tool(
                pdf_url=pdf_url,
                provider=provider,
                model=model,
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in extract_pdf_data: {e}")
            return json.dumps({"error": str(e), "status": "error"})
else:
    logger.warning(
        "No API keys found for Anthropic, PDF extraction tool will not be available"
    )


if (
    config.get("ANTHROPIC_API_KEY")
    or config.get("OPENAI_API_KEY")
    or config.get("GEMINI_API_KEY")
):

    @mcp.tool(
        description="Extract structured data from HTML content. Automatically identifies tables, lists, key-value pairs, and other data structures."
    )
    async def extract_html_data(
        url: str,
        focus_areas: List[str] = None,
    ) -> str:
        """
        Extract structured data from HTML content.

        Args:
            url: URL of the HTML page to analyze and extract data from
            focus_areas: Optional list of areas to focus analysis on (e.g., ["pricing table", "product specifications"])

        Returns:
            JSON string containing extracted data with confidence scores and metadata
        """
        # Determine which provider and model to use
        if config.get("OPENAI_API_KEY"):
            provider = "openai"
            model = "o3"
        elif config.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"
        else:
            provider = "gemini"
            model = "gemini-pro-2.5"

        try:
            # Initialize the HTML data extractor
            extractor = HTMLDataExtractor(
                analysis_provider=provider,
                analysis_model=model,
                extraction_provider=provider,
                extraction_model=model,
                enable_caching=True,
                max_parallel_extractions=5,
            )

            # Fetch HTML content from URL
            async with httpx.AsyncClient() as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/",
                    "DNT": "1",  # Do Not Track
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    return json.dumps(
                        {
                            "error": f"Failed to fetch URL: HTTP {response.status_code}",
                            "status": "error",
                        },
                        indent=2,
                    )
                html_content = response.text

            # Extract all data
            result = await extractor.extract_all_data(
                html_content=html_content,
                focus_areas=focus_areas,
            )

            # Convert result to dict for JSON serialization
            return json.dumps(result.model_dump(), indent=2, default=str)

        except Exception as e:
            logger.error(f"Error in extract_html_data: {e}")
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

    @mcp.tool(
        description="Extract structured data from plain text. Identifies Q&A pairs, tables, lists, key-value pairs, and other data structures."
    )
    async def extract_text_data(
        file_path: str,
        focus_areas: List[str] = None,
        datapoint_filter: List[str] = None,
    ) -> str:
        """
        Extract structured data from unstructured text.

        Args:
            file_path: Path to the text file to analyze and extract data from
            focus_areas: Optional list of areas to focus analysis on (e.g., ["Q&A section", "financial metrics"])
            datapoint_filter: Optional list of specific datapoint names to extract

        Returns:
            JSON string containing extracted data with confidence scores and metadata
        """
        # Determine which provider and model to use
        if config.get("OPENAI_API_KEY"):
            provider = "openai"
            model = "o3"
        elif config.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"
        else:
            provider = "gemini"
            model = "gemini-pro-2.5"

        try:
            # Initialize the text data extractor
            extractor = TextDataExtractor(
                analysis_provider=provider,
                analysis_model=model,
                extraction_provider=provider,
                extraction_model=model,
                enable_caching=True,
                max_parallel_extractions=5,
            )

            # Read text content from file
            if not os.path.exists(file_path):
                return json.dumps(
                    {"error": f"File not found: {file_path}", "status": "error"},
                    indent=2,
                )

            async with aiofiles.open(file_path, mode="r", encoding="utf-8") as f:
                text_content = await f.read()

            # Extract all data
            result = await extractor.extract_all_data(
                text_content=text_content,
                focus_areas=focus_areas,
                datapoint_filter=datapoint_filter,
            )

            # Convert result to dict for JSON serialization
            return json.dumps(result.model_dump(), indent=2, default=str)

        except Exception as e:
            logger.error(f"Error in extract_text_data: {e}")
            return json.dumps({"error": str(e), "status": "error"}, indent=2)

else:
    logger.warning(
        "No API keys found for LLM providers, HTML and text extraction tools will not be available"
    )


def run_server(transport=None, port=None):
    """Run the MCP server.

    Args:
        transport: Transport type (e.g., 'stdio', 'streamable-http')
        port: Port number for streamable-http transport
    """
    logger.info("Starting Defog Browser")

    # Log database validation status
    if validation_result["valid"]:
        logger.info(f"Database credentials validated: {validation_result['message']}")
    else:
        logger.warning(
            f"Database credentials validation failed: {validation_result['error']}"
        )
        logger.warning("SQL tool will not be available")

    # List all registered tools and resources
    try:
        import asyncio

        # List tools
        tools = asyncio.run(mcp._list_tools())
        tool_names = [tool.name for tool in tools]
        logger.info(f"Registered {len(tool_names)} tools: {', '.join(tool_names)}")

        # List resources
        resources = asyncio.run(mcp._list_resources())
        resource_templates = asyncio.run(mcp._list_resource_templates())
        resource_uris = [str(resource.uri) for resource in resources] + [
            str(resource_template.uri_template)
            for resource_template in resource_templates
        ]
        logger.info(
            f"Registered {len(resource_uris)} resources: {', '.join(resource_uris)}"
        )
    except Exception as e:
        logger.warning(f"Could not list tools/resources: {e}")

    # Run with specified transport and port, or defaults
    if transport == "streamable-http":
        if port:
            logger.info(f"Starting MCP server with streamable-http on port {port}")
            mcp.run(transport="streamable-http", port=port, host="0.0.0.0")
        else:
            logger.info("Starting MCP server with streamable-http on default port")
            mcp.run(transport="streamable-http", host="0.0.0.0")
    else:
        # Default to stdio
        logger.info("Starting MCP server with stdio transport")
        mcp.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Defog MCP Server - Provides tools for SQL queries, code interpretation, and more"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default=None,
        help="Transport type (e.g., 'stdio', 'streamable-http'). Default: stdio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number for streamable-http transport",
    )

    args = parser.parse_args()
    run_server(transport=args.transport, port=args.port)
