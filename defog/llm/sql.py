"""
SQL execution tools for local database operations.
"""

from typing import Dict, List, Optional, Any
from defog.llm.llm_providers import LLMProvider
from defog.llm.utils_logging import (
    ToolProgressTracker,
    SubTaskLogger,
    NoOpToolProgressTracker,
    NoOpSubTaskLogger,
)
from defog.llm.sql_generator import generate_sql_query_local
from defog.local_metadata_extractor import extract_metadata_from_db_async
from defog.query import async_execute_query
from defog.llm.utils import chat_async
from defog.llm.config import LLMConfig
import json


class SQLAgentConfig:
    """Configuration settings for SQL agent operations."""

    # Table filtering thresholds
    TABLE_FILTER_COLUMN_THRESHOLD: int = 1000
    TABLE_FILTER_TABLE_THRESHOLD: int = 5

    # Maximum number of tables to return from relevance analysis
    MAX_RELEVANCE_TABLES: int = 10


async def sql_answer_tool(
    question: str,
    db_type: str,
    db_creds: Dict[str, Any],
    model: str,
    provider: LLMProvider,
    glossary: Optional[str] = None,
    hard_filters: Optional[str] = None,
    previous_context: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Answer a natural language question by generating and executing SQL on a local database.

    Args:
        question: Natural language question to answer
        db_type: Database type (postgres, mysql, bigquery, etc.)
        db_creds: Database connection credentials
        model: LLM model name
        provider: LLM provider
        glossary: Optional business glossary
        hard_filters: Optional hard filters to apply
        previous_context: Optional previous conversation context
        temperature: LLM temperature setting
        config: Optional LLM configuration
        verbose: Whether to show progress logging

    Returns:
        Dictionary with query results and metadata
    """
    tracker_class = ToolProgressTracker if verbose else NoOpToolProgressTracker
    logger_class = SubTaskLogger if verbose else NoOpSubTaskLogger

    async with tracker_class(
        "SQL Query Answer",
        f"Answering: {question[:50]}{'...' if len(question) > 50 else ''}",
    ) as tracker:
        subtask_logger = logger_class()
        subtask_logger.log_provider_info(
            provider.value if hasattr(provider, "value") else str(provider), model
        )

        # Track total cost across all operations
        total_cost_in_cents = 0

        try:
            # Extract database metadata
            tracker.update(20, "Extracting database schema")
            subtask_logger.log_subtask("Extracting table metadata", "processing")
            table_metadata = await extract_metadata_from_db_async(db_type, db_creds)

            # Check if we need to filter tables due to large database
            total_tables = len(table_metadata)
            total_columns = sum(len(columns) for columns in table_metadata.values())

            # Use table relevance filtering if database is large
            if (
                total_columns > SQLAgentConfig.TABLE_FILTER_COLUMN_THRESHOLD
                and total_tables > SQLAgentConfig.TABLE_FILTER_TABLE_THRESHOLD
            ):
                tracker.update(30, "Database is large, identifying relevant tables")
                subtask_logger.log_subtask(
                    f"Filtering {total_tables} tables with {total_columns} columns",
                    "processing",
                )

                relevance_result = await identify_relevant_tables_tool(
                    question=question,
                    db_type=db_type,
                    db_creds=db_creds,
                    model=model,
                    provider=provider,
                    max_tables=10,
                    temperature=temperature,
                    config=config,
                    verbose=verbose,
                )

                if relevance_result.get("success"):
                    table_metadata = relevance_result["filtered_metadata"]
                    subtask_logger.log_subtask(
                        f"Filtered to {len(table_metadata)} relevant tables",
                        "completed",
                    )
                    # Track cost from relevance analysis
                    if relevance_result.get("cost_in_cents") is not None:
                        total_cost_in_cents += relevance_result["cost_in_cents"]
                else:
                    # If filtering fails, continue with all tables but log warning
                    subtask_logger.log_subtask(
                        f"Table filtering failed: {relevance_result.get('error')}, using all tables",
                        "warning",
                    )

            # Generate SQL query
            tracker.update(50, "Generating SQL query")
            subtask_logger.log_subtask("Converting question to SQL", "processing")
            sql_result = await generate_sql_query_local(
                question=question,
                table_metadata=table_metadata,
                db_type=db_type,
                provider=provider,
                model=model,
                glossary=glossary,
                hard_filters=hard_filters,
                previous_context=previous_context,
                temperature=temperature,
                config=config,
            )

            if not sql_result.get("ran_successfully"):
                return {
                    "success": False,
                    "error": f"SQL generation failed: {sql_result.get('error_message')}",
                    "query": None,
                    "results": None,
                    "tables_used": len(table_metadata),
                    "columns_analyzed": sum(
                        len(columns) for columns in table_metadata.values()
                    ),
                }

            generated_sql = sql_result["query_generated"]

            # Track cost from SQL generation
            if sql_result.get("cost_in_cents") is not None:
                total_cost_in_cents += sql_result["cost_in_cents"]

            # Execute the SQL query
            tracker.update(80, "Executing SQL query")
            subtask_logger.log_subtask("Running query on database", "processing")
            try:
                colnames, results = await async_execute_query(
                    query=generated_sql,
                    db_type=db_type,
                    db_creds=db_creds,
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Query execution failed: {str(e)}",
                    "query": generated_sql,
                    "results": None,
                    "columns": None,
                }

            # Include total cost from all operations
            result = {
                "success": True,
                "error": None,
                "query": generated_sql,
                "columns": colnames,
                "results": results,
            }

            # Add total cost if any operations incurred costs
            if total_cost_in_cents > 0:
                result["cost_in_cents"] = total_cost_in_cents

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": None,
                "results": None,
                "columns": None,
            }


async def identify_relevant_tables_tool(
    question: str,
    db_type: str,
    db_creds: Dict[str, Any],
    model: str,
    provider: LLMProvider,
    max_tables: int = SQLAgentConfig.MAX_RELEVANCE_TABLES,
    temperature: float = 0.0,
    config: Optional[LLMConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Identify the most relevant tables in a database for answering a specific question.
    This is particularly useful for databases with hundreds or thousands of tables.

    Args:
        question: Natural language question
        db_type: Database type (postgres, mysql, bigquery, etc.)
        db_creds: Database connection credentials
        model: LLM model name
        provider: LLM provider
        max_tables: Maximum number of tables to return
        temperature: LLM temperature setting
        config: Optional LLM configuration
        verbose: Whether to show progress logging

    Returns:
        Dictionary with relevant tables and their relevance scores
    """
    tracker_class = ToolProgressTracker if verbose else NoOpToolProgressTracker
    logger_class = SubTaskLogger if verbose else NoOpSubTaskLogger

    async with tracker_class(
        "Table Relevance Analysis",
        f"Finding relevant tables for: {question[:50]}{'...' if len(question) > 50 else ''}",
    ) as tracker:
        subtask_logger = logger_class()
        subtask_logger.log_provider_info(
            provider.value if hasattr(provider, "value") else str(provider), model
        )

        try:
            # Extract all database metadata
            tracker.update(30, "Extracting complete database schema")
            subtask_logger.log_subtask("Getting all table metadata", "processing")
            all_table_metadata = await extract_metadata_from_db_async(db_type, db_creds)

            # Build table summary for LLM analysis
            tracker.update(50, "Analyzing table relevance")
            subtask_logger.log_subtask("Scoring table relevance", "processing")

            # Create a concise summary of each table
            table_summaries = []
            for table_name, columns in all_table_metadata.items():
                column_names = [col.get("column_name", "") for col in columns]
                column_types = [col.get("data_type", "") for col in columns]

                # Create a brief description
                summary = f"Table: {table_name}\n"
                summary += (
                    f"Columns: {', '.join(column_names[:10])}"  # First 10 columns
                )
                if len(column_names) > 10:
                    summary += f" ... and {len(column_names) - 10} more"
                summary += f"\nColumn types: {', '.join(set(column_types))}"

                # Add column descriptions if available
                descriptions = [
                    col.get("column_description", "")
                    for col in columns
                    if col.get("column_description")
                ]
                if descriptions:
                    summary += f"\nDescriptions available: {len(descriptions)} columns"

                table_summaries.append(summary)

            # Use LLM to identify relevant tables
            analysis_prompt = f"""Given the following question and database tables, identify the {max_tables} most relevant tables that would be needed to answer the question.

Question: {question}

Database Tables:
{chr(10).join(table_summaries)}

Please analyze each table's relevance to the question and return a JSON response with the following format:
{{
    "relevant_tables": [
        {{
            "table_name": "table_name",
            "relevance_score": 0.9,
            "reason": "Brief explanation of why this table is relevant"
        }}
    ]
}}

Focus on tables that contain data directly related to the question. Consider:
1. Tables with columns that match entities mentioned in the question
2. Tables that would contain the metrics or values being asked about
3. Tables needed for joining to get complete information
4. Tables with temporal data if the question involves time-based analysis

Return only the JSON response."""

            analysis_messages = [
                {
                    "role": "system",
                    "content": "You are a database analyst expert at identifying relevant tables for SQL queries.",
                },
                {"role": "user", "content": analysis_prompt},
            ]

            analysis_response = await chat_async(
                provider=provider,
                model=model,
                messages=analysis_messages,
                temperature=temperature,
                config=config,
            )

            # Parse the JSON response
            try:
                analysis_result = json.loads(analysis_response.content.strip())
                relevant_tables = analysis_result.get("relevant_tables", [])

                # Sort by relevance score and limit to max_tables
                relevant_tables.sort(
                    key=lambda x: x.get("relevance_score", 0), reverse=True
                )
                relevant_tables = relevant_tables[:max_tables]

                # Extract the filtered metadata for only relevant tables
                relevant_table_names = [
                    table["table_name"] for table in relevant_tables
                ]
                filtered_metadata = {
                    name: metadata
                    for name, metadata in all_table_metadata.items()
                    if name in relevant_table_names
                }

                tracker.update(100, "Analysis complete")
                subtask_logger.log_result_summary(
                    "Table Relevance Analysis",
                    {
                        "total_tables": len(all_table_metadata),
                        "relevant_tables": len(relevant_tables),
                        "top_score": (
                            relevant_tables[0]["relevance_score"]
                            if relevant_tables
                            else 0
                        ),
                    },
                )

                result = {
                    "success": True,
                    "error": None,
                    "relevant_tables": relevant_tables,
                    "filtered_metadata": filtered_metadata,
                    "total_tables_analyzed": len(all_table_metadata),
                    "tables_selected": len(relevant_tables),
                }

                # Add cost if available
                if analysis_response.cost_in_cents is not None:
                    result["cost_in_cents"] = analysis_response.cost_in_cents

                return result

            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse LLM response as JSON: {str(e)}",
                    "relevant_tables": [],
                    "filtered_metadata": {},
                    "total_tables_analyzed": len(all_table_metadata),
                    "tables_selected": 0,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "relevant_tables": [],
                "filtered_metadata": {},
                "total_tables_analyzed": 0,
                "tables_selected": 0,
            }
