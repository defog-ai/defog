"""
Helper functions to extract database metadata directly from connected databases
for use with local SQL generation.
"""

from typing import Dict, List, Optional, Any
from defog import Defog
from defog import AsyncDefog

def extract_metadata_from_db(
    db_type: str,
    db_creds: Dict[str, Any],
    tables: Optional[List[str]] = None,
    cache: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Extract database metadata directly from the connected database.

    Args:
        db_type: Database type (postgres, mysql, bigquery, etc.)
        db_creds: Database connection credentials
        tables: Optional list of specific tables to extract
        cache: Optional cache instance for storing results
        api_key: Optional API key for cache key generation

    Returns:
        Dictionary mapping table names to column metadata
    """

    # Create instance with the provided credentials
    temp_defog = Defog(api_key=api_key, db_type=db_type, db_creds=db_creds)

    try:
        # Use the existing generate_db_schema method but don't upload to server
        schema_result = temp_defog.generate_db_schema(
            tables=tables or [],
            scan=False,  # Skip scanning for performance
            upload=False,  # Don't upload to Defog servers
            return_format="json",  # Get as dictionary
        )

        # Cache the result if cache is provided
        if cache:
            cache.set(api_key, db_type, schema_result, dev=False, db_creds=db_creds)

        return schema_result

    except Exception as e:
        raise RuntimeError(
            f"Failed to extract metadata from {db_type} database: {str(e)}"
        )


async def extract_metadata_from_db_async(
    db_type: str,
    db_creds: Dict[str, Any],
    tables: Optional[List[str]] = None,
    cache: Optional[Any] = None,
    api_key: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Async version using AsyncDefog class.

    Args:
        db_type: Database type (postgres, mysql, bigquery, etc.)
        db_creds: Database connection credentials
        tables: Optional list of specific tables to extract
        cache: Optional cache instance for storing results
        api_key: Optional API key for cache key generation

    Returns:
        Dictionary mapping table names to column metadata
    """

    # Create async instance with the provided credentials
    temp_defog = AsyncDefog(api_key=api_key, db_type=db_type, db_creds=db_creds)

    try:
        # Use the async generate_db_schema method
        schema_result = await temp_defog.generate_db_schema(
            tables=tables or [],
            scan=False,  # Skip scanning for performance
            upload=False,  # Don't upload to Defog servers
            return_format="json",  # Get as dictionary
        )

        # Cache the result if cache is provided
        if cache:
            cache.set(api_key, db_type, schema_result, dev=False, db_creds=db_creds)

        return schema_result

    except Exception as e:
        raise RuntimeError(
            f"Failed to extract metadata from {db_type} database: {str(e)}"
        )
