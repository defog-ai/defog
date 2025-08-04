import json
import pandas as pd
import psycopg2
import mysql.connector
import pyodbc
import snowflake.connector
from defog.metadata_cache import get_global_cache
from defog.local_storage import LocalStorage
from google.cloud import bigquery
from typing import Dict, List, Optional
from databricks import sql


def update_db_schema(self, path_to_csv, dev=False, temp=False):
    """
    Update the DB schema via a CSV - now saves to local storage
    """
    schema_df = pd.read_csv(path_to_csv).fillna("")
    # check columns
    if not all(
        col in schema_df.columns
        for col in ["table_name", "column_name", "data_type", "column_description"]
    ):
        raise ValueError(
            "The CSV must contain the following columns: table_name, column_name, data_type, column_description"
        )
    schema = {}
    for table_name in schema_df["table_name"].unique():
        schema[table_name] = schema_df[schema_df["table_name"] == table_name][
            ["column_name", "data_type", "column_description"]
        ].to_dict(orient="records")

    # Save to local storage instead of API
    storage = LocalStorage()
    resp = storage.save_metadata(
        metadata={"table_metadata": schema, "db_type": self.db_type, "dev": dev},
        api_key=self.api_key,
        db_type=self.db_type,
    )

    # Invalidate cache after updating schema
    cache = get_global_cache()
    cache.invalidate(self.api_key, self.db_type, dev)

    return resp


def update_glossary(
    self,
    glossary: str = "",
    customized_glossary: dict = None,
    glossary_compulsory: str = "",
    glossary_prunable_units: List[str] = [],
    dev: bool = False,
):
    """
    Updates the glossary in local storage.
    :param glossary: The glossary to be used.
    """
    # Save glossary to local storage
    storage = LocalStorage()

    # Save the main glossary
    resp = storage.save_glossary(
        glossary=glossary, api_key=self.api_key, db_type=self.db_type
    )

    # If there's customized glossary, save it as metadata
    if customized_glossary or glossary_compulsory or glossary_prunable_units:
        metadata = storage.get_metadata(self.api_key, self.db_type).get("metadata", {})
        metadata["customized_glossary"] = customized_glossary or {}
        metadata["glossary_compulsory"] = glossary_compulsory
        metadata["glossary_prunable_units"] = glossary_prunable_units
        storage.save_metadata(metadata, self.api_key, self.db_type)

    return resp


def delete_glossary(self, user_type=None, dev=False):
    """
    Deletes the glossary from local storage.
    """
    storage = LocalStorage()
    resp = storage.delete_glossary(api_key=self.api_key, db_type=self.db_type)
    print("Glossary deleted successfully.")
    return resp


def get_glossary(self, mode="general", dev=False):
    """
    Gets the glossary from local storage.
    """
    storage = LocalStorage()

    if mode == "general":
        return storage.get_glossary(self.api_key, self.db_type)
    elif mode == "customized":
        metadata = storage.get_metadata(self.api_key, self.db_type).get("metadata", {})
        return metadata.get("customized_glossary", {})


def get_metadata(self, format="markdown", export_path=None, dev=False):
    """
    Gets the metadata from local storage.
    """
    storage = LocalStorage()
    resp = storage.get_metadata(self.api_key, self.db_type)
    metadata = resp.get("metadata", {})
    table_metadata = metadata.get("table_metadata", {})

    items = []
    for table in table_metadata:
        for item in table_metadata[table]:
            item["table_name"] = table
            items.append(item)

    if format == "markdown":
        return pd.DataFrame(items)[
            ["table_name", "column_name", "data_type", "column_description"]
        ].to_markdown(index=False)
    elif format == "csv":
        if export_path is None:
            export_path = "metadata.csv"
        pd.DataFrame(items)[
            ["table_name", "column_name", "data_type", "column_description"]
        ].to_csv(export_path, index=False)
        print(f"Metadata exported to {export_path}")
        return True
    elif format == "json":
        return table_metadata


def get_feedback(self, n_rows: int = 50, start_from: int = 0):
    """
    This method is deprecated as feedback is no longer tracked locally.
    """
    print("Warning: get_feedback is deprecated. Feedback tracking has been removed.")
    return pd.DataFrame().to_markdown(index=False)


def get_quota(self) -> Optional[Dict]:
    """
    This method is deprecated as quota management is no longer needed for local generation.
    """
    print(
        "Warning: get_quota is deprecated. Quota management is not needed for local generation."
    )
    return {"quota": "unlimited", "usage": "n/a"}


def update_golden_queries(
    self,
    golden_queries: List[Dict] = None,
    golden_queries_path: str = None,
    scrub: bool = True,
    dev: bool = False,
):
    """
    Updates the golden queries in local storage.
    :param golden_queries: The golden queries to be used.
    :param golden_queries_path: The path to the golden queries CSV.
    :param scrub: Whether to scrub the golden queries.
    """
    if golden_queries is None and golden_queries_path is None:
        raise ValueError("Please provide either golden_queries or golden_queries_path.")

    if golden_queries is None:
        golden_queries = (
            pd.read_csv(golden_queries_path).fillna("").to_dict(orient="records")
        )

    # Save to local storage
    storage = LocalStorage()
    resp = storage.save_golden_queries(
        golden_queries=golden_queries, api_key=self.api_key, db_type=self.db_type
    )

    print("Golden queries have been saved locally and are now available for use.")
    return resp


def delete_golden_queries(
    self,
    golden_queries: dict = None,
    golden_queries_path: str = None,
    all: bool = False,
    dev: bool = False,
):
    """
    Deletes golden queries from local storage.
    :param golden_queries: The golden queries to be deleted.
    :param golden_queries_path: The path to the golden queries CSV.
    :param all: Whether to delete all golden queries.
    """
    if golden_queries is None and golden_queries_path is None and not all:
        raise ValueError(
            "Please provide either golden_queries or golden_queries_path, or set all=True."
        )

    storage = LocalStorage()

    if all:
        # Delete all by removing the file
        storage.delete_golden_queries([], self.api_key, self.db_type)
        print("All golden queries have now been deleted.")
        return {"status": "success", "message": "All golden queries deleted"}
    else:
        if golden_queries is None:
            golden_queries = (
                pd.read_csv(golden_queries_path).fillna("").to_dict(orient="records")
            )

        # Extract questions to delete
        questions_to_delete = [q["question"] for q in golden_queries]
        resp = storage.delete_golden_queries(
            golden_queries=questions_to_delete,
            api_key=self.api_key,
            db_type=self.db_type,
        )
        return resp


def get_golden_queries(
    self, format: str = "csv", export_path: str = None, dev: bool = False
):
    """
    Gets the golden queries from local storage.
    """
    storage = LocalStorage()
    golden_queries = storage.get_golden_queries(self.api_key, self.db_type)

    if format == "csv":
        if export_path is None:
            export_path = "golden_queries.csv"
        pd.DataFrame(golden_queries).to_csv(export_path, index=False)
        print(f"{len(golden_queries)} golden queries exported to {export_path}")
        return golden_queries
    elif format == "json":
        if export_path is None:
            export_path = "golden_queries.json"
        with open(export_path, "w") as f:
            json.dump({"golden_queries": golden_queries}, f, indent=4)
        print(f"{len(golden_queries)} golden queries exported to {export_path}")
        return golden_queries
    else:
        raise ValueError("format must be either 'csv' or 'json'.")


def create_table_ddl(
    table_name: str, columns: List[Dict[str, str]], add_exists=True
) -> str:
    """
    Return a DDL statement for creating a table from a list of columns
    `columns` is a list of dictionaries with the following keys:
    - column_name: str
    - data_type: str
    - column_description: str
    """
    md_create = ""
    if add_exists:
        md_create += f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    else:
        md_create += f"CREATE TABLE {table_name} (\n"
    for i, column in enumerate(columns):
        col_name = column["column_name"]
        # if column name has spaces and hasn't been wrapped in double quotes, wrap it in double quotes
        if " " in col_name and not col_name.startswith('"'):
            col_name = f'"{col_name}"'
        dtype = column["data_type"]
        if i < len(columns) - 1:
            md_create += f"  {col_name} {dtype},\n"
        else:
            # avoid the trailing comma for the last line
            md_create += f"  {col_name} {dtype}\n"
    md_create += ");\n"
    return md_create


def create_ddl_from_metadata(
    metadata: Dict[str, List[Dict[str, str]]], add_exists=True
) -> str:
    """
    Return a DDL statement for creating tables from metadata
    `metadata` is a dictionary with table names as keys and lists of dictionaries as values.
    Each dictionary in the list has the following keys:
    - column_name: str
    - data_type: str
    - column_description: str
    """
    md_create = ""
    for table_name, columns in metadata.items():
        if "." in table_name:
            # table_name = table_name.split(".", 1)[1]
            schema_name = table_name.split(".")[0]

            md_create += f"CREATE SCHEMA IF NOT EXISTS {schema_name};\n"
        md_create += create_table_ddl(table_name, columns, add_exists=add_exists)
    return md_create


def create_empty_tables(self, dev: bool = False):
    """
    Create empty tables based on metadata
    """
    metadata = self.get_metadata(format="json", dev=dev)
    if self.db_type == "sqlserver":
        ddl = create_ddl_from_metadata(metadata, add_exists=False)
    else:
        ddl = create_ddl_from_metadata(metadata)

    try:
        if self.db_type == "postgres" or self.db_type == "redshift":

            conn = psycopg2.connect(**self.db_creds)
            cur = conn.cursor()
            cur.execute(ddl)
            conn.commit()
            conn.close()
            return True

        elif self.db_type == "mysql":

            conn = mysql.connector.connect(**self.db_creds)
            cur = conn.cursor()
            for statement in ddl.split(";"):
                cur.execute(statement)
            conn.commit()
            conn.close()
            return True

        elif self.db_type == "databricks":

            con = sql.connect(**self.db_creds)
            con.execute(ddl)
            conn.commit()
            conn.close()
            return True

        elif self.db_type == "snowflake":

            conn = snowflake.connector.connect(
                user=self.db_creds["user"],
                password=self.db_creds["password"],
                account=self.db_creds["account"],
            )
            cur = conn.cursor()
            cur.execute(
                f"USE WAREHOUSE {self.db_creds['warehouse']}"
            )  # set the warehouse
            for statement in ddl.split(";"):
                cur.execute(statement)
            conn.commit()
            conn.close()
            return True

        elif self.db_type == "bigquery":

            client = bigquery.Client.from_service_account_json(
                self.db_creds["json_key_path"]
            )
            for statement in ddl.split(";"):
                client.query(statement)
            return True

        elif self.db_type == "sqlserver":

            if self.db_creds["database"] != "":
                connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};DATABASE={self.db_creds['database']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
            else:
                connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.db_creds['server']};UID={self.db_creds['user']};PWD={self.db_creds['password']};TrustServerCertificate=yes;Connection Timeout=120;"
            conn = pyodbc.connect(connection_string)
            cur = conn.cursor()
            for statement in ddl.split(";"):
                cur.execute(statement)
            conn.commit()
            conn.close()
            return True
        else:
            raise ValueError(f"Unsupported DB type: {self.db_type}")
    except Exception as e:
        print(f"Error: {e}")
        return False
