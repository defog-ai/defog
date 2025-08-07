from setuptools import find_packages, setup

extras = {
    "postgres": ["psycopg2-binary"],
    "mysql": ["mysql-connector-python"],
    "snowflake": ["snowflake-connector-python"],
    "bigquery": ["google-cloud-bigquery"],
    "redshift": ["psycopg2-binary"],
    "databricks": ["databricks-sql-connector"],
    "sqlserver": ["pyodbc"],
    "duckdb": ["duckdb>=1.3.0"],
    "async-postgres": ["asyncpg"],
    "async-mysql": ["aiomysql"],
    "async-odbc": ["aioodbc"],
}


setup(
    name="defog",
    packages=find_packages(),
    version="1.2.8",
    description="Defog is a Python library that helps you generate data queries from natural language questions.",
    author="Full Stack Data Pte. Ltd.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "defog=defog.cli:main",
        ],
    },
    install_requires=[
        "httpx>=0.28.1",
        "psycopg2-binary>=2.9.5",
        "prompt-toolkit>=3.0.38",
        "tqdm",
        "pydantic",
        "portalocker>=3.2.0",
        "pandas",
        "requests>=2.28.2",
        "anthropic>=0.52.2",
        "google-genai>=1.16.1",
        "openai>=1.84.0",
        "together>=1.3.11",
        "mistralai>=1.3.6",
        "tiktoken>=0.9.0",
        "bleach>=6.0.0",
        "beautifulsoup4>=4.12.0",
        "mcp",
        "rich",
        "jsonref",
        "fastmcp",
        "pwinput>=1.0.3",
        "aiofiles",
    ],
    author_email="founders@defog.ai",
    url="https://github.com/defog-ai/defog-python",
    long_description="Defog is a Python library that helps you generate data queries from natural language questions.",
    long_description_content_type="text/markdown",
    extras_require=extras,
)
