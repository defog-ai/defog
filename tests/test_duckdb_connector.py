import os
import tempfile
import unittest

import pytest

duckdb = pytest.importorskip("duckdb", reason="needs the duckdb extra")

from defog.query import execute_query_once, async_execute_query_once
from defog import Defog, AsyncDefog


class DuckDBConnectorTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary DuckDB database for testing
        # Use tempfile.mktemp() to get a path without creating the file
        self.db_path = tempfile.mktemp(suffix=".duckdb")

        # Create test data
        conn = duckdb.connect(self.db_path)

        # Create test tables with various DuckDB data types
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER,
                name VARCHAR NOT NULL,
                email VARCHAR UNIQUE,
                age INTEGER,
                is_active BOOLEAN DEFAULT true,
                salary DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id)
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE products (
                id INTEGER,
                name VARCHAR NOT NULL,
                price DECIMAL(8,2),
                category VARCHAR,
                tags VARCHAR[],
                metadata JSON,
                PRIMARY KEY (id)
            )
        """
        )

        # Create a table with schema prefix to test multi-schema support
        conn.execute("CREATE SCHEMA test_schema")
        conn.execute(
            """
            CREATE TABLE test_schema.orders (
                id INTEGER,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                order_date DATE,
                PRIMARY KEY (id)
            )
        """
        )

        # Insert test data into users
        conn.execute(
            """
            INSERT INTO users (id, name, email, age, is_active, salary) VALUES 
            (1, 'John Doe', 'john@example.com', 30, true, 75000.50),
            (2, 'Jane Smith', 'jane@example.com', 25, true, 68000.00),
            (3, 'Bob Johnson', 'bob@example.com', 35, false, 82000.75)
        """
        )

        # Insert test data into products
        conn.execute(
            """
            INSERT INTO products (id, name, price, category, tags, metadata) VALUES 
            (1, 'Laptop', 999.99, 'Electronics', ['computer', 'portable'], '{"brand": "TechCorp", "warranty": 2}'),
            (2, 'Mouse', 29.99, 'Electronics', ['accessory', 'wireless'], '{"brand": "ClickCorp", "warranty": 1}'),
            (3, 'Book', 19.99, 'Education', ['paperback', 'technical'], '{"author": "Tech Writer", "pages": 300}')
        """
        )

        # Insert test data into orders
        conn.execute(
            """
            INSERT INTO test_schema.orders (id, user_id, product_id, quantity, order_date) VALUES 
            (1, 1, 1, 1, '2024-01-15'),
            (2, 2, 2, 2, '2024-01-16'),
            (3, 1, 3, 1, '2024-01-17')
        """
        )

        conn.commit()
        conn.close()

        self.db_creds = {"database": self.db_path}

    def tearDown(self):
        # Clean up temporary database
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_execute_query_once_select(self):
        query = "SELECT name, email FROM users WHERE age > 25"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "email"])
        self.assertEqual(len(rows), 2)
        self.assertIn(("John Doe", "john@example.com"), rows)
        self.assertIn(("Bob Johnson", "bob@example.com"), rows)

    def test_execute_query_once_count(self):
        query = "SELECT COUNT(*) FROM products"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["count_star()"])
        self.assertEqual(rows[0][0], 3)

    def test_execute_query_once_aggregate(self):
        query = (
            "SELECT category, AVG(price) as avg_price FROM products GROUP BY category"
        )
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["category", "avg_price"])
        self.assertEqual(len(rows), 2)

        # Check that we have Electronics and Education categories
        categories = [row[0] for row in rows]
        self.assertIn("Electronics", categories)
        self.assertIn("Education", categories)

    def test_execute_query_once_decimal_and_boolean(self):
        query = "SELECT name, salary, is_active FROM users WHERE is_active = true"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "salary", "is_active"])
        self.assertEqual(len(rows), 2)

        # Check that all returned users are active
        for row in rows:
            self.assertTrue(row[2])  # is_active column

    def test_execute_query_once_json_operations(self):
        query = "SELECT name, metadata::JSON ->> 'brand' as brand FROM products WHERE category = 'Electronics'"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "brand"])
        self.assertEqual(len(rows), 2)

        brands = [row[1] for row in rows]
        self.assertIn("TechCorp", brands)
        self.assertIn("ClickCorp", brands)

    def test_execute_query_once_array_operations(self):
        query = "SELECT name FROM products WHERE 'computer' = ANY(tags)"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "Laptop")

    def test_execute_query_once_multi_schema(self):
        query = "SELECT COUNT(*) FROM test_schema.orders"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["count_star()"])
        self.assertEqual(rows[0][0], 3)

    def test_execute_query_once_join_across_schemas(self):
        query = """
            SELECT u.name, COUNT(o.id) as order_count 
            FROM users u 
            LEFT JOIN test_schema.orders o ON u.id = o.user_id 
            GROUP BY u.name
        """
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "order_count"])
        self.assertEqual(len(rows), 3)

    def test_execute_query_once_empty_result(self):
        query = "SELECT * FROM users WHERE age > 100"
        colnames, rows = execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(
            colnames,
            ["id", "name", "email", "age", "is_active", "salary", "created_at"],
        )
        self.assertEqual(len(rows), 0)

    def test_execute_query_once_invalid_query(self):
        query = "SELECT * FROM nonexistent_table"
        with self.assertRaises(Exception) as context:
            execute_query_once("duckdb", self.db_creds, query)
        self.assertIn("table", str(context.exception).lower())

    def test_execute_query_once_invalid_syntax(self):
        query = "SELECT * FROM users WHERE"
        with self.assertRaises(Exception) as context:
            execute_query_once("duckdb", self.db_creds, query)
        self.assertIn("syntax", str(context.exception).lower())

    def test_generate_duckdb_schema_basic(self):
        # Create a real Defog instance for schema generation
        defog = Defog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test schema generation without upload
        schema = defog.generate_db_schema([], upload=False, scan=False)

        self.assertIn("users", schema)
        self.assertIn("products", schema)
        self.assertIn("test_schema.orders", schema)

        # Check users table schema
        users_schema = schema["users"]
        columns = users_schema["columns"]
        column_names = [col["column_name"] for col in columns]
        self.assertIn("id", column_names)
        self.assertIn("name", column_names)
        self.assertIn("email", column_names)
        self.assertIn("age", column_names)
        self.assertIn("is_active", column_names)
        self.assertIn("salary", column_names)
        self.assertIn("created_at", column_names)

        # Check data types are properly detected
        for col in columns:
            if col["column_name"] == "id":
                self.assertEqual(col["data_type"], "INTEGER")
            elif col["column_name"] == "name":
                self.assertEqual(col["data_type"], "VARCHAR")
            elif col["column_name"] == "is_active":
                self.assertEqual(col["data_type"], "BOOLEAN")
            elif col["column_name"] == "salary":
                self.assertIn("DECIMAL", col["data_type"])

    def test_generate_duckdb_schema_specific_tables(self):
        defog = Defog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test schema generation for specific table
        schema = defog.generate_db_schema(["users"], upload=False, scan=False)

        self.assertIn("users", schema)
        self.assertNotIn("products", schema)
        self.assertNotIn("test_schema.orders", schema)

    def test_generate_duckdb_schema_multi_schema_specific(self):
        defog = Defog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test schema generation for schema-prefixed table
        schema = defog.generate_db_schema(
            ["test_schema.orders"], upload=False, scan=False
        )

        self.assertIn("test_schema.orders", schema)
        self.assertNotIn("users", schema)
        self.assertNotIn("products", schema)

    def test_generate_duckdb_schema_return_tables_only(self):
        defog = Defog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test returning table names only
        tables = defog.generate_db_schema([], upload=False, return_tables_only=True)

        self.assertIsInstance(tables, list)
        self.assertIn("users", tables)
        self.assertIn("products", tables)
        self.assertIn("test_schema.orders", tables)

    def test_memory_database_credentials(self):
        # Test with in-memory database
        memory_db_creds = {"database": ":memory:"}

        # Create an in-memory database with test data
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE test_table (id INTEGER, name VARCHAR)")
        conn.execute("INSERT INTO test_table VALUES (1, 'test')")

        # Note: In-memory databases are connection-specific in DuckDB
        # So we'll test the credentials validation instead
        try:
            Defog(api_key="test_key", db_type="duckdb", db_creds=memory_db_creds)
            # If no exception is raised, credentials are valid
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Memory database credentials should be valid: {e}")

        conn.close()

    def test_file_path_validation(self):
        # Test with invalid file path
        invalid_db_creds = {"database": "/nonexistent/path/test.duckdb"}

        # This should work for file paths that don't exist yet (DuckDB creates them)
        try:
            Defog(api_key="test_key", db_type="duckdb", db_creds=invalid_db_creds)
            # Should not raise exception as DuckDB can create new files
            self.assertTrue(True)
        except Exception as e:
            # Only fail if it's not a permission or path issue
            if "permission" not in str(e).lower() and "directory" not in str(e).lower():
                self.fail(f"Unexpected error for new file path: {e}")


class AsyncDuckDBConnectorTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a temporary DuckDB database for testing
        # Use tempfile.mktemp() to get a path without creating the file
        self.db_path = tempfile.mktemp(suffix=".duckdb")

        # Create test data
        conn = duckdb.connect(self.db_path)

        # Create test tables
        conn.execute(
            """
            CREATE TABLE async_users (
                id INTEGER,
                name VARCHAR NOT NULL,
                email VARCHAR UNIQUE,
                age INTEGER,
                score REAL,
                PRIMARY KEY (id)
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE async_products (
                id INTEGER,
                name VARCHAR NOT NULL,
                price DECIMAL(8,2),
                in_stock BOOLEAN,
                PRIMARY KEY (id)
            )
        """
        )

        # Insert test data
        conn.execute(
            """
            INSERT INTO async_users (id, name, email, age, score) VALUES 
            (1, 'Alice', 'alice@example.com', 28, 95.5),
            (2, 'Charlie', 'charlie@example.com', 32, 87.2),
            (3, 'Diana', 'diana@example.com', 29, 92.8)
        """
        )

        conn.execute(
            """
            INSERT INTO async_products (id, name, price, in_stock) VALUES 
            (1, 'Widget A', 25.99, true),
            (2, 'Widget B', 35.99, false),
            (3, 'Widget C', 45.99, true)
        """
        )

        conn.commit()
        conn.close()

        self.db_creds = {"database": self.db_path}

    async def asyncTearDown(self):
        # Clean up temporary database
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    async def test_async_execute_query_once_select(self):
        query = "SELECT name, email FROM async_users WHERE age > 25"
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "email"])
        self.assertEqual(len(rows), 3)
        self.assertIn(("Alice", "alice@example.com"), rows)
        self.assertIn(("Charlie", "charlie@example.com"), rows)
        self.assertIn(("Diana", "diana@example.com"), rows)

    async def test_async_execute_query_once_count(self):
        query = "SELECT COUNT(*) FROM async_users"
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["count_star()"])
        self.assertEqual(rows[0][0], 3)

    async def test_async_execute_query_once_aggregation(self):
        query = "SELECT AVG(score) as avg_score, MAX(age) as max_age FROM async_users"
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["avg_score", "max_age"])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0][0], 91.83, places=1)  # Average score
        self.assertEqual(rows[0][1], 32)  # Max age

    async def test_async_execute_query_once_join(self):
        query = """
            SELECT u.name, p.name as product_name, p.price 
            FROM async_users u 
            CROSS JOIN async_products p 
            WHERE u.id = 1 AND p.in_stock = true
        """
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "product_name", "price"])
        self.assertEqual(len(rows), 2)  # 2 products in stock

    async def test_async_execute_query_once_boolean_filter(self):
        query = "SELECT name, price FROM async_products WHERE in_stock = true"
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["name", "price"])
        self.assertEqual(len(rows), 2)
        product_names = [row[0] for row in rows]
        self.assertIn("Widget A", product_names)
        self.assertIn("Widget C", product_names)

    async def test_async_execute_query_once_empty_result(self):
        query = "SELECT * FROM async_users WHERE age > 100"
        colnames, rows = await async_execute_query_once("duckdb", self.db_creds, query)

        self.assertEqual(colnames, ["id", "name", "email", "age", "score"])
        self.assertEqual(len(rows), 0)

    async def test_async_execute_query_once_invalid_query(self):
        query = "SELECT * FROM nonexistent_table"
        with self.assertRaises(Exception) as context:
            await async_execute_query_once("duckdb", self.db_creds, query)
        self.assertIn("table", str(context.exception).lower())

    async def test_async_execute_query_once_invalid_column(self):
        query = "SELECT nonexistent_column FROM async_users"
        with self.assertRaises(Exception) as context:
            await async_execute_query_once("duckdb", self.db_creds, query)
        self.assertIn("column", str(context.exception).lower())

    async def test_async_generate_duckdb_schema(self):
        # Create a real AsyncDefog instance for schema generation
        defog = AsyncDefog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test async schema generation without upload
        schema = await defog.generate_db_schema([], upload=False, scan=False)

        self.assertIn("async_users", schema)
        self.assertIn("async_products", schema)

        # Check async_users table schema
        users_schema = schema["async_users"]
        column_names = [col["column_name"] for col in users_schema]
        self.assertIn("id", column_names)
        self.assertIn("name", column_names)
        self.assertIn("email", column_names)
        self.assertIn("age", column_names)
        self.assertIn("score", column_names)

        # Check data types
        for col in users_schema:
            if col["column_name"] == "id":
                self.assertEqual(col["data_type"], "INTEGER")
            elif col["column_name"] == "name":
                self.assertEqual(col["data_type"], "VARCHAR")
            elif col["column_name"] == "score":
                self.assertEqual(col["data_type"], "FLOAT")

    async def test_async_generate_duckdb_schema_specific_tables(self):
        defog = AsyncDefog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test async schema generation for specific table
        schema = await defog.generate_db_schema(
            ["async_users"], upload=False, scan=False
        )

        self.assertIn("async_users", schema)
        self.assertNotIn("async_products", schema)

    async def test_async_generate_duckdb_schema_return_tables_only(self):
        defog = AsyncDefog(api_key="test_key", db_type="duckdb", db_creds=self.db_creds)

        # Test returning table names only
        tables = await defog.generate_db_schema(
            [], upload=False, return_tables_only=True
        )

        self.assertIsInstance(tables, list)
        self.assertIn("async_users", tables)
        self.assertIn("async_products", tables)


if __name__ == "__main__":
    unittest.main()
