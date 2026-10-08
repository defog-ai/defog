import importlib.util
import os
import tempfile
import unittest
import sqlite3

from defog.query import execute_query_once, async_execute_query_once
from defog import Defog, AsyncDefog


class SQLiteConnectorTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary SQLite database for testing
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_file.close()
        self.db_path = self.db_file.name

        # Create test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create test tables
        cursor.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER,
                is_active BOOLEAN DEFAULT 1
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL,
                category TEXT
            )
        """
        )

        # Insert test data
        cursor.execute(
            "INSERT INTO users (name, email, age, is_active) VALUES (?, ?, ?, ?)",
            ("John Doe", "john@example.com", 30, True),
        )
        cursor.execute(
            "INSERT INTO users (name, email, age, is_active) VALUES (?, ?, ?, ?)",
            ("Jane Smith", "jane@example.com", 25, True),
        )
        cursor.execute(
            "INSERT INTO users (name, email, age, is_active) VALUES (?, ?, ?, ?)",
            ("Bob Johnson", "bob@example.com", 35, False),
        )

        cursor.execute(
            "INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
            ("Laptop", 999.99, "Electronics"),
        )
        cursor.execute(
            "INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
            ("Mouse", 29.99, "Electronics"),
        )
        cursor.execute(
            "INSERT INTO products (name, price, category) VALUES (?, ?, ?)",
            ("Book", 19.99, "Education"),
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
        colnames, rows = execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["name", "email"])
        self.assertEqual(len(rows), 2)
        self.assertIn(("John Doe", "john@example.com"), rows)
        self.assertIn(("Bob Johnson", "bob@example.com"), rows)

    def test_execute_query_once_count(self):
        query = "SELECT COUNT(*) FROM products"
        colnames, rows = execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["COUNT(*)"])
        self.assertEqual(rows[0][0], 3)

    def test_execute_query_once_aggregate(self):
        query = (
            "SELECT category, AVG(price) as avg_price FROM products GROUP BY category"
        )
        colnames, rows = execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["category", "avg_price"])
        self.assertEqual(len(rows), 2)

        # Check that we have Electronics and Education categories
        categories = [row[0] for row in rows]
        self.assertIn("Electronics", categories)
        self.assertIn("Education", categories)

    def test_execute_query_once_empty_result(self):
        query = "SELECT * FROM users WHERE age > 100"
        colnames, rows = execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["id", "name", "email", "age", "is_active"])
        self.assertEqual(len(rows), 0)

    def test_execute_query_once_invalid_query(self):
        query = "SELECT * FROM nonexistent_table"
        with self.assertRaises(Exception) as context:
            execute_query_once("sqlite", self.db_creds, query)
        self.assertIn("no such table", str(context.exception).lower())

    def test_generate_sqlite_schema_basic(self):
        # Create a real Defog instance for schema generation
        defog = Defog(api_key="test_key", db_type="sqlite", db_creds=self.db_creds)

        # Test schema generation without upload
        schema = defog.generate_db_schema([], upload=False, scan=False)

        self.assertIn("users", schema)
        self.assertIn("products", schema)

        # Check users table schema
        users_schema = schema["users"]
        column_names = [col["column_name"] for col in users_schema]
        self.assertIn("id", column_names)
        self.assertIn("name", column_names)
        self.assertIn("email", column_names)
        self.assertIn("age", column_names)
        self.assertIn("is_active", column_names)

        # Check data types
        for col in users_schema:
            if col["column_name"] == "id":
                self.assertEqual(col["data_type"], "INTEGER")
            elif col["column_name"] == "name":
                self.assertEqual(col["data_type"], "TEXT")

    def test_generate_sqlite_schema_specific_tables(self):
        defog = Defog(api_key="test_key", db_type="sqlite", db_creds=self.db_creds)

        # Test schema generation for specific table
        schema = defog.generate_db_schema(["users"], upload=False, scan=False)

        self.assertIn("users", schema)
        self.assertNotIn("products", schema)

    def test_generate_sqlite_schema_return_tables_only(self):
        defog = Defog(api_key="test_key", db_type="sqlite", db_creds=self.db_creds)

        # Test returning table names only
        tables = defog.generate_db_schema([], upload=False, return_tables_only=True)

        self.assertIsInstance(tables, list)
        self.assertIn("users", tables)
        self.assertIn("products", tables)


@unittest.skipUnless(importlib.util.find_spec("aiosqlite"), "needs aiosqlite")
class AsyncSQLiteConnectorTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a temporary SQLite database for testing
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_file.close()
        self.db_path = self.db_file.name

        # Create test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create test tables
        cursor.execute(
            """
            CREATE TABLE async_users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                age INTEGER
            )
        """
        )

        # Insert test data
        cursor.execute(
            "INSERT INTO async_users (name, email, age) VALUES (?, ?, ?)",
            ("Alice", "alice@example.com", 28),
        )
        cursor.execute(
            "INSERT INTO async_users (name, email, age) VALUES (?, ?, ?)",
            ("Charlie", "charlie@example.com", 32),
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
        colnames, rows = await async_execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["name", "email"])
        self.assertEqual(len(rows), 2)
        self.assertIn(("Alice", "alice@example.com"), rows)
        self.assertIn(("Charlie", "charlie@example.com"), rows)

    async def test_async_execute_query_once_count(self):
        query = "SELECT COUNT(*) FROM async_users"
        colnames, rows = await async_execute_query_once("sqlite", self.db_creds, query)

        self.assertEqual(colnames, ["COUNT(*)"])
        self.assertEqual(rows[0][0], 2)

    async def test_async_execute_query_once_invalid_query(self):
        query = "SELECT * FROM nonexistent_table"
        with self.assertRaises(Exception) as context:
            await async_execute_query_once("sqlite", self.db_creds, query)
        self.assertIn("no such table", str(context.exception).lower())

    async def test_async_generate_sqlite_schema(self):
        # Create a real AsyncDefog instance for schema generation
        defog = AsyncDefog(api_key="test_key", db_type="sqlite", db_creds=self.db_creds)

        # Test async schema generation without upload
        schema = await defog.generate_db_schema([], upload=False, scan=False)

        self.assertIn("async_users", schema)

        # Check async_users table schema
        users_schema = schema["async_users"]
        column_names = [col["column_name"] for col in users_schema]
        self.assertIn("id", column_names)
        self.assertIn("name", column_names)
        self.assertIn("email", column_names)
        self.assertIn("age", column_names)


if __name__ == "__main__":
    unittest.main()
