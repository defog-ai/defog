import importlib.util
import os
import shutil
import unittest
from unittest import mock

from defog.query import (
    is_connection_error,
    execute_query,
    async_execute_query_once,
    async_execute_query,
)


class ExecuteQueryOnceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # if connection.json exists, copy it to /tmp since we'll be overwriting it
        home_dir = os.path.expanduser("~")
        self.logs_path = os.path.join(home_dir, ".defog", "logs")
        self.tmp_dir = os.path.join("/tmp")
        self.moved = False
        if os.path.exists(self.logs_path):
            print("Moving logs to /tmp")
            if os.path.exists(os.path.join(self.tmp_dir, "logs")):
                os.remove(os.path.join(self.tmp_dir, "logs"))
            shutil.move(self.logs_path, self.tmp_dir)
            self.moved = True

    @classmethod
    def tearDownClass(self):
        # copy back the original after all tests have completed
        if self.moved:
            print("Moving logs back to ~/.defog")
            shutil.move(os.path.join(self.tmp_dir, "logs"), self.logs_path)

    @mock.patch("requests.post")
    @mock.patch("defog.query.execute_query_once")
    def test_execute_query_success(self, mock_execute_query_once, mock_requests_post):
        # Mock the execute_query_once function
        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query1 = "SELECT * FROM table_name;"
        query2 = "SELECT * FROM new_table_name;"
        api_key = "your_api_key"
        question = "your_question"
        hard_filters = "your_hard_filters"
        retries = 3

        # Set up the mock responses
        mock_execute_query_once.return_value = (
            ["col1", "col2"],
            [("data1", "data2"), ("data3", "data4")],
        )
        mock_response = mock.Mock()
        mock_response.json.return_value = {"new_query": query2}
        mock_requests_post.return_value = mock_response

        # Call the function being tested
        colnames, results = execute_query(
            query=query1,
            api_key=api_key,
            db_type=db_type,
            db_creds=db_creds,
            question=question,
            hard_filters=hard_filters,
            retries=retries,
        )

        # Assert the expected behavior
        mock_execute_query_once.assert_called_once_with(db_type, db_creds, query1)
        mock_requests_post.assert_not_called()
        self.assertEqual(colnames, ["col1", "col2"])
        self.assertEqual(results, [("data1", "data2"), ("data3", "data4")])


class ExecuteAsyncQueryOnceTestCase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(self):
        # if connection.json exists, copy it to /tmp since we'll be overwriting it
        home_dir = os.path.expanduser("~")
        self.logs_path = os.path.join(home_dir, ".defog", "logs")
        self.tmp_dir = os.path.join("/tmp")
        self.moved = False
        if os.path.exists(self.logs_path):
            print("Moving logs to /tmp")
            if os.path.exists(os.path.join(self.tmp_dir, "logs")):
                os.remove(os.path.join(self.tmp_dir, "logs"))
            shutil.move(self.logs_path, self.tmp_dir)
            self.moved = True

    @classmethod
    def tearDownClass(self):
        # copy back the original after all tests have completed
        if self.moved:
            print("Moving logs back to ~/.defog")
            shutil.move(os.path.join(self.tmp_dir, "logs"), self.logs_path)

    @unittest.skipUnless(
        importlib.util.find_spec("asyncpg"), "needs the async-postgres extra"
    )
    @mock.patch("asyncpg.connect")
    async def test_async_execute_query_once_success(self, mock_connect):
        # Mock the asyncpg.connect function
        mock_cursor = mock_connect.return_value.fetch
        mock_cursor.return_value = [
            {"col1": "data1", "col2": "data2"},
            {"col1": "data3", "col2": "data4"},
        ]

        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query = "SELECT * FROM table_name;"

        colnames, results = await async_execute_query_once(db_type, db_creds, query)

        # Add your assertions here to validate the results
        self.assertEqual(colnames, ["col1", "col2"])
        self.assertEqual(results, [["data1", "data2"], ["data3", "data4"]])
        print("Postgres async query execution test passed!")

    @mock.patch("httpx.AsyncClient.post")
    @mock.patch("defog.query.async_execute_query_once")
    async def test_async_execute_query_success(
        self, mock_execute_query_once, mock_aiohttp_post
    ):
        # Mock the execute_query_once function
        db_type = "postgres"
        db_creds = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
        }
        query1 = "SELECT * FROM table_name;"
        query2 = "SELECT * FROM new_table_name;"
        api_key = "your_api_key"
        question = "your_question"
        hard_filters = "your_hard_filters"
        retries = 3

        # Set up the mock responses
        mock_execute_query_once.return_value = (
            ["col1", "col2"],
            [["data1", "data2"], ["data3", "data4"]],
        )

        # Mock the async httpx response
        mock_response = mock.Mock()
        mock_response.json = mock.Mock(return_value={"new_query": query2})
        mock_aiohttp_post.return_value = mock_response

        # Call the function being tested
        colnames, results = await async_execute_query(
            query=query1,
            api_key=api_key,
            db_type=db_type,
            db_creds=db_creds,
            question=question,
            hard_filters=hard_filters,
            retries=retries,
        )

        # Assert the expected behavior
        mock_execute_query_once.assert_called_once_with(
            db_type="postgres",  # Use keyword arguments for db_type
            db_creds={
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "user": "test_user",
                "password": "test_password",
            },  # Use keyword arguments for db_creds
            query="SELECT * FROM table_name;",  # Use keyword arguments for query
        )

        self.assertEqual(colnames, ["col1", "col2"])
        self.assertEqual(results, [["data1", "data2"], ["data3", "data4"]])


class TestConnectionError(unittest.TestCase):
    def test_connection_failed(self):
        self.assertTrue(
            is_connection_error(
                """connection to server on socket "/tmp/.s.PGSQL.5432" failed: No such file or directory
    Is the server running locally and accepting connections on that socket?"""
            )
        )

    def test_not_connection_failed(self):
        self.assertFalse(
            is_connection_error(
                'psycopg2.errors.UndefinedTable: relation "nonexistent_table" does not exist'
            )
        )
        self.assertFalse(
            is_connection_error(
                'psycopg2.errors.SyntaxError: syntax error at or near "nonexistent_table"'
            )
        )
        self.assertFalse(
            is_connection_error(
                'psycopg2.errors.UndefinedColumn: column "nonexistent_column" does not exist'
            )
        )

    def test_empty_string(self):
        self.assertFalse(is_connection_error(""))
        self.assertFalse(is_connection_error(None))


if __name__ == "__main__":
    unittest.main()
