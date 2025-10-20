"""Tests for the schema documenter functionality."""

import pytest
from unittest.mock import Mock, patch
from defog.schema_documenter import (
    SchemaDocumenter,
    DocumentationConfig,
    TableDocumentation,
    document_schema_for_defog,
)


class TestDocumentationConfig:
    """Test DocumentationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DocumentationConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-haiku-4-5"
        assert config.sample_size == 100
        assert config.min_confidence_score == 0.7
        assert config.include_data_patterns is True
        assert config.dry_run is False
        assert config.concurrent_requests == 3
        assert config.rate_limit_delay == 1.0


class TestTableDocumentation:
    """Test TableDocumentation class."""

    def test_table_documentation_init(self):
        """Test TableDocumentation initialization."""
        doc = TableDocumentation("test_table")
        assert doc.table_name == "test_table"
        assert doc.table_description is None
        assert doc.table_confidence == 0.0
        assert doc.column_descriptions == {}
        assert doc.column_confidences == {}
        assert doc.existing_table_comment is None
        assert doc.existing_column_comments == {}


class TestSchemaDocumenter:
    """Test SchemaDocumenter class."""

    def test_init_postgres(self):
        """Test SchemaDocumenter initialization for Postgres."""
        db_creds = {"host": "localhost", "database": "test"}
        documenter = SchemaDocumenter("postgres", db_creds)
        assert documenter.db_type == "postgres"
        assert documenter.db_creds == db_creds
        assert isinstance(documenter.config, DocumentationConfig)

    def test_init_duckdb(self):
        """Test SchemaDocumenter initialization for DuckDB."""
        db_creds = {"database": ":memory:"}
        documenter = SchemaDocumenter("duckdb", db_creds)
        assert documenter.db_type == "duckdb"
        assert documenter.db_creds == db_creds

    def test_init_unsupported_db(self):
        """Test SchemaDocumenter initialization with unsupported database."""
        with pytest.raises(ValueError, match="Database type 'mysql' not supported"):
            SchemaDocumenter("mysql", {})

    def test_init_unsupported_provider(self):
        """Test SchemaDocumenter initialization with unsupported LLM provider."""
        config = DocumentationConfig(provider="unsupported")
        # Note: Provider validation happens in chat_async, not in SchemaDocumenter init
        # This test creates the documenter successfully but would fail on actual LLM calls
        documenter = SchemaDocumenter("postgres", {}, config)
        assert documenter.config.provider == "unsupported"


class TestPatternAnalysis:
    """Test data pattern analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.documenter = SchemaDocumenter("postgres", {"host": "localhost"})

    def test_is_email(self):
        """Test email pattern detection."""
        assert self.documenter._is_email("test@example.com") is True
        assert self.documenter._is_email("user.name+tag@domain.co.uk") is True
        assert self.documenter._is_email("invalid-email") is False
        assert self.documenter._is_email("@example.com") is False

    def test_is_url(self):
        """Test URL pattern detection."""
        assert self.documenter._is_url("https://example.com") is True
        assert self.documenter._is_url("http://sub.domain.com/path?query=1") is True
        assert self.documenter._is_url("not-a-url") is False
        assert self.documenter._is_url("ftp://example.com") is False

    def test_is_uuid(self):
        """Test UUID pattern detection."""
        assert self.documenter._is_uuid("123e4567-e89b-12d3-a456-426614174000") is True
        assert self.documenter._is_uuid("invalid-uuid") is False
        assert self.documenter._is_uuid("123e4567-e89b-12d3-a456") is False

    def test_is_phone(self):
        """Test phone number pattern detection."""
        assert self.documenter._is_phone("+1234567890") is True
        assert self.documenter._is_phone("123-456-7890") is True
        assert self.documenter._is_phone("123") is False

    def test_analyze_string_patterns(self):
        """Test string pattern analysis."""
        # Email data
        email_data = ["user1@example.com", "user2@test.org", "user3@domain.net"]
        patterns = self.documenter._analyze_string_patterns(email_data)
        assert patterns.get("email") is True

        # Categorical data
        categorical_data = ["red", "blue", "green", "red", "blue"] * 10
        patterns = self.documenter._analyze_string_patterns(categorical_data)
        assert patterns.get("categorical") is True
        assert "categories" in patterns

    def test_analyze_numeric_patterns(self):
        """Test numeric pattern analysis."""
        # Unique ID-like data
        id_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        patterns = self.documenter._analyze_numeric_patterns(id_data)
        assert patterns.get("unique") is True
        assert patterns.get("id_like") is True

    def test_analyze_general_patterns(self):
        """Test general pattern analysis."""
        data_with_nulls = ["value1", None, "value2", "", "value3"]
        patterns = self.documenter._analyze_general_patterns(data_with_nulls)
        assert "null_percentage" in patterns
        assert patterns["null_percentage"] == 40.0


class TestLLMIntegration:
    """Test LLM integration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.documenter = SchemaDocumenter("postgres", {"host": "localhost"})

    def test_build_documentation_prompt(self):
        """Test building documentation prompt."""
        table_name = "users"
        analysis = {
            "columns": [
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "email", "type": "varchar", "nullable": True},
            ],
            "sample_data": {
                "id": [1, 2, 3],
                "email": ["user1@example.com", "user2@example.com", None],
            },
            "data_patterns": {"id": {"id_like": True}, "email": {"email": True}},
            "row_count": 1000,
        }

        prompt = self.documenter._build_documentation_prompt(table_name, analysis)
        assert "users" in prompt
        assert "1,000" in prompt
        assert "id (integer, NOT NULL)" in prompt
        assert "email (varchar, NULL)" in prompt
        assert "[EMAIL ADDRESSES]" in prompt
        assert "[ID/SEQUENCE]" in prompt

    def test_structured_response_format(self):
        """Test that structured response format is used instead of JSON parsing."""
        from defog.schema_documenter import (
            ColumnDocumentation,
            SchemaDocumentationResponse,
        )

        # Test that the Pydantic models are properly defined
        col_doc = ColumnDocumentation(description="User ID", confidence=0.95)
        assert col_doc.description == "User ID"
        assert col_doc.confidence == 0.95

        response = SchemaDocumentationResponse(
            table_description="Stores user information",
            table_confidence=0.9,
            columns={"id": col_doc},
        )

        assert response.table_description == "Stores user information"
        assert response.table_confidence == 0.9
        assert "id" in response.columns
        assert response.columns["id"].description == "User ID"


class TestPostgresComments:
    """Test Postgres comment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.documenter = SchemaDocumenter("postgres", {"host": "localhost"})

    @patch("psycopg2.connect")
    def test_get_postgres_comments(self, mock_connect):
        """Test retrieving existing Postgres comments."""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock table comment query
        mock_cursor.fetchone.side_effect = [
            ("Table stores user data",),  # table comment
        ]

        # Mock column comments query
        mock_cursor.fetchall.return_value = [
            ("id", "User identifier"),
            ("email", "User email address"),
        ]

        table_comment, column_comments = self.documenter._get_postgres_comments(
            mock_conn, "users"
        )

        assert table_comment == "Table stores user data"
        assert column_comments["id"] == "User identifier"
        assert column_comments["email"] == "User email address"

    @patch("psycopg2.connect")
    def test_apply_postgres_comments(self, mock_connect):
        """Test applying comments to Postgres database."""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Test table comment
        success = self.documenter._apply_table_comment(
            mock_conn, "users", "Stores user information"
        )
        assert success is True
        mock_cursor.execute.assert_called()

        # Test column comment
        success = self.documenter._apply_column_comment(
            mock_conn, "users", "email", "User email address"
        )
        assert success is True


class TestDuckDBComments:
    """Test DuckDB comment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.documenter = SchemaDocumenter("duckdb", {"database": ":memory:"})

    def test_get_duckdb_comments(self):
        """Test retrieving DuckDB comments (limited support)."""
        table_comment, column_comments = self.documenter._get_duckdb_comments(
            None, "test_table"
        )
        assert table_comment is None
        assert column_comments == {}


class TestAsyncOperations:
    """Test async functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DocumentationConfig(dry_run=True)
        self.documenter = SchemaDocumenter(
            "postgres", {"host": "localhost"}, self.config
        )

    @pytest.mark.asyncio
    async def test_document_schema_for_defog(self):
        """Test the utility function for Defog integration."""
        with (
            patch.object(SchemaDocumenter, "_get_connection"),
            patch.object(
                SchemaDocumenter, "_get_all_tables", return_value=["test_table"]
            ),
            patch.object(SchemaDocumenter, "_document_single_table") as mock_doc,
        ):
            mock_doc.return_value = TableDocumentation("test_table")

            result = await document_schema_for_defog(
                "postgres", {"host": "localhost"}, None, self.config
            )

            assert "test_table" in result
            assert isinstance(result["test_table"], TableDocumentation)


class TestSchemaGeneration:
    """Test integration with schema generation."""

    def test_preserve_existing_comments(self):
        """Test that existing comments are preserved."""
        doc = TableDocumentation(
            table_name="users",
            existing_table_comment="Existing table description",
            existing_column_comments={"id": "Existing column description"},
        )

        # When existing comments exist, new descriptions should not override them
        assert doc.existing_table_comment == "Existing table description"
        assert doc.existing_column_comments["id"] == "Existing column description"


if __name__ == "__main__":
    pytest.main([__file__])
