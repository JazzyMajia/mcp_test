"""Unit tests for QueryOrchestrator.

This module tests the orchestrator's coordination of the query pipeline,
including retry logic, error handling, and integration with all components.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    QueryRequest,
    ResultValidationResult,
    ReturnType,
)
from pg_mcp.models.schema import ColumnInfo, DatabaseSchema, TableInfo
from pg_mcp.observability.tracing import get_request_id
from pg_mcp.resilience.circuit_breaker import CircuitState
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


class TestDatabaseResolution:
    """Test database name resolution logic."""

    @pytest.fixture
    def mock_pools(self) -> dict[str, MagicMock]:
        """Create mock connection pools."""
        return {
            "db1": MagicMock(),
            "db2": MagicMock(),
        }

    @pytest.fixture
    def orchestrator(self, mock_pools: dict[str, MagicMock]) -> QueryOrchestrator:
        """Create orchestrator with mocked components."""
        return QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools=mock_pools,
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

    def test_resolve_database_specified_valid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified valid database."""
        result = orchestrator._resolve_database("db1")
        assert result == "db1"

    def test_resolve_database_specified_invalid(self, orchestrator: QueryOrchestrator) -> None:
        """Test resolving a specified but invalid database."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database("nonexistent")

        assert "not found" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]
        assert "db2" in exc_info.value.details["available_databases"]

    def test_resolve_database_auto_select_single(self) -> None:
        """Test auto-selecting when only one database available."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"only_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        result = orchestrator._resolve_database(None)
        assert result == "only_db"

    def test_resolve_database_auto_select_multiple_fails(
        self, orchestrator: QueryOrchestrator
    ) -> None:
        """Test that auto-select fails when multiple databases available."""
        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "multiple databases" in str(exc_info.value).lower()
        assert "db1" in exc_info.value.details["available_databases"]

    def test_resolve_database_no_databases(self) -> None:
        """Test error when no databases configured."""
        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(DatabaseError) as exc_info:
            orchestrator._resolve_database(None)

        assert "no databases configured" in str(exc_info.value).lower()


class TestSQLGenerationWithRetry:
    """Test SQL generation with retry logic."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_generate_sql_success_first_attempt(self, mock_schema: DatabaseSchema) -> None:
        """Test successful SQL generation on first attempt."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None  # No exception = valid

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert validation_result.is_select is True
        mock_generator.generate.assert_called_once()
        mock_validator.validate_or_raise.assert_called_once_with("SELECT * FROM users;")

    @pytest.mark.asyncio
    async def test_generate_sql_retry_on_validation_failure(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test retry logic when validation fails."""
        # Setup mocks - first attempt fails validation, second succeeds
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = [
            "SELECT * FROM user;",  # First attempt (wrong table name)
            "SELECT * FROM users;",  # Second attempt (correct)
        ]

        mock_validator = MagicMock()
        # First call raises error, second call succeeds
        mock_validator.validate_or_raise.side_effect = [
            SQLParseError('relation "user" does not exist'),
            None,  # Success on second attempt
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=3),
            validation_config=ValidationConfig(),
        )

        # Execute
        sql, validation_result, _tokens = await orchestrator._generate_sql_with_retry(
            question="Get all users",
            schema=mock_schema,
            request_id="test-123",
        )

        # Verify
        assert sql == "SELECT * FROM users;"
        assert validation_result.is_valid is True
        assert mock_generator.generate.call_count == 2
        assert mock_validator.validate_or_raise.call_count == 2

        # Verify retry included error feedback
        second_call = mock_generator.generate.call_args_list[1]
        assert second_call.kwargs["previous_attempt"] == "SELECT * FROM user;"
        assert 'relation "user" does not exist' in second_call.kwargs["error_feedback"]

    @pytest.mark.asyncio
    async def test_generate_sql_fails_after_max_retries(self, mock_schema: DatabaseSchema) -> None:
        """Test failure after exhausting all retries."""
        # Setup mocks - all attempts fail validation
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError(
            "DELETE statements are not allowed"
        )

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=2),
            validation_config=ValidationConfig(),
        )

        # Execute and verify exception
        with pytest.raises(SecurityViolationError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Delete all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "DELETE statements are not allowed" in str(exc_info.value)
        # Should attempt max_retries + 1 times (initial + retries)
        assert mock_generator.generate.call_count == 3
        assert orchestrator.circuit_breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_generate_sql_circuit_breaker_open(self, mock_schema: DatabaseSchema) -> None:
        """Test that open circuit breaker prevents SQL generation."""
        orchestrator = QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(circuit_breaker_threshold=1),
            validation_config=ValidationConfig(),
        )

        # Manually open the circuit breaker
        orchestrator.circuit_breaker._state = CircuitState.OPEN
        orchestrator.circuit_breaker._failure_count = 5

        # Attempt should fail immediately
        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert "circuit breaker" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_sql_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors during generation."""
        mock_generator = AsyncMock()
        mock_generator.generate.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        with pytest.raises(LLMError) as exc_info:
            await orchestrator._generate_sql_with_retry(
                question="Get all users",
                schema=mock_schema,
                request_id="test-123",
            )

        assert "unexpectedly" in str(exc_info.value).lower()
        assert orchestrator.circuit_breaker.failure_count == 1


class TestResultValidation:
    """Test result validation logic."""

    @pytest.mark.asyncio
    async def test_validate_results_success(self) -> None:
        """Test successful result validation."""
        mock_validator = AsyncMock()
        mock_validator.validate.return_value = ResultValidationResult(
            confidence=85,
            explanation="Results match the question well",
            suggestion=None,
            is_acceptable=True,
        )

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 85
        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_results_disabled(self) -> None:
        """Test that validation is skipped when disabled."""
        mock_validator = AsyncMock()

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100
        mock_validator.validate.assert_not_called()

    @pytest.mark.asyncio
    async def test_validate_results_failure_does_not_raise(self) -> None:
        """Test that validation failures don't raise exceptions."""
        mock_validator = AsyncMock()
        mock_validator.validate.side_effect = Exception("Validation failed")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=mock_validator,
            schema_cache=MagicMock(),
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Should not raise, returns default confidence
        confidence = await orchestrator._validate_results_safely(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 42}],
            row_count=1,
            request_id="test-123",
        )

        assert confidence == 100


class TestExecuteQueryFlow:
    """Test complete query execution flow."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="test_db",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnInfo(
                            name="name",
                            data_type="varchar(255)",
                            is_nullable=False,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    @pytest.mark.asyncio
    async def test_execute_query_sql_only(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=SQL."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT * FROM users;"
        assert response.validation is not None
        assert response.validation.is_valid is True
        assert response.data is None  # No execution for SQL-only
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_with_results(self, mock_schema: DatabaseSchema) -> None:
        """Test executing query with return_type=RESULT."""
        # Setup mocks
        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT id, name FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = (
            [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            2,  # total count
        )

        mock_result_validator = AsyncMock()
        mock_result_validator.validate.return_value = ResultValidationResult(
            confidence=90,
            explanation="Good results",
            suggestion=None,
            is_acceptable=True,
        )

        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=mock_executor,
            result_validator=mock_result_validator,
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=True),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        assert response.generated_sql == "SELECT id, name FROM users;"
        assert response.data is not None
        assert response.data.row_count == 2
        assert len(response.data.rows) == 2
        assert response.data.columns == ["id", "name"]
        assert response.confidence == 90
        assert response.error is None

    @pytest.mark.asyncio
    async def test_execute_query_schema_not_cached(self) -> None:
        """Test loading schema when not in cache."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Not in cache
        mock_cache.load = AsyncMock(return_value=mock_schema)

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_pool = MagicMock()

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": mock_pool},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify schema was loaded
        mock_cache.load.assert_called_once_with("test_db", mock_pool)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_execute_query_schema_load_fails(self) -> None:
        """Test handling of schema load failure."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.load = AsyncMock(side_effect=Exception("DB connection failed"))

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Test query",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "schema" in response.error.message.lower()
        assert response.generated_sql is None

    @pytest.mark.asyncio
    async def test_execute_query_validation_error(self) -> None:
        """Test handling of SQL validation errors."""
        mock_schema = DatabaseSchema(
            database_name="test_db",
            tables=[],
            version="15.0",
        )

        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "DELETE FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.side_effect = SecurityViolationError("DELETE not allowed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(max_retries=1),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Delete all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "DELETE not allowed" in response.error.message
        assert response.error.code == "security_violation"

    @pytest.mark.asyncio
    async def test_execute_query_execution_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of SQL execution errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT * FROM users;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        mock_executor = AsyncMock()
        mock_executor.execute.side_effect = DatabaseError("Query execution failed")

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=mock_executor,
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.RESULT,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert "execution failed" in response.error.message.lower()
        assert response.error.code == "database_error"

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_error(self, mock_schema: DatabaseSchema) -> None:
        """Test handling of unexpected errors."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.side_effect = RuntimeError("Unexpected error")

        orchestrator = QueryOrchestrator(
            sql_generator=MagicMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"test_db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute
        request = QueryRequest(
            question="Get all users",
            database="test_db",
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify error response
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "internal_error"
        assert "internal server error" in response.error.message.lower()

    @pytest.mark.asyncio
    async def test_execute_query_auto_select_database(self, mock_schema: DatabaseSchema) -> None:
        """Test auto-selecting database when only one available."""
        # Setup mocks
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_schema

        mock_generator = AsyncMock()
        mock_generator.generate.return_value = "SELECT 1;"

        mock_validator = MagicMock()
        mock_validator.validate_or_raise.return_value = None

        orchestrator = QueryOrchestrator(
            sql_generator=mock_generator,
            sql_validator=mock_validator,
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=mock_cache,
            pools={"only_db": MagicMock()},  # Only one database
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(),
        )

        # Execute without specifying database
        request = QueryRequest(
            question="Test query",
            database=None,  # No database specified
            return_type=ReturnType.SQL,
        )
        response = await orchestrator.execute_query(request)

        # Verify
        assert response.success is True
        # Verify schema was fetched for auto-selected database
        mock_cache.get.assert_called_once_with("only_db")


class TestMultiDatabaseRouting:
    """Test per-database executor and validator routing."""

    @pytest.fixture
    def mock_schema(self) -> DatabaseSchema:
        """Create mock database schema."""
        return DatabaseSchema(
            database_name="db1",
            tables=[
                TableInfo(
                    schema_name="public",
                    table_name="users",
                    columns=[
                        ColumnInfo(
                            name="id",
                            data_type="integer",
                            is_nullable=False,
                            is_primary_key=True,
                        ),
                    ],
                )
            ],
            version="15.0",
        )

    def _build(
        self,
        mock_schema: DatabaseSchema,
        executors: dict[str, AsyncMock],
        validators: dict[str, MagicMock],
    ) -> QueryOrchestrator:
        """Build an orchestrator with per-database components."""
        return QueryOrchestrator(
            sql_generator=AsyncMock(),
            sql_validator=MagicMock(),
            sql_executor=MagicMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db1": MagicMock(), "db2": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            sql_executors=executors,
            sql_validators=validators,
        )

    @pytest.mark.asyncio
    async def test_executor_for_requested_database_is_used(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test the executor bound to the requested database executes the SQL."""
        executor_db1, executor_db2 = AsyncMock(), AsyncMock()
        executor_db1.execute.return_value = ([{"id": 1}], 1)

        generator = AsyncMock()
        generator.generate.return_value = "SELECT * FROM users"

        orchestrator = self._build(
            mock_schema,
            executors={"db1": executor_db1, "db2": executor_db2},
            validators={"db1": MagicMock(), "db2": MagicMock()},
        )
        orchestrator.sql_generator = generator
        orchestrator.schema_cache.get.return_value = mock_schema

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db1")
        )

        assert response.success is True
        executor_db1.execute.assert_awaited_once()
        executor_db2.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validator_for_requested_database_is_used(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test the per-database validator enforces its security policy."""
        validator_db1, validator_db2 = MagicMock(), MagicMock()

        generator = AsyncMock()
        generator.generate.return_value = "SELECT * FROM users"

        orchestrator = self._build(
            mock_schema,
            executors={"db1": AsyncMock(), "db2": AsyncMock()},
            validators={"db1": validator_db1, "db2": validator_db2},
        )
        orchestrator.sql_generator = generator
        orchestrator.schema_cache.get.return_value = mock_schema

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db2", return_type=ReturnType.SQL)
        )

        assert response.success is True
        validator_db2.validate_or_raise.assert_called_once_with("SELECT * FROM users")
        validator_db1.validate_or_raise.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_executor_for_database_fails(
        self, mock_schema: DatabaseSchema
    ) -> None:
        """Test error response when a database has no executor configured."""
        orchestrator = self._build(
            mock_schema,
            executors={"db1": AsyncMock()},  # db2 missing
            validators={"db1": MagicMock(), "db2": MagicMock()},
        )
        orchestrator.schema_cache.get.return_value = mock_schema

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db2", return_type=ReturnType.SQL)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.DATABASE_ERROR.value
        assert "db2" in response.error.message


class TestRateLimiting:
    """Test rate limiting integration in the query flow."""

    def _build(self, resilience_config: ResilienceConfig) -> QueryOrchestrator:
        """Build an orchestrator with a real rate limiter."""
        generator = AsyncMock()
        generator.generate.return_value = "SELECT 1"
        validator = MagicMock()

        return QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=resilience_config,
            validation_config=ValidationConfig(enabled=False),
            rate_limiter=MultiRateLimiter(query_limit=1, llm_limit=1),
        )

    @pytest.mark.asyncio
    async def test_query_rejected_when_limit_reached(self) -> None:
        """Test query returns rate_limit_exceeded when no slot is available."""
        orchestrator = self._build(ResilienceConfig(rate_limit_timeout=0.1))

        # Exhaust the single query slot
        assert await orchestrator.rate_limiter.query_limiter.acquire() is True

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db")
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMIT_EXCEEDED.value
        assert response.tokens_used == 0

    @pytest.mark.asyncio
    async def test_query_slot_released_after_success(self) -> None:
        """Test the query slot is released after a successful request."""
        orchestrator = self._build(ResilienceConfig(rate_limit_timeout=0.1))

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        )

        assert response.success is True
        await asyncio.sleep(0)  # Allow release bookkeeping task to run
        stats = orchestrator.rate_limiter.query_limiter.get_stats()
        assert stats["active_count"] == 0
        assert stats["total_requests"] == 1
        assert stats["total_rejections"] == 0

    @pytest.mark.asyncio
    async def test_query_slot_released_after_error(self) -> None:
        """Test the query slot is released even when the query fails."""
        orchestrator = self._build(ResilienceConfig(rate_limit_timeout=0.1))
        orchestrator.sql_validator.validate_or_raise.side_effect = SQLParseError("bad")

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db")
        )

        assert response.success is False
        await asyncio.sleep(0)
        stats = orchestrator.rate_limiter.query_limiter.get_stats()
        assert stats["active_count"] == 0

    @pytest.mark.asyncio
    async def test_llm_rate_limit_timeout_raises_rate_limit_error(self) -> None:
        """Test LLM slot exhaustion produces rate_limit_exceeded, not an LLM error."""
        orchestrator = self._build(ResilienceConfig(rate_limit_timeout=0.1))

        # Exhaust the single LLM slot
        assert await orchestrator.rate_limiter.llm_limiter.acquire() is True

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMIT_EXCEEDED.value


class TestRetryBackoff:
    """Test exponential backoff between generation retries."""

    @pytest.mark.asyncio
    async def test_backoff_sleep_between_attempts(self) -> None:
        """Test sleep is called with retry_delay * backoff_factor**attempt."""
        schema = DatabaseSchema(database_name="db", tables=[], version="15.0")

        generator = AsyncMock()
        generator.generate.side_effect = ["SELECT blocked_table FROM x", "SELECT 1"]

        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SecurityViolationError("blocked table"),
            None,
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(
                max_retries=2, retry_delay=0.5, backoff_factor=3.0
            ),
            validation_config=ValidationConfig(enabled=False),
        )

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            sql, _, _ = await orchestrator._generate_sql_with_retry(
                question="q", schema=schema, request_id="req-1"
            )

        assert sql == "SELECT 1"
        mock_sleep.assert_awaited_once_with(0.5)  # retry_delay * backoff_factor ** 0

    @pytest.mark.asyncio
    async def test_backoff_escalates_with_attempt(self) -> None:
        """Test the delay grows exponentially across consecutive retries."""
        schema = DatabaseSchema(database_name="db", tables=[], version="15.0")

        generator = AsyncMock()
        generator.generate.side_effect = ["SELECT a", "SELECT b", "SELECT c"]

        validator = MagicMock()
        validator.validate_or_raise.side_effect = [
            SecurityViolationError("e1"),
            SecurityViolationError("e2"),
            None,
        ]

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator,
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(
                max_retries=3, retry_delay=1.0, backoff_factor=2.0
            ),
            validation_config=ValidationConfig(enabled=False),
        )

        with patch(
            "pg_mcp.services.orchestrator.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await orchestrator._generate_sql_with_retry(
                question="q", schema=schema, request_id="req-1"
            )

        delays = [call.args[0] for call in mock_sleep.await_args_list]
        assert delays == [1.0, 2.0]


class TestMetricsIntegration:
    """Test metrics collection in the query flow."""

    def _build(
        self,
        metrics: MagicMock,
        validator: MagicMock | None = None,
    ) -> QueryOrchestrator:
        """Build an orchestrator with mocked metrics collector."""
        generator = AsyncMock()
        generator.generate.return_value = "SELECT 1"

        return QueryOrchestrator(
            sql_generator=generator,
            sql_validator=validator or MagicMock(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
            metrics=metrics,
        )

    @pytest.mark.asyncio
    async def test_successful_query_records_metrics(self) -> None:
        """Test success counters and histograms are recorded."""
        metrics = MagicMock()

        orchestrator = self._build(metrics)
        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        )

        assert response.success is True
        metrics.increment_query_request.assert_called_once_with(
            status="success", database="db"
        )
        metrics.increment_llm_call.assert_called_with("generate_sql")
        metrics.observe_llm_latency.assert_called_once()
        metrics.query_duration.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_violation_records_rejection_metric(self) -> None:
        """Test SQL rejections increment the sql_rejected counter."""
        metrics = MagicMock()
        validator = MagicMock()
        validator.validate_or_raise.side_effect = SecurityViolationError("blocked")

        orchestrator = self._build(metrics, validator=validator)
        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db")
        )

        assert response.success is False
        metrics.increment_sql_rejected.assert_called_with("security_violation")
        metrics.increment_query_request.assert_called_once_with(
            status="security_violation", database="db"
        )

    @pytest.mark.asyncio
    async def test_db_query_duration_recorded_on_execution(self) -> None:
        """Test db query duration histogram is observed when SQL executes."""
        metrics = MagicMock()
        orchestrator = self._build(metrics)
        orchestrator.sql_executor.execute.return_value = ([{"id": 1}], 1)

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db")
        )

        assert response.success is True
        metrics.observe_db_query_duration.assert_called_once()


class TestRequestTracing:
    """Test request context propagation through the pipeline."""

    @pytest.mark.asyncio
    async def test_request_id_available_during_generation(self) -> None:
        """Test request_id is set in the tracing context during SQL generation."""
        captured: dict[str, str | None] = {}

        async def fake_generate(**kwargs: object) -> str:
            captured["request_id"] = get_request_id()
            return "SELECT 1"

        generator = AsyncMock()
        generator.generate.side_effect = fake_generate

        orchestrator = QueryOrchestrator(
            sql_generator=generator,
            sql_validator=MagicMock(),
            sql_executor=AsyncMock(),
            result_validator=MagicMock(),
            schema_cache=MagicMock(),
            pools={"db": MagicMock()},
            resilience_config=ResilienceConfig(),
            validation_config=ValidationConfig(enabled=False),
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="q", database="db", return_type=ReturnType.SQL)
        )

        assert response.success is True
        assert captured["request_id"] is not None
        # Request ID must be a valid UUID string
        assert str(uuid.UUID(captured["request_id"])) == captured["request_id"]
        # Context is cleaned up after the request completes
        assert get_request_id() is None
