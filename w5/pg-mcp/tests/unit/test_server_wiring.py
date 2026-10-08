"""Tests for MCP server wiring.

These tests verify that the lifespan correctly wires up multi-database
components (per-database validators/executors with security policies),
resilience components (rate limiter), and observability (metrics) into the
query orchestrator, and that the query tool handles invalid input gracefully.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp import server as server_module
from pg_mcp.models.query import QueryResponse
from pg_mcp.models.schema import DatabaseSchema


def _make_schema(db_name: str) -> DatabaseSchema:
    """Create an empty schema for a database."""
    return DatabaseSchema(database_name=db_name, tables=[], version="PostgreSQL 16.0")


@pytest.fixture
def two_db_env(monkeypatch: pytest.MonkeyPatch):
    """Configure two databases via environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_NAME", "dbmain")
    monkeypatch.setenv(
        "DATABASES",
        json.dumps([{"name": "db2", "host": "remote.example.com"}]),
    )
    # Per-database policy for db2
    monkeypatch.setenv(
        "SECURITY_DATABASE_POLICIES",
        json.dumps({"db2": {"blocked_tables": "secrets", "allow_explain": True}}),
    )


@pytest.fixture
def fake_pools(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace real pool creation/closing with fakes."""
    pools = {"dbmain": MagicMock(), "db2": MagicMock()}

    async def fake_create_pools(configs):
        return pools

    async def fake_close_pools(pools_dict, **kwargs):
        fake_close_pools.called = True

    fake_close_pools.called = False  # type: ignore[attr-defined]
    monkeypatch.setattr(server_module, "create_pools", fake_create_pools)
    monkeypatch.setattr(server_module, "close_pools", fake_close_pools)
    return pools


class TestLifespanWiring:
    """Test lifespan startup wires all components correctly."""

    @pytest.mark.asyncio
    async def test_per_database_components_wired(
        self, two_db_env, fake_pools: dict[str, MagicMock]
    ) -> None:
        """Test validators/executors are created per database and passed to the orchestrator."""
        with patch("pg_mcp.cache.schema_cache.SchemaIntrospector") as mock_cls:
            introspector = AsyncMock()
            introspector.introspect.side_effect = lambda: _make_schema("db")
            mock_cls.return_value = introspector

            async with server_module.lifespan(None):
                orchestrator = server_module._orchestrator
                assert orchestrator is not None
                assert set(orchestrator.sql_executors) == {"dbmain", "db2"}
                assert set(orchestrator.sql_validators) == {"dbmain", "db2"}

                # Rate limiter and metrics must be integrated, not just constructed
                assert orchestrator.rate_limiter is not None
                assert orchestrator.metrics is not None

                # Pools loaded for both databases
                assert set(orchestrator.pools) == {"dbmain", "db2"}

    @pytest.mark.asyncio
    async def test_per_database_policy_enforced(
        self, two_db_env, fake_pools: dict[str, MagicMock]
    ) -> None:
        """Test the db2 validator enforces its configured security policy."""
        with patch("pg_mcp.cache.schema_cache.SchemaIntrospector") as mock_cls:
            introspector = AsyncMock()
            introspector.introspect.side_effect = lambda: _make_schema("db")
            mock_cls.return_value = introspector

            async with server_module.lifespan(None):
                validator_db2 = server_module._orchestrator.sql_validators["db2"]
                assert validator_db2.allow_explain is True
                assert "secrets" in validator_db2.blocked_tables

                # Access to the blocked table on db2 must be rejected
                is_valid, error = validator_db2.validate("SELECT * FROM secrets")
                assert not is_valid
                assert "secrets" in error

                # dbmain keeps the default policy
                validator_main = server_module._orchestrator.sql_validators["dbmain"]
                assert validator_main.allow_explain is False
                assert validator_main.validate("SELECT * FROM secrets")[0] is True

    @pytest.mark.asyncio
    async def test_shutdown_closes_pools(
        self, two_db_env, fake_pools: dict[str, MagicMock]
    ) -> None:
        """Test pools are closed on shutdown."""
        with patch("pg_mcp.cache.schema_cache.SchemaIntrospector") as mock_cls:
            introspector = AsyncMock()
            introspector.introspect.side_effect = lambda: _make_schema("db")
            mock_cls.return_value = introspector

            async with server_module.lifespan(None):
                pass

        assert fake_pools is not None
        assert server_module.close_pools.called  # type: ignore[attr-defined]


class TestQueryToolValidation:
    """Test the MCP query tool's input validation and error handling."""

    @pytest.fixture
    def orchestrator_mock(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Attach a mock orchestrator to the server module."""
        mock = AsyncMock()
        monkeypatch.setattr(server_module, "_orchestrator", mock)
        return mock

    @pytest.mark.asyncio
    async def test_server_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test the tool reports SERVER_NOT_INITIALIZED when orchestrator is missing."""
        monkeypatch.setattr(server_module, "_orchestrator", None)
        result = await server_module.query(question="count users")
        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_invalid_return_type(self, orchestrator_mock: AsyncMock) -> None:
        """Test invalid return_type is rejected without calling the orchestrator."""
        result = await server_module.query(question="q", return_type="bogus")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_PARAMETER"
        orchestrator_mock.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_question_rejected(self, orchestrator_mock: AsyncMock) -> None:
        """Test empty question is rejected as INVALID_REQUEST."""
        result = await server_module.query(question="   ")
        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"
        orchestrator_mock.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_successful_query_returns_to_dict(
        self, orchestrator_mock: AsyncMock
    ) -> None:
        """Test the tool returns the response's to_dict shape with tokens_used."""
        orchestrator_mock.execute_query.return_value = QueryResponse(
            success=True,
            generated_sql="SELECT 1",
            confidence=100,
        )
        result = await server_module.query(question="q", return_type="sql")
        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1"
        assert result["tokens_used"] == 0
        orchestrator_mock.execute_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_orchestrator_exception_handled(
        self, orchestrator_mock: AsyncMock
    ) -> None:
        """Test unexpected orchestrator exceptions become INTERNAL_ERROR responses."""
        orchestrator_mock.execute_query.side_effect = RuntimeError("boom")
        result = await server_module.query(question="q")
        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["tokens_used"] == 0
