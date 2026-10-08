"""Tests for the Prometheus metrics collector."""

import pytest

from pg_mcp.observability.metrics import MetricsCollector, metrics


class TestMetricsCollector:
    """Test metrics collector counters, histograms, and gauges."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Get the singleton metrics collector."""
        return MetricsCollector()

    def test_singleton(self, collector: MetricsCollector) -> None:
        """Test MetricsCollector returns the same instance."""
        assert MetricsCollector() is collector
        assert metrics is collector

    def test_increment_query_request(self, collector: MetricsCollector) -> None:
        """Test query request counter increments per status/database labels."""
        before = (
            collector.query_requests.labels(status="success", database="metricsdb")
            ._value.get()
        )
        collector.increment_query_request(status="success", database="metricsdb")
        after = (
            collector.query_requests.labels(status="success", database="metricsdb")
            ._value.get()
        )
        assert after == before + 1

    def test_increment_llm_call_and_tokens(self, collector: MetricsCollector) -> None:
        """Test LLM call and token counters increment."""
        op = "generate_sql"
        before_calls = collector.llm_calls.labels(operation=op)._value.get()
        collector.increment_llm_call(op)
        assert collector.llm_calls.labels(operation=op)._value.get() == before_calls + 1

        before_tokens = collector.llm_tokens_used.labels(operation=op)._value.get()
        collector.increment_llm_tokens(op, 42)
        assert collector.llm_tokens_used.labels(operation=op)._value.get() == before_tokens + 42

    def test_observe_llm_latency(self, collector: MetricsCollector) -> None:
        """Test LLM latency observations are recorded."""
        op = "validate_result"
        before = collector.llm_latency.labels(operation=op)._sum.get()
        collector.observe_llm_latency(op, 0.25)
        assert collector.llm_latency.labels(operation=op)._sum.get() == before + 0.25

    def test_increment_sql_rejected(self, collector: MetricsCollector) -> None:
        """Test SQL rejection counter increments per reason."""
        before = collector.sql_rejected.labels(reason="security_violation")._value.get()
        collector.increment_sql_rejected("security_violation")
        after = collector.sql_rejected.labels(reason="security_violation")._value.get()
        assert after == before + 1

    def test_db_query_duration_and_gauges(self, collector: MetricsCollector) -> None:
        """Test db duration histogram and gauge setters."""
        before = collector.db_query_duration._sum.get()
        collector.observe_db_query_duration(0.5)
        assert collector.db_query_duration._sum.get() == before + 0.5

        collector.set_db_connections_active("metricsdb", 7)
        assert (
            collector.db_connections_active.labels(database="metricsdb")._value.get() == 7
        )

        collector.set_schema_cache_age("metricsdb", 120.0)
        assert (
            collector.schema_cache_age.labels(database="metricsdb")._value.get() == 120.0
        )
