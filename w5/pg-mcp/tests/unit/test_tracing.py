"""Tests for observability tracing: request context propagation and helpers."""

import logging

import pytest

from pg_mcp.observability.tracing import (
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestId:
    """Test request ID helpers."""

    def test_generate_request_id_is_uuid(self) -> None:
        """Test generated IDs are valid UUID strings."""
        request_id = generate_request_id()
        assert len(request_id) == 36
        assert request_id.count("-") == 4

    def test_generate_request_id_unique(self) -> None:
        """Test generated IDs are unique."""
        assert generate_request_id() != generate_request_id()

    def test_default_request_id_is_none(self) -> None:
        """Test request ID is None outside of a request context."""
        assert get_request_id() is None

    def test_set_and_clear_request_id(self) -> None:
        """Test manually setting and clearing the request ID."""
        set_request_id("custom-id")
        assert get_request_id() == "custom-id"
        clear_request_id()
        assert get_request_id() is None


class TestRequestContext:
    """Test the request_context context manager."""

    @pytest.mark.asyncio
    async def test_generates_id_when_missing(self) -> None:
        """Test a new request ID is generated when none is provided."""
        async with request_context() as request_id:
            assert request_id is not None
            assert get_request_id() == request_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_uses_provided_id(self) -> None:
        """Test the provided request ID is used as-is."""
        async with request_context("fixed-id") as request_id:
            assert request_id == "fixed-id"
            assert get_request_id() == "fixed-id"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_nested_contexts_isolated(self) -> None:
        """Test nested contexts restore the outer request ID on exit."""
        async with request_context("outer") as outer_id:
            async with request_context("inner") as inner_id:
                assert inner_id == "inner"
                assert get_request_id() == "inner"
            assert get_request_id() == outer_id
        assert get_request_id() is None
        assert outer_id == "outer"

    @pytest.mark.asyncio
    async def test_context_propagates_to_async_tasks(self) -> None:
        """Test the request ID propagates into spawned async operations."""
        import asyncio

        observed: dict[str, str | None] = {}

        async def child() -> None:
            observed["request_id"] = get_request_id()

        async with request_context("parent-id"):
            await asyncio.create_task(child())

        assert observed["request_id"] == "parent-id"


class TestTraceDecorators:
    """Test the trace_async and trace_sync decorators."""

    @pytest.mark.asyncio
    async def test_trace_async_returns_result(self) -> None:
        """Test decorated async functions execute and return their result."""

        @trace_async(operation="op")
        async def work(value: int) -> int:
            return value * 2

        assert await work(21) == 42

    @pytest.mark.asyncio
    async def test_trace_async_without_context(self) -> None:
        """Test decorated async functions work outside a request context."""

        @trace_async()
        async def work() -> str:
            return "ok"

        assert await work() == "ok"

    def test_trace_sync_returns_result(self) -> None:
        """Test decorated sync functions execute and return their result."""

        @trace_sync(operation="op")
        def work(value: int) -> int:
            return value + 1

        assert work(1) == 2

    def test_trace_async_log_records_carry_request_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test log records emitted inside a traced call carry the request ID."""
        import asyncio

        @trace_async(operation="logged_op")
        async def work() -> None:
            logging.getLogger("trace_test").info("hello")

        set_request_id("trace-123")
        try:
            with caplog.at_level(logging.INFO):
                asyncio.run(work())
        finally:
            clear_request_id()

        records = [r for r in caplog.records if r.name == "trace_test"]
        assert records, "expected at least one log record"
        assert getattr(records[0], "request_id", None) == "trace-123"
        assert getattr(records[0], "operation", None) == "logged_op"


class TestTracingLogger:
    """Test the TracingLogger wrapper."""

    def test_adds_request_id_to_extra(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test the logger injects the current request ID into extras."""
        tlog = TracingLogger("tracing_logger_test")
        with caplog.at_level(logging.INFO):
            set_request_id("tlog-1")
            try:
                tlog.info("message with context")
            finally:
                clear_request_id()

        record = next(r for r in caplog.records if r.name == "tracing_logger_test")
        assert getattr(record, "request_id", None) == "tlog-1"

    def test_respects_existing_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test an explicitly provided request_id is not overwritten."""
        tlog = TracingLogger("tracing_logger_test2")
        with caplog.at_level(logging.INFO):
            set_request_id("ctx-id")
            try:
                tlog.info("msg", extra={"request_id": "explicit-id"})
            finally:
                clear_request_id()

        record = next(r for r in caplog.records if r.name == "tracing_logger_test2")
        assert getattr(record, "request_id", None) == "explicit-id"

    def test_exception_logs_traceback(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test .exception() records exc_info."""
        tlog = TracingLogger("tracing_logger_test3")
        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("boom")
            except ValueError:
                tlog.exception("caught an error")

        record = next(r for r in caplog.records if r.name == "tracing_logger_test3")
        assert record.exc_info is not None
