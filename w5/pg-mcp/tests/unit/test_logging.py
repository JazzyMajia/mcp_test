"""Tests for observability logging: formatters and sensitive data filtering."""

import json
import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture
def log_record() -> logging.LogRecord:
    """Create a basic log record."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )


class TestSensitiveDataFilter:
    """Test sensitive data redaction in log records."""

    @pytest.fixture
    def log_filter(self) -> SensitiveDataFilter:
        """Create the sensitive data filter."""
        return SensitiveDataFilter()

    def test_record_passes_through(self, log_filter: SensitiveDataFilter) -> None:
        """Test the filter always allows records through."""
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="p", lineno=1,
            msg="m", args=None, exc_info=None,
        )
        assert log_filter.filter(record) is True

    def test_redacts_sensitive_extra_fields(
        self, log_filter: SensitiveDataFilter
    ) -> None:
        """Test sensitive extra fields are redacted."""
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="p", lineno=1,
            msg="m", args=None, exc_info=None,
        )
        record.__dict__["password"] = "hunter2"
        record.__dict__["api_key"] = "sk-secret"
        log_filter.filter(record)
        assert record.__dict__["password"] == "***REDACTED***"
        assert record.__dict__["api_key"] == "***REDACTED***"

    def test_redacts_nested_dicts_in_extra(
        self, log_filter: SensitiveDataFilter
    ) -> None:
        """Test nested dicts inside extras are sanitized recursively."""
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="p", lineno=1,
            msg="m", args=None, exc_info=None,
        )
        record.__dict__["details"] = {"user": "bob", "token": "abc123"}
        log_filter.filter(record)
        assert record.__dict__["details"]["user"] == "bob"
        assert record.__dict__["details"]["token"] == "***REDACTED***"

    def test_sanitizes_args_dicts(self, log_filter: SensitiveDataFilter) -> None:
        """Test sensitive keys inside %-format args are redacted."""
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="p", lineno=1,
            msg="connect %s (%s)", args=({"host": "h", "password": "pw"}, "ctx"),
            exc_info=None,
        )
        log_filter.filter(record)
        assert record.args[0]["password"] == "***REDACTED***"
        assert record.args[0]["host"] == "h"
        assert record.args[1] == "ctx"

    def test_case_insensitive_key_matching(
        self, log_filter: SensitiveDataFilter
    ) -> None:
        """Test key matching ignores case (PASSWORD, Api_Key, ...)."""
        record = logging.LogRecord(
            name="t", level=logging.INFO, pathname="p", lineno=1,
            msg="m", args=None, exc_info=None,
        )
        record.__dict__["PASSWORD"] = "x"
        log_filter.filter(record)
        assert record.__dict__["PASSWORD"] == "***REDACTED***"

    def test_lists_and_tuples_in_args(self, log_filter: SensitiveDataFilter) -> None:
        """Test lists/tuples in args are sanitized element-wise."""
        sanitized = log_filter._sanitize_data([{"secret": "s"}, ("token", "keep")])
        assert sanitized[0]["secret"] == "***REDACTED***"
        assert sanitized[1] == ("token", "keep")


class TestJSONFormatter:
    """Test the JSON log formatter."""

    def test_formats_as_json(self, log_record: logging.LogRecord) -> None:
        """Test output is a valid JSON object with core fields."""
        formatted = JSONFormatter().format(log_record)
        data = json.loads(formatted)
        assert data["level"] == "INFO"
        assert data["message"] == "hello world"

    def test_includes_extra_fields(self, log_record: logging.LogRecord) -> None:
        """Test custom extras appear in the JSON output."""
        log_record.__dict__["request_id"] = "rid-1"
        data = json.loads(JSONFormatter().format(log_record))
        assert data["request_id"] == "rid-1"


class TestTextFormatter:
    """Test the text log formatter."""

    def test_formats_as_text(self, log_record: logging.LogRecord) -> None:
        """Test output is a human-readable line."""
        formatted = TextFormatter().format(log_record)
        assert "INFO" in formatted
        assert "hello world" in formatted

    def test_includes_request_id(self, log_record: logging.LogRecord) -> None:
        """Test request_id appears when present."""
        log_record.__dict__["request_id"] = "rid-2"
        formatted = TextFormatter().format(log_record)
        assert "rid-2" in formatted


class TestConfigureLogging:
    """Test logging configuration."""

    def teardown_method(self) -> None:
        """Restore root logger state after each test."""
        logging.disable(logging.NOTSET)

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a named logger."""
        logger = get_logger("configured_test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "configured_test_logger"

    def test_configure_json_format(self) -> None:
        """Test configuring logging with JSON format installs the JSON formatter."""
        configure_logging(level="WARNING", log_format="json")
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert any(
            isinstance(getattr(h, "formatter", None), JSONFormatter) for h in root.handlers
        )
