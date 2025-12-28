"""Unit tests for structured logging utilities."""

import json
import logging
import pytest
from io import StringIO

from src.utils.logging import (
    get_logger,
    get_request_id,
    set_request_id,
    setup_logging,
    log_llm_call,
    StructuredFormatter,
    HumanFormatter,
    ContextLogger,
)


class TestRequestId:
    """Test request ID management."""

    def test_set_and_get_request_id(self):
        """Test setting and getting request ID."""
        set_request_id("test-123")
        assert get_request_id() == "test-123"

    def test_generate_request_id(self):
        """Test auto-generating request ID."""
        generated = set_request_id()
        assert generated is not None
        assert len(generated) == 8

    def test_none_request_id_default(self):
        """Test that request ID can be cleared."""
        set_request_id(None)
        generated = set_request_id()
        # A new ID should be generated
        assert get_request_id() == generated


class TestStructuredFormatter:
    """Test JSON log formatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_includes_request_id(self):
        """Test that request ID is included if set."""
        set_request_id("req-456")
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data.get("request_id") == "req-456"

    def test_includes_extra_data(self):
        """Test that extra_data is included."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.extra_data = {"key1": "value1", "key2": 42}

        output = formatter.format(record)
        data = json.loads(output)

        assert data["key1"] == "value1"
        assert data["key2"] == 42

    def test_includes_exception(self):
        """Test that exception info is included."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestHumanFormatter:
    """Test human-readable log formatter."""

    def test_basic_format(self):
        """Test basic human formatting."""
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_includes_extra_data(self):
        """Test that extra_data is appended."""
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.extra_data = {"key": "value"}

        output = formatter.format(record)

        assert "key=value" in output


class TestContextLogger:
    """Test context-aware logger."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("test.module")
        assert isinstance(logger, ContextLogger)

    def test_with_context(self):
        """Test creating logger with additional context."""
        logger = get_logger("test")
        child = logger.with_context(component="validator")

        assert child.extra.get("component") == "validator"

    def test_logging_with_extra_data(self):
        """Test logging with extra_data parameter."""
        # Set up a handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        base_logger = logging.getLogger("test_extra_data")
        base_logger.handlers.clear()
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = ContextLogger(base_logger, {})
        logger.info("Test message", extra_data={"custom": "value"})

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["custom"] == "value"


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_info_level(self):
        """Test setting up INFO level logging."""
        setup_logging(level="INFO", json_format=False)
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_setup_debug_level(self):
        """Test setting up DEBUG level logging."""
        setup_logging(level="DEBUG", json_format=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_json_format(self):
        """Test setting up JSON format."""
        setup_logging(level="INFO", json_format=True)
        root = logging.getLogger()
        # Check that handler has JSON formatter
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, StructuredFormatter)

    def test_setup_human_format(self):
        """Test setting up human-readable format."""
        setup_logging(level="INFO", json_format=False)
        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, HumanFormatter)


class TestLogLLMCall:
    """Test LLM call logging helper."""

    def test_log_successful_call(self):
        """Test logging a successful LLM call."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        base_logger = logging.getLogger("test_llm_success")
        base_logger.handlers.clear()
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = ContextLogger(base_logger, {})
        log_llm_call(
            logger=logger,
            provider="deepseek",
            model="deepseek-chat",
            input_tokens=100,
            output_tokens=50,
            duration_ms=1500,
            success=True
        )

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["provider"] == "deepseek"
        assert data["model"] == "deepseek-chat"
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["duration_ms"] == 1500
        assert data["success"] is True

    def test_log_failed_call(self):
        """Test logging a failed LLM call."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        base_logger = logging.getLogger("test_llm_fail")
        base_logger.handlers.clear()
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        logger = ContextLogger(base_logger, {})
        log_llm_call(
            logger=logger,
            provider="ollama",
            model="llama3",
            input_tokens=100,
            output_tokens=0,
            duration_ms=500,
            success=False,
            error="Connection timeout"
        )

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["success"] is False
        assert data["error"] == "Connection timeout"
        assert data["level"] == "ERROR"
