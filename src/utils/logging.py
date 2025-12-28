"""Structured logging infrastructure for the style transfer pipeline."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable for request tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID."""
    return _request_id.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set a new request ID, generating one if not provided."""
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
    _request_id.set(request_id)
    return request_id


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """Human-readable log output for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        request_id = get_request_id()
        req_str = f"[{request_id}] " if request_id else ""

        color = self.COLORS.get(record.levelname, "")
        level = f"{color}{record.levelname:8}{self.RESET}"

        message = record.getMessage()

        # Add extra data if present
        extra_str = ""
        if hasattr(record, "extra_data") and record.extra_data:
            extra_items = [f"{k}={v}" for k, v in record.extra_data.items()]
            extra_str = f" | {', '.join(extra_items)}"

        return f"{level} {req_str}{record.name}: {message}{extra_str}"


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that supports structured extra data."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Extract extra_data from kwargs
        extra_data = kwargs.pop("extra_data", {})

        # Add extra_data to the record
        extra = kwargs.get("extra", {})
        extra["extra_data"] = extra_data
        kwargs["extra"] = extra

        return msg, kwargs

    def with_context(self, **context) -> "ContextLogger":
        """Create a new logger with additional default context."""
        new_extra = {**self.extra, **context}
        return ContextLogger(self.logger, new_extra)


def get_logger(name: str) -> ContextLogger:
    """Get a structured logger for the given name."""
    logger = logging.getLogger(name)
    return ContextLogger(logger, {})


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, use JSON format; otherwise human-readable
        log_file: Optional file path for logging output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if json_format:
        formatter = StructuredFormatter()
    else:
        formatter = HumanFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        # Always use JSON format for file logs
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


# Convenience function for logging LLM calls
def log_llm_call(
    logger: ContextLogger,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """Log an LLM API call with structured data."""
    extra = {
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "duration_ms": duration_ms,
        "success": success,
    }
    if error:
        extra["error"] = error

    if success:
        logger.info(
            f"LLM call completed: {input_tokens}+{output_tokens} tokens in {duration_ms}ms",
            extra_data=extra
        )
    else:
        logger.error(
            f"LLM call failed: {error}",
            extra_data=extra
        )
