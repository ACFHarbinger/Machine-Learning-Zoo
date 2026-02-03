"""
Structured JSON logging utilities for ELK stack integration.

Provides handlers and formatters to output logs in JSON format,
optionally sending them via TCP to Logstash.
"""

import datetime
import json
import logging
import socket
from pathlib import Path
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def format(self, record):
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class LogstashTcpHandler(logging.Handler):
    """Handler that sends JSON logs over TCP to Logstash."""

    def __init__(self, host: str = "localhost", port: int = 5000):
        """Initialize the TCP handler for Logstash.

        Args:
            host: Logstash server hostname.
            port: Logstash server port.
        """
        super().__init__()
        self.host = host
        self.port = port
        self.formatter = JsonFormatter()

    def emit(self, record):
        """Emit a log record by sending it to Logstash over TCP."""
        try:
            msg = self.format(record) + "\n"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((self.host, self.port))
                s.sendall(msg.encode("utf-8"))
        except Exception:
            # Silent failure to avoid breaking the application
            pass


def get_structured_logger(
    name: str = "wsmart.structured",
    level: int = logging.INFO,
    logstash_host: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger configured for structured JSON output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler for local debugging
        console = logging.StreamHandler()
        console.setFormatter(JsonFormatter())
        logger.addHandler(console)

        # File handler if path provided
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(JsonFormatter())
            logger.addHandler(file_handler)

        # Logstash handler if host provided
        if logstash_host:
            ls_handler = LogstashTcpHandler(host=logstash_host)
            logger.addHandler(ls_handler)

    return logger


def log_test_metric(name: str, value: Any, logger_name: str = "wsmart.structured"):
    """Convenience function to log a test metric."""
    logger = logging.getLogger(logger_name)
    extra = {"type": "test_metric", "metric_name": name, "metric_value": value}
    logger.info(f"Metric: {name}={value}", extra={"extra_fields": extra})


def log_benchmark_metric(
    benchmark: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    logger_name: str = "wsmart.benchmark",
):
    """
    Convenience function to log benchmark metrics.
    """

    # Ensure logger is initialized with a file if not already
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        get_structured_logger(name=logger_name, log_file="logs/benchmarks/benchmarks.jsonl")

    extra = {
        "benchmark": benchmark,
        "metrics": metrics,
        "metadata": metadata or {},
        "type": "performance_benchmark",
    }
    logger.info(f"Benchmark: {benchmark}", extra={"extra_fields": extra})
