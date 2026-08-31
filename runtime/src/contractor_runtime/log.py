"""Small JSON logging setup that never serializes process settings."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, str] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info is not None and record.exc_info[0] is not None:
            event["exceptionType"] = record.exc_info[0].__name__
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party INFO records include complete request/listener URLs. Runtime
    # Agent emits its own bounded lifecycle events, so keep dependency logs at
    # warning or above rather than publishing private endpoints incidentally.
    for dependency_logger in ("httpx", "httpcore", "uvicorn", "uvicorn.error"):
        logging.getLogger(dependency_logger).setLevel(logging.WARNING)
