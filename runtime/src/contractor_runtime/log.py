"""Small JSON logging setup that never serializes process settings."""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime

# Structured ``extra={...}`` fields a Runtime log record may carry. Each one is
# a reviewed, non-secret identifier, outcome or counter: capability probes,
# registration identity and the verified Podman policy. Any other extra is
# dropped so dependency records, and an accidental secret-bearing attribute,
# never reach the log sink. Add a field here only after checking every caller.
LOG_EXTRA_FIELDS = frozenset(
    {
        "capabilityKind",
        "capabilityRef",
        "durationMs",
        "instanceId",
        "podmanBindDiskQuotaEnforced",
        "podmanCPUs",
        "podmanImageDigest",
        "podmanMemoryBytes",
        "podmanNetwork",
        "podmanPids",
        "podmanSwapMaxBytes",
        "podmanTmpfsBytes",
        "probeOutcome",
    }
)
MAX_LOG_EXTRA_TEXT = 256
_UNSAFE = object()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info is not None and record.exc_info[0] is not None:
            event["exceptionType"] = record.exc_info[0].__name__
        for name in sorted(LOG_EXTRA_FIELDS):
            if name in record.__dict__:
                value = _json_scalar(record.__dict__[name])
                if value is not _UNSAFE:
                    event[name] = value
        return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


def _json_scalar(value: object) -> object:
    """Keep only bounded JSON scalars; containers and objects are never rendered."""

    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else _UNSAFE
    if type(value) is str and len(value) <= MAX_LOG_EXTRA_TEXT:
        return value
    return _UNSAFE


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
