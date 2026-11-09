"""Execution-local exact credential replacement for exported telemetry copies."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from typing import Any

from contractor_runtime.redaction import REDACTED


class TelemetryRedactor:
    def __init__(self, values: Iterable[str]) -> None:
        self._values = tuple(
            sorted({value for value in values if value}, key=lambda v: (-len(v), v))
        )

    def __repr__(self) -> str:
        return "TelemetryRedactor([REDACTED])"

    def text(self, value: str) -> str:
        # Literal matching keeps credentials out of process-global regex caches.
        # Keep only the next match per value, rather than all match positions.
        pending = [
            (index, -len(secret), secret)
            for secret in self._values
            if (index := value.find(secret)) >= 0
        ]
        heapq.heapify(pending)
        parts = []
        cursor = 0
        while pending:
            index, length, secret = heapq.heappop(pending)
            if index < cursor:
                index = value.find(secret, cursor)
            else:
                parts.extend((value[cursor:index], REDACTED))
                cursor = index + len(secret)
                index = value.find(secret, cursor)
            if index >= 0:
                heapq.heappush(pending, (index, length, secret))
        parts.append(value[cursor:])
        return "".join(parts)

    def value(self, value: Any, *, redact: bool = True) -> Any:
        """Copy JSON data with bounded traversal; keys are structure, not data."""
        remaining = 100000

        def visit(item: Any, depth: int = 0) -> Any:
            nonlocal remaining
            remaining -= 1
            if depth > 64 or remaining < 0:
                raise ValueError("telemetry content exceeds traversal bound")
            if isinstance(item, str):
                return self.text(item) if redact else item
            if isinstance(item, dict):
                return {key: visit(child, depth + 1) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                return [visit(child, depth + 1) for child in item]
            dump = getattr(item, "model_dump", None)
            if callable(dump):
                return visit(dump(mode="json", exclude_none=True, by_alias=True), depth + 1)
            return item

        return visit(value)
