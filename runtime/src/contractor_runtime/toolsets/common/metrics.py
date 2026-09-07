"""Tool-call metrics interface shared by built-in toolsets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class ToolMetrics(Protocol):
    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        secrets: tuple[str, ...] = (),
        duration_ms: int | None = None,
    ) -> None: ...
