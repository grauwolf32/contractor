"""Checks shared by built-in toolset factories before they create tools."""

from __future__ import annotations

from collections.abc import Iterable, Set
from typing import Any

from contractor_runtime.toolsets.common.metrics import ToolMetrics


def require_selected_tools(
    selected: Iterable[str],
    available: Set[str],
    *,
    description: str = "unknown selected tools",
) -> None:
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"{description}: {', '.join(missing)}")


def require_metrics(state: Any, ref: str) -> ToolMetrics:
    metrics = getattr(state, "metrics", None)
    if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
        raise TypeError(f"{ref} requires State.metrics")
    return metrics
