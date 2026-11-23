"""Checks shared by built-in toolset factories before they create tools."""

from __future__ import annotations

from collections.abc import Iterable, Set

from contractor_runtime.core.tool_metrics import ToolMetrics
from contractor_runtime.toolsets.common.resources import ToolResources


def require_selected_tools(
    selected: Iterable[str],
    available: Set[str],
    *,
    description: str = "unknown selected tools",
) -> None:
    missing = sorted(set(selected) - available)
    if missing:
        raise ValueError(f"{description}: {', '.join(missing)}")


def require_metrics(resources: ToolResources, ref: str) -> ToolMetrics:
    return resources.metrics
