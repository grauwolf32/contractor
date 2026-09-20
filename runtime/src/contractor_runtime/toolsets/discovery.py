"""Local discovery facts for registered toolset contracts, without endpoint IO."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

# Exact contracts, not patterns/wildcards. A new allocation-discovered ref needs
# matching Server metadata and a code-owned Runtime factory.
ALLOCATION_DISCOVERY_REFS = frozenset({"mcp@1"})
MCP_COMMON_CHANNELS = frozenset({"runtime-http-client"})
MCP_WORKER_RUNTIMES = frozenset({"adk@1"})
ALLOCATION_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
MAXIMUM_ALLOCATION_TOOLS = 64


@dataclass(frozen=True, slots=True)
class ToolsetDiscovery:
    mode: Literal["static", "allocation"]
    tools: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.mode not in {"static", "allocation"} or not isinstance(self.tools, frozenset):
            raise ValueError("invalid toolset discovery result")
        if (self.mode == "allocation" and self.tools) or (self.mode == "static" and not self.tools):
            raise ValueError("tool inventory does not match discovery mode")


def valid_allocation_tool_names(names: Iterable[str]) -> bool:
    selected = tuple(names)
    return (
        1 <= len(selected) <= MAXIMUM_ALLOCATION_TOOLS
        and len(set(selected)) == len(selected)
        and all(isinstance(name, str) and ALLOCATION_TOOL_NAME.fullmatch(name) for name in selected)
    )
