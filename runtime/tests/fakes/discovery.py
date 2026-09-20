"""A local allocation-discovery factory; deliberately performs no network IO."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from contractor_runtime.toolsets.discovery import (
    MCP_COMMON_CHANNELS,
    MCP_WORKER_RUNTIMES,
    ToolsetDiscovery,
)


class DiscoveredTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def declared_toolsets(factories):
    """Full fixture inventory for tests deliberately bypassing startup probes."""
    return {
        ref: ToolsetDiscovery("allocation")
        if getattr(factory, "tool_discovery", "static") == "allocation"
        else factory.exported_tools
        for ref, factory in factories.toolsets.items()
    }


class FakeDiscoveryFactory:
    ref = "mcp@1"
    tool_discovery = "allocation"
    exported_tools = frozenset()
    infrastructure_channels = MappingProxyType({})
    common_infrastructure_channels = MCP_COMMON_CHANNELS
    supported_worker_runtimes = MCP_WORKER_RUNTIMES
    configuration_required = True
    configuration_schema = "mcp-streamable-http@1"

    def __init__(
        self, *, available: bool = True, output: list[tuple[str, str]] | None = None
    ) -> None:
        self.available = available
        self.output = output
        self.probes = 0
        self.created: dict[str, DiscoveredTool] = {}

    async def probe(self) -> ToolsetDiscovery:
        self.probes += 1
        if not self.available:
            raise ModuleNotFoundError("local SDK unavailable")
        return ToolsetDiscovery("allocation")

    async def create_selected(
        self, *, selected: Sequence[str], **_: object
    ) -> dict[str, DiscoveredTool]:
        self.created = {
            key: DiscoveredTool(name)
            for key, name in (
                self.output if self.output is not None else [(name, name) for name in selected]
            )
        }
        return self.created
