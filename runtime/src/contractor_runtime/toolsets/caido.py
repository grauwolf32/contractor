"""Descriptor shell for the staged caido@1 Toolset implementation."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

CAIDO_TOOL_NAMES = frozenset(
    {
        "caido_automate_results",
        "caido_automate_run",
        "caido_history",
        "caido_replay",
        "caido_request_detail",
        "caido_scope",
        "caido_sitemap",
        "caido_workflow_findings",
        "caido_workflow_list",
        "caido_workflow_run",
    }
)


class CaidoToolsetFactory:
    """Advertise the closed contract without claiming unfinished tools usable."""

    ref = "caido@1"
    exported_tools = CAIDO_TOOL_NAMES
    infrastructure_channels = MappingProxyType(
        {tool: frozenset({"caido-graphql-client"}) for tool in sorted(exported_tools)}
    )

    async def probe(self) -> frozenset[str]:
        # V12-004/V12-005 enable exact read/action subsets as they land.
        return frozenset()

    async def create_selected(self, **_values: Any) -> dict[str, Any]:
        raise RuntimeError("Caido Toolset implementation is unavailable")
