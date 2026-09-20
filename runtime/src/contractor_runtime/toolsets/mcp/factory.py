"""Local factory metadata; optional SDK imports occur inside the readiness probe."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES, AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.contracts.redaction import runtime_setting_values
from contractor_runtime.contracts.toolset_settings import MCPToolsetSettings
from contractor_runtime.toolsets.discovery import (
    MCP_COMMON_CHANNELS,
    MCP_WORKER_RUNTIMES,
    ToolsetDiscovery,
    valid_allocation_tool_names,
)
from contractor_runtime.toolsets.mcp.errors import MCPError


def _implementation():
    from contractor_runtime.toolsets.mcp.session import MCPSessionOwner
    from contractor_runtime.toolsets.mcp.tools import MCPTool

    return MCPSessionOwner, MCPTool


class MCPToolsetFactory:
    ref = "mcp@1"
    tool_discovery = "allocation"
    exported_tools = frozenset()
    infrastructure_channels = MappingProxyType({})
    common_infrastructure_channels = MCP_COMMON_CHANNELS
    supported_worker_runtimes = MCP_WORKER_RUNTIMES
    configuration_required = True
    configuration_schema = "mcp-streamable-http@1"

    async def probe(self) -> ToolsetDiscovery:
        _implementation()
        return ToolsetDiscovery("allocation")

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        runtime_settings: RuntimeSettings,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        private_bypass_hosts: Sequence[str] = (),
        **_: Any,
    ) -> dict[str, Any]:
        if not valid_allocation_tool_names(selected):
            raise MCPError("invalid_selection")
        settings = runtime_settings.toolsets.get(self.ref)
        if not isinstance(settings, MCPToolsetSettings):
            raise MCPError("configuration_required")
        proxy = adapter_handles.tool_http
        if (proxy is not None and not isinstance(proxy, ProxyHTTPClient)) or (
            runtime_settings.http_proxy is not None
            and "tool-http" in runtime_settings.http_proxy.targets
            and proxy is None
        ):
            raise MCPError("invalid_transport")
        owner_class, tool_class = _implementation()
        owner = owner_class(
            settings,
            selected,
            proxy=proxy,
            forbidden_hosts=private_bypass_hosts,
            private_values=runtime_setting_values(runtime_settings),
        )
        # No await or network after construction begins: return every owner
        # reference to allocation cleanup before its optional prepare hook.
        return {
            name: tool_class(name, owner, timeout_seconds=runtime_settings.request_timeout_seconds)
            for name in selected
        }
