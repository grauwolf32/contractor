from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import MappingProxyType

import pytest
from fakes.mcp import MCPFixture, remote_tool
from test_adk_runtime import runtime_settings
from test_mcp_session import SECRET, configuration, deadline

from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.mcp.errors import MCPError
from contractor_runtime.toolsets.mcp.factory import MCPToolsetFactory


async def construct_tools(server, selected=("search", "fetch")):
    settings = runtime_settings().model_copy(
        update={"toolsets": {"mcp@1": configuration(server.url)}}
    )
    tools = await MCPToolsetFactory().create_selected(selected=selected, runtime_settings=settings)
    assert not server.requests
    assert all(tool._owner._task is None for tool in tools.values())
    return tools


@asynccontextmanager
async def prepared_tools(server, selected=("search", "fetch")):
    tools = await construct_tools(server, selected)
    try:
        for tool in tools.values():
            await tool.prepare_allocation(deadline=deadline())
        yield tools
    finally:
        for tool in tools.values():
            await tool.close_before(deadline=deadline())


def test_factory_owns_exact_frozen_tools_before_any_network_or_model_call():
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            assert set(tools) == {"search", "fetch"}
            assert tools["search"]._owner is tools["fetch"]._owner
            first = tools["search"]._get_declaration()
            assert first.parameters_json_schema == server.pages[0][0]["inputSchema"]
            digest = tools["search"].schema_digest
            server.pages[0][0]["inputSchema"] = {"type": "object"}
            server.pages[0][0]["description"] = "mutated remote description"
            # GET delivers tools/list_changed; neither it nor a call refreshes.
            await tools["search"].run_async(args={"query": "hello"}, tool_context=None)
            assert tools["search"]._get_declaration() == first
            assert tools["fetch"].schema_digest == digest
            assert len([r for r in server.requests if r["rpc"] == "tools/list"]) == 2
            assert len(server.sessions) == 1
            assert SECRET not in repr(tools)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"type": "object", "$ref": "#/properties/a"},
        {"type": "object", "properties": {"a": {"type": "string", "pattern": ".*"}}},
        {"type": "object", "if": True},
        {"type": "object", "required": "bad"},
        {"type": "object", "additionalProperties": {"format": "uri"}},
    ],
)
def test_unsupported_frozen_schemas_fail_prepare_and_close_owner(schema):
    async def scenario():
        async with MCPFixture(pages=[[remote_tool("search", inputSchema=schema)]]) as server:
            tools = await construct_tools(server, ["search"])
            with pytest.raises(MCPError, match="unsupported_schema"):
                await tools["search"].prepare_allocation(deadline=deadline())
            await tools["search"].close()
            assert tools["search"]._owner.closed and not server.calls

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "args", [{}, {"query": 1}, {"query": None}, {"query": "ok", "extra": 1}, {"query": "x" * 65536}]
)
def test_arguments_rejected_locally_without_coercion(args):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            with pytest.raises(MCPError, match="invalid_arguments"):
                await tools["search"].run_async(args=args, tool_context=None)
            assert server.calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (
            {"content": [{"type": "image", "data": "YQ==", "mimeType": "image/png"}]},
            "unsupported_content",
        ),
        (
            {"content": [{"type": "audio", "data": "YQ==", "mimeType": "audio/wav"}]},
            "unsupported_content",
        ),
        (
            {
                "content": [
                    {"type": "resource_link", "name": "external", "uri": "https://external.example"}
                ]
            },
            "unsupported_content",
        ),
        (
            {"content": [{"type": "resource", "resource": {"uri": "file:///secret", "text": "x"}}]},
            "unsupported_content",
        ),
        ({"content": [{"type": "future_unknown"}]}, "unsupported_content"),
        ({"content": [{"type": "text", "text": 123}]}, "invalid_result"),
        ({"content": [], "structuredContent": []}, "invalid_result"),
        ({"content": [], "isError": "false"}, "invalid_result"),
        ({"content": [{"type": "text", "text": "x"}] * 65}, "result_too_large"),
        ({"content": [{"type": "text", "text": "x" * (256 * 1024)}]}, "result_too_large"),
        ({"content": [{"type": "text", "text": SECRET}]}, "private_content"),
        (
            {"content": [], "structuredContent": {"echo": "private-mcp-session-1"}},
            "private_content",
        ),
    ],
)
def test_result_contract_rejects_unsupported_invalid_oversized_and_private_content(result, error):
    async def scenario():
        async with MCPFixture() as server, prepared_tools(server) as tools:
            server.result = result
            with pytest.raises(MCPError, match=error) as caught:
                await tools["search"].run_async(args={"query": "safe"}, tool_context=None)
            assert SECRET not in str(caught.value)
            assert len(server.calls) == 1
            assert all(r["target"] == "/mcp" for r in server.requests)

    asyncio.run(scenario())


@pytest.mark.parametrize("is_error", [False, True])
def test_supported_envelope_and_output_schema(is_error):
    async def scenario():
        schema = {
            "type": "object",
            "properties": {"found": {"type": "boolean"}},
            "required": ["found"],
        }
        async with (
            MCPFixture(pages=[[remote_tool("search", outputSchema=schema)]]) as server,
            prepared_tools(server, ["search"]) as tools,
        ):
            server.result = {
                "content": [{"type": "text", "text": "https://inert.example/path"}],
                "structuredContent": {"found": True},
                "isError": is_error,
            }
            assert (
                await tools["search"].run_async(args={"query": "x"}, tool_context=None)
                == server.result
            )
            server.result["structuredContent"] = {"found": "true"}
            if is_error:
                assert (await tools["search"].run_async(args={"query": "x"}, tool_context=None))[
                    "isError"
                ] is True
            else:
                with pytest.raises(MCPError, match="invalid_result"):
                    await tools["search"].run_async(args={"query": "x"}, tool_context=None)
            assert len(server.calls) == 2

    asyncio.run(scenario())


def test_private_declarations_fail_before_worker_can_receive_them():
    async def scenario():
        async with MCPFixture(pages=[[remote_tool("search", description=SECRET)]]) as server:
            tools = await construct_tools(server, ["search"])
            with pytest.raises(MCPError, match="private_content"):
                await tools["search"].prepare_allocation(deadline=deadline())
            await tools["search"].close()
            assert SECRET not in repr(tools)

    asyncio.run(scenario())


def test_local_sdk_failure_omits_only_mcp_capability(tmp_path, monkeypatch):
    import contractor_runtime.toolsets.mcp.factory as factory

    async def scenario():
        registry = built_in_factories(tmp_path)
        snapshot = await discover_capabilities(registry)
        assert snapshot.has_toolset("mcp@1")
        assert json.loads(
            next(t for t in snapshot.wire_toolsets() if t.ref == "mcp@1").model_dump_json(
                by_alias=True
            )
        ) == {"ref": "mcp@1", "tools": [], "toolDiscovery": "allocation"}

        def unavailable():
            raise ModuleNotFoundError("controlled SDK missing")

        monkeypatch.setattr(factory, "_implementation", unavailable)
        without = await discover_capabilities(registry)
        assert not without.has_toolset("mcp@1")
        assert without.has_toolset("http-tools@1")
        assert without.runtimes == snapshot.runtimes

    asyncio.run(scenario())


def test_closed_schema_subset_supports_nested_boolean_nullable_and_array_constraints():
    async def scenario():
        schema = {
            "type": "object",
            "properties": {
                "nullable": {"type": ["string", "null"], "maxLength": 12},
                "values": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "free": True,
                "forbidden": False,
            },
            "required": ["nullable", "values"],
            "additionalProperties": False,
        }
        async with (
            MCPFixture(pages=[[remote_tool("search", inputSchema=schema)]]) as server,
            prepared_tools(server, ["search"]) as tools,
        ):
            server.result = {"content": []}
            result = await tools["search"].run_async(
                args={"nullable": None, "values": [1, 2], "free": [True, {}]}, tool_context=None
            )
            assert result == {"content": [], "isError": False}
            for args in (
                {"nullable": None, "values": [True]},
                {"nullable": "ok", "values": [1, 1]},
                {"nullable": "ok", "values": [1], "forbidden": 1},
            ):
                with pytest.raises(MCPError, match="invalid_arguments"):
                    await tools["search"].run_async(args=args, tool_context=None)
            assert len(server.calls) == 1

    asyncio.run(scenario())


def test_native_factory_receives_typed_configuration_through_common_allocation_plumbing(tmp_path):
    from google.adk.tools import FunctionTool
    from pydantic import BaseModel, Field
    from test_allocation import make_spec

    from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
    from contractor_runtime.allocation import AllocationService, WorkerState
    from contractor_runtime.contracts import RuntimeSettings, ToolsetRef, ToolsetSelection
    from contractor_runtime.factories import FactoryRegistry, StubADKWorkerRuntimeFactory
    from contractor_runtime.state import RuntimeState
    from contractor_runtime.worker.runtime import _adk_tool
    from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

    class NativeConfig(BaseModel):
        region: str
        limit: int = 5

    # Test-only code-owned schema; documents cannot install a decoder/factory.
    class NativeSettings(RuntimeSettings):
        toolsets: dict[str, NativeConfig] = Field(default_factory=dict, repr=False)

    class NativeTool:
        name = __name__ = "native_lookup"
        closed = False

        def __init__(self, config):
            self.config = config

        async def __call__(self, query: str) -> dict:
            return {"region": self.config.region, "limit": self.config.limit, "query": query}

        async def close(self):
            self.closed = True

    class NativeFactory:
        ref = "test-native@2"
        exported_tools = frozenset({"native_lookup"})
        infrastructure_channels = MappingProxyType({})
        configuration_required = True
        configuration_schema = "test-native@2"

        async def probe(self):
            return self.exported_tools

        async def create_selected(self, *, selected, runtime_settings, **kwargs):
            config = runtime_settings.toolsets[self.ref]
            assert isinstance(config, NativeConfig)
            return {name: NativeTool(config.model_copy(deep=True)) for name in selected}

    async def scenario():
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
            toolsets={"test-native@2": NativeFactory()},
        )
        service = AllocationService(
            RuntimeState(instance_id="native"),
            registry,
            await discover_capabilities(registry),
            a2a_base_url="https://runtime.example",
        )
        spec = make_spec()
        spec.agent_template.toolsets = [
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="test-native", version="2"), tools=["native_lookup"]
            )
        ]
        spec.runtime_settings = NativeSettings(
            **runtime_settings().model_dump(),
            toolsets={"test-native@2": NativeConfig(region="local")},
        )
        tools = {}
        await service._create_tools(
            spec,
            AllocationWorkspace(root=tmp_path, path=tmp_path),
            None,
            WorkerState(),
            EMPTY_ADAPTER_HANDLES,
            None,
            tools,
        )
        tool = tools["native_lookup"]
        assert isinstance(_adk_tool(tool), FunctionTool)
        assert await tool(query="hello") == {"region": "local", "limit": 5, "query": "hello"}
        assert "mcp@1" not in registry.toolsets
        await tool.close()
        assert tool.closed

    asyncio.run(scenario())
