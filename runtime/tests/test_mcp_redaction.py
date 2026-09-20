"""MCP allocation headers stay private throughout admission and reporting."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from fakes.discovery import FakeDiscoveryFactory
from pydantic import ValidationError
from test_allocation import NOW, make_spec, resign_template
from test_otlp_adapter import telemetry_settings

from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    ToolsetRef,
    ToolsetSelection,
    encode_private,
)
from contractor_runtime.contracts.redaction import runtime_setting_values
from contractor_runtime.contracts.toolset_settings import MCPToolsetSettings
from contractor_runtime.factories import FactoryRegistry, StubADKWorkerRuntimeFactory
from contractor_runtime.state import RuntimeState
from contractor_runtime.worker.runtime import _summarizer_secrets
from contractor_runtime.workspace import LocalWorkdirFactory

SECRET = "MCP-allocation-header-private-canary"


def settings() -> MCPToolsetSettings:
    return MCPToolsetSettings(
        endpoint="https://knowledge.example/mcp",
        headers={"Authorization": SECRET},
        connect_timeout_seconds=10,
        request_timeout_seconds=60,
    )


def test_mcp_headers_private_wire_repr_and_validation_errors() -> None:
    value = settings()
    assert SECRET not in repr(value)
    assert SECRET.encode() in encode_private(value)
    restored = MCPToolsetSettings.model_validate_json(encode_private(value))
    assert restored.headers["Authorization"].get_secret_value() == SECRET
    raw = value.model_dump(mode="json", by_alias=True)
    raw["headers"]["Host"] = SECRET
    with pytest.raises(ValidationError) as failure:
        MCPToolsetSettings.model_validate(raw)
    assert SECRET not in str(failure.value)


@pytest.mark.parametrize("mode", ["valid", "missing", "unselected", "leaking_card"])
def test_mcp_allocation_admission_redaction_and_report_cleanup(tmp_path: Path, mode: str) -> None:
    payloads: list[bytes] = []

    async def collect(request: httpx.Request) -> httpx.Response:
        payloads.append(await request.aread())
        return httpx.Response(200)

    class WorkerFactory(StubADKWorkerRuntimeFactory):
        async def create(self, context):
            assert SECRET in runtime_setting_values(context.runtime_settings)
            assert SECRET in _summarizer_secrets(context, "")
            assert SECRET not in repr(context)
            span = context.adapter_handles.instrumentation.start_span(
                "contractor.worker.tool", attributes={"tool.name": SECRET}
            )
            span.end(outcome="succeeded")
            worker = await super().create(context)
            if mode == "leaking_card":
                worker._agent_card["description"] = SECRET
            return worker

    async def scenario() -> None:
        factory = FakeDiscoveryFactory()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": WorkerFactory()},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
            toolsets={"mcp@1": factory},
            runtime_adapters={"otlp-http@1": OTLPHTTPAdapterFactory(httpx.MockTransport(collect))},
        )
        capabilities = await discover_capabilities(registry)
        state = RuntimeState(instance_id="mcp-redaction")
        await state.mark_registered()
        service = AllocationService(
            state,
            registry,
            capabilities,
            a2a_base_url="https://runtime.example",
            now=lambda: NOW,
            force_exit=lambda _: None,
        )
        spec = make_spec()
        spec.agent_template.toolsets = [
            ToolsetSelection(ref=ToolsetRef(toolset_id="mcp", version="1"), tools=["search"])
        ]
        spec.runtime_settings.toolsets = {"mcp@1": settings()}
        spec.runtime_settings.telemetry = telemetry_settings()
        spec.resolved_runtime_config_provenance.runtime_adapters = ["otlp-http@1"]
        if mode == "missing":
            spec.runtime_settings.toolsets = {}
        elif mode == "unselected":
            spec.agent_template.toolsets = []
        resign_template(spec.agent_template)
        if mode != "valid":
            with pytest.raises(AllocationError) as failure:
                await service.prepare(spec)
            assert (
                failure.value.code
                == {
                    "missing": "toolset_configuration_required",
                    "unselected": "unselected_toolset_configuration",
                    "leaking_card": "unsafe_worker_handle",
                }[mode]
            )
            assert SECRET not in str(failure.value)
        else:
            prepared = await service.prepare(spec)
            assert SECRET.encode() not in encode_private(prepared)
            finalized = await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="mcp-finalize",
                    deadline=NOW + timedelta(seconds=10),
                )
            )
            assert SECRET.encode() not in encode_private(finalized)
            await service.release(
                ReleaseAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                )
            )
            assert not (await service.snapshot()).has_runtime_settings
        assert all(tool.close_calls == 1 for tool in factory.created.values())

    asyncio.run(scenario())
    assert SECRET.encode() not in b"".join(payloads)
    if mode == "valid":
        assert payloads
