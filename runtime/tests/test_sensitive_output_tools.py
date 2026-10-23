"""Sensitive-output tools are declared by toolset factories, not listed by the Worker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from fakes.spec import allocation_spec
from test_adapter_host import CapturingRuntimeFactory, CapturingToolset

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    built_in_factories,
    declared_sensitive_output_tools,
)
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings
from contractor_runtime.state import RuntimeState
from contractor_runtime.toolsets.caido.tools import CAIDO_TOOL_NAMES
from contractor_runtime.toolsets.http.tools import HTTPToolsetFactory
from contractor_runtime.toolsets.interactsh.tools import TOOL_NAMES as INTERACTSH_TOOL_NAMES
from contractor_runtime.workspace import LocalWorkdirFactory

# worker/runtime.py UNDECLARED_SENSITIVE_OUTPUT_TOOLS before the declaration
# moved to the toolset factories: every http-tools@1 and Caido tool.
PREVIOUS_UNDECLARED_SENSITIVE_OUTPUT_TOOLS = frozenset(
    {
        "http_request",
        "http_read_body",
        "http_history",
        "http_session_set",
        "http_session_get",
        "http_session_clear",
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


@pytest.mark.parametrize("with_execution", [False, True])
def test_production_registry_derives_exactly_the_previous_sensitive_output_tools(
    tmp_path: Path, with_execution: bool
) -> None:
    registry = built_in_factories(
        tmp_path,
        execution_lifecycle=PodmanLifecycle(PodmanSettings()) if with_execution else None,
    )
    assert ("code-execution@1" in registry.toolsets) is with_execution

    assert registry.sensitive_output_tools == (
        PREVIOUS_UNDECLARED_SENSITIVE_OUTPUT_TOOLS | INTERACTSH_TOOL_NAMES
    )
    # The previous definition, which also covered tools added to either toolset later.
    assert registry.sensitive_output_tools == frozenset(
        HTTPToolsetFactory.exported_tools | CAIDO_TOOL_NAMES | INTERACTSH_TOOL_NAMES
    )
    declaring = {
        ref: declared_sensitive_output_tools(factory)
        for ref, factory in registry.toolsets.items()
        if declared_sensitive_output_tools(factory)
    }
    assert declaring == {
        "http-tools@1": HTTPToolsetFactory.exported_tools,
        "caido@1": CAIDO_TOOL_NAMES,
        "interactsh@1": INTERACTSH_TOOL_NAMES,
    }


class DeclaringToolset:
    ref = "declaring@1"
    exported_tools = frozenset({"capture_probe", "plain_probe"})
    infrastructure_channels = MappingProxyType({})
    sensitive_output_tools = frozenset({"capture_probe"})

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(self, **_: Any) -> dict[str, Any]:
        raise AssertionError("an unselected toolset must not create tools")


def registry(tmp_path: Path, runtime: CapturingRuntimeFactory, *toolsets: Any) -> FactoryRegistry:
    return FactoryRegistry(
        worker_runtimes={"adk@1": runtime},
        toolsets={toolset.ref: toolset for toolset in toolsets},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path / "work")},
    )


def test_toolsets_without_a_declaration_declare_none(tmp_path: Path) -> None:
    factories = registry(tmp_path, CapturingRuntimeFactory(), CapturingToolset())
    assert not hasattr(CapturingToolset, "sensitive_output_tools")
    assert factories.sensitive_output_tools == frozenset()


@pytest.mark.parametrize(
    "declared",
    [
        frozenset({"not_exported"}),
        {"capture_probe"},
        ("capture_probe",),
        frozenset({1}),
    ],
)
def test_registry_rejects_invalid_sensitive_output_declarations(
    tmp_path: Path, declared: Any
) -> None:
    toolset = DeclaringToolset()
    toolset.sensitive_output_tools = declared
    with pytest.raises(ValueError, match="declares invalid sensitive-output tools"):
        registry(tmp_path, CapturingRuntimeFactory(), toolset)


def test_allocation_builds_the_worker_with_the_registered_declarations(tmp_path: Path) -> None:
    runtime = CapturingRuntimeFactory()

    async def scenario() -> None:
        state = RuntimeState(instance_id="sensitive-output-test")
        await state.mark_registered()
        factories = registry(tmp_path, runtime, CapturingToolset(), DeclaringToolset())
        capabilities = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={
                "run-artifacts@1": ["read_artifact"],
                "declaring@1": sorted(DeclaringToolset.exported_tools),
            },
            sandbox_profiles=["local-workdir@1"],
        )
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
            force_exit=lambda _: None,
        )
        spec = allocation_spec(tools=["read_artifact"])
        await service.prepare(spec)

        assert runtime.context is not None
        assert set(runtime.context.tools) == {"read_artifact"}
        # Every registered declaration applies by tool name, as the previous
        # Worker-side list did, whether or not its toolset is selected.
        assert runtime.context.sensitive_output_tools == frozenset({"capture_probe"})

        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-sensitive-output",
                deadline=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())
