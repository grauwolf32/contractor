"""Deterministic allocation-boundary gate for workspace Workflow fixtures.

The PostgreSQL process suite owns Server scheduling.  This test deliberately
keeps the model and Artifact service in-process while exercising the same
AllocationSpec -> hydration -> ADK tool calls -> export -> finalize/release
boundary for both immutable Runtime workspace backends.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import json_result, scripted_model, tool_call
from fakes.spec import allocation_spec
from test_projectfs_zip import REVISION, archive, settings
from test_workspace_auto_export import MemoryArtifactClient, StoredArtifact

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationWorkspaceExportV2,
    AllocationWorkspaceSourceV2,
    AllocationWorkspaceSpecV2,
    AllocationWorkspaceStateV2,
    ArtifactRef,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    StageContentRequest,
    ToolsetRef,
    ToolsetSelection,
)
from contractor_runtime.digests import _agent_template_digest
from contractor_runtime.factories import built_in_factories
from contractor_runtime.projectfs import ManagedWorkspaceTree, decode_workspace_state
from contractor_runtime.state import ProcessState, RuntimeState


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_exact_source_edit_export_import_and_slot_reuse(tmp_path: Path, storage: str) -> None:
    async def scenario() -> None:
        artifacts = MemoryArtifactClient()
        artifacts.bindings["source"] = StoredArtifact(
            REVISION,
            "application/zip",
            archive({"source.txt": b"before\n", "nested/readme.txt": b"evidence\n"}),
        )
        first_model = edit_model("source.txt", "before", "after", "first")
        state, service = await make_service(
            tmp_path / storage, storage, artifacts, first_model, "runtime-first"
        )
        first_spec = workspace_spec("allocation-first", "stage-first")

        first = await invoke(service, first_spec)

        assert first.result is not None
        assert set(first.result.artifacts) == {"workspace_state", "workspace_diff"}
        state_ref = first.result.artifacts["workspace_state"]
        assert state_ref.revision == "revision-1"
        assert b"-before" in artifacts.binding("workspace_diff").data
        assert b"+after" in artifacts.binding("workspace_diff").data
        state_value = artifacts.binding("workspace_state")
        assert (
            decode_workspace_state(
                state_value.data,
                ManagedWorkspaceTree(
                    directories={"nested"},
                    text_files={"source.txt": "before\n", "nested/readme.txt": "evidence\n"},
                ),
                limits=service._factories.workspace_provider.capability.limits,  # type: ignore[union-attr]
            ).text_files["source.txt"]
            == "after\n"
        )
        await terminate_and_release(service, first_spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None

        # A later allocation consumes only the exact exported state revision.
        second_model = edit_model("source.txt", "after", "after again", "second")
        service._factories.worker_runtimes["adk@1"]._model_factory = (  # type: ignore[attr-defined,index]
            lambda _: second_model
        )
        second_spec = workspace_spec("allocation-second", "stage-second", state=state_ref)
        second = await invoke(service, second_spec)
        assert second.result is not None
        assert second.result.artifacts["workspace_state"].revision == "revision-2"
        assert b"after again" in artifacts.binding("workspace_diff").data
        await terminate_and_release(service, second_spec.allocation_id)

        # Reusing the same single slot without an imported state starts from the
        # source archive again; prior allocation data is not observable.
        third_model = edit_model("source.txt", "before", "clean reuse", "third")
        service._factories.worker_runtimes["adk@1"]._model_factory = (  # type: ignore[attr-defined,index]
            lambda _: third_model
        )
        third_spec = workspace_spec("allocation-third", "stage-third")
        third = await invoke(service, third_spec)
        assert third.result is not None
        assert b"clean reuse" in artifacts.binding("workspace_diff").data
        assert b"after again" not in artifacts.binding("workspace_diff").data
        await terminate_and_release(service, third_spec.allocation_id)

    asyncio.run(scenario())


def test_router_siblings_with_multiple_sources_are_isolated_until_exact_state_import(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        left_artifacts = multi_source_artifacts()
        right_artifacts = multi_source_artifacts()
        left_state, left = await make_service(
            tmp_path / "left",
            "memory",
            left_artifacts,
            edit_model("backend/app.py", "before", "left-only", "left"),
            "runtime-left",
        )
        right_state, right = await make_service(
            tmp_path / "right",
            "memory",
            right_artifacts,
            read_only_model("backend/app.py", "right"),
            "runtime-right",
        )
        left_spec = workspace_spec(
            "allocation-left", "stage-router", namespace="builder", multiple_sources=True
        )
        right_spec = workspace_spec(
            "allocation-right", "stage-router", namespace="reviewer", multiple_sources=True
        )
        await left.prepare(left_spec)
        await right.prepare(right_spec)
        assert left._context is not None and right._context is not None
        assert await left._context.project_workspace.read_text("backend/app.py") == "before\n"  # type: ignore[union-attr]
        assert await right._context.project_workspace.read_text("backend/app.py") == "before\n"  # type: ignore[union-attr]

        left_result = await left._context.worker.invoke(stage_request())  # type: ignore[union-attr]
        assert left_result.result is not None
        assert await left._context.project_workspace.read_text("backend/app.py") == "left-only\n"  # type: ignore[union-attr]
        assert await right._context.project_workspace.read_text("backend/app.py") == "before\n"  # type: ignore[union-attr]

        right_result = await right._context.worker.invoke(stage_request())  # type: ignore[union-attr]
        assert right_result.result is not None
        assert right_result.result.artifacts["workspace_state"].revision is not None
        await terminate_and_release(left, left_spec.allocation_id)
        await terminate_and_release(right, right_spec.allocation_id)
        assert (await left_state.snapshot()).process_state is ProcessState.IDLE
        assert (await right_state.snapshot()).process_state is ProcessState.IDLE

        # A fresh later Stage sees the left edit only after selecting that exact
        # exported state, never because Router siblings shared a filesystem.
        imported_model = edit_model("backend/app.py", "left-only", "explicit-import", "imported")
        _, imported = await make_service(
            tmp_path / "imported", "local", left_artifacts, imported_model, "runtime-imported"
        )
        imported_spec = workspace_spec(
            "allocation-imported",
            "stage-later",
            state=left_result.result.artifacts["workspace_state"],
            multiple_sources=True,
        )
        imported_result = await invoke(imported, imported_spec)
        assert imported_result.result is not None
        await terminate_and_release(imported, imported_spec.allocation_id)

    asyncio.run(scenario())


def multi_source_artifacts() -> MemoryArtifactClient:
    artifacts = MemoryArtifactClient()
    artifacts.bindings["backend"] = StoredArtifact(
        REVISION, "application/zip", archive({"app.py": b"before\n"})
    )
    artifacts.bindings["frontend"] = StoredArtifact(
        REVISION, "application/zip", archive({"ui.ts": b"unchanged\n"})
    )
    return artifacts


async def make_service(
    root: Path,
    storage: str,
    artifacts: MemoryArtifactClient,
    model: object,
    instance_id: str,
) -> tuple[RuntimeState, AllocationService]:
    workspace_settings = settings(
        storage,
        root / "project-workspaces" if storage == "local" else None,
    )
    factories = built_in_factories(
        root / "scratch",
        artifact_client_factory=lambda *_: artifacts,  # type: ignore[arg-type,return-value]
        model_factory=lambda _: model,  # type: ignore[arg-type,return-value]
        workspace_settings=workspace_settings,
    )
    provider = factories.workspace_provider
    assert provider is not None
    capabilities = CapabilitySnapshot.create(
        runtimes=factories.worker_runtimes,
        toolsets={ref: factory.exported_tools for ref, factory in factories.toolsets.items()},
        sandbox_profiles=factories.sandbox_profiles,
        workspace=provider.capability,
    )
    state = RuntimeState(instance_id=instance_id)
    await state.mark_registered()
    return state, AllocationService(
        state,
        factories,
        capabilities,
        a2a_base_url="https://runtime.example",
    )


def workspace_spec(
    allocation_id: str,
    stage_execution_id: str,
    *,
    state: ArtifactRef | None = None,
    namespace: str = "editor",
    multiple_sources: bool = False,
) -> Any:
    spec = allocation_spec(allocation_id=allocation_id)
    spec.stage_execution_id = stage_execution_id
    spec.logical_agent_name = namespace
    spec.namespace = namespace
    spec.agent_template.ref.template_id = "workspace_editor"
    spec.agent_template.description = "Edits one private project workspace"
    spec.agent_template.toolsets = [
        ToolsetSelection(ref=ToolsetRef(toolsetId="filesystem", version="1"), tools=["read_file"]),
        ToolsetSelection(ref=ToolsetRef(toolsetId="edit-files", version="1"), tools=["edit"]),
        ToolsetSelection(
            ref=ToolsetRef(toolsetId="workspace-changes", version="1"),
            tools=["changed_paths", "diff"],
        ),
    ]
    spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
    sources = (
        [
            AllocationWorkspaceSourceV2(
                artifact=ArtifactRef(namespace="inputs", name="backend", revision=REVISION),
                target="backend",
            ),
            AllocationWorkspaceSourceV2(
                artifact=ArtifactRef(namespace="inputs", name="frontend", revision=REVISION),
                target="frontend",
            ),
        ]
        if multiple_sources
        else [
            AllocationWorkspaceSourceV2(
                artifact=ArtifactRef(namespace="inputs", name="source", revision=REVISION),
                target="",
            )
        ]
    )
    spec.workspace = AllocationWorkspaceSpecV2(
        mode="overlay",
        sources=sources,
        state=AllocationWorkspaceStateV2(artifact=state) if state is not None else None,
        export=AllocationWorkspaceExportV2(state="workspace_state", diff="workspace_diff"),
    )
    return spec


def edit_model(path: str, old: str, new: str, prefix: str) -> object:
    return scripted_model(
        [
            tool_call(
                "read_file",
                {"path": path, "start_line": 1, "max_lines": 20},
                call_id=f"{prefix}-read",
            ),
            tool_call(
                "edit",
                {"path": path, "old": old, "new": new, "replace_all": False},
                call_id=f"{prefix}-edit",
            ),
            tool_call(
                "changed_paths",
                {"cursor": "", "limit": 100},
                call_id=f"{prefix}-changes",
            ),
            tool_call(
                "diff",
                {"path": path, "cursor": "", "max_bytes": 65536},
                call_id=f"{prefix}-diff",
            ),
            json_result(success_result()),
        ]
    )


def read_only_model(path: str, prefix: str) -> object:
    return scripted_model(
        [
            tool_call(
                "read_file",
                {"path": path, "start_line": 1, "max_lines": 20},
                call_id=f"{prefix}-read",
            ),
            json_result(success_result()),
        ]
    )


def success_result() -> dict[str, object]:
    return {
        "subtaskId": "0",
        "result": "Workspace operation complete",
    }


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        subtaskId="0",
        objective="Exercise the private workspace",
        instructions="Use only selected tools and return a strict result.",
        parameters={},
        artifacts={},
    )


async def invoke(service: AllocationService, spec: Any) -> Any:
    await service.prepare(spec)
    assert service._context is not None and service._context.worker is not None
    return await service._context.worker.invoke(stage_request())


async def terminate_and_release(service: AllocationService, allocation_id: str) -> None:
    await service.finalize(
        FinalizeAllocationRequest(
            apiVersion=API_VERSION,
            allocationId=allocation_id,
            finalizationId=f"finalize-{allocation_id}",
            deadline=datetime.now(UTC) + timedelta(seconds=3),
        )
    )
    await service.release(
        ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=allocation_id)
    )
    await service.confirm_release(allocation_id)
