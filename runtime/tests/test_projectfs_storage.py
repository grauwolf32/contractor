from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fakes.spec import allocation_spec
from test_projectfs_zip import archive, workspace_inputs

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TerminationError,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
)
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    WorkspaceStorageError,
)
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.state import RuntimeState
from contractor_runtime.workspace import LocalWorkdirFactory


def test_snapshots_are_sorted_deterministic_and_hide_backend_details(tmp_path: Path) -> None:
    async def scenario() -> None:
        provider = LocalWorkspaceProvider(local_settings(tmp_path / "work"))
        storage = await provider.create("allocation")
        root = Path(storage.root) / "run_workdir"
        (root / "a").mkdir(parents=True)
        (root / "z").mkdir()
        (root / "a/a.txt").write_bytes(b"one")
        (root / "z/b.txt").write_bytes(b"two")
        (root / "z/image.bin").write_bytes(b"\x00image")
        session = DirectWorkspaceSession(
            mode="direct",
            storage=storage,
            content_root=f"{storage.root}/run_workdir",
            limits=limits(),
            directories={"z", "a"},
            text_files={"z/b.txt": "two", "a/a.txt": "one"},
            binary_paths={"z/image.bin"},
            stored_binary_paths={"z/image.bin"},
        )

        first = await session.snapshot()
        second = await session.snapshot()
        assert first == second
        assert first.directories == ("a", "z")
        assert [item.path for item in first.files] == ["a/a.txt", "z/b.txt"]
        assert first.binary_paths == ("z/image.bin",)
        assert first.digest.startswith("sha256:")
        assert str(tmp_path) not in repr(first)
        assert await session.read_text("a/a.txt") == "one"
        with pytest.raises(WorkspaceStorageError, match="binary"):
            await session.read_text("z/image.bin")
        with pytest.raises(WorkspaceStorageError, match="not_found"):
            await session.read_text("missing.txt")

        await session.close()
        with pytest.raises(WorkspaceStorageError):
            await session.snapshot()
        await provider.cleanup(storage)

    asyncio.run(scenario())


def test_allocation_prepare_hydrates_before_ready_and_release_erases_project_tree(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source_spec, reader = workspace_inputs(
            [("source", "", archive({"src/main.py": b"print('ready')\n"}))]
        )
        provider = LocalWorkspaceProvider(local_settings(tmp_path / "project"))
        sandbox = LocalWorkdirFactory(tmp_path / "sandbox")
        runtime = StubADKWorkerRuntimeFactory()
        artifact_tools = RunArtifactsToolsetFactory(lambda *_: reader)  # type: ignore[arg-type]
        factories = FactoryRegistry(
            worker_runtimes={runtime.ref: runtime},
            toolsets={artifact_tools.ref: artifact_tools},
            sandbox_profiles={sandbox.ref: sandbox},
            workspace_provider=provider,
            artifact_client_factory=lambda *_: reader,  # type: ignore[arg-type]
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=factories.worker_runtimes,
            toolsets={artifact_tools.ref: artifact_tools.exported_tools},
            sandbox_profiles=factories.sandbox_profiles,
            workspace=provider.capability,
        )
        state = RuntimeState(instance_id="workspace-runtime")
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
        )
        spec = allocation_spec(tools=["read_artifact"])
        spec.workspace = source_spec

        await service.prepare(spec)
        snapshot = await service.snapshot()
        assert snapshot is not None and snapshot.has_project_workspace
        assert service._context is not None
        project_path = Path(service._context.project_workspace.storage.root)  # type: ignore[union-attr]
        assert (project_path / "run_workdir/src/main.py").read_text() == "print('ready')\n"

        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalize-workspace",
                deadline=spec.lease_expires_at,
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        assert not project_path.exists()
        await service.confirm_release(spec.allocation_id)
        assert await service.snapshot() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("termination", ["abort", "lease"])
def test_abort_and_lease_loss_erase_project_tree_before_release(
    tmp_path: Path, termination: str
) -> None:
    async def scenario() -> None:
        source_spec, reader = workspace_inputs(
            [("source", "", archive({"src/main.py": b"private source\n"}))]
        )
        source_spec.mode = "overlay"
        provider = LocalWorkspaceProvider(local_settings(tmp_path / "project"))
        sandbox = LocalWorkdirFactory(tmp_path / "sandbox")
        runtime = StubADKWorkerRuntimeFactory()
        artifact_tools = RunArtifactsToolsetFactory(lambda *_: reader)  # type: ignore[arg-type]
        factories = FactoryRegistry(
            worker_runtimes={runtime.ref: runtime},
            toolsets={artifact_tools.ref: artifact_tools},
            sandbox_profiles={sandbox.ref: sandbox},
            workspace_provider=provider,
            artifact_client_factory=lambda *_: reader,  # type: ignore[arg-type]
        )
        capabilities = CapabilitySnapshot.create(
            runtimes=factories.worker_runtimes,
            toolsets={artifact_tools.ref: artifact_tools.exported_tools},
            sandbox_profiles=factories.sandbox_profiles,
            workspace=provider.capability,
        )
        state = RuntimeState(instance_id="workspace-runtime")
        await state.mark_registered()
        service = AllocationService(
            state,
            factories,
            capabilities,
            a2a_base_url="https://runtime.example",
        )
        spec = allocation_spec(tools=["read_artifact"])
        spec.workspace = source_spec
        await service.prepare(spec)
        assert service._context is not None
        assert service._context.project_workspace is not None
        project_path = Path(service._context.project_workspace.storage.root)

        if termination == "abort":
            await service.abort(
                AbortAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    abortId="abort-workspace",
                    reason=TerminationError(
                        code="run_cancelled",
                        message="WorkflowRun was cancelled",
                        retryable=False,
                    ),
                    deadline=spec.lease_expires_at,
                )
            )
        else:
            await service.expire_control_lease(1)

        assert not project_path.exists()
        snapshot = await service.snapshot()
        assert snapshot is not None and not snapshot.has_project_workspace
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("storage_kind", ["local", "memory"])
def test_private_storage_isolated_for_simultaneous_and_later_allocations(
    tmp_path: Path, storage_kind: str
) -> None:
    async def scenario() -> None:
        provider = (
            LocalWorkspaceProvider(local_settings(tmp_path / "work"))
            if storage_kind == "local"
            else MemoryWorkspaceProvider(memory_settings())
        )
        left = await provider.create("left")
        right = await provider.create("right")
        left_path = f"{left.root}/run_workdir/value.txt"
        right_path = f"{right.root}/run_workdir/value.txt"
        left.filesystem.makedirs(f"{left.root}/run_workdir")
        right.filesystem.makedirs(f"{right.root}/run_workdir")
        left.filesystem.pipe(left_path, b"changed")
        right.filesystem.pipe(right_path, b"initial")
        assert left.filesystem.cat(left_path) == b"changed"
        assert right.filesystem.cat(right_path) == b"initial"

        await provider.cleanup(left)
        later = await provider.create("later")
        assert not later.filesystem.exists(f"{later.root}/run_workdir/value.txt")
        assert right.filesystem.cat(right_path) == b"initial"
        await provider.cleanup(right)
        await provider.cleanup(later)

    asyncio.run(scenario())


def local_settings(root: Path) -> WorkspaceSettings:
    return WorkspaceSettings(storage="local", work_root=root, limits=limits())


def memory_settings() -> WorkspaceSettings:
    return WorkspaceSettings(storage="memory", limits=limits())


def limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=4096,
        max_managed_text_bytes=2048,
        max_file_bytes=1024,
    )
