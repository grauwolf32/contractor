"""Release-gate faults at the physical and allocation ownership boundaries."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_projectfs_local_direct import workspace
from test_projectfs_zip import REVISION, archive
from test_workspace_auto_export import MemoryArtifactClient, StoredArtifact
from test_workspace_process_e2e import direct_workspace_spec, make_service, read_only_model

from contractor_runtime.allocation import AllocationError
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
)
from contractor_runtime.projectfs import WorkspaceStorageError
from contractor_runtime.state import ProcessState


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "socket"])
def test_unsupported_external_types_cannot_use_stale_memory_or_escape(
    tmp_path: Path, kind: str
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            previous = await session.snapshot()
            target = root / "src/a.txt"
            outside = tmp_path / "outside"
            outside.write_bytes(b"outside-secret")
            target.unlink()
            endpoint = None
            try:
                if kind == "symlink":
                    target.symlink_to(outside)
                elif kind == "hardlink":
                    os.link(outside, target)
                elif kind == "fifo":
                    os.mkfifo(target)
                else:
                    endpoint = socket.socket(socket.AF_UNIX)
                    # Keep Unix socket paths below the platform length limit.
                    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        endpoint.bind(f"/proc/self/fd/{descriptor}/sock")
                    finally:
                        os.close(descriptor)
                    (root / "sock").rename(target)
                for operation in (
                    session.snapshot,
                    lambda: session.read_text("src/a.txt"),
                    lambda: session.write_text("src/a.txt", "unsafe"),
                    lambda: session.copy_path("src", "copied", recursive=True),
                    lambda: session.move_path("src", "moved"),
                    lambda: session.delete_path("src", recursive=True),
                ):
                    with pytest.raises(WorkspaceStorageError) as failure:
                        await asyncio.wait_for(operation(), timeout=1)
                    assert str(failure.value) == "workspace_type_conflict"
                assert outside.read_bytes() == b"outside-secret"
                assert previous.files[0].text == "source\r\n"
                assert not (root / "copied").exists() and not (root / "moved").exists()
            finally:
                if endpoint is not None:
                    endpoint.close()

    asyncio.run(scenario())


def test_leaf_replaced_after_stat_cannot_redirect_a_read_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            target = root / "src/a.txt"
            outside = tmp_path / "outside"
            outside.write_bytes(b"outside-secret")
            original = os.open
            swapped = False

            def replacing(path: object, flags: int, *args: object, **kwargs: object) -> int:
                nonlocal swapped
                if path == "a.txt" and not swapped:
                    swapped = True
                    target.unlink()
                    target.symlink_to(outside)
                return original(path, flags, *args, **kwargs)

            monkeypatch.setattr(os, "open", replacing)
            with pytest.raises(WorkspaceStorageError, match="workspace_type_conflict"):
                await session.write_text("src/a.txt", "must not escape")
            assert swapped and target.is_symlink()
            assert outside.read_bytes() == b"outside-secret"

    asyncio.run(scenario())


def test_disappearing_scan_is_not_returned_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            previous = await session.snapshot()
            assert session._local is not None and session._local._filesystem is not None
            fs = session._local._filesystem
            original = fs._read

            def disappearing(
                descriptor: int, path: str, deadline: float, **kwargs: object
            ) -> bytes:
                if path == "src/a.txt":
                    (root / path).unlink()
                return original(descriptor, path, deadline, **kwargs)

            monkeypatch.setattr(fs, "_read", disappearing)
            with pytest.raises(WorkspaceStorageError, match=r"^workspace_not_found$"):
                await session.snapshot()
            assert previous.files[0].text == "source\r\n"
            assert not (root / "src/a.txt").exists()
            monkeypatch.setattr(fs, "_read", original)
            current = await session.snapshot()
            assert current.files == () and current.digest != previous.digest

    asyncio.run(scenario())


def test_partial_copy_failure_fences_without_restoring_unrelated_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "src/z.txt").write_bytes(b"later")
            assert session._local is not None and session._local._filesystem is not None
            fs = session._local._filesystem
            original = fs._write

            def failing(
                descriptor: int, path: str, data: bytes, deadline: float, **kwargs: object
            ) -> None:
                original(descriptor, path, data, deadline, **kwargs)
                (root / "src/a.txt").write_bytes(b"external write after partial copy")
                raise OSError("unconfirmed physical failure /private/host")

            monkeypatch.setattr(fs, "_write", failing)
            with pytest.raises(WorkspaceStorageError, match=r"^workspace_unavailable$"):
                await session.copy_path("src", "copy", recursive=True)
            assert (root / "copy/a.txt").read_bytes() == b"source\r\n"
            assert not (root / "copy/z.txt").exists()
            assert (root / "src/a.txt").read_bytes() == b"external write after partial copy"
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await session.snapshot()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_release", [False, True])
def test_owned_mutation_blocks_allocation_release_and_slot_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_release: bool
) -> None:
    async def scenario() -> None:
        artifacts = MemoryArtifactClient()
        artifacts.bindings["source"] = StoredArtifact(
            REVISION, "application/zip", archive({"source.txt": b"before\n"})
        )
        state, service = await make_service(
            tmp_path, "local", artifacts, read_only_model("source.txt", "release"), "owned-io"
        )
        spec = direct_workspace_spec("owned-io")
        spec.runtime_settings.request_timeout_seconds = 1
        await service.prepare(spec)
        context = service._context
        assert context is not None and context.project_workspace is not None
        project = context.project_workspace
        assert project._local is not None and project._local._filesystem is not None
        fs = project._local._filesystem
        root = Path(project.storage.root)
        started, finish = threading.Event(), threading.Event()
        original_write = fs.write
        provider = service._factories.workspace_provider
        assert provider is not None
        original_cleanup = provider.cleanup
        cleanup_calls = 0

        def blocked(path: str, data: bytes, *, deadline: float) -> None:
            started.set()
            assert finish.wait(8)
            original_write(path, data, deadline=deadline)

        async def cleanup(storage: object) -> None:
            nonlocal cleanup_calls
            assert finish.is_set()
            cleanup_calls += 1
            await original_cleanup(storage)

        monkeypatch.setattr(fs, "write", blocked)
        monkeypatch.setattr(provider, "cleanup", cleanup)
        owner = asyncio.create_task(project.write_text("source.txt", "owned write"))
        releasing = retry = None
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="finalize-owned",
                    deadline=datetime.now(UTC) + timedelta(seconds=5),
                )
            )
            request = ReleaseAllocationRequest(
                apiVersion=API_VERSION, allocationId=spec.allocation_id
            )
            releasing = asyncio.create_task(service.release(request))
            while context.release_cleanup_task is None:
                await asyncio.sleep(0)
            heartbeat = await asyncio.wait_for(state.heartbeat(1, 0), timeout=0.2)
            assert heartbeat.allocation_id == spec.allocation_id
            if cancel_release:
                releasing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await releasing
            else:
                with pytest.raises(AllocationError, match="allocation cleanup failed"):
                    await releasing
            assert (await state.snapshot()).process_state is ProcessState.FENCED
            # The bounded release attempt may finish on its deadline. Physical
            # disposal still belongs to the original guard task until I/O settles.
            retained = project._local.guard._cleanup
            assert retained is not None and not retained.done()
            release_attempt = context.release_cleanup_task
            if cancel_release:
                assert release_attempt is not None and not release_attempt.done()
            assert root.exists() and cleanup_calls == 0
            with pytest.raises(AllocationError):
                await service.prepare(direct_workspace_spec("must-not-reuse"))
            retry = asyncio.create_task(service.release(request))
            await asyncio.sleep(0)
            assert project._local.guard._cleanup is retained
            if cancel_release:
                assert context.release_cleanup_task is release_attempt
            finish.set()
            await asyncio.wait_for(retry, timeout=2)
            await service.release(request)
            assert cleanup_calls == 1 and not root.exists()
            assert (await state.snapshot()).process_state is ProcessState.FENCED
            await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE
        finally:
            finish.set()
            await asyncio.gather(
                owner,
                *[task for task in (releasing, retry) if task is not None],
                return_exceptions=True,
            )

    asyncio.run(scenario())
