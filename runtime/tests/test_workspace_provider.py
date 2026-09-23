from __future__ import annotations

import asyncio
import stat
import threading
import uuid
from pathlib import Path

import pytest

import contractor_runtime.projectfs.provider as provider_module
import contractor_runtime.workspace as workspace_module
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    ProjectWorkspaceStorage,
)
from contractor_runtime.projectfs.provider import LOCAL_DIRECTORY_PREFIX
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.workspace import LocalWorkdirFactory


def test_local_provider_removes_only_marker_owned_immediate_stale_children(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "project-workspaces"
        settings = local_settings(root)
        first = LocalWorkspaceProvider(settings)
        stale = await first.create("allocation-stale")
        stale_path = Path(stale.root)

        marker_free = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        marker_free.mkdir()
        (marker_free / "keep").write_text("operator data", encoding="utf-8")
        malformed = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        malformed.mkdir()
        (malformed / ".contractor-workspace-owner").write_text("wrong", encoding="ascii")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("outside", encoding="utf-8")
        linked = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        linked.symlink_to(outside, target_is_directory=True)

        replacement = LocalWorkspaceProvider(settings)
        assert await replacement.probe()
        assert not stale_path.exists()
        assert (marker_free / "keep").read_text(encoding="utf-8") == "operator data"
        assert malformed.exists()
        assert linked.is_symlink()
        assert (outside / "keep").read_text(encoding="utf-8") == "outside"

    asyncio.run(scenario())


def _revoke_access(root: Path, outside: Path) -> None:
    """What a keep-id sandbox workload can do to its own project tree."""
    locked = root / "run_workdir" / "build" / "locked"
    locked.mkdir(parents=True)
    (locked / "deep").mkdir()
    (locked / "deep" / "object.o").write_bytes(b"x")
    (locked / "secret").write_text("x", encoding="utf-8")
    (locked / "secret").chmod(0o000)
    (locked / "outside").symlink_to(outside, target_is_directory=True)
    (locked / "deep").chmod(0o000)
    locked.chmod(0o000)
    readonly = root / "run_workdir" / "readonly"
    readonly.mkdir()
    (readonly / "file").write_text("x", encoding="utf-8")
    readonly.chmod(0o500)
    (root / "run_workdir").chmod(0o500)


def test_cleanup_restores_access_the_workload_revoked_without_following_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("outside", encoding="utf-8")
        outside.chmod(0o500)
        provider = LocalWorkspaceProvider(local_settings(tmp_path / "project-workspaces"))
        storage = await provider.create("allocation-1")
        path = Path(storage.root)
        _revoke_access(path, outside)

        await provider.cleanup(storage)

        assert not path.exists()
        assert (outside / "keep").read_text(encoding="utf-8") == "outside"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o500

    asyncio.run(scenario())


def test_undeletable_stale_workspace_cannot_block_provider_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        root = tmp_path / "project-workspaces"
        predecessor = LocalWorkspaceProvider(local_settings(root))
        stuck = Path((await predecessor.create("allocation-stuck")).root)
        stale = Path((await predecessor.create("allocation-stale")).root)
        locked = Path((await predecessor.create("allocation-locked")).root)
        outside = tmp_path / "outside"
        outside.mkdir()
        _revoke_access(locked, outside)

        remove = provider_module._remove_tree

        def failing(path: Path) -> None:
            if path == stuck:
                raise PermissionError("still busy")
            remove(path)

        monkeypatch.setattr(provider_module, "_remove_tree", failing)
        replacement = LocalWorkspaceProvider(local_settings(root))
        with caplog.at_level("WARNING", logger=provider_module.__name__):
            assert await replacement.probe()
            storage = await replacement.create("allocation-next")

        assert not stale.exists() and not locked.exists()
        # Retained with its marker so a later start retries it.
        assert provider_module._is_owned_directory(stuck)
        assert "PermissionError" in caplog.text and str(root) not in caplog.text
        await replacement.cleanup(storage)

    asyncio.run(scenario())


def test_local_provider_handles_are_private_and_cleanup_is_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "project-workspaces"
        provider = LocalWorkspaceProvider(local_settings(root))
        storage = await provider.create("allocation-1")
        path = Path(storage.root)
        assert path.parent == root
        assert str(root) not in repr(storage)
        (path / "data.txt").write_text("temporary", encoding="utf-8")

        outside = tmp_path / "outside"
        outside.mkdir()
        forged = ProjectWorkspaceStorage(
            storage="local",
            filesystem=storage.filesystem,
            root=str(outside),
            provider_id=storage.provider_id,
            owner_token=storage.owner_token,
        )
        with pytest.raises(ValueError, match="unowned"):
            await provider.cleanup(forged)
        assert outside.exists()

        await provider.cleanup(storage)
        assert not path.exists()
        await provider.cleanup(storage)

    asyncio.run(scenario())


def test_local_provider_rejects_root_replaced_by_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project-workspaces"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)
    provider = LocalWorkspaceProvider(local_settings(root))
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(provider.probe())


def test_memory_provider_isolates_allocations_and_provider_instances() -> None:
    async def scenario() -> None:
        settings = memory_settings()
        first = MemoryWorkspaceProvider(settings)
        second = MemoryWorkspaceProvider(settings)
        left = await first.create("allocation-left")
        right = await first.create("allocation-right")
        foreign = await second.create("allocation-foreign")
        left.filesystem.pipe(f"{left.root}/value.txt", b"left")
        right.filesystem.pipe(f"{right.root}/value.txt", b"right")
        assert left.filesystem.cat(f"{left.root}/value.txt") == b"left"
        assert right.filesystem.cat(f"{right.root}/value.txt") == b"right"
        assert not foreign.filesystem.exists(f"{foreign.root}/value.txt")
        assert not foreign.filesystem.exists(f"{left.root}/value.txt")
        assert left.root != right.root != foreign.root
        assert left.root not in repr(left)

        with pytest.raises(ValueError, match="another provider"):
            await second.cleanup(left)
        await first.cleanup(left)
        assert not left.filesystem.exists(left.root)
        assert right.filesystem.exists(right.root)
        await first.cleanup(right)
        await second.cleanup(foreign)

    asyncio.run(scenario())


def test_recursive_local_cleanup_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        loop_thread = threading.get_ident()
        sandbox = LocalWorkdirFactory(tmp_path / "scratch")
        scratch = await sandbox.prepare()
        (scratch.path / "nested").mkdir()
        (scratch.path / "nested" / "data.txt").write_text("data", encoding="utf-8")

        provider = LocalWorkspaceProvider(local_settings(tmp_path / "projects"))
        project = await provider.create("allocation-1")
        project_path = Path(project.root)
        (project_path / "data.txt").write_text("data", encoding="utf-8")

        scratch_threads: list[int] = []
        project_threads: list[int] = []
        original_scratch_remove = workspace_module._remove_workspace_path
        original_project_remove = provider_module._remove_owned_local_workspace

        def tracked_scratch_remove(path: Path) -> None:
            scratch_threads.append(threading.get_ident())
            original_scratch_remove(path)

        def tracked_project_remove(path: Path) -> None:
            project_threads.append(threading.get_ident())
            original_project_remove(path)

        monkeypatch.setattr(workspace_module, "_remove_workspace_path", tracked_scratch_remove)
        monkeypatch.setattr(
            provider_module,
            "_remove_owned_local_workspace",
            tracked_project_remove,
        )

        await sandbox.cleanup(scratch)
        await provider.cleanup(project)

        assert scratch_threads and scratch_threads[0] != loop_thread
        assert project_threads and project_threads[0] != loop_thread
        assert not scratch.path.exists()
        assert not project_path.exists()

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
