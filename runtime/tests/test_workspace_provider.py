from __future__ import annotations

import asyncio
import stat
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from filesystem_audit import FilesystemAudit

import contractor_runtime.projectfs.provider as provider_module
import contractor_runtime.workspace as workspace_module
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    ProjectWorkspaceStorage,
)
from contractor_runtime.projectfs.config import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.projectfs.provider import LOCAL_DIRECTORY_PREFIX
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
    from contractor_runtime.projectfs import DirectWorkspaceSession

    async def scenario() -> None:
        settings = memory_settings()
        first = MemoryWorkspaceProvider(settings)
        second = MemoryWorkspaceProvider(settings)
        left = await first.create("allocation-left")
        right = await first.create("allocation-right")
        foreign = await second.create("allocation-foreign")
        sessions = [
            DirectWorkspaceSession(
                mode="direct",
                storage=handle,
                limits=settings.limits,
                directories=set(),
                text_files={},
                binary_paths=set(),
            )
            for handle in (left, right, foreign)
        ]
        await sessions[0].write_text("value.txt", "left")
        await sessions[1].write_text("value.txt", "right")
        assert await sessions[0].read_text("value.txt") == "left"
        assert await sessions[1].read_text("value.txt") == "right"
        assert (await sessions[2].snapshot()).files == ()
        assert left.root != right.root != foreign.root
        assert left.root not in repr(left)
        with pytest.raises(ValueError, match="another provider"):
            await second.cleanup(left)
        await sessions[0].close()
        await first.cleanup(left)
        assert await sessions[1].read_text("value.txt") == "right"
        for session, provider in zip(sessions[1:], (first, second), strict=True):
            await session.close()
            await provider.cleanup(session.storage)

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


def test_local_preparation_writes_owned_directories_off_the_event_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        loop_thread = threading.get_ident()
        provider = LocalWorkspaceProvider(local_settings(tmp_path / "projects"))
        with FilesystemAudit(str(tmp_path), str(tmp_path.resolve())) as audit:
            scratch = await LocalWorkdirFactory(tmp_path / "scratch").prepare()
            project = await provider.create("allocation-1")
        scratch_root = scratch.root
        marker = scratch_root / f"{scratch.path.name}{workspace_module.ALLOCATION_OWNER_SUFFIX}"
        project_path = Path(project.root)
        project_marker = project_path / provider_module.LOCAL_OWNER_MARKER

        assert {
            ("os.mkdir", str(scratch_root)),
            ("open", str(marker)),
            ("os.mkdir", str(scratch.path)),
            ("os.mkdir", str(project_path)),
            ("open", str(project_marker)),
            ("os.chmod", str(project_marker)),
        } <= set(audit.events())
        assert audit.on_thread(loop_thread) == []
        assert stat.S_IMODE(scratch_root.stat().st_mode) == 0o700
        assert stat.S_IMODE(scratch.path.stat().st_mode) == 0o700
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
        assert workspace_module._owns(scratch_root, scratch.path.name)
        assert stat.S_IMODE(project_path.stat().st_mode) == 0o700
        assert stat.S_IMODE(project_marker.stat().st_mode) == 0o600
        assert provider_module._is_owned_directory(project_path)

    asyncio.run(scenario())


def test_cancelled_workdir_preparation_joins_its_thread_before_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    events: list[tuple[str, int, bool, bool]] = []
    original_create = workspace_module._create_owned_workdir
    original_remove = workspace_module._remove_owned_workdir

    def owned(root: Path, name: str) -> tuple[bool, bool]:
        marker = root / f"{name}{workspace_module.ALLOCATION_OWNER_SUFFIX}"
        return (root / name).is_dir(), marker.is_file()

    def blocking_create(root: Path, name: str) -> None:
        original_create(root, name)
        events.append(("created", threading.get_ident(), *owned(root, name)))
        started.set()
        if not release.wait(10):
            raise RuntimeError("the test never released preparation")

    def tracked_remove(root: Path, name: str) -> None:
        events.append(("rollback", threading.get_ident(), *owned(root, name)))
        original_remove(root, name)

    monkeypatch.setattr(workspace_module, "_create_owned_workdir", blocking_create)
    monkeypatch.setattr(workspace_module, "_remove_owned_workdir", tracked_remove)
    root = tmp_path / "scratch"

    async def scenario() -> None:
        loop_thread = threading.get_ident()
        with FilesystemAudit(str(tmp_path), str(tmp_path.resolve())) as audit:
            await prepare_and_cancel(loop_thread)
        assert audit.events() and audit.on_thread(loop_thread) == []

    async def prepare_and_cancel(loop_thread: int) -> None:
        task = asyncio.create_task(LocalWorkdirFactory(root).prepare())
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        # The cancelled owner waits for the thread and rolls nothing back yet.
        assert not task.done()
        assert [event[0] for event in events] == ["created"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [event[0] for event in events] == ["created", "rollback"]
        assert all(thread != loop_thread for _, thread, _, _ in events)
        # The rollback saw the directory and marker that the joined thread made.
        assert events[1][2:] == (True, True)
        assert list(root.iterdir()) == []

    asyncio.run(scenario())


def test_failed_workdir_rollback_keeps_cancellation_and_marker_for_startup_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    original_create = workspace_module._create_owned_workdir

    def blocking_create(root: Path, name: str) -> None:
        original_create(root, name)
        started.set()
        if not release.wait(10):
            raise RuntimeError("the test never released preparation")

    def failing_removal(path: Path) -> None:
        raise PermissionError("directory removal failed")

    monkeypatch.setattr(workspace_module, "_create_owned_workdir", blocking_create)
    monkeypatch.setattr(workspace_module, "_remove_workspace_path", failing_removal)
    root = tmp_path / "scratch"

    async def scenario() -> None:
        task = asyncio.create_task(LocalWorkdirFactory(root).prepare())
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        release.set()
        # The cancellation, not the removal failure, ends preparation.
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    (directory,) = [entry for entry in root.iterdir() if entry.is_dir()]
    assert workspace_module._owns(root, directory.name)
    monkeypatch.undo()
    workspace_module.cleanup_orphan_workdirs(root)
    assert list(root.iterdir()) == []


def test_cancelled_project_workspace_creation_joins_its_thread_before_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    events: list[tuple[str, int, bool, bool]] = []
    original_create = provider_module._create_owned_local_workspace
    original_discard = provider_module._discard_new_local_workspace

    def owned(path: Path) -> tuple[bool, bool]:
        return path.is_dir(), (path / provider_module.LOCAL_OWNER_MARKER).is_file()

    def blocking_create(path: Path) -> None:
        original_create(path)
        events.append(("created", threading.get_ident(), *owned(path)))
        started.set()
        if not release.wait(10):
            raise RuntimeError("the test never released creation")

    def tracked_discard(path: Path) -> None:
        events.append(("rollback", threading.get_ident(), *owned(path)))
        original_discard(path)

    monkeypatch.setattr(provider_module, "_create_owned_local_workspace", blocking_create)
    monkeypatch.setattr(provider_module, "_discard_new_local_workspace", tracked_discard)
    root = tmp_path / "projects"
    provider = LocalWorkspaceProvider(local_settings(root))

    async def scenario() -> None:
        loop_thread = threading.get_ident()
        with FilesystemAudit(str(tmp_path), str(tmp_path.resolve())) as audit:
            await create_and_cancel(loop_thread)
        assert audit.events() and audit.on_thread(loop_thread) == []

    async def create_and_cancel(loop_thread: int) -> None:
        task = asyncio.create_task(provider.create("allocation-1"))
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done()
        assert [event[0] for event in events] == ["created"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [event[0] for event in events] == ["created", "rollback"]
        assert all(thread != loop_thread for _, thread, _, _ in events)
        assert events[1][2:] == (True, True)
        assert list(root.iterdir()) == []

    asyncio.run(scenario())


def test_failed_local_preparation_rolls_back_in_its_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        loop_thread = threading.get_ident()
        # A directory that cannot be created after its marker was written.
        scratch_root = tmp_path / "scratch"
        scratch_root.mkdir()
        token = "0" * 32
        monkeypatch.setattr(workspace_module.uuid, "uuid4", lambda: SimpleNamespace(hex=token))
        blocker = scratch_root / f"{workspace_module.ALLOCATION_DIRECTORY_PREFIX}{token}"
        blocker.write_text("operator file", encoding="utf-8")
        with pytest.raises(FileExistsError):
            await LocalWorkdirFactory(scratch_root).prepare()
        assert sorted(entry.name for entry in scratch_root.iterdir()) == [blocker.name]
        assert blocker.read_text(encoding="utf-8") == "operator file"
        monkeypatch.undo()

        # A project workspace whose marker cannot be written is removed.
        discarded: list[int] = []
        original_discard = provider_module._discard_new_local_workspace

        def failing_marker(name: str) -> str:
            raise OSError("marker write failed")

        def tracked_discard(path: Path) -> None:
            discarded.append(threading.get_ident())
            original_discard(path)

        monkeypatch.setattr(provider_module, "_marker_contents", failing_marker)
        monkeypatch.setattr(provider_module, "_discard_new_local_workspace", tracked_discard)
        project_root = tmp_path / "projects"
        with pytest.raises(OSError, match="marker write failed"):
            await LocalWorkspaceProvider(local_settings(project_root)).create("allocation-1")
        assert list(project_root.iterdir()) == []
        assert discarded and loop_thread not in discarded

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
