"""Local direct sessions use current disk state, including unknown external writes."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    WorkspaceStorageError,
    hydrate_workspace,
    local_direct,
)
from contractor_runtime.projectfs.local_io import RootedLocalFilesystem
from contractor_runtime.settings import WorkspaceLimits


@asynccontextmanager
async def workspace(
    tmp_path: Path, *, limits: WorkspaceLimits | None = None
) -> AsyncIterator[tuple[DirectWorkspaceSession, Path]]:
    spec, reader = workspace_inputs(
        [("source", "", archive({"src/a.txt": b"source\r\n", "binary": b"\x00x"}))]
    )
    spec.mode = "direct"
    provider = LocalWorkspaceProvider(settings("local", tmp_path / "project", limits=limits))
    session = await hydrate_workspace(
        provider=provider,
        spec=spec,
        artifact_reader=reader,
        allocation_id="disk-authority",
        timeout_seconds=5,
    )
    try:
        yield session, Path(session.storage.root) / "run_workdir"
    finally:
        await session.close()
        await provider.cleanup(session.storage)


def test_external_changes_refresh_snapshots_metadata_and_classification(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            initial = await session.snapshot()
            path = root / "src/a.txt"
            before = path.stat()
            path.write_bytes(b"edited\r\n")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert path.stat().st_size == before.st_size
            assert path.stat().st_mtime_ns == before.st_mtime_ns
            assert await session.read_text("src/a.txt") == "edited\r\n"
            second = await session.snapshot()
            assert second.digest != initial.digest
            assert initial.files[0].text == "source\r\n"
            path.rename(root / "renamed")
            (root / "src").rmdir()
            (root / "empty").mkdir()
            (root / "binary").write_bytes(b"now text")
            (root / "renamed").write_bytes(b"\xffbinary")
            current = await session.snapshot()
            assert current.directories == ("empty",)
            assert current.binary_paths == ("renamed",)
            assert [(file.path, file.text) for file in current.files] == [("binary", "now text")]
            metadata = await session.observation_metadata()
            assert metadata.digest == current.digest
            assert metadata.managed_text_paths == ("binary",)
            with pytest.raises(WorkspaceStorageError, match="binary_file_unsupported"):
                await session.read_text("renamed")
            with pytest.raises(WorkspaceStorageError, match="workspace_not_found"):
                await session.read_text("src/a.txt")
            # Constructor/hydration texts are not retained as a fallback tree.
            assert not session._tree.paths()

    asyncio.run(scenario())


def test_mutations_use_current_bytes_and_do_not_resurrect_or_overwrite_others(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "src/a.txt").write_bytes(b"external\r\n")
            (root / "unrelated").write_bytes(b"keep")
            await session.update_text("src/a.txt", lambda text: text + "append\r\n")
            assert (root / "src/a.txt").read_bytes() == b"external\r\nappend\r\n"
            await session.copy_path("src", "copied", recursive=True)
            assert (root / "copied/a.txt").read_bytes() == b"external\r\nappend\r\n"
            (root / "copied/a.txt").write_bytes(b"newer")
            await session.move_path("copied", "moved")
            assert (root / "moved/a.txt").read_bytes() == b"newer"
            (root / "src/a.txt").unlink()
            with pytest.raises(WorkspaceStorageError, match="workspace_not_found"):
                await session.update_text("src/a.txt", lambda _: "resurrected")
            await session.write_text("new", "created")
            assert not (root / "src/a.txt").exists()
            await session.make_directory("external")
            (root / "external").rmdir()
            await session.make_directory("external/nested", parents=True)
            await session.delete_path("moved", recursive=True)
            assert not (root / "moved").exists()
            assert (root / "unrelated").read_bytes() == b"keep"

    asyncio.run(scenario())


def test_concurrent_transforms_serialize_the_whole_read_modify_write(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "src/a.txt").write_bytes(b"external\n")
            await asyncio.gather(
                *(
                    session.update_text("src/a.txt", lambda text, i=i: text + f"{i}\n")
                    for i in range(32)
                )
            )
            lines = (root / "src/a.txt").read_text().splitlines()
            assert lines[0] == "external"
            assert sorted(map(int, lines[1:])) == list(range(32))

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["write", "update", "copy", "move", "mkdir", "remove"])
def test_current_type_conflicts_fail_before_any_mutation(tmp_path: Path, operation: str) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "destination").write_bytes(b"external destination")
            if operation in {"write", "update", "remove"}:
                (root / "src/a.txt").write_bytes(b"\x00binary now")
            baseline = await session.snapshot()
            with pytest.raises(WorkspaceStorageError):
                if operation == "write":
                    await session.write_text("src/a.txt", "bad")
                elif operation == "update":
                    await session.update_text("src/a.txt", lambda _: "bad")
                elif operation == "copy":
                    await session.copy_path("src/a.txt", "destination")
                elif operation == "move":
                    await session.move_path("src/a.txt", "destination")
                elif operation == "mkdir":
                    await session.make_directory("destination/nested", parents=True)
                else:
                    await session.delete_path("src", recursive=True)
            assert await session.snapshot() == baseline
            assert (root / "destination").read_bytes() == b"external destination"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bound", ["max_files", "max_file_bytes", "max_expanded_bytes", "max_managed_text_bytes"]
)
def test_external_quota_violation_has_no_stale_snapshot_or_unrelated_read(
    tmp_path: Path, bound: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        limits = dict(
            max_files=10, max_file_bytes=100, max_expanded_bytes=100, max_managed_text_bytes=100
        )
        limits[bound] = 4 if bound == "max_files" else 20
        async with workspace(tmp_path, limits=WorkspaceLimits(**limits)) as (session, root):
            if bound == "max_files":
                (root / "extra1").touch()
                (root / "extra2").touch()
            else:
                (root / "extra").write_bytes(b"x" * 21)
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.snapshot()
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.write_text("new", "no effects")
            assert not (root / "new").exists()
            assert session._local is not None
            assert session._local._filesystem is not None

            def no_scan(**_: object) -> None:
                pytest.fail("single-file reads must not scan unrelated entries")

            monkeypatch.setattr(session._local._filesystem, "scan", no_scan)
            assert await session.read_text("src/a.txt") == "source\r\n"

    asyncio.run(scenario())


def test_mutation_quota_includes_unchanged_binary_bytes(tmp_path: Path) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=10, max_file_bytes=100, max_expanded_bytes=15, max_managed_text_bytes=100
        )
        async with workspace(tmp_path, limits=limits) as (session, root):
            # Eight text bytes plus two binary bytes: copying the text would
            # exceed the physical bound even though managed bytes can grow.
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.copy_path("src/a.txt", "copy")
            assert not (root / "copy").exists()
            await session.write_text("src/a.txt", "x" * 13)
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.write_text("src/a.txt", "x" * 14)
            assert (root / "src/a.txt").read_bytes() == b"x" * 13

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_cancel_timeout_and_close_share_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            started, finish = threading.Event(), threading.Event()

            def transform(text: str) -> str:
                started.set()
                assert finish.wait(3)
                return text + "done"

            monkeypatch.setattr(local_direct, "_OPERATION_SECONDS", 2 if cancel else 0.05)
            owner = asyncio.create_task(session.update_text("src/a.txt", transform))
            closing = None
            try:
                while not started.is_set():
                    await asyncio.sleep(0.001)
                if cancel:
                    owner.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await owner
                else:
                    with pytest.raises(WorkspaceStorageError):
                        await owner
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.write_text("new", "denied")
                monkeypatch.setattr(local_direct, "_OPERATION_SECONDS", 2)
                closing = asyncio.create_task(session.close())
                for _ in range(5):
                    await asyncio.sleep(0)
                assert not closing.done()
                assert (root / "src/a.txt").read_bytes() == b"source\r\n"
                finish.set()
                await closing
                assert not (root / "new").exists()
                with pytest.raises(WorkspaceStorageError):
                    await session.read_text("src/a.txt")
            finally:
                finish.set()
                await asyncio.gather(owner, *([closing] if closing else []), return_exceptions=True)

    asyncio.run(scenario())


def test_partial_physical_failure_fences_without_reverse_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            assert session._local is not None
            assert session._local._filesystem is not None
            original = session._local._filesystem.write

            def fail(path: str, data: bytes, *, deadline: float) -> None:
                original(path, data, deadline=deadline)
                (root / "external").write_bytes(b"do not overwrite")
                raise OSError("host path /private must not escape")

            monkeypatch.setattr(session._local._filesystem, "write", fail)
            with pytest.raises(WorkspaceStorageError, match=r"^workspace_unavailable$"):
                await session.write_text("src/a.txt", "already committed")
            assert (root / "src/a.txt").read_bytes() == b"already committed"
            assert (root / "external").read_bytes() == b"do not overwrite"
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await session.snapshot()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "storage,mode", [("memory", "direct"), ("memory", "overlay"), ("local", "overlay")]
)
def test_other_modes_keep_managed_view_semantics(tmp_path: Path, storage: str, mode: str) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs([("source", "", archive({"file": b"source"}))])
        spec.mode = mode
        provider = (
            LocalWorkspaceProvider(settings("local", tmp_path / "project"))
            if storage == "local"
            else MemoryWorkspaceProvider(settings("memory"))
        )
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="other-modes",
            timeout_seconds=5,
        )
        try:
            session.storage.filesystem.pipe(f"{session.storage.root}/run_workdir/file", b"external")
            assert await session.read_text("file") == "source"
            await session.write_text("file", "managed")
            assert await session.read_text("file") == "managed"
        finally:
            await session.close()
            await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_cancelled_hydration_joins_initial_scan_before_erasing_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs([("source", "", archive({"file": b"source"}))])
        spec.mode = "direct"
        provider = LocalWorkspaceProvider(settings("local", tmp_path / "project"))
        started, finish = threading.Event(), threading.Event()
        original = RootedLocalFilesystem.scan

        def blocked(fs: RootedLocalFilesystem, **kwargs: object) -> object:
            started.set()
            assert finish.wait(3)
            return original(fs, **kwargs)

        monkeypatch.setattr(RootedLocalFilesystem, "scan", blocked)
        preparing = asyncio.create_task(
            hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="cancelled-prepare",
                timeout_seconds=5,
            )
        )
        try:
            while not started.is_set():
                await asyncio.sleep(0.001)
            preparing.cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            assert not preparing.done()
            assert len(list((tmp_path / "project").iterdir())) == 1
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await preparing
            assert list((tmp_path / "project").iterdir()) == []
        finally:
            finish.set()
            await asyncio.gather(preparing, return_exceptions=True)

    asyncio.run(scenario())
