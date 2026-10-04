"""Local direct sessions use current disk state, including unknown external writes."""

from __future__ import annotations

import asyncio
import errno
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.projectfs.local_io import RootedLocalFilesystem
from contractor_runtime.settings import WorkspaceLimits


@asynccontextmanager
async def workspace(
    tmp_path: Path,
    *,
    limits: WorkspaceLimits | None = None,
    operation_timeout_seconds: float = 30.0,
) -> AsyncIterator[tuple[DirectWorkspaceSession, Path]]:
    spec, reader = workspace_inputs(
        [("source", "", archive({"src/a.txt": b"source\r\n", "binary": b"\x00x"}))]
    )
    spec.mode = "direct"
    provider = LocalWorkspaceProvider(
        replace(
            settings("local", tmp_path / "project", limits=limits),
            operation_timeout_seconds=operation_timeout_seconds,
        )
    )
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


@pytest.mark.parametrize(
    ("operation", "expected_reads"),
    [
        ("mkdir", set()),
        ("write_new", set()),
        ("write_existing", {"src/a.txt"}),
        ("update", {"src/a.txt"}),
        ("delete", {"src/a.txt"}),
        ("copy", {"src/a.txt"}),
        ("move", {"src/a.txt"}),
    ],
)
def test_mutation_preflight_reads_only_selected_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_reads: set[str],
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "unrelated.txt").write_text("unrelated")
            assert session._local is not None and session._local._filesystem is not None
            filesystem = session._local._filesystem
            original = filesystem.read_classified
            reads: list[str] = []

            def counted(path: str, **kwargs: object) -> tuple[str | None, bool]:
                reads.append(path)
                return original(path, **kwargs)

            monkeypatch.setattr(filesystem, "read_classified", counted)
            if operation == "mkdir":
                await session.make_directory("new-directory")
            elif operation == "write_new":
                await session.write_text("new.txt", "new")
            elif operation == "write_existing":
                await session.write_text("src/a.txt", "replaced")
            elif operation == "update":
                await session.update_text("src/a.txt", lambda text: text + "updated")
            elif operation == "delete":
                await session.delete_path("src/a.txt")
            elif operation == "copy":
                await session.copy_path("src", "copied", recursive=True)
            else:
                await session.move_path("src", "moved")
            assert set(reads) == expected_reads
            assert "unrelated.txt" not in reads

    asyncio.run(scenario())


def test_large_binary_leaves_keep_mutation_preflight_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=10, max_file_bytes=100, max_expanded_bytes=100, max_managed_text_bytes=12
        )
        async with workspace(tmp_path, limits=limits) as (session, root):
            (root / "large-binary").write_bytes(b"\x00" + b"x" * 39)
            assert session._local is not None and session._local._filesystem is not None
            filesystem = session._local._filesystem
            original_scan = filesystem.scan
            original_read = filesystem.read_classified
            scans: list[bool] = []
            reads: list[str] = []

            def counted_scan(**kwargs: object):
                scans.append(bool(kwargs.get("contents", True)))
                return original_scan(**kwargs)

            def counted_read(path: str, **kwargs: object):
                reads.append(path)
                return original_read(path, **kwargs)

            monkeypatch.setattr(filesystem, "scan", counted_scan)
            monkeypatch.setattr(filesystem, "read_classified", counted_read)
            await session.write_text("src/a.txt", "edited\r\n")
            assert scans == [False]
            assert reads == ["large-binary", "src/a.txt"]
            scans.clear()
            reads.clear()
            await session.make_directory("new-directory")
            assert scans == [False]
            assert reads == ["large-binary"]

    asyncio.run(scenario())


def test_large_binary_does_not_hide_excess_current_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=10, max_file_bytes=100, max_expanded_bytes=100, max_managed_text_bytes=12
        )
        async with workspace(tmp_path, limits=limits) as (session, root):
            (root / "large-binary").write_bytes(b"\x00" + b"x" * 39)
            (root / "too-much.txt").write_text("x" * 13)
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.make_directory("not-created")
            assert not (root / "not-created").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("marker", [b"\x00", b"\xff"])
def test_content_scan_stops_after_first_binary_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: bytes
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            path = root / "large-binary"
            path.write_bytes(marker + b"x" * (4 * 65536 - 1))
            original = os.read
            read_bytes = 0

            def counted(descriptor: int, count: int) -> bytes:
                nonlocal read_bytes
                data = original(descriptor, count)
                if os.readlink(f"/proc/self/fd/{descriptor}") == str(path):
                    read_bytes += len(data)
                return data

            with monkeypatch.context() as patch:
                patch.setattr(os, "read", counted)
                snapshot = await session.snapshot()
            assert "large-binary" in snapshot.binary_paths
            assert 0 < read_bytes <= 65536

    asyncio.run(scenario())


def test_content_scan_accepts_utf8_split_across_chunks(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            text = "a" * (65536 - 1) + "é" + "b"
            (root / "chunk-boundary.txt").write_text(text)
            snapshot = await session.snapshot()
            assert (
                next(file.text for file in snapshot.files if file.path == "chunk-boundary.txt")
                == text
            )

    asyncio.run(scenario())


def test_ambiguous_text_bound_classifies_once_without_repeating_transform(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=10, max_file_bytes=100, max_expanded_bytes=100, max_managed_text_bytes=11
        )
        async with workspace(tmp_path, limits=limits) as (session, root):
            calls = 0

            def transform(text: str) -> str:
                nonlocal calls
                calls += 1
                return text + "ab"

            await session.update_text("src/a.txt", transform)
            assert calls == 1
            assert (root / "src/a.txt").read_bytes() == b"source\r\nab"

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


def test_external_oversize_file_is_an_opaque_leaf_not_a_workspace_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=10, max_file_bytes=20, max_expanded_bytes=100, max_managed_text_bytes=100
        )
        async with workspace(tmp_path, limits=limits) as (session, root):
            (root / "build.log").write_bytes(b"x" * 21)
            current = await session.snapshot()
            assert "build.log" in current.binary_paths
            await session.write_text("new", "fits")
            assert (root / "new").read_bytes() == b"fits"
            with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
                await session.read_text("build.log")
            with pytest.raises(WorkspaceStorageError, match="workspace_type_conflict"):
                await session.write_text("build.log", "replace")
            assert (root / "build.log").read_bytes() == b"x" * 21

    asyncio.run(scenario())


@pytest.mark.parametrize("bound", ["max_files", "max_expanded_bytes", "max_managed_text_bytes"])
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
def test_cancel_timeout_and_close_share_ownership(tmp_path: Path, cancel: bool) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path, operation_timeout_seconds=2 if cancel else 0.05) as (
            session,
            root,
        ):
            started, finish = threading.Event(), threading.Event()

            def transform(text: str) -> str:
                started.set()
                assert finish.wait(3)
                return text + "done"

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


@pytest.mark.parametrize("operation", ["write", "update"])
@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT])
def test_failed_temp_write_preserves_workspace_and_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, code: int
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            before = await session.snapshot()
            listing = sorted(path.name for path in (root / "src").iterdir())
            original = os.write

            def disk_full(descriptor: int, data: bytes) -> int:
                target = os.readlink(f"/proc/self/fd/{descriptor}")
                if "/.contractor-write-" in target:
                    raise OSError(code, "private disk detail")
                return original(descriptor, data)

            with monkeypatch.context() as patch:
                patch.setattr(os, "write", disk_full)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    if operation == "write":
                        await session.write_text("src/a.txt", "changed")
                    else:
                        await session.update_text("src/a.txt", lambda _text: "changed")
            assert session._local is not None and not session._local.guard.fenced
            assert sorted(path.name for path in (root / "src").iterdir()) == listing
            assert await session.read_text("src/a.txt") == "source\r\n"
            assert await session.snapshot() == before
            await session.delete_path("src/a.txt")
            assert not (root / "src/a.txt").exists()

    asyncio.run(scenario())


def test_failed_temp_creation_and_first_unlink_do_not_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            original_open = os.open

            def denied_temp(path: object, flags: int, *args: object, **kwargs: object) -> int:
                if isinstance(path, str) and path.startswith(".contractor-write-"):
                    raise PermissionError(errno.EACCES, "private path")
                return original_open(path, flags, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(os, "open", denied_temp)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.write_text("src/a.txt", "changed")
            assert session._local is not None and not session._local.guard.fenced
            assert await session.read_text("src/a.txt") == "source\r\n"

            original_unlink = os.unlink

            def denied_unlink(path: object, *args: object, **kwargs: object) -> None:
                if path == "a.txt":
                    raise PermissionError(errno.EACCES, "private path")
                original_unlink(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(os, "unlink", denied_unlink)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.delete_path("src/a.txt")
            assert not session._local.guard.fenced
            assert await session.snapshot()
            await session.delete_path("src/a.txt")
            assert not (root / "src/a.txt").exists()

    asyncio.run(scenario())


def test_failed_first_mkdir_and_failed_temp_cleanup_have_distinct_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            original_mkdir = os.mkdir

            def denied_mkdir(path: object, *args: object, **kwargs: object) -> None:
                if path == "new":
                    raise PermissionError(errno.EACCES, "private path")
                original_mkdir(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(os, "mkdir", denied_mkdir)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.make_directory("new")
            assert session._local is not None and not session._local.guard.fenced
            assert not (root / "new").exists()
            await session.make_directory("new")

            original_write = os.write
            original_unlink = os.unlink

            def fail_write(descriptor: int, data: bytes) -> int:
                if "/.contractor-write-" in os.readlink(f"/proc/self/fd/{descriptor}"):
                    raise OSError(errno.ENOSPC, "private disk detail")
                return original_write(descriptor, data)

            def fail_cleanup(path: object, *args: object, **kwargs: object) -> None:
                if isinstance(path, str) and path.startswith(".contractor-write-"):
                    raise PermissionError(errno.EACCES, "private path")
                original_unlink(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(os, "write", fail_write)
                patch.setattr(os, "unlink", fail_cleanup)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.write_text("src/a.txt", "changed")
            assert session._local.guard.fenced
            assert any(
                path.name.startswith(".contractor-write-")
                for path in root.joinpath("src").iterdir()
            )

    asyncio.run(scenario())


def test_recursive_remove_fences_after_first_file_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            (root / "src/z.txt").write_text("second")
            original = os.unlink
            removed = 0

            def fail_second(path: object, *args: object, **kwargs: object) -> None:
                nonlocal removed
                if path in {"a.txt", "z.txt"}:
                    if removed:
                        raise PermissionError(errno.EACCES, "private path")
                    removed += 1
                original(path, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(os, "unlink", fail_second)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.delete_path("src", recursive=True)
            assert removed == 1
            assert session._local is not None and session._local.guard.fenced
            assert len(list((root / "src").iterdir())) == 1
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await session.snapshot()

    asyncio.run(scenario())


def test_first_unlink_that_changes_then_reports_error_still_fences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with workspace(tmp_path) as (session, root):
            original = os.unlink

            def removed_then_failed(path: object, *args: object, **kwargs: object) -> None:
                original(path, *args, **kwargs)
                if path == "a.txt":
                    raise PermissionError(errno.EACCES, "private path")

            with monkeypatch.context() as patch:
                patch.setattr(os, "unlink", removed_then_failed)
                with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                    await session.delete_path("src/a.txt")
            assert not (root / "src/a.txt").exists()
            assert session._local is not None and session._local.guard.fenced

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
            physical_root = f"{session.storage.root}/run_workdir"
            assert not session.storage.filesystem.exists(physical_root)
            session.storage.filesystem.makedirs(physical_root, exist_ok=False)
            session.storage.filesystem.pipe(f"{physical_root}/file", b"external")
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


@pytest.mark.parametrize("operation", ["read", "copy"])
def test_unclassified_reads_fail_explicitly_without_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    root = tmp_path / "content"
    root.mkdir()
    (root / "a.txt").write_bytes(b"a\n")
    fs = RootedLocalFilesystem(
        root,
        WorkspaceLimits(
            max_files=10, max_expanded_bytes=1024, max_managed_text_bytes=1024, max_file_bytes=1024
        ),
    )
    # Only a classifying read may report binary content as None. The invariant
    # is an explicit error, not an assert that python -O strips.
    monkeypatch.setattr(RootedLocalFilesystem, "_read", lambda *_args, **_kwargs: None)
    deadline = time.monotonic() + 5
    with pytest.raises(RuntimeError, match="returned no content"):
        if operation == "read":
            fs.read("a.txt", deadline=deadline)
        else:
            fs.copy("a.txt", "b.txt", deadline=deadline)
    assert sorted(path.name for path in root.iterdir()) == ["a.txt"]
