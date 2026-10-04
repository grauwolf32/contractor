from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.factories import built_in_factories
from contractor_runtime.projectfs import (
    ManagedWorkspaceTree,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
)
from contractor_runtime.projectfs import overlay as overlay_module
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.toolsets.filesystem.tools import FilesystemToolError
from contractor_runtime.toolsets.workspace_changes.tools import WorkspaceChangesToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace


def test_change_classification_pagination_diff_and_rollback_are_deterministic(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = await overlay("classification")
        state = WorkerState()
        tools = await make_tools(
            tmp_path,
            session.changes_view(),
            state,
            ["changed_paths", "diff", "rollback_changes"],
        )
        await session.write_text("modified.txt", "after\n")
        await session.delete_path("deleted.txt")
        await session.write_text("created.txt", "created\n")
        await session.delete_path("type-file")
        await session.make_directory("type-file")
        await session.write_text("type-file/child.txt", "new child\n")
        await session.delete_path("type-dir", recursive=True)
        await session.write_text("type-dir", "directory became file\n")

        changes: list[dict[str, str]] = []
        cursor = ""
        while True:
            page = await tools["changed_paths"](cursor, 2)
            changes.extend(page["changes"])
            cursor = page["nextCursor"] or ""
            if not cursor:
                break
        assert changes == [
            {"path": "created.txt", "change": "created"},
            {"path": "deleted.txt", "change": "deleted"},
            {"path": "modified.txt", "change": "modified"},
            {"path": "type-dir", "change": "type_changed"},
            {"path": "type-dir/child.txt", "change": "deleted"},
            {"path": "type-file", "change": "type_changed"},
            {"path": "type-file/child.txt", "change": "created"},
        ]

        chunks: list[str] = []
        cursor = ""
        while True:
            page = await tools["diff"]("", cursor, 37)
            chunks.append(page["text"])
            assert not page["authoritative"]
            assert page["truncated"] == (page["nextCursor"] is not None)
            serialized = json.dumps(page)
            for forbidden in ("baseWorkspaceDigest", "operations", "revision"):
                assert forbidden not in serialized
            cursor = page["nextCursor"] or ""
            if not cursor:
                break
        combined = "".join(chunks)
        full = await session.diff(max_bytes=1 << 20)
        assert combined == full.text
        assert "a/modified.txt" in combined
        assert "b/created.txt" in combined

        await tools["rollback_changes"]("modified.txt")
        assert await session.read_text("modified.txt") == "before\n"
        await tools["rollback_changes"]()
        assert await session.changed_paths() == ()
        assert (await tools["changed_paths"]())["changes"] == []
        assert all(call.arguments == {} for call in state.metrics.tool_calls)

    asyncio.run(scenario())


def test_rollback_uses_imported_and_later_checkpoint_not_original_source(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first = await overlay("first")
        await first.write_text("modified.txt", "imported baseline\n")
        imported_state = await first.export_state()

        second = await overlay("second")
        await second.import_state(imported_state)
        tools = await make_tools(
            tmp_path,
            second.changes_view(),
            WorkerState(),
            ["rollback_changes", "diff"],
        )
        await second.write_text("modified.txt", "invocation one\n")
        await tools["rollback_changes"]("modified.txt")
        assert await second.read_text("modified.txt") == "imported baseline\n"

        await second.write_text("modified.txt", "checkpoint two\n")
        await second.commit_checkpoint()
        await second.write_text("modified.txt", "invocation two\n")
        assert "checkpoint two" in (await tools["diff"]())["text"]
        await tools["rollback_changes"]()
        assert await second.read_text("modified.txt") == "checkpoint two\n"

    asyncio.run(scenario())


def test_factory_rejects_absent_and_direct_views_and_cursors_track_content(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with pytest.raises(FilesystemToolError, match="workspace_required"):
            await make_tools(tmp_path, None, WorkerState(), ["diff"])

        provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits()))
        storage = await provider.create("direct")
        direct = OverlayWorkspaceSession(
            storage=storage,
            content_root=f"{storage.root}/run_workdir",
            limits=limits(),
            directories=set(),
            text_files={"file.txt": "one\n"},
            binary_paths=set(),
        )
        # A writer-only handle deliberately lacks the changes interface, which
        # is the same shape a direct session would provide to a factory.
        with pytest.raises(FilesystemToolError, match="mode_unsupported"):
            await make_tools(tmp_path, direct.writer_view(), WorkerState(), ["diff"])

        tools = await make_tools(
            tmp_path,
            direct.changes_view(),
            WorkerState(),
            ["changed_paths"],
        )
        await direct.write_text("file.txt", "two\n")
        first = await tools["changed_paths"]("", 1)
        cursor = first["nextCursor"]
        # Add a second change so a cursor exists, then mutate content while its
        # classification remains "modified"; the internal token still invalidates it.
        if cursor is None:
            await direct.write_text("other.txt", "new\n")
            first = await tools["changed_paths"]("", 1)
            cursor = first["nextCursor"]
        assert cursor
        tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
        with pytest.raises(FilesystemToolError, match="cursor_invalid"):
            await tools["changed_paths"](tampered, 1)
        await direct.write_text("file.txt", "three\n")
        with pytest.raises(FilesystemToolError, match="cursor_invalid"):
            await tools["changed_paths"](cursor, 1)

        with pytest.raises(ValueError, match="unknown selected tools"):
            await make_tools(tmp_path, direct.changes_view(), WorkerState(), ["export_state"])

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["workspace-changes@1"].exported_tools == {
        "changed_paths",
        "diff",
        "rollback_changes",
    }


def test_diff_pages_render_once_per_generation_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        session = await overlay("diff-pages")
        tools = await make_tools(tmp_path, session.changes_view(), WorkerState(), ["diff"])
        await session.write_text("modified.txt", "".join(f"after {i}\n" for i in range(40)))
        await session.write_text("created.txt", "".join(f"créé {i}\n" for i in range(40)))
        expected = uncached_diff(session)
        rendered = record_rendering(monkeypatch)
        hashed: list[int] = []
        change_token = overlay_module._change_token

        def counted_token(*args: Any) -> str:
            hashed.append(threading.get_ident())
            return change_token(*args)

        monkeypatch.setattr(overlay_module, "_change_token", counted_token)

        pages = await page_through(tools, "", 64)
        assert len(pages) > 10
        assert "".join(pages) == expected
        # Each changed text renders once for all pages, always in a worker thread.
        assert sorted(path for path, _ in rendered) == ["created.txt", "modified.txt"]
        assert threading.get_ident() not in {thread for _, thread in rendered}
        assert hashed and threading.get_ident() not in hashed

        await session.write_text("created.txt", "replaced\n")
        expected = uncached_diff(session)
        rendered.clear()
        pages = await page_through(tools, "", 64)
        assert "".join(pages) == expected
        assert sorted(path for path, _ in rendered) == ["created.txt", "modified.txt"]

    asyncio.run(scenario())


def test_diff_cache_restarts_for_new_generation_root_or_earlier_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session = await overlay("diff-restart")
        await session.write_text("modified.txt", "".join(f"after {i}\n" for i in range(40)))
        await session.write_text("created.txt", "created\n")
        rendered = record_rendering(monkeypatch)

        def paths() -> list[str]:
            return [path for path, _ in rendered]

        first = await session.diff(max_bytes=32)
        assert paths() == ["created.txt"]
        second = await session.diff(max_bytes=32, offset_bytes=first.next_offset or 0)
        assert paths() == ["created.txt", "modified.txt"]
        # A retried page and another page size reuse the retained bytes.
        assert await session.diff(max_bytes=32, offset_bytes=second.offset_bytes) == second
        await session.diff(max_bytes=7, offset_bytes=second.offset_bytes)
        assert len(rendered) == 2

        # An earlier offset or another root renders again.
        assert await session.diff(max_bytes=32) == first
        assert paths()[2:] == ["created.txt"]
        await session.diff("modified.txt", max_bytes=32)
        assert paths()[3:] == ["modified.txt"]

        offset = first.next_offset or 0
        await session.diff(max_bytes=32, offset_bytes=offset)
        # Same length, different bytes: a stale window would return "+created".
        await session.write_text("created.txt", "changed\n")
        changed = await session.diff(max_bytes=32, offset_bytes=offset)
        assert "+changed\n" in changed.text
        assert changed == overlay_module._workspace_diff(
            session._checkpoint, session._tree, "", 32, offset
        )

    asyncio.run(scenario())


def test_diff_cache_does_not_retain_more_than_its_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        session = await overlay("diff-bound")
        await session.write_text("modified.txt", "".join(f"after {i}\n" for i in range(40)))
        expected = uncached_diff(session)
        rendered = record_rendering(monkeypatch)
        # Smaller than any truncated page plus its look-ahead byte.
        monkeypatch.setattr(overlay_module, "MAX_DIFF_CACHE_BYTES", 31)
        chunks: list[str] = []
        offset = 0
        while True:
            page = await session.diff(max_bytes=32, offset_bytes=offset)
            chunks.append(page.text)
            if page.next_offset is None:
                break
            assert session._diff_cache is None
            offset = page.next_offset
        assert "".join(chunks) == expected
        # Without a retained window every page renders the file again.
        assert len(rendered) == len(chunks) > 1

    asyncio.run(scenario())


def test_diff_thread_holds_the_session_lock_through_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        session = await overlay("diff-lock")
        await session.write_text("modified.txt", "after\n")
        started, release = threading.Event(), threading.Event()
        unified_diff = overlay_module.difflib.unified_diff

        def blocked(*args: Any, **kwargs: Any) -> Iterator[str]:
            started.set()
            if not release.wait(5):
                raise AssertionError("diff worker was never released")
            yield from unified_diff(*args, **kwargs)

        monkeypatch.setattr(overlay_module.difflib, "unified_diff", blocked)
        cancelled = asyncio.create_task(session.diff())
        assert await asyncio.to_thread(started.wait, 2)
        writer = asyncio.create_task(session.write_text("modified.txt", "concurrent\n"))
        cancelled.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        # The edit waits for the diff thread although its caller was cancelled.
        assert not writer.done() and not cancelled.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await writer
        monkeypatch.undo()
        diff = (await session.diff()).text
        assert "+concurrent\n" in diff and "+after\n" not in diff

    asyncio.run(scenario())


def record_rendering(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Record each difflib run by project path and the thread that ran it."""

    calls: list[tuple[str, int]] = []
    unified_diff = overlay_module.difflib.unified_diff

    def counted(*args: Any, **kwargs: Any) -> Iterator[str]:
        label = kwargs["tofile"] if kwargs["tofile"] != "/dev/null" else kwargs["fromfile"]
        calls.append((label.split("/", 1)[1], threading.get_ident()))
        yield from unified_diff(*args, **kwargs)

    monkeypatch.setattr(overlay_module.difflib, "unified_diff", counted)
    return calls


def uncached_diff(session: OverlayWorkspaceSession) -> str:
    return overlay_module._workspace_diff(
        session._checkpoint, session._tree, "", overlay_module.MAX_DIFF_BYTES, 0
    ).text


async def page_through(tools: dict[str, Any], path: str, max_bytes: int) -> list[str]:
    chunks: list[str] = []
    cursor = ""
    while True:
        page = await tools["diff"](path, cursor, max_bytes)
        chunks.append(page["text"])
        cursor = page["nextCursor"] or ""
        if not cursor:
            return chunks


async def overlay(name: str) -> OverlayWorkspaceSession:
    provider = MemoryWorkspaceProvider(WorkspaceSettings(storage="memory", limits=limits()))
    storage = await provider.create(name)
    source = source_tree()
    return OverlayWorkspaceSession(
        storage=storage,
        content_root=f"{storage.root}/run_workdir",
        limits=limits(),
        directories=source.directories,
        text_files=source.text_files,
        binary_paths=source.binary_paths,
    )


def source_tree() -> ManagedWorkspaceTree:
    return ManagedWorkspaceTree(
        directories={"type-dir"},
        text_files={
            "modified.txt": "before\n",
            "deleted.txt": "delete\n",
            "type-file": "file\n",
            "type-dir/child.txt": "child\n",
            "unchanged.txt": "same\n",
        },
    )


async def make_tools(
    tmp_path: Path,
    changes: Any,
    state: WorkerState,
    selected: list[str],
) -> dict[str, Any]:
    result = await WorkspaceChangesToolsetFactory().create_selected(
        selected=selected,
        allocation_id="allocation-changes",
        run_id="run-changes",
        namespace="editor",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=state,
        project_workspace=changes,
    )
    return dict(result)


def limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 19,
        max_file_bytes=1 << 18,
    )
