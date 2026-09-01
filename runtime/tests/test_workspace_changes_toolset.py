from __future__ import annotations

import asyncio
import json
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
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.toolsets.filesystem import FilesystemToolError
from contractor_runtime.toolsets.workspace_changes import WorkspaceChangesToolsetFactory
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
            stored_binary_paths=set(),
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
        stored_binary_paths=source.stored_binary_paths,
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
