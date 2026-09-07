"""Retained contractor-old filesystem/Edit behavior on the narrower v2 API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_edit_files_toolset import hydrated_workspace
from test_edit_files_toolset import make_tools as make_edit_tools
from test_filesystem_toolset import create_tools as make_read_tools

from contractor_runtime.allocation import WorkerState
from contractor_runtime.projectfs import OverlayWorkspaceSession
from contractor_runtime.toolsets.filesystem.tools import FilesystemToolsetFactory


@pytest.mark.parametrize("storage", ["local", "memory"])
@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_retained_read_edit_tree_and_newline_behavior(
    tmp_path: Path, storage: str, mode: str
) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(
            tmp_path, storage, mode, f"legacy-{storage}-{mode}"
        )
        state = WorkerState()
        reads = await make_read_tools(
            FilesystemToolsetFactory(),
            session.reader_view(),
            state,
            tmp_path,
            ["ls", "glob", "read_file", "grep"],
        )
        edits = await make_edit_tools(
            tmp_path,
            session.writer_view(),
            state,
            [
                "append_file",
                "cp",
                "edit",
                "insert_line",
                "mkdir",
                "mv",
                "replace_range",
                "rm",
                "write_file",
            ],
        )

        listing = await reads["ls"]("", "", 100)
        assert [entry["path"] for entry in listing["entries"]] == [
            "crlf.txt",
            "duplicate.txt",
            "empty",
            "image.bin",
            "lf.txt",
            "tree",
        ]
        assert [item["path"] for item in (await reads["glob"]("**/*.txt"))["matches"]] == [
            "crlf.txt",
            "duplicate.txt",
            "lf.txt",
            "tree/child/file.txt",
        ]
        assert (await reads["grep"]("beta", "", "**/*.txt"))["matches"][0]["path"] == "lf.txt"

        await edits["append_file"]("crlf.txt", "four\nfive")
        await edits["insert_line"]("crlf.txt", 2, "inserted")
        await edits["edit"]("crlf.txt", "two\r\n", "TWO\n")
        await edits["replace_range"]("crlf.txt", 3, 4, "middle")
        assert await session.read_text("crlf.txt") == "one\r\ninserted\r\nmiddle\r\nfour\r\nfive"

        await edits["mkdir"]("generated/deep", True)
        await edits["write_file"]("generated/new.txt", "new\n")
        await edits["cp"]("generated/new.txt", "generated/deep/copied.txt")
        await edits["mv"]("generated/deep/copied.txt", "generated/deep/moved.txt")
        await edits["cp"]("tree", "tree-copy", True)
        await edits["rm"]("tree", True)
        snapshot = await session.snapshot()
        paths = {item.path for item in snapshot.files}
        assert "generated/deep/moved.txt" in paths
        assert "tree-copy/child/file.txt" in paths
        assert "tree/child/file.txt" not in paths

        if isinstance(session, OverlayWorkspaceSession):
            changes = {entry.path: entry.change for entry in await session.change_entries()}
            assert changes["crlf.txt"] == "modified"
            assert changes["generated"] == "created"
            assert changes["tree"] == "deleted"
            await session.rollback_changes("crlf.txt")
            assert await session.read_text("crlf.txt") == "one\r\ntwo\r\nthree"
        await provider.cleanup(session.storage)

        # Tool telemetry keeps only bounded counts/outcomes, never paths,
        # patterns or source contents inherited from contractor-old fixtures.
        rendered = repr(state.metrics)
        for forbidden in ("generated/new.txt", "beta", "one\\r\\ntwo", "recognizable"):
            assert forbidden not in rendered

    asyncio.run(scenario())


def test_removed_legacy_escape_hatches_are_not_present(tmp_path: Path) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "memory", "overlay", "narrow")
        for view in (session.reader_view(), session.writer_view(), session.changes_view()):
            for removed in (
                "materialize",
                "fork_overlay",
                "merge_overlay",
                "open",
                "filesystem",
                "root",
                "__getattr__",
            ):
                assert not hasattr(view, removed)
        await provider.cleanup(session.storage)

    asyncio.run(scenario())
