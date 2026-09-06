from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import contractor_runtime.toolsets.filesystem as filesystem_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.factories import built_in_factories
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
)
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings
from contractor_runtime.toolsets.filesystem import (
    FilesystemToolError,
    FilesystemToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET_PATTERN = "recognizable-file-content-secret"


def test_factory_exposes_only_selected_tools_and_requires_narrow_workspace(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session = await workspace("direct", "selection")
        view = session.reader_view()
        assert not hasattr(view, "storage")
        state = WorkerState()
        factory = FilesystemToolsetFactory()
        selected = await create_tools(factory, view, state, tmp_path, ["read_file"])
        assert set(selected) == {"read_file"}
        with pytest.raises(ValueError, match="unknown selected tools"):
            await create_tools(factory, view, state, tmp_path, ["write_file"])
        with pytest.raises(FilesystemToolError, match="workspace_required"):
            await create_tools(factory, None, state, tmp_path, ["ls"])

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["filesystem@1"].exported_tools == {
        "ls",
        "glob",
        "read_file",
        "grep",
    }


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_read_tools_are_sorted_paginated_and_preserve_newlines(tmp_path: Path, mode: str) -> None:
    async def scenario() -> None:
        session = await workspace(mode, f"read-{mode}")
        state = WorkerState()
        tools = await create_tools(
            FilesystemToolsetFactory(),
            session.reader_view(),
            state,
            tmp_path,
            ["ls", "glob", "read_file", "grep"],
        )

        root_entries: list[dict[str, Any]] = []
        cursor = ""
        while True:
            page = await tools["ls"]("", cursor, 2)
            root_entries.extend(page["entries"])
            assert page["truncated"] == (page["nextCursor"] is not None)
            cursor = page["nextCursor"] or ""
            if not cursor:
                break
        assert [entry["path"] for entry in root_entries] == [
            "docs",
            "empty",
            "image.bin",
            "long.txt",
            "root.py",
            "src",
        ]
        assert [entry["type"] for entry in root_entries] == [
            "directory",
            "directory",
            "binary",
            "file",
            "file",
            "directory",
        ]

        first_glob = await tools["glob"]("**/*.py", "", 2)
        second_glob = await tools["glob"]("**/*.py", first_glob["nextCursor"], 2)
        assert [item["path"] for item in first_glob["matches"] + second_glob["matches"]] == [
            "root.py",
            "src/a.py",
            "src/nested/b.py",
        ]

        read = await tools["read_file"]("docs/readme.txt", 1, 10)
        assert read["lines"] == [
            {"number": 1, "text": "first", "newline": "crlf", "truncated": False},
            {"number": 2, "text": "second", "newline": "lf", "truncated": False},
            {"number": 3, "text": "last", "newline": "none", "truncated": False},
        ]
        assert not read["truncated"] and read["nextLine"] is None

        literal = await tools["grep"]("needle", "", "**/*.py", False, True, "", 10)
        assert [(item["path"], item["line"]) for item in literal["matches"]] == [
            ("root.py", 1),
            ("src/a.py", 2),
        ]
        regex = await tools["grep"](r"NEE[D]LE", "src", "**/*.py", True, False, "", 10)
        assert [item["path"] for item in regex["matches"]] == [
            "src/a.py",
            "src/nested/b.py",
        ]

        assert len(state.metrics.tool_calls) >= 8
        assert all(call.arguments == {} for call in state.metrics.tool_calls)
        assert SECRET_PATTERN not in repr(state.metrics)

    asyncio.run(scenario())


def test_cursor_is_query_and_snapshot_bound_and_scan_truncation_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        session = await workspace("overlay", "cursor")
        assert isinstance(session, OverlayWorkspaceSession)
        tools = await create_tools(
            FilesystemToolsetFactory(),
            session.reader_view(),
            WorkerState(),
            tmp_path,
            ["glob"],
        )
        first = await tools["glob"]("**/*", "", 1)
        cursor = first["nextCursor"]
        assert cursor and "/" not in cursor
        with pytest.raises(FilesystemToolError, match="cursor"):
            await tools["glob"]("**/*.py", cursor, 1)
        with pytest.raises(FilesystemToolError, match="cursor"):
            await tools["glob"]("**/*", cursor[:-1] + ("A" if cursor[-1] != "A" else "B"), 1)

        await session.write_text("root.py", "changed")
        with pytest.raises(FilesystemToolError, match="cursor"):
            await tools["glob"]("**/*", cursor, 1)

        monkeypatch.setattr(filesystem_module, "MAX_SCAN_PATHS", 2)
        empty = await tools["glob"]("does-not-match-*", "", 100)
        assert empty["matches"] == []
        assert empty["truncated"] and empty["nextCursor"]

    asyncio.run(scenario())


def test_invalid_paths_binary_reads_regex_and_visible_output_bounds(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await workspace("direct", "bounds")
        tools = await create_tools(
            FilesystemToolsetFactory(),
            session.reader_view(),
            WorkerState(),
            tmp_path,
            ["ls", "glob", "read_file", "grep"],
        )
        for operation in (
            lambda: tools["ls"]("../escape"),
            lambda: tools["glob"]("../**"),
            lambda: tools["read_file"]("../escape"),
            lambda: tools["grep"]("x", "../escape"),
        ):
            with pytest.raises(FilesystemToolError, match="path_invalid"):
                await operation()
        with pytest.raises(FilesystemToolError, match="binary"):
            await tools["read_file"]("image.bin")
        with pytest.raises(FilesystemToolError, match="search_invalid"):
            await tools["grep"]("[", regex=True)
        with pytest.raises(FilesystemToolError, match="limit"):
            await tools["ls"]("", "", 101)

        bounded = await tools["read_file"]("long.txt", 1, 1)
        assert bounded["returnedBytes"] <= filesystem_module.MAX_READ_BYTES
        assert bounded["truncated"] and bounded["lines"][0]["truncated"]

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_scoped_read_preserves_errors_and_does_not_snapshot_memory_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    async def scenario() -> None:
        session = await workspace(mode, f"scoped-{mode}")
        tools = await create_tools(
            FilesystemToolsetFactory(),
            session.reader_view(),
            WorkerState(),
            tmp_path,
            ["read_file"],
        )

        async def no_snapshot() -> None:
            pytest.fail("read_file should acquire one text, not a workspace snapshot")

        monkeypatch.setattr(session, "snapshot", no_snapshot)
        try:
            result = await tools["read_file"]("docs/readme.txt")
            assert [line["text"] for line in result["lines"]] == ["first", "second", "last"]
            for path, code in (
                ("src", "workspace_type_conflict"),
                ("missing", "workspace_not_found"),
                ("image.bin", "binary_file_unsupported"),
            ):
                with pytest.raises(FilesystemToolError) as failure:
                    await tools["read_file"](path)
                assert failure.value.code == code

            async def failing_read(_: str) -> str:
                raise OSError(f"private host path {tmp_path} and {SECRET_PATTERN}")

            monkeypatch.setattr(session, "read_text", failing_read)
            with pytest.raises(FilesystemToolError) as failure:
                await tools["read_file"]("docs/readme.txt")
            assert failure.value.code == "workspace_unavailable"
            assert str(tmp_path) not in str(failure.value)
            assert SECRET_PATTERN not in str(failure.value)
        finally:
            await tools["read_file"].close()
            await session.close()

    asyncio.run(scenario())


async def workspace(mode: str, name: str) -> DirectWorkspaceSession:
    provider = MemoryWorkspaceProvider(
        WorkspaceSettings(storage="memory", limits=workspace_limits())
    )
    storage = await provider.create(name)
    arguments = dict(
        storage=storage,
        content_root=f"{storage.root}/run_workdir",
        limits=workspace_limits(),
        directories={"src", "src/nested", "docs", "empty"},
        text_files={
            "root.py": f"needle {SECRET_PATTERN}\n",
            "src/a.py": "first\nneedle here\n",
            "src/nested/b.py": "NEEDLE nested\n",
            "docs/readme.txt": "first\r\nsecond\nlast",
            "long.txt": "x" * (filesystem_module.MAX_READ_BYTES + 100),
        },
        binary_paths={"image.bin"},
        stored_binary_paths=set(),
    )
    if mode == "overlay":
        return OverlayWorkspaceSession(**arguments)
    return DirectWorkspaceSession(mode="direct", **arguments)


async def create_tools(
    factory: FilesystemToolsetFactory,
    project_workspace: Any,
    state: WorkerState,
    tmp_path: Path,
    selected: list[str],
) -> dict[str, Any]:
    result = await factory.create_selected(
        selected=selected,
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="reader",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=state,
        project_workspace=project_workspace,
    )
    return dict(result)


def workspace_limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=1 << 20,
        max_managed_text_bytes=1 << 20,
        max_file_bytes=1 << 20,
    )
