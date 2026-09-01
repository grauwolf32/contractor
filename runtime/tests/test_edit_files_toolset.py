from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.factories import built_in_factories
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    hydrate_workspace,
)
from contractor_runtime.toolsets.edit_files import EditFilesToolsetFactory
from contractor_runtime.toolsets.filesystem import FilesystemToolError
from contractor_runtime.workspace import AllocationWorkspace

SECRET_CONTENT = "recognizable-edit-content-secret"


def test_factory_constructs_only_selected_writer_tools(tmp_path: Path) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "memory", "direct", "factory")
        view = session.writer_view()
        assert not hasattr(view, "storage")
        selected = await make_tools(tmp_path, view, WorkerState(), ["write_file", "edit"])
        assert set(selected) == {"write_file", "edit"}
        with pytest.raises(ValueError, match="unknown selected tools"):
            await make_tools(tmp_path, view, WorkerState(), ["shell"])
        with pytest.raises(FilesystemToolError, match="workspace_required"):
            await make_tools(tmp_path, None, WorkerState(), ["write_file"])
        await provider.cleanup(session.storage)

    asyncio.run(scenario())
    assert built_in_factories(tmp_path).toolsets["edit-files@1"].exported_tools == {
        "write_file",
        "append_file",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "insert_line",
        "edit",
        "replace_range",
    }


def test_failed_edits_and_concurrent_updates_leave_no_partial_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "memory", "overlay", "failures")
        tools = await make_tools(
            tmp_path,
            session.writer_view(),
            WorkerState(),
            sorted(EditFilesToolsetFactory.exported_tools),
        )
        baseline = await session.snapshot()
        for operation in (
            lambda: tools["edit"]("duplicate.txt", "same", "changed"),
            lambda: tools["edit"]("duplicate.txt", "missing", "changed", True),
            lambda: tools["replace_range"]("lf.txt", 4, 2, "bad"),
            lambda: tools["insert_line"]("lf.txt", 999, "bad"),
            lambda: tools["write_file"]("too-large.txt", "x" * (session.limits.max_file_bytes + 1)),
            lambda: tools["mv"]("tree", "tree/child/moved"),
            lambda: tools["rm"]("tree", False),
        ):
            with pytest.raises(FilesystemToolError):
                await operation()
            assert await session.snapshot() == baseline

        for operation in (
            lambda: tools["write_file"]("image.bin", "text"),
            lambda: tools["cp"]("image.bin", "image-copy.bin"),
            lambda: tools["rm"]("image.bin"),
        ):
            with pytest.raises(FilesystemToolError, match="binary"):
                await operation()
            assert await session.snapshot() == baseline

        def cancelled(_: str) -> str:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await session.writer_view().update_text("lf.txt", cancelled)
        assert await session.snapshot() == baseline

        await asyncio.gather(
            tools["append_file"]("lf.txt", "parallel-a"),
            tools["append_file"]("lf.txt", "parallel-b"),
        )
        content = await session.read_text("lf.txt")
        assert content.count("parallel-a") == 1
        assert content.count("parallel-b") == 1
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


async def hydrated_workspace(
    tmp_path: Path,
    storage: str,
    mode: str,
    allocation_id: str,
) -> tuple[Any, Any]:
    spec, reader = workspace_inputs(
        [
            (
                "source",
                "",
                archive(
                    {
                        "crlf.txt": b"one\r\ntwo\r\nthree",
                        "lf.txt": f"alpha\nbeta\n{SECRET_CONTENT}\n".encode(),
                        "duplicate.txt": b"same same\n",
                        "tree/child/file.txt": b"tree\n",
                        "empty/": None,
                        "image.bin": b"\x00\xffbinary",
                    }
                ),
            )
        ]
    )
    spec.mode = mode  # type: ignore[assignment]
    provider = (
        LocalWorkspaceProvider(settings("local", tmp_path / f"local-{allocation_id}"))
        if storage == "local"
        else MemoryWorkspaceProvider(settings("memory"))
    )
    session = await hydrate_workspace(
        provider=provider,
        spec=spec,
        artifact_reader=reader,
        allocation_id=allocation_id,
        timeout_seconds=5,
    )
    return session, provider


async def make_tools(
    tmp_path: Path,
    writer: Any,
    state: WorkerState,
    selected: list[str],
) -> dict[str, Any]:
    factory = EditFilesToolsetFactory()
    result = await factory.create_selected(
        selected=selected,
        allocation_id="allocation-edit",
        run_id="run-edit",
        namespace="editor",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://llm.example/v1",
            llmGatewayToken="test-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=30,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=tmp_path),
        state=state,
        project_workspace=writer,
    )
    return dict(result)
