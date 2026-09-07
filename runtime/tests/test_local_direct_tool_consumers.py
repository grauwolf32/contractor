"""Real external writes reach narrow tools, derived analysis and evidence."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from test_code_analysis_graph import _close as close_graph
from test_code_analysis_graph import _tools as graph_tools
from test_code_analysis_shallow import _tools as shallow_tools
from test_edit_files_toolset import make_tools as edit_tools
from test_filesystem_observations import Context, call
from test_filesystem_toolset import create_tools
from test_openapi_toolset import MemoryArtifactClient, clean_vacuum
from test_openapi_toolset import make_tools as openapi_tools
from test_projectfs_local_direct import workspace as local_workspace
from test_taint_annotations import make_tools as annotation_tools

import contractor_runtime.toolsets.openapi.tools as openapi_module
import contractor_runtime.toolsets.taint_annotations.tools as annotations_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.settings import WorkspaceLimits
from contractor_runtime.toolsets.code_analysis.tools import CodeAnalysisError
from contractor_runtime.toolsets.filesystem.tools import (
    FilesystemToolError,
    FilesystemToolsetFactory,
)
from contractor_runtime.toolsets.taint_annotations.tools import TaintAnnotationError
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker.observations import lean_workspace_summary
from contractor_runtime.worker.state import WorkerStateStore


def test_filesystem_and_edits_see_external_create_write_rename_delete(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            state = WorkerState()
            tools = await create_tools(
                FilesystemToolsetFactory(),
                session.reader_view(),
                state,
                tmp_path,
                ["ls", "glob", "grep", "read_file"],
            )
            edits = await edit_tools(
                tmp_path, session.writer_view(), state, ["edit", "append_file"]
            )
            try:
                (root / "src/a.txt").write_bytes(b"external\r\nsecond\r\n")
                (root / "created").mkdir()
                (root / "created/new.txt").write_bytes(b"new external\n")
                listing = await tools["ls"]("created")
                assert [item["path"] for item in listing["entries"]] == ["created/new.txt"]
                matched = await tools["glob"]("**/*.txt")
                assert [item["path"] for item in matched["matches"]] == [
                    "created/new.txt",
                    "src/a.txt",
                ]
                found = await tools["grep"]("external")
                assert {item["path"] for item in found["matches"]} == {
                    "created/new.txt",
                    "src/a.txt",
                }
                read = await tools["read_file"]("src/a.txt", 2, 1)
                assert read["lines"][0]["text"] == "second"
                await edits["edit"]("src/a.txt", "external", "edited")
                assert (root / "src/a.txt").read_bytes() == b"edited\r\nsecond\r\n"
                (root / "src/a.txt").write_bytes(b"latest\r\n")
                await edits["append_file"]("src/a.txt", "appended")
                assert (root / "src/a.txt").read_bytes().startswith(b"latest\r\n")
                (root / "created/new.txt").rename(root / "renamed.txt")
                (root / "src/a.txt").unlink()
                current = await tools["glob"]("**/*.txt")
                assert [item["path"] for item in current["matches"]] == ["renamed.txt"]
                with pytest.raises(FilesystemToolError, match="workspace_not_found"):
                    await tools["read_file"]("src/a.txt")
                assert str(tmp_path) not in json.dumps([listing, matched, found, read, current])
                assert "external" not in state.metrics.tool_calls[-1].model_dump_json()
            finally:
                for tool in (*tools.values(), *edits.values()):
                    await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("tool_name", ["ls", "glob", "grep"])
def test_external_same_size_restored_mtime_invalidates_filesystem_cursor(
    tmp_path: Path, tool_name: str
) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            (root / "first.txt").write_bytes(b"needle one\n")
            (root / "second.txt").write_bytes(b"needle two\n")
            tools = await create_tools(
                FilesystemToolsetFactory(),
                session.reader_view(),
                WorkerState(),
                tmp_path,
                [tool_name],
            )
            tool = tools[tool_name]
            argument = {"ls": "", "glob": "**/*", "grep": "needle"}[tool_name]
            try:
                first = await tool(argument, limit=1)
                assert first["nextCursor"]
                path = root / "first.txt"
                before = path.stat()
                path.write_bytes(b"needle new\n")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                with pytest.raises(FilesystemToolError, match="workspace_cursor_invalid"):
                    await tool(argument, cursor=first["nextCursor"], limit=1)
                assert await tool(argument, limit=1)
            finally:
                await tool.close()

    asyncio.run(scenario())


def test_read_file_stays_scoped_when_an_external_file_exceeds_workspace_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        limits = WorkspaceLimits(
            max_files=20, max_file_bytes=64, max_expanded_bytes=100, max_managed_text_bytes=100
        )
        async with local_workspace(tmp_path, limits=limits) as (session, root):
            tools = await create_tools(
                FilesystemToolsetFactory(),
                session.reader_view(),
                WorkerState(),
                tmp_path,
                ["read_file", "ls"],
            )
            try:
                (root / "oversized").write_bytes(b"x" * 65)
                with pytest.raises(FilesystemToolError, match="workspace_limit_exceeded"):
                    await tools["ls"]()

                async def no_snapshot() -> None:
                    pytest.fail("read_file must not acquire a whole-workspace snapshot")

                monkeypatch.setattr(session, "snapshot", no_snapshot)
                result = await tools["read_file"]("src/a.txt")
                assert result["lines"][0]["text"] == "source"
                with pytest.raises(FilesystemToolError, match="workspace_limit_exceeded"):
                    await tools["read_file"]("oversized")
                with pytest.raises(FilesystemToolError, match="binary_file_unsupported"):
                    await tools["read_file"]("binary")
                with pytest.raises(FilesystemToolError, match="workspace_not_found"):
                    await tools["read_file"]("missing")
                with pytest.raises(FilesystemToolError, match="workspace_type_conflict"):
                    await tools["read_file"]("src")
                await tools["read_file"].close()
                with pytest.raises(FilesystemToolError, match="workspace_not_found"):
                    await tools["read_file"]("src/a.txt")
            finally:
                for tool in tools.values():
                    await tool.close()

    asyncio.run(scenario())


def test_shallow_analysis_checks_disk_digest_before_using_derived_cache(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            (root / "a.py").write_bytes(b"def first():\n    pass\n")
            (root / "b.py").write_bytes(b"def other():\n    pass\n")
            tools, _ = await shallow_tools(session.reader_view(), tmp_path)
            try:
                initial = await tools["list_symbols"](limit=1)
                assert initial["nextCursor"]
                assert tools["list_symbols"]._session._file_cache
                path = root / "a.py"
                before = path.stat()
                path.write_bytes(b"def fresh():\n    pass\n")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                with pytest.raises(CodeAnalysisError) as stale:
                    await tools["list_symbols"](cursor=initial["nextCursor"], limit=1)
                assert stale.value.code == "code_analysis_workspace_changed"
                assert tools["list_symbols"]._session._file_cache == {}
                fresh = await tools["search_def"]("fresh")
                assert [item["name"] for item in fresh["items"]] == ["fresh"]
                assert (await tools["search_def"]("first"))["items"] == []
                path.unlink()
                assert (await tools["search_def"]("fresh"))["items"] == []
            finally:
                for tool in tools.values():
                    await tool.close()

    asyncio.run(scenario())


def test_external_edit_retires_trailmark_graph_cursor_symbol_and_mirror(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            (root / "a.py").write_bytes(b"def target():\n    return 1\n")
            (root / "b.py").write_bytes(b"def target():\n    return 2\n")
            tools, _, scratch = await graph_tools(tmp_path / "graph", session.reader_view())
            try:
                first = await tools["find_symbol"]("target", limit=1)
                old_id, cursor = first["items"][0]["symbolId"], first["nextCursor"]
                assert cursor and list(scratch.glob("code-analysis-mirror-*"))
                (root / "a.py").write_bytes(b"def replacement():\n    return 3\n")
                with pytest.raises(CodeAnalysisError) as stale:
                    await tools["find_callers"](old_id)
                assert stale.value.code == "code_analysis_stale_symbol"
                assert list(scratch.iterdir()) == []
                with pytest.raises(CodeAnalysisError) as stale_page:
                    await tools["find_symbol"]("target", cursor=cursor, limit=1)
                assert stale_page.value.code == "code_analysis_workspace_changed"
                fresh = await tools["find_symbol"]("replacement")
                assert fresh["observedTotal"] == 1
                assert fresh["items"][0]["symbolId"] != old_id
                assert list(scratch.glob("code-analysis-mirror-*"))
            finally:
                await close_graph(tools)
            assert list(scratch.iterdir()) == []

    asyncio.run(scenario())


def test_observation_scope_and_read_coverage_acquire_current_disk_each_invocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            tools = await create_tools(
                FilesystemToolsetFactory(),
                session.reader_view(),
                WorkerState(),
                tmp_path,
                ["read_file"],
            )
            state = WorkerStateStore()
            plugin = WorkerInstrumentationPlugin(
                state=state,
                budget=lambda: None,
                instrumentation=None,
                model_alias="model",
                observe_artifacts=lambda _owner, _cursor: None,
                workspace_observation_source=session.reader_view(),
            )
            previous_digest = None
            try:
                for index in range(2):
                    if index:
                        (root / "src/a.txt").unlink()
                        (root / "new.py").write_bytes(b"new external source\n")
                    path = "new.py" if index else "src/a.txt"
                    invocation = f"disk-observation-{index}"
                    plugin.prepare_invocation(invocation_id=invocation, subtask_id="1")
                    await plugin.before_run_callback(invocation_context=Context(invocation))
                    result = await tools["read_file"](path)
                    await call(
                        plugin, invocation, "read_file", tools["read_file"], {"path": path}, result
                    )
                    completed = await plugin.complete_invocation(
                        invocation_id=invocation, phase="succeeded"
                    )
                    assert completed is not None
                    observation = completed["lastCompletedInvocation"]["workspace"]
                    assert observation["scopePaths"] == [path]
                    assert (
                        observation["workspaceDigest"]
                        == (await session.observation_metadata()).digest
                    )
                    assert observation["workspaceDigest"] != previous_digest
                    previous_digest = observation["workspaceDigest"]
                    summary, truncated = lean_workspace_summary(observation)
                    assert summary is not None and not truncated
                    assert summary.read_files == 1 and summary.unread_files == 0
                    assert "new external source" not in json.dumps(observation)
            finally:
                await tools["read_file"].close()

    asyncio.run(scenario())


def test_annotation_reads_external_definition_and_never_restores_missing_source(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            (root / "app.py").write_bytes(b"def original(req):\n    return req\n")
            tools, _ = await annotation_tools(tmp_path, session.writer_view())
            try:
                (root / "app.py").write_bytes(b"def changed(req):\r\n    return req\r\n")
                result = await tools["annotate_trace"]("app.py", "changed", target="test")
                updated = (root / "app.py").read_bytes()
                assert result["changed"]
                assert b"# @trace target=test\r\ndef changed(req):\r\n" in updated
                assert b"original" not in updated
                (root / "app.py").unlink()
                with pytest.raises(TaintAnnotationError):
                    await tools["annotate_trace"]("app.py", "changed", target="test")
                assert not (root / "app.py").exists()
            finally:
                for tool in tools.values():
                    await tool.close()

    asyncio.run(scenario())


def test_openapi_evidence_uses_current_project_not_stale_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            scratch_source = tmp_path / "source"
            scratch_source.mkdir()
            (scratch_source / "app.py").write_bytes(b"stale scratch")
            (root / "app.py").write_bytes(b"def handler(): pass\n")
            tools = await openapi_tools(
                tmp_path,
                MemoryArtifactClient(),
                WorkerState(),
                namespace="openapi",
                project_workspace=session.reader_view(),
            )
            monkeypatch.setattr(openapi_module, "_run_vacuum", clean_vacuum)
            try:
                await tools["initialize_openapi"](title="Current disk")
                await tools["upsert_openapi_component"](
                    "schemas", "Response", {"type": "object"}, ["app.py"]
                )
                assert (await tools["validate_openapi"]())["valid"]
                (root / "app.py").rename(root / "renamed.py")
                with pytest.raises(ValueError, match="source evidence file does not exist"):
                    await tools["upsert_openapi_component"](
                        "schemas", "Response", {"type": "object"}, ["app.py"]
                    )
                validation = await tools["validate_openapi"]()
                assert not validation["valid"]
                assert any("does not exist" in error for error in validation["structuralErrors"])
                await tools["upsert_openapi_component"](
                    "schemas", "Response", {"type": "object"}, ["renamed.py"]
                )
                assert (await tools["validate_openapi"]())["valid"]
                assert str(tmp_path) not in json.dumps(validation)
            finally:
                for tool in tools.values():
                    await tool.close()

    asyncio.run(scenario())


def test_annotation_compare_rejects_a_real_external_write_during_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        async with local_workspace(tmp_path) as (session, root):
            source = b"def handler(req):\n    return req\n"
            (root / "app.py").write_bytes(source)
            tools, _ = await annotation_tools(tmp_path, session.writer_view())
            parse = annotations_module._parse_target_file

            def competing_parse(*args: object, **kwargs: object) -> object:
                parsed = parse(*args, **kwargs)
                (root / "app.py").write_bytes(source + b"# external\n")
                return parsed

            monkeypatch.setattr(annotations_module, "_parse_target_file", competing_parse)
            try:
                with pytest.raises(TaintAnnotationError) as changed:
                    await tools["annotate_trace"]("app.py", "handler")
                assert changed.value.code == "taint_annotation_workspace_changed"
                assert changed.value.retryable
                assert (root / "app.py").read_bytes() == source + b"# external\n"
                # A rejected compare is a validation failure, not an uncertain
                # mutation: subsequent scoped reads remain available.
                assert await session.read_text("app.py") == (source + b"# external\n").decode()
            finally:
                for tool in tools.values():
                    await tool.close()

    asyncio.run(scenario())
