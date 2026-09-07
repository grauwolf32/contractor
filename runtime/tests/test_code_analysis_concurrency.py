from __future__ import annotations

import asyncio
import threading
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest
from test_projectfs_zip import archive, workspace_inputs
from test_projectfs_zip import settings as workspace_settings

import contractor_runtime.toolsets.code_analysis.languages as code_analysis_languages
import contractor_runtime.toolsets.code_analysis.tools as code_analysis
import contractor_runtime.toolsets.code_analysis.trailmark_host as host_module
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    hydrate_workspace,
)
from contractor_runtime.projectfs.storage import ManagedWorkspaceTree, WorkspaceSnapshot
from contractor_runtime.telemetry.metrics import MetricsState
from contractor_runtime.toolsets.code_analysis.tools import (
    GRAPH_TOOLS,
    CodeAnalysisError,
    CodeAnalysisToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_edit_during_shallow_parse_keeps_each_call_on_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    started = threading.Event()
    release = threading.Event()
    original = code_analysis_languages.parse_symbols

    def blocked_parse(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(code_analysis_languages, "parse_symbols", blocked_parse)

    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [("source", "", archive({"app.py": b"def before_edit():\n    return 1\n"}))]
        )
        spec.mode = mode  # type: ignore[assignment]
        provider = MemoryWorkspaceProvider(workspace_settings("memory"))
        workspace = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id=f"shallow-race-{mode}",
            timeout_seconds=5,
        )
        tools = await _shallow_tools(workspace.reader_view(), tmp_path)

        first_call = asyncio.create_task(tools["list_symbols"]())
        assert await asyncio.to_thread(started.wait, 2)
        await workspace.write_text("app.py", "def after_edit():\n    return 2\n")
        release.set()

        first = await first_call
        assert [item["name"] for item in first["items"]] == ["before_edit"]
        second = await tools["list_symbols"]()
        assert [item["name"] for item in second["items"]] == ["after_edit"]

        await _close(tools)
        await workspace.close()
        await provider.cleanup(workspace.storage)

    asyncio.run(scenario())


def test_cancelled_parse_keeps_owner_until_thread_finishes_and_cannot_repopulate_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    original = code_analysis_languages.parse_symbols

    def blocked_parse(*args: Any, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(code_analysis_languages, "parse_symbols", blocked_parse)

    async def scenario() -> None:
        tools = await _shallow_tools(
            MutableReader({"private.py": "def private_symbol():\n    return 1\n"}),
            tmp_path,
        )
        session = tools["list_symbols"]._session
        metrics = tools["list_symbols"]._metrics
        operation = asyncio.create_task(tools["list_symbols"]())
        assert await asyncio.to_thread(started.wait, 2)

        operation.cancel()
        close = asyncio.create_task(tools["list_symbols"].close())
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.005)
                ticks += 1

        await ticker()
        assert ticks == 10
        assert not operation.done()
        assert not close.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await close
        await asyncio.sleep(0.01)
        assert session._file_cache == {}
        assert session._parsers == {}
        assert set(session._cursor_key) == {0}
        cancelled = metrics.snapshot()["toolCalls"][-1]
        assert cancelled["error"] == {
            "code": "code_analysis_cancelled",
            "message": "Tool list_symbols failed (CodeAnalysisError)",
            "retryable": True,
        }
        assert "private_symbol" not in repr(metrics.snapshot())

    asyncio.run(scenario())


def test_compact_cache_has_independent_file_and_symbol_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(code_analysis, "MAX_COMPACT_CACHE_FILES", 1)
        tools = await _shallow_tools(
            MutableReader(
                {
                    "a.py": "def first():\n    return 1\n",
                    "b.py": "def second():\n    return 2\n",
                }
            ),
            tmp_path,
        )
        result = await tools["list_symbols"]()
        assert [item["name"] for item in result["items"]] == ["first", "second"]
        session = tools["list_symbols"]._session
        assert len(session._file_cache) == 1
        assert session._cached_symbols == 1
        await _close(tools)

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["direct", "overlay"])
@pytest.mark.parametrize("winner", ["analysis", "edit"])
def test_graph_build_and_edit_have_deterministic_snapshot_winners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winner: str,
    mode: str,
) -> None:
    async def scenario() -> None:
        root = tmp_path / f"{winner}-{mode}"
        root.mkdir()
        factory = CodeAnalysisToolsetFactory(
            workspace_storage="local",
            graph_probe_root=root / "probe",
        )
        assert await factory.probe() >= GRAPH_TOOLS
        spec, artifact_reader = workspace_inputs(
            [("source", "", archive({"app.py": b"def before_edit():\n    return 1\n"}))]
        )
        spec.mode = mode  # type: ignore[assignment]
        provider = LocalWorkspaceProvider(workspace_settings("local", root / "workspaces"))
        workspace = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=artifact_reader,
            allocation_id=f"graph-race-{winner}-{mode}",
            timeout_seconds=5,
        )

        if winner == "edit":
            await workspace.write_text("app.py", "def after_edit():\n    return 2\n")
            tools = await _graph_tools(factory, workspace.reader_view(), root)
            fresh = await tools["find_symbol"]("after_edit")
            assert [item["name"] for item in fresh["items"]] == ["after_edit"]
            await _close(tools)
            await workspace.close()
            await provider.cleanup(workspace.storage)
            return

        started = threading.Event()
        release = threading.Event()
        original = host_module._materialize_snapshot

        def blocked_materialize(
            snapshot: WorkspaceSnapshot, scratch_root: Path
        ) -> host_module._PreparedMirror:
            started.set()
            assert release.wait(timeout=5)
            return original(snapshot, scratch_root)

        monkeypatch.setattr(host_module, "_materialize_snapshot", blocked_materialize)
        tools = await _graph_tools(factory, workspace.reader_view(), root)
        first_call = asyncio.create_task(tools["find_symbol"]("before_edit"))
        assert await asyncio.to_thread(started.wait, 2)
        await workspace.write_text("app.py", "def after_edit():\n    return 2\n")
        release.set()

        first = await first_call
        assert [item["name"] for item in first["items"]] == ["before_edit"]
        stale_id = first["items"][0]["symbolId"]
        fresh = await tools["find_symbol"]("after_edit")
        assert [item["name"] for item in fresh["items"]] == ["after_edit"]
        with pytest.raises(CodeAnalysisError) as stale:
            await tools["find_callers"](stale_id)
        assert stale.value.code == "code_analysis_stale_symbol"
        await _close(tools)
        assert list((root / "allocation").iterdir()) == []
        await workspace.close()
        await provider.cleanup(workspace.storage)

    asyncio.run(scenario())


class MutableReader:
    def __init__(self, files: dict[str, str]) -> None:
        self._tree = ManagedWorkspaceTree(
            directories=_directories(files),
            text_files=dict(files),
        )

    async def snapshot(self) -> WorkspaceSnapshot:
        return self._tree.snapshot()

    async def read_text(self, path: str) -> str:
        return self._tree.text_files[path]

    def write(self, path: str, text: str) -> None:
        self._tree.directories.update(_directories({path: text}))
        self._tree.text_files[path] = text


async def _shallow_tools(reader: Any, root: Path) -> dict[str, Any]:
    return dict(
        await CodeAnalysisToolsetFactory().create_selected(
            selected=("list_symbols", "search_def"),
            allocation_id="allocation-shallow-race",
            run_id="run-shallow-race",
            namespace="analysis",
            runtime_settings=_runtime_settings(),
            workspace=AllocationWorkspace(root=root, path=root),
            state=SimpleNamespace(metrics=MetricsState()),
            project_workspace=reader,
        )
    )


async def _graph_tools(
    factory: CodeAnalysisToolsetFactory,
    reader: Any,
    root: Path,
) -> dict[str, Any]:
    scratch = root / "allocation"
    scratch.mkdir()
    return dict(
        await factory.create_selected(
            selected=("find_symbol", "find_callers"),
            allocation_id=f"allocation-{root.name}",
            run_id="run-graph-race",
            namespace="analysis",
            runtime_settings=_runtime_settings(),
            workspace=AllocationWorkspace(root=root, path=scratch),
            state=SimpleNamespace(metrics=MetricsState()),
            project_workspace=reader,
        )
    )


async def _close(tools: dict[str, Any]) -> None:
    for tool in reversed(tuple(tools.values())):
        await tool.close()


def _runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llm_gateway_url="https://llm.example/v1",
        llm_gateway_token="temporary-token",
        artifact_api_url="https://server.example/private/v1/artifacts",
        request_timeout_seconds=10,
    )


def _directories(files: dict[str, str]) -> set[str]:
    result: set[str] = set()
    for path in files:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            result.add(str(parent))
            parent = parent.parent
    return result
