from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import scripted_model, text_result, tool_call
from test_adk_runtime import create_runtime, stage_request
from test_filesystem_toolset import create_tools, workspace

from contractor_runtime.allocation import WorkerState
from contractor_runtime.projectfs import WorkspaceObservationMetadata
from contractor_runtime.toolsets.code_analysis.tools import SearchDefinitionTool
from contractor_runtime.toolsets.edit_files.tools import EditTextTool
from contractor_runtime.toolsets.filesystem.tools import (
    FilesystemToolsetFactory,
    ReadWorkspaceFileTool,
)
from contractor_runtime.toolsets.taint_annotations.tools import AnnotateTraceTool
from contractor_runtime.toolsets.workspace_changes.tools import ChangedPathsTool
from contractor_runtime.worker.instrumentation import WorkerInstrumentationPlugin
from contractor_runtime.worker.observations import (
    WorkspaceToolObservation,
    annotation_tool_observation,
    edit_tool_observation,
    filesystem_tool_observation,
    lean_workspace_summary,
    validate_workspace_observation,
    workspace_changes_observation,
)
from contractor_runtime.worker.state import WorkerStateStore

SECRET = "recognizable-observation-content-secret"
REVISION = "recognizable-observation-revision"


@dataclass
class Session:
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    invocation_id: str
    session: Session = field(default_factory=Session)


@dataclass
class Tool:
    name: str
    func: Any


class MetadataSource:
    def __init__(self) -> None:
        self.metadata = WorkspaceObservationMetadata(
            digest="sha256:" + "1" * 64,
            managed_text_paths=("a.py", "b.py", "c.py"),
        )

    async def observation_metadata(self) -> WorkspaceObservationMetadata:
        return self.metadata


class Extractor:
    def __init__(self, operation: str) -> None:
        self.operation = operation

    def contractor_observation(
        self,
        tool_args: dict[str, Any],
        result: Any,
    ) -> WorkspaceToolObservation | None:
        if self.operation in {"ls", "glob", "read_file", "grep"}:
            return filesystem_tool_observation(self.operation, tool_args, result)
        if self.operation in {
            "write_file", "append_file", "rm", "insert_line", "edit", "replace_range", "cp", "mv"
        }:
            return edit_tool_observation(self.operation, tool_args, result)
        if self.operation == "annotate_trace":
            return annotation_tool_observation(tool_args, result)
        if self.operation == "changed_paths":
            return workspace_changes_observation(self.operation, tool_args, result)
        return None


class BrokenExtractor:
    def contractor_observation(
        self,
        _tool_args: dict[str, Any],
        _result: Any,
    ) -> WorkspaceToolObservation:
        raise RuntimeError(f"extractor must not leak {SECRET}")


class AnalyzerWithoutWorkspaceExtractor:
    pass


def test_only_reviewed_workspace_tools_expose_observation_extractors() -> None:
    read = object.__new__(ReadWorkspaceFileTool)
    edit = object.__new__(EditTextTool)
    annotation = object.__new__(AnnotateTraceTool)
    changes = object.__new__(ChangedPathsTool)
    analyzer = object.__new__(SearchDefinitionTool)

    assert read.contractor_observation({}, {"path": "a.py"}) == WorkspaceToolObservation(
        read=("a.py",)
    )
    assert edit.contractor_observation(
        {"path": "b.py", "old": SECRET, "new": "safe"},
        {"changed": True},
    ) == WorkspaceToolObservation(modified=("b.py",))
    assert annotation.contractor_observation(
        {"path": "c.py", "symbol": SECRET},
        {"path": "c.py", "changed": True},
    ) == WorkspaceToolObservation(modified=("c.py",))
    assert changes.contractor_observation(
        {},
        {"changes": [{"path": "d.py", "change": "created"}]},
    ) == WorkspaceToolObservation(modified=("d.py",))
    assert not hasattr(analyzer, "contractor_observation")


@pytest.mark.parametrize(
    ("operation", "arguments", "expected"),
    [
        ("write_file", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("append_file", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("rm", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("insert_line", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("edit", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("replace_range", {"path": "cafe\u0301.py"}, ("café.py",)),
        ("cp", {"source": "a.py", "destination": "cafe\u0301.py"}, ("café.py",)),
        (
            "mv",
            {"source": "cafe\u0301.py", "destination": "re\u0301sume\u0301.py"},
            ("café.py", "résumé.py"),
        ),
    ],
)
def test_successful_edits_observe_canonical_argument_paths(
    operation: str, arguments: dict[str, str], expected: tuple[str, ...]
) -> None:
    assert edit_tool_observation(operation, arguments, {"changed": True}) == (
        WorkspaceToolObservation(modified=expected)
    )
    assert edit_tool_observation(operation, arguments, {"changed": False}) == (
        WorkspaceToolObservation()
    )


@pytest.mark.parametrize("path", ["", "../escape.py", "/absolute.py", "a//b.py", "C:\\bad.py"])
def test_edit_observation_rejects_invalid_argument_paths(path: str) -> None:
    with pytest.raises(ValueError, match="workspace observation path is invalid"):
        edit_tool_observation("write_file", {"path": path}, {"changed": True})


def test_decomposed_edit_path_keeps_workspace_observation_complete() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        source = MetadataSource()
        source.metadata = WorkspaceObservationMetadata(
            digest="sha256:" + "1" * 64,
            managed_text_paths=("a.py", "b.py", "café.py"),
        )
        plugin = WorkerInstrumentationPlugin(
            state=state,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            workspace_observation_source=source,
        )
        invocation_id = "worker-decomposed-edit"
        plugin.prepare_invocation(invocation_id=invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=Context(invocation_id))
        await call(
            plugin,
            invocation_id,
            "write_file",
            Extractor("write_file"),
            {"path": "cafe\u0301.py", "content": SECRET},
            {"changed": True},
        )

        completed = await plugin.complete_invocation(invocation_id=invocation_id, phase="succeeded")
        assert completed is not None
        workspace = completed["lastCompletedInvocation"]["workspace"]
        assert workspace["detailComplete"] is True
        assert [(item["path"], item["mutationCalls"]) for item in workspace["interactions"]] == [
            ("café.py", 1)
        ]
        summary, truncated = lean_workspace_summary(workspace)
        assert summary is not None
        assert summary.modified_files == 1
        assert summary.unread_files == 3
        assert summary.detail_complete is True
        assert truncated is False
        assert plugin.projection_failed is False

        invalid = dict(workspace)
        invalid["interactions"] = [dict(workspace["interactions"][0], path="cafe\u0301.py")]
        with pytest.raises(ValueError, match="workspace observation path is not normalized"):
            validate_workspace_observation(invalid)

    asyncio.run(scenario())


def test_plugin_records_only_successful_visible_workspace_facts() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        source = MetadataSource()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            workspace_observation_source=source,
        )
        invocation_id = "worker-observations"
        context = Context(invocation_id)
        plugin.prepare_invocation(invocation_id=invocation_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=context)

        await call(
            plugin,
            invocation_id,
            "glob",
            Extractor("glob"),
            {"pattern": f"**/*{SECRET}*"},
            {
                "matches": [
                    {"path": "a.py", "type": "file", "size": 12},
                    {"path": "docs", "type": "directory", "size": None},
                    {"path": "image.bin", "type": "binary", "size": None},
                ]
            },
        )
        for _ in range(2):
            await call(
                plugin,
                invocation_id,
                "read_file",
                Extractor("read_file"),
                {"path": "a.py", "revision": REVISION},
                {"path": "a.py", "lines": [{"text": SECRET}]},
            )
        await call(
            plugin,
            invocation_id,
            "grep",
            Extractor("grep"),
            {"pattern": SECRET},
            {
                "matches": [
                    {"path": "b.py", "line": 1, "excerpt": SECRET},
                    {"path": "b.py", "line": 2, "excerpt": SECRET},
                ]
            },
        )
        await call(
            plugin,
            invocation_id,
            "edit",
            Extractor("edit"),
            {"path": "c.py", "old": SECRET, "new": "safe"},
            {"changed": True},
        )
        await call(
            plugin,
            invocation_id,
            "annotate_trace",
            Extractor("annotate_trace"),
            {"path": "b.py", "symbol": SECRET},
            {"path": "b.py", "changed": True},
        )
        await call(
            plugin,
            invocation_id,
            "changed_paths",
            Extractor("changed_paths"),
            {},
            {"changes": [{"path": "c.py", "change": "modified"}]},
        )
        await call(
            plugin,
            invocation_id,
            "read_file",
            Extractor("read_file"),
            {"path": "secret.py"},
            {"ok": False, "error": {"code": "workspace_not_found"}},
        )
        await call(
            plugin,
            invocation_id,
            "search_def",
            AnalyzerWithoutWorkspaceExtractor(),
            {"symbol": SECRET},
            {
                "items": [{"path": "not-model-read.py", "source": SECRET}],
                "coverage": {"analyzedFiles": 999},
            },
        )

        completed = await plugin.complete_invocation(
            invocation_id=invocation_id,
            phase="succeeded",
        )
        assert completed is not None
        workspace = completed["lastCompletedInvocation"]["workspace"]
        by_path = {item["path"]: item for item in workspace["interactions"]}
        assert set(by_path) == {"a.py", "b.py", "c.py"}
        assert by_path["a.py"]["discoveryCalls"] == 1
        assert by_path["a.py"]["readCalls"] == 2
        assert by_path["b.py"]["matchCalls"] == 1
        assert by_path["b.py"]["readCalls"] == 0
        assert by_path["b.py"]["mutationCalls"] == 1
        assert by_path["c.py"]["mutationCalls"] == 2
        summary, truncated = lean_workspace_summary(workspace)
        assert summary is not None
        assert summary.read_files == 1
        assert summary.matched_files == 1
        assert summary.modified_files == 2
        assert summary.unread_files == 2
        assert truncated is False

        retained = json.dumps(completed)
        assert SECRET not in retained
        assert REVISION not in retained
        assert "not-model-read.py" not in retained

    asyncio.run(scenario())


def test_extractor_failure_is_non_fatal_and_sequential_invocations_reset() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        source = MetadataSource()
        plugin = WorkerInstrumentationPlugin(
            state=state,
            budget=lambda: None,
            instrumentation=None,
            model_alias="model",
            observe_artifacts=lambda _owner, _cursor: None,
            workspace_observation_source=source,
        )

        first_id = "worker-first-observation"
        first_context = Context(first_id)
        plugin.prepare_invocation(invocation_id=first_id, subtask_id="1")
        await plugin.before_run_callback(invocation_context=first_context)
        returned = {"path": "a.py", "lines": [{"text": SECRET}]}
        assert (
            await call(
                plugin,
                first_id,
                "read_file",
                BrokenExtractor(),
                {"path": "a.py"},
                returned,
            )
            is returned
        )
        first = await plugin.complete_invocation(invocation_id=first_id, phase="succeeded")
        assert first is not None
        assert first["lastCompletedInvocation"]["workspace"]["detailComplete"] is False

        source.metadata = WorkspaceObservationMetadata(
            digest="sha256:" + "2" * 64,
            managed_text_paths=("b.py", "new.py"),
        )
        second_id = "worker-second-observation"
        second_context = Context(second_id)
        plugin.prepare_invocation(invocation_id=second_id, subtask_id="2")
        await plugin.before_run_callback(invocation_context=second_context)
        current = await state.snapshot()
        current_workspace = current["currentInvocation"]["workspace"]
        assert current_workspace["workspaceDigest"] == "sha256:" + "2" * 64
        assert current_workspace["interactions"] == []
        await call(
            plugin,
            second_id,
            "read_file",
            Extractor("read_file"),
            {"path": "new.py"},
            {"path": "new.py", "lines": []},
        )
        second = await plugin.complete_invocation(invocation_id=second_id, phase="succeeded")
        assert second is not None
        paths = [
            item["path"] for item in second["lastCompletedInvocation"]["workspace"]["interactions"]
        ]
        assert paths == ["new.py"]

    asyncio.run(scenario())


def test_real_adk_runtime_projects_workspace_observations_into_worker_result(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        project_workspace = await workspace("overlay", "runtime-observations")
        state = WorkerState()
        tools = await create_tools(
            FilesystemToolsetFactory(),
            project_workspace.reader_view(),
            state,
            tmp_path,
            ["read_file", "grep"],
        )
        model = scripted_model(
            [
                tool_call("read_file", {"path": "docs/readme.txt"}, call_id="read-1"),
                tool_call(
                    "grep",
                    {"pattern": "needle", "path": "", "glob": "**/*.py"},
                    call_id="grep-1",
                ),
                text_result("Workspace inspected"),
            ]
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            tools,
            model,
            project_workspace=project_workspace,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        observed = completion.result.observations.workspace
        assert observed is not None
        assert observed.scoped_files == 5
        assert observed.read_files == 1
        assert observed.matched_files == 2
        assert observed.modified_files == 0
        assert observed.unread_files == 4
        assert observed.files_read == ["docs/readme.txt"]
        assert completion.result.observations.truncated is False
        snapshot = await state.snapshot()
        assert (
            snapshot["lastCompletedInvocation"]["workspace"]["workspaceDigest"]
            == (await project_workspace.observation_metadata()).digest
        )
        retained = completion.model_dump_json(by_alias=True) + json.dumps(snapshot)
        assert SECRET not in retained
        assert REVISION not in retained
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))
        await project_workspace.close()

    asyncio.run(scenario())


def test_real_adk_runtime_preserves_success_after_long_file_read(tmp_path: Path) -> None:
    async def scenario() -> None:
        long_path = "x" * 1025
        project_workspace = await workspace("overlay", "runtime-long-read")
        await project_workspace.write_text(long_path, "long path file\n")
        state = WorkerState()
        tools = await create_tools(
            FilesystemToolsetFactory(),
            project_workspace.reader_view(),
            state,
            tmp_path,
            ["read_file"],
        )
        model = scripted_model(
            [
                tool_call("read_file", {"path": long_path}, call_id="long-read"),
                text_result("Long path file inspected"),
            ]
        )
        runtime = await create_runtime(
            tmp_path, state, tools, model, project_workspace=project_workspace
        )

        completion = await runtime.invoke(stage_request())

        assert completion.failure is None
        assert completion.result is not None
        observed = completion.result.observations.workspace
        assert observed is not None
        assert observed.read_files == 1
        assert observed.files_read == []
        assert observed.files_read_truncated is True
        assert completion.result.observations.truncated is True
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))
        await project_workspace.close()

    asyncio.run(scenario())


async def call(
    plugin: WorkerInstrumentationPlugin,
    invocation_id: str,
    name: str,
    owner: Any,
    arguments: dict[str, Any],
    result: Any,
) -> Any:
    context = Context(invocation_id)
    tool = Tool(name, owner)
    assert (
        await plugin.before_tool_callback(
            tool=tool,
            tool_args=arguments,
            tool_context=context,
        )
        is None
    )
    await plugin.after_tool_callback(
        tool=tool,
        tool_args=arguments,
        tool_context=context,
        result=result,
    )
    return result
