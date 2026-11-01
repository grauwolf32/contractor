"""Real-Runner contracts that dependency preprocessing must not weaken."""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.model import scripted_model, text_result, thought_result, tool_call
from fakes.worker_state import WorkerState
from google.genai import types
from test_adk_runtime import (
    CapturingScriptedLlm,
    FakeArtifactClient,
    assert_balanced_tool_calls,
    create_runtime,
    selected_tools,
    stage_request,
)

from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.code_analysis.tools import CodeAnalysisToolsetFactory
from contractor_runtime.toolsets.code_execution.tools import ExecCommandTool
from contractor_runtime.toolsets.filesystem.tools import FilesystemToolsetFactory
from contractor_runtime.toolsets.security_findings.reader import ListFindingsTool
from contractor_runtime.toolsets.taint_annotations.tools import TaintAnnotationsToolsetFactory


def empty_response(kind: str, *, limited: bool = False):
    response = (
        thought_result() if kind == "thought" else text_result(" \n" if kind == "space" else "")
    )
    if kind == "absent":
        response.content = None
    response.finish_reason = types.FinishReason.MAX_TOKENS if limited else types.FinishReason.STOP
    return response


@pytest.mark.parametrize("field", ["offset", "length"])
@pytest.mark.parametrize("value", [1.0, True, "1", None])
def test_runner_preserves_raw_artifact_integer_rejection(
    tmp_path: Path, field: str, value: object
) -> None:
    async def scenario() -> None:
        state = WorkerState()
        client = FakeArtifactClient()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", field: value},
                    call_id="invalid-integer",
                ),
                text_result("Rejected invalid read"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            assert client.calls == []
            assert (
                state.metrics.build_report(report_id="raw-read", duration_ms=1)
                .metrics.tools["read_artifact"]
                .failed
                == 1
            )
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize("values", [[1.0], [True], ["1"], [None]])
def test_runner_preserves_raw_integer_list_rejection(tmp_path: Path, values: list[object]) -> None:
    async def scenario() -> None:
        effects: list[list[int]] = []

        async def strict_list(values: list[int]) -> dict[str, bool]:
            """Record a list only when each original element is an integer."""
            if any(type(value) is not int for value in values):
                raise ToolInputError("values must contain only integers")
            effects.append(values)
            return {"ok": True}

        model = scripted_model(
            [
                tool_call("strict_list", {"values": values}, call_id="invalid-list"),
                text_result("Rejected invalid list"),
            ]
        )
        runtime = await create_runtime(tmp_path, WorkerState(), {"strict_list": strict_list}, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            assert effects == []
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (text_result(""), "worker_result_invalid"),
        (text_result(" \n"), "worker_result_invalid"),
        (thought_result(), "worker_result_missing"),
    ],
)
def test_runner_empty_visible_output_keeps_missing_result_outcome(
    tmp_path: Path, response: object, code: str
) -> None:
    async def scenario() -> None:
        model = scripted_model([response])
        runtime = await create_runtime(tmp_path, WorkerState(), {}, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is None
            assert result.failure is not None
            assert result.failure.code == code
            assert len(model.requests) == 1
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["ordinary", "finalizer", "summarizer"])
@pytest.mark.parametrize(
    ("kind", "limited"),
    [("absent", False), ("empty", False), ("space", False), ("thought", False), ("absent", True)],
    ids=["absent-stop", "empty-stop", "space-stop", "thought-stop", "output-limit"],
)
def test_runner_terminal_empty_response_has_finite_outcome_and_known_usage(
    tmp_path: Path, phase: str, kind: str, limited: bool
) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            """Complete the normal loop before requesting terminal summarization."""
            return {"ok": True}

        state = WorkerState()
        response = empty_response(kind, limited=limited)
        summary_model = None
        if phase == "summarizer":
            responses = [tool_call("probe", {}, call_id="before-empty-summary")]
            summary_model = scripted_model([response], model="worker-summary-model")
        elif phase == "finalizer":
            responses = [text_result("Completed ordinary work"), response]
        else:
            responses = [response]
        model = scripted_model(responses, auto_result_finalizer=False)
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe} if phase == "summarizer" else {},
            model,
            summary_model=summary_model,
            cumulative_budget=10 if summary_model is not None else None,
        )
        try:
            completion = await runtime.invoke(stage_request())
            assert completion.result is None and completion.failure is not None
            assert completion.failure.retryable
            missing = kind in {"absent", "thought"}
            expected = "worker_result_missing" if missing else "worker_result_invalid"
            if limited:
                expected = "worker_output_limit_exceeded"
            if phase == "summarizer":
                expected = "worker_summarization_failed"
            assert completion.failure.code == expected
            expected_main_calls = 2 if phase == "finalizer" else 1
            assert len(model.requests) == expected_main_calls
            snapshot = await state.snapshot()
            assert snapshot["currentInvocation"] is None
            completed = snapshot["lastCompletedInvocation"]
            assert completed["phase"] == "failed"
            metrics = completed["metrics"]
            assert metrics["modelCalls"] == expected_main_calls
            assert metrics["inputTokens"] == 7 * expected_main_calls
            assert metrics["outputTokens"] == 3 * expected_main_calls
            assert metrics["totalTokens"] == 10 * expected_main_calls
            assert metrics["tokenUsageUnavailable"] == 0
            if summary_model is not None:
                assert len(summary_model.requests) == 1
                cause = "result_missing" if missing else "result_invalid"
                if limited:
                    cause = "output_limit_exceeded"
                summary = completed["summarizer"]
                assert summary["phase"] == "failed" and summary["failureCode"] == cause
                assert summary["modelCalls"] == 1
                assert summary["inputTokens"] == 7 and summary["outputTokens"] == 3
                assert summary["totalTokens"] == 10 and summary["tokenUsageUnavailable"] == 0
                report = state.metrics.build_report(report_id="empty-summary", duration_ms=1)
                assert report.metrics.summarizer is not None
                assert report.metrics.summarizer.failure_codes == {cause: 1}
                assert report.metrics.summarizer.attempts == report.metrics.summarizer.failed == 1
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize("provider_id", [None, "", "provider-supplied-call"])
def test_runner_tool_ids_remain_exactly_paired_across_shared_invocations(
    tmp_path: Path, provider_id: str | None
) -> None:
    async def scenario() -> None:
        effects = []

        async def probe(value: str) -> dict[str, str]:
            """Return one value without replaying a previous invocation's effect."""
            effects.append(value)
            return {"value": value}

        model = CapturingScriptedLlm(
            model="paired-ids",
            responses=[
                tool_call("probe", {"value": "first"}, call_id=provider_id),
                text_result("First complete"),
                tool_call("probe", {"value": "second"}, call_id=provider_id),
                text_result("Second complete"),
            ],
        )
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {"probe": probe}, model)
        try:
            first = await runtime.invoke(stage_request())
            assert first.failure is None and first.result is not None
            session_id = runtime._session_lifecycle.shared_session_id
            assert session_id is not None
            session = await runtime._session_service.get_session(
                app_name=runtime._app_name, user_id=runtime._user_id, session_id=session_id
            )
            assert session is not None
            calls = [call for event in session.events for call in event.get_function_calls()]
            responses = [
                part for event in session.events for part in event.get_function_responses()
            ]
            assert len(calls) == len(responses) == 1
            first_id = calls[0].id
            assert first_id and responses[0].id == first_id
            if provider_id:
                assert first_id == provider_id
            else:
                assert first_id.startswith("contractor-tool-")
            next_payload = len(model.payloads)
            second = await runtime.invoke(stage_request().model_copy(update={"subtask_id": "1"}))
            assert second.failure is None and second.result is not None
            carried = model.payloads[next_payload]["messages"]
            assert [
                call["id"] for message in carried for call in message.get("tool_calls", [])
            ] == [first_id]
            assert [
                message["tool_call_id"] for message in carried if message["role"] == "tool"
            ] == [first_id]
            session = await runtime._session_service.get_session(
                app_name=runtime._app_name, user_id=runtime._user_id, session_id=session_id
            )
            calls = [call for event in session.events for call in event.get_function_calls()]
            responses = [
                part for event in session.events for part in event.get_function_responses()
            ]
            assert len(calls) == len(responses) == 2
            assert [part.id for part in responses] == [call.id for call in calls]
            assert calls[0].id == first_id
            if provider_id:
                assert calls[1].id == provider_id
            else:
                assert calls[1].id and calls[1].id != first_id
                assert calls[1].id.startswith("contractor-tool-")
            assert effects == ["first", "second"]
            assert first.invocation_id != second.invocation_id
            assert len(model.requests) == 6
            for payload in model.payloads:
                assert_balanced_tool_calls(payload)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("family", "name", "arguments", "code"),
    [
        (
            "filesystem",
            "read_file",
            {"path": "root.py", "start_line": 1.0},
            "workspace_limit_exceeded",
        ),
        ("source", "list_source_files", {"offset": 1.0}, "tool_input_invalid"),
        ("code", "search_def", {"symbol": "target", "limit": 1.0}, "code_analysis_input_invalid"),
        ("http", "http_history", {"limit": 1.0}, "http_request_invalid"),
        ("caido", "caido_history", {"limit": 1.0}, "caido_request_invalid"),
        ("findings", "list_findings", {"limit": 1.0}, "findings_arguments_invalid"),
        ("findings", "list_findings", {"limit": None}, None),
        (
            "taint",
            "annotate_trace",
            {"path": "root.py", "symbol": "target", "definition_line": 1.0},
            "taint_annotation_input_invalid",
        ),
        (
            "execution",
            "exec_command",
            {"command": "true", "timeout_seconds": 1.0},
            "sandbox_invalid_command",
        ),
    ],
    ids=[
        "filesystem",
        "source",
        "code",
        "http",
        "caido",
        "optional-int",
        "valid-null",
        "taint",
        "exec",
    ],
)
def test_runner_preserves_production_numeric_sibling_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    name: str,
    arguments: dict,
    code: str | None,
) -> None:
    async def scenario() -> None:
        state = WorkerState()
        effects = []

        async def unexpected_request(request):
            effects.append(request.url.path)
            raise AssertionError("Invalid numeric argument reached external I/O")

        async with AsyncExitStack() as cleanup:
            if family in {"filesystem", "code", "taint"}:
                from test_filesystem_toolset import create_tools as workspace_tools
                from test_filesystem_toolset import workspace

                project = await workspace("overlay", f"raw-numeric-{family}")
                cleanup.push_async_callback(project.close)
                factory = {
                    "filesystem": FilesystemToolsetFactory,
                    "code": CodeAnalysisToolsetFactory,
                    "taint": TaintAnnotationsToolsetFactory,
                }[family]()
                view = project.writer_view() if family == "taint" else project.reader_view()
                tools = await workspace_tools(factory, view, state, tmp_path, [name])

                async def unexpected_workspace(*_args, **_kwargs):
                    effects.append("workspace operation")
                    raise AssertionError("Invalid numeric argument reached the workspace")

                for operation in ("read_text", "snapshot") + (
                    ("update_text",) if family == "taint" else ()
                ):
                    monkeypatch.setattr(type(view), operation, unexpected_workspace)
            elif family == "source":
                from test_source_analysis_toolset import ReadOnlyArtifactClient, make_zip
                from test_source_analysis_toolset import make_tools as source_tools

                client = ReadOnlyArtifactClient()
                source = client.seed(
                    "inputs",
                    "source",
                    "application/zip",
                    make_zip({"a.py": "def target(): pass\n"}),
                )
                tools = await source_tools(tmp_path, client, state)
                await tools["open_source_archive"]("inputs", "source", source.revision)
            elif family == "http":
                from test_http_toolset import create_tools as http_tools

                tools, state = await http_tools(tmp_path, unexpected_request)
            elif family == "caido":
                from test_caido_read_tools import FakeArtifactClient as CaidoArtifacts
                from test_caido_read_tools import create_tools as caido_tools

                tools, state, handle = await caido_tools(
                    tmp_path, unexpected_request, CaidoArtifacts(), selected={name}
                )
                cleanup.push_async_callback(handle.close)
            elif family == "execution":
                from test_code_execution_toolset import Executor

                executor = Executor()
                tools = {name: ExecCommandTool(executor, state)}
            else:
                tools = {name: ListFindingsTool((), "sha256:" + "1" * 64, state.metrics, ())}
            for tool in tools.values():
                cleanup.push_async_callback(tool.close)
            model = CapturingScriptedLlm(
                model="numeric-sibling",
                responses=[
                    tool_call(name, arguments, call_id="raw-numeric"),
                    text_result("Finished"),
                ],
            )
            runtime = await create_runtime(tmp_path, state, tools, model)
            cleanup.push_async_callback(runtime.finalize, datetime.now(UTC) + timedelta(seconds=2))
            result = await runtime.invoke(stage_request())
            assert result.failure is None and result.result is not None
            response = next(
                json.loads(message["content"])
                for message in model.payloads[1]["messages"]
                if message["role"] == "tool"
            )
            if code is not None:
                assert response["ok"] is False and response["error"]["code"] == code
            else:
                assert "error" not in response
            metric = state.metrics.build_report(report_id="numeric-sibling", duration_ms=1).metrics
            assert metric.tools[name].calls == 1
            assert metric.tools[name].failed == (1 if code is not None else 0)
            assert len(model.requests) == 3
            assert effects == []
            if family == "source":
                assert client.read_calls == 1  # Fixture preparation; rejected call fetched nothing.
            if family == "execution":
                assert executor.calls == []

    asyncio.run(scenario())
