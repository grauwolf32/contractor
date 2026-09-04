from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes.model import json_result, scripted_model, text_result, thought_result, tool_call
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest

from contractor_runtime.adapters import (
    AdapterHandles,
    RuntimeInstrumentation,
    RuntimeSpan,
    TelemetryAttribute,
)
from contractor_runtime.adk_runtime import (
    AdkWorkerRuntime,
    AdkWorkerRuntimeFactory,
    GatewayLiteLlm,
    GatewayModelError,
    gateway_model,
)
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactValue
from contractor_runtime.contracts import (
    API_VERSION,
    AgentTemplateRef,
    ArtifactRef,
    ArtifactWriteResult,
    ModelPolicyRef,
    ResolvedAgentTemplate,
    ResolvedInstructions,
    ResolvedModelPolicy,
    RuntimeSettings,
    SandboxProfileRef,
    StageContentRequest,
    ToolsetRef,
    ToolsetSelection,
    WorkerRuntimeRef,
    WorkerSummarizerConfig,
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.toolsets.memory import MemoryToolsetFactory
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-adk-gateway-token"


def structured_result(result: str, *, subtask_id: str = "0", **extra: object) -> object:
    return json_result({"subtaskId": subtask_id, "result": result, **extra})


def terminal_text(result: str) -> object:
    return text_result(result)


def test_adk_worker_executes_selected_tools_and_validates_exact_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-1",
                ),
                tool_call(
                    "write_artifact",
                    {
                        "namespace": "builder",
                        "name": "report",
                        "media_type": "application/json",
                        "data_base64": base64.b64encode(b"{}").decode(),
                        "expected_revision": None,
                    },
                    call_id="write-1",
                ),
                terminal_text("Report created"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)
        assert runtime._agent is not None
        assert runtime._agent.output_schema is None
        assert runtime._app.plugins == [runtime._plugin]
        assert runtime._agent.before_model_callback is None
        assert runtime._agent.after_model_callback is None
        assert runtime._agent.on_model_error_callback is None
        assert runtime._result_finalizer is not None

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert result.failure is None
        assert result.result.result == "Report created"
        assert result.result.subtask_id == "0"
        assert result.result.artifacts["report"].revision == "write-r1"
        assert result.result.summarized is False
        assert result.result.observations.profile == "lean@1"
        assert {
            name: value.model_dump() for name, value in result.result.observations.tools.items()
        } == {
            "read_artifact": {"calls": 1, "failures": 0},
            "write_artifact": {"calls": 1, "failures": 0},
        }
        assert result.state_revision > 1
        assert result.invocation_id.startswith("worker-")
        assert client.calls == ["read_artifact", "write_artifact"]
        assert state.metrics.counters == {
            "input_tokens": 28,
            "llm_calls": 4,
            "output_tokens": 12,
            "outcomes.succeeded": 1,
            "tool_calls": 2,
            "tool_calls.read_artifact": 1,
            "tool_calls.write_artifact": 1,
            "total_tokens": 40,
        }
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.max_model_calls == 8
        assert budget.max_tool_calls == 16
        assert budget.max_total_tokens == 32768
        assert budget.observed_model_calls == 4
        assert budget.observed_tool_calls == 2
        assert budget.observed_total_tokens == 40
        assert budget.token_usage_unavailable == 0
        assert budget.exhausted is None
        assert all(request["maxOutputTokens"] == 4096 for request in model.requests)
        assert all(request["temperature"] == 0.1 for request in model.requests)
        assert all(request["responseMimeType"] is None for request in model.requests[:-1])
        assert all(request["hasResponseSchema"] is False for request in model.requests[:-1])
        assert model.requests[-1]["responseMimeType"] == "application/json"
        assert model.requests[-1]["hasResponseSchema"] is True
        assert all("Subtask ID:\n0" in request["contentText"] for request in model.requests[:-1])
        assert all(
            request["toolNames"] == ["read_artifact", "write_artifact"]
            for request in model.requests[:-1]
        )
        assert model.requests[-1]["toolNames"] == []
        assert "result finalization input" in model.requests[-1]["contentText"]
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=runtime._session_id,
        )
        assert session is not None
        contractor_state = session.state["contractor"]
        assert contractor_state["metrics"]["finalOutcome"] == "succeeded"
        assert contractor_state["metrics"]["counters"]["tool_calls"] == 2
        assert contractor_state["currentInvocation"] is None
        assert contractor_state["lastCompletedInvocation"]["subtaskId"] == "0"
        serialized = repr(runtime) + repr(state.metrics) + result.model_dump_json(by_alias=True)
        assert SECRET not in serialized
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_emits_content_free_model_tool_and_a2a_hooks(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-for-telemetry",
                ),
                terminal_text("Telemetry-safe result"),
            ]
        )
        instrumentation = RecordingInstrumentation()
        runtime = await create_runtime(
            tmp_path,
            state,
            tools,
            model,
            instrumentation=instrumentation,
        )

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert [span.name for span in instrumentation.spans] == [
            "contractor.worker.a2a_task",
            "contractor.worker.model",
            "contractor.worker.tool",
            "contractor.worker.model",
            "contractor.worker.model",
        ]
        assert [span.outcome for span in instrumentation.spans] == [
            "succeeded",
            "succeeded",
            "succeeded",
            "succeeded",
            "succeeded",
        ]
        model_spans = [
            span for span in instrumentation.spans if span.name == "contractor.worker.model"
        ]
        assert [span.attributes["tokens.total"] for span in model_spans] == [10, 10, 10]
        assert model_spans[-1].attributes["model.phase"] == "result_finalizer"
        assert instrumentation.spans[2].attributes["tool.name"] == "read_artifact"
        assert instrumentation.spans[0].attributes["counts.model_calls"] == 3
        assert instrumentation.spans[0].attributes["counts.tool_calls"] == 1
        rendered = repr([span.attributes for span in instrumentation.spans])
        for forbidden in (SECRET, "inputs", "source", "input-r1", "Telemetry-safe result"):
            assert forbidden not in rendered
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_treats_terminal_text_as_opaque_and_blocks_secrets(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        shaped = json.dumps(
            {
                "subtaskId": "0",
                "result": "guessed",
                "outcome": "succeeded",
                "artifacts": {"report": {"revision": "guessed"}},
            },
            separators=(",", ":"),
        )
        model = scripted_model([text_result(shaped)])
        state = WorkerState()
        runtime = await create_runtime(
            tmp_path / "shaped",
            state,
            {},
            model,
        )
        result = await runtime.invoke(stage_request())
        assert result.result is not None
        assert result.failure is None
        assert result.result.result == shaped
        assert result.result.artifacts == {}
        assert len(model.requests) == 2
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        secret_bearing = scripted_model([terminal_text(SECRET)])
        secret_state = WorkerState()
        secret_runtime = await create_runtime(tmp_path / "secret", secret_state, {}, secret_bearing)
        secret_result = await secret_runtime.invoke(stage_request())
        assert secret_result.failure is not None
        assert secret_result.failure.code == "unsafe_worker_result"
        assert len(secret_bearing.requests) == 1
        assert SECRET not in secret_result.model_dump_json(by_alias=True)
        assert SECRET not in repr(secret_state.metrics.snapshot())
        await secret_runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "expected_code", "retryable"),
    [
        (text_result("not-json"), "worker_result_invalid", True),
        (thought_result("no final value"), "worker_result_missing", True),
        (
            json_result({"subtaskId": "0", "result": "original", "outcome": "succeeded"}),
            "worker_result_invalid",
            True,
        ),
        (
            json_result({"subtaskId": "1", "result": "stale"}),
            "worker_result_subtask_mismatch",
            True,
        ),
        (json_result({"subtaskId": "0", "result": ""}), "worker_result_invalid", True),
        (
            json_result({"subtaskId": "0", "result": "x" * (64 * 1024 + 1)}),
            "worker_result_too_large",
            False,
        ),
        (
            json_result({"subtaskId": "0", "result": "rewritten"}),
            "worker_result_finalizer_mismatch",
            True,
        ),
    ],
    ids=[
        "malformed",
        "missing",
        "unknown-field",
        "subtask-mismatch",
        "empty-result",
        "oversized-result",
        "changed-result",
    ],
)
def test_adk_worker_rejects_invalid_result_finalizer_boundaries(
    tmp_path: Path,
    response: object,
    expected_code: str,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        model = scripted_model(
            [terminal_text("original"), response],
            auto_result_finalizer=False,
        )
        runtime = await create_runtime(
            tmp_path / expected_code,
            WorkerState(),
            {},
            model,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == expected_code
        assert completion.failure.retryable is retryable
        assert completion.failure.message
        assert completion.state_revision > 1
        assert completion.failure.code != "worker_execution_failed"
        assert len(model.requests) == 2
        assert len(completion.model_dump_json(by_alias=True).encode("utf-8")) < 256 * 1024
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_unhandled_model_error_to_safe_worker_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {}, scripted_model([]))

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_execution_failed"
        assert completion.failure.retryable is True
        assert completion.invocation_id.startswith("worker-")
        snapshot = await state.snapshot()
        assert snapshot["lastCompletedInvocation"]["invocationId"] == completion.invocation_id
        assert snapshot["lastCompletedInvocation"]["phase"] == "failed"
        assert SECRET not in completion.model_dump_json(by_alias=True)
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_gateway_error_to_safe_retryable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = scripted_model([])

    async def gateway_failure(_model: object, _request: object, stream: bool = False) -> Any:
        del stream
        if False:
            yield None
        raise GatewayModelError("TimeoutError")

    monkeypatch.setattr(type(model), "generate_content_async", gateway_failure)

    async def scenario() -> None:
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {}, model)

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_gateway_unavailable"
        assert completion.failure.retryable is True
        assert "TimeoutError" not in completion.failure.message
        snapshot = await state.snapshot()
        assert snapshot["lastCompletedInvocation"]["invocationId"] == completion.invocation_id
        assert snapshot["lastCompletedInvocation"]["phase"] == "failed"
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_only_declared_runtime_result_bindings(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        hidden = ArtifactRef(namespace="builder", name="memory.hidden", revision="memory-r1")
        hidden_model = scripted_model(
            [
                tool_call("ref_probe", {}, call_id="hidden-ref"),
                terminal_text("Notebook update completed"),
            ]
        )
        first = await create_runtime(
            tmp_path / "hidden",
            WorkerState(),
            {"ref_probe": RefExposingTool(hidden)},
            hidden_model,
        )
        hidden_result = await first.invoke(stage_request())
        assert hidden_result.result is not None
        assert hidden_result.result.artifacts == {}
        await first.finalize(datetime.now(UTC) + timedelta(seconds=1))

        allowed = ArtifactRef(namespace="builder", name="report", revision="report-r1")
        allowed_model = scripted_model(
            [
                tool_call("ref_probe", {}, call_id="allowed-ref"),
                terminal_text("Purpose output selected"),
            ]
        )
        second = await create_runtime(
            tmp_path / "allowed",
            WorkerState(),
            {"ref_probe": RefExposingTool(allowed)},
            allowed_model,
        )
        request = stage_request().model_copy(
            update={"result_artifacts": {"report": ArtifactRef(namespace="builder", name="report")}}
        )
        accepted = await second.invoke(request)
        assert accepted.result is not None
        assert accepted.result.artifacts["report"] == allowed
        await second.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_builds_typed_result_with_trusted_artifacts(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "write_artifact",
                    {
                        "namespace": "builder",
                        "name": "report",
                        "media_type": "application/json",
                        "data_base64": base64.b64encode(b"{}").decode(),
                        "expected_revision": None,
                    },
                    call_id="write-before-invalid-result",
                ),
                terminal_text("Report created"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert result.result.artifacts["report"].revision == "write-r1"
        assert result.result.result == "Report created"
        assert len(model.requests) == 3
        assert all(request["hasResponseSchema"] is False for request in model.requests[:-1])
        assert model.requests[-1]["hasResponseSchema"] is True
        for request in model.requests:
            model_context = request["contentText"] + repr(request["systemInstruction"])
            for forbidden in (
                "StageContentRequest",
                "StageContentResult",
                "apiVersion",
                "result slot",
                "output bindings",
                "protocol envelope",
                "AgentTemplate",
            ):
                assert forbidden not in model_context
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_select_artifact_from_failed_tool_call(tmp_path: Path) -> None:
    class FailingRefTool(RefExposingTool):
        async def __call__(self) -> dict[str, bool]:
            self._observations += 1
            raise RuntimeError("synthetic failure after observation")

    async def scenario() -> None:
        ref = ArtifactRef(namespace="builder", name="report", revision="failed-r1")
        model = scripted_model(
            [
                tool_call("ref_probe", {}, call_id="failed-ref"),
                terminal_text("Could not update the report"),
            ]
        )
        runtime = await create_runtime(
            tmp_path,
            WorkerState(),
            {"ref_probe": FailingRefTool(ref)},
            model,
        )

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert result.result.artifacts == {}
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_reuse_artifact_observed_by_an_earlier_invocation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        tools = await selected_tools(tmp_path, client, WorkerState())
        model = scripted_model(
            [
                tool_call(
                    "write_artifact",
                    {
                        "namespace": "builder",
                        "name": "report",
                        "media_type": "application/json",
                        "data_base64": base64.b64encode(b"{}").decode(),
                        "expected_revision": None,
                    },
                    call_id="first-write",
                ),
                terminal_text("First task completed"),
                terminal_text("Second task completed without touching the result"),
            ]
        )
        runtime = await create_runtime(tmp_path, WorkerState(), tools, model)

        first = await runtime.invoke(stage_request())
        second = await runtime.invoke(stage_request())

        assert first.result is not None
        assert first.result.artifacts["report"].revision == "write-r1"
        assert second.result is not None
        assert second.result.artifacts == {}
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_finalizes_plain_terminal_text_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([text_result("not-json")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_model_calls=2)

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert result.failure is None
        assert result.result.result == "not-json"
        assert len(model.requests) == 2
        assert model.requests[0]["hasResponseSchema"] is False
        assert model.requests[1]["hasResponseSchema"] is True
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_model_calls == 2
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_start_required_finalizer_without_model_budget(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([terminal_text("Terminal text is ready")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_model_calls=1)

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_budget_exhausted"
        assert len(model.requests) == 1
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.observed_model_calls == 1
        assert budget.exhausted == "model_calls"
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_result_finalizer_gateway_failure_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model(
            [terminal_text("Terminal text is ready")],
            result_finalizer_error=GatewayModelError("TimeoutError"),
        )
        runtime = await create_runtime(tmp_path, state, {}, model)

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_gateway_unavailable"
        assert completion.failure.retryable is True
        assert len(model.requests) == 2
        assert state.metrics.counters["llm_calls"] == 2
        assert state.metrics.counters["llm_errors"] == 1
        snapshot = await state.snapshot()
        metrics = snapshot["lastCompletedInvocation"]["metrics"]
        assert metrics["modelCalls"] == 2
        assert metrics["modelErrors"] == 1
        retained = completion.model_dump_json(by_alias=True) + repr(state.metrics.snapshot())
        assert "Terminal text is ready" not in retained
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_unexpected_result_finalizer_failure_safely(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model(
            [terminal_text("Terminal text is ready")],
            result_finalizer_error=RuntimeError("sensitive finalizer failure"),
        )
        runtime = await create_runtime(tmp_path, state, {}, model)

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_result_finalizer_failed"
        assert completion.failure.retryable is True
        assert completion.failure.message == "Worker result finalizer failed (runtime_failed)"
        assert len(model.requests) == 2
        assert state.metrics.counters["llm_errors"] == 1
        assert "sensitive finalizer failure" not in completion.model_dump_json(by_alias=True)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_abort_cancels_active_result_finalizer(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        instrumentation = RecordingInstrumentation()
        model = scripted_model(
            [terminal_text("Terminal text is ready")],
            block_call_number=2,
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            {},
            model,
            instrumentation=instrumentation,
        )
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(model.blocked.wait(), timeout=1)

        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert len(model.requests) == 2
        assert state.metrics.counters["llm_calls"] == 2
        assert state.metrics.counters.get("llm_errors", 0) == 0
        snapshot = await state.snapshot()
        assert snapshot["lastCompletedInvocation"]["phase"] == "cancelled"
        finalizer_spans = [
            span
            for span in instrumentation.spans
            if span.attributes.get("model.phase") == "result_finalizer"
        ]
        assert len(finalizer_spans) == 1
        assert finalizer_spans[0].outcome == "cancelled"

    asyncio.run(scenario())


def test_adk_worker_bounds_result_finalizer_document_before_model_call(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([terminal_text("\x00" * (64 * 1024))])
        runtime = await create_runtime(tmp_path, state, {}, model)

        completion = await runtime.invoke(stage_request())

        assert completion.result is None
        assert completion.failure is not None
        assert completion.failure.code == "worker_result_too_large"
        assert completion.failure.retryable is False
        assert len(model.requests) == 1
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.observed_model_calls == 1
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_can_recover_from_a_safe_tool_exception(tmp_path: Path) -> None:
    class MissingArtifact(RuntimeError):
        code = "not_found"
        retryable = False

    async def load_optional(name: str) -> dict[str, object]:
        """Load an optional artifact."""

        raise MissingArtifact(f"missing {name}: {SECRET}")

    async def scenario() -> None:
        model = scripted_model(
            [
                tool_call("load_optional", {"name": "candidate"}, call_id="load-1"),
                terminal_text("Recovered from optional absence"),
            ]
        )
        runtime = await create_runtime(
            tmp_path, WorkerState(), {"load_optional": load_optional}, model
        )

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert len(model.requests) == 3
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=runtime._session_id,
        )
        assert session is not None
        assert SECRET not in repr(session.events)
        assert "load_optional failed (MissingArtifact)" in repr(session.events)
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_counts_unknown_model_tool_without_retaining_its_name(tmp_path: Path) -> None:
    async def scenario() -> None:
        model_authored_name = "recognizable_model_authored_tool_name"
        model = scripted_model(
            [
                tool_call(model_authored_name, {"body": SECRET}, call_id="unknown-1"),
                terminal_text("Recovered from an unknown tool"),
            ]
        )
        state = WorkerState()
        instrumentation = RecordingInstrumentation()
        runtime = await create_runtime(
            tmp_path,
            state,
            {},
            model,
            instrumentation=instrumentation,
        )

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert set(report.metrics.tools) == {"unknown_tool"}
        assert report.metrics.tools["unknown_tool"].failed == 1
        exported = repr(state.metrics.snapshot()) + repr(
            [span.attributes for span in instrumentation.spans]
        )
        assert model_authored_name not in exported
        assert SECRET not in exported
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "raw_arguments",
    [
        {"content": "body"},
        {
            "name": "safe_note",
            "content": "body",
            "unexpected": "recognizable-raw-argument-canary",
        },
        {"name": 7, "content": "body"},
    ],
    ids=["missing", "extra", "wrong-type"],
)
def test_adk_worker_reduces_malformed_raw_memory_arguments_before_binding(
    tmp_path: Path,
    raw_arguments: dict[str, object],
) -> None:
    async def scenario() -> None:
        client = NoArtifactAccessClient()
        state = WorkerState()
        tools = await selected_memory_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call("write_memory", raw_arguments, call_id="malformed-memory"),
                terminal_text("Handled bounded memory failure"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        assert client.calls == []
        assert state.metrics.counters["tool_calls"] == 1
        assert state.metrics.counters["tool_errors"] == 1
        assert len(state.metrics.tool_calls) == 1
        call = state.metrics.tool_calls[0]
        assert call.error is not None
        assert call.error.code == "memory_invalid"
        assert call.error.retryable is False
        assert call.arguments is not None
        assert set(call.arguments) == {
            "name",
            "content_bytes",
            "description_bytes",
            "tag_count",
        }
        assert call.arguments["content_bytes"] in {0, 4}
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None and budget.observed_tool_calls == 1
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=runtime._session_id,
        )
        assert session is not None
        function_responses = [
            part.function_response.response
            for event in session.events
            for part in (event.content.parts if event.content is not None else [])
            if part.function_response is not None
        ]
        rendered_response = repr(function_responses)
        rendered_metrics = repr(state.metrics.snapshot())
        assert "memory_invalid" in rendered_response
        assert "mandatory input parameters" not in rendered_response
        assert "recognizable-raw-argument-canary" not in rendered_response
        assert "recognizable-raw-argument-canary" not in rendered_metrics
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_start_a_serializer_when_structured_result_is_missing(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "write_artifact",
                    {
                        "namespace": "builder",
                        "name": "report",
                        "media_type": "application/json",
                        "data_base64": base64.b64encode(b"{}").decode(),
                        "expected_revision": None,
                    },
                    call_id="write-before-missing-result",
                ),
                thought_result("The report is complete."),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "worker_result_missing"
        assert len(model.requests) == 2
        assert model.requests[0]["toolNames"] == ["read_artifact", "write_artifact"]
        assert model.requests[1]["toolNames"] == ["read_artifact", "write_artifact"]
        assert all("result_slot" not in request["contentText"] for request in model.requests)
        assert all("StageContentResult" not in request["contentText"] for request in model.requests)
        assert SECRET not in repr(state.metrics.snapshot())
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_missing_result_fails_without_an_extra_model_turn(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model(
            [
                thought_result("The task is complete."),
            ]
        )
        runtime = await create_runtime(tmp_path, state, {}, model)

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "worker_result_missing"
        assert result.failure.retryable
        assert len(model.requests) == 1
        assert model.requests[0]["toolNames"] == []
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_missing_result_does_not_exhaust_model_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([thought_result("Task complete")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_model_calls=1)

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "worker_result_missing"
        assert result.failure.retryable
        assert len(model.requests) == 1
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.complete
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_model_calls == 1
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_tool_budget_stops_before_extra_side_effect(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-1",
                ),
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                    call_id="read-2",
                ),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model, max_tool_calls=1)

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "worker_budget_exhausted"
        assert client.calls == ["read_artifact"]
        assert len(model.requests) == 2
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_tool_calls == 1
        assert report.metrics.worker_budget.exhausted == "tool_calls"
        assert report.metrics.tools["read_artifact"].calls == 1
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize("max_total_tokens", [15, 20])
def test_adk_worker_token_budget_stops_before_response_tool_side_effect(
    tmp_path: Path, max_total_tokens: int
) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        response = tool_call(
            "read_artifact",
            {"namespace": "inputs", "name": "source", "revision": "input-r1"},
            call_id="read-over-token-budget",
        )
        assert response.usage_metadata is not None
        response.usage_metadata.total_token_count = 20
        model = scripted_model([response])
        runtime = await create_runtime(
            tmp_path, state, tools, model, max_total_tokens=max_total_tokens
        )

        result = await runtime.invoke(stage_request())

        assert result.failure is not None
        assert result.failure.code == "worker_budget_exhausted"
        assert client.calls == []
        assert len(model.requests) == 1
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_total_tokens == 20
        assert report.metrics.worker_budget.exhausted == "total_tokens"
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_missing_token_usage_keeps_call_limits_effective(tmp_path: Path) -> None:
    async def scenario() -> None:
        response = terminal_text("Done without provider token usage")
        response.usage_metadata = None
        state = WorkerState()
        model = scripted_model([response])
        runtime = await create_runtime(
            tmp_path, state, {}, model, max_model_calls=2, max_total_tokens=10
        )

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_model_calls == 2
        assert report.metrics.worker_budget.observed_total_tokens == 10
        assert report.metrics.worker_budget.token_usage_unavailable == 1
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_accepts_finalized_result_exactly_at_token_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([terminal_text("Finished at the exact token ceiling")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_total_tokens=20)

        result = await runtime.invoke(stage_request())

        assert result.result is not None
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_total_tokens == 20
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "trigger",
    ["cumulative-budget", "context-window"],
)
def test_adk_worker_uses_one_terminal_summarizer_at_a_safe_tool_boundary(
    tmp_path: Path,
    trigger: str,
) -> None:
    async def scenario() -> None:
        client = FakeArtifactClient()
        state = WorkerState()
        tools = await selected_tools(tmp_path, client, state)
        first_response = tool_call(
            "read_artifact",
            {"namespace": "inputs", "name": "source", "revision": "input-r1"},
            call_id="read-before-summary",
        )
        expected_prompt_tokens = 7
        expected_normal_total = 10
        if trigger == "context-window":
            assert first_response.usage_metadata is not None
            # min(floor(8192 * .9), 8192 - 1024) == 7168.
            first_response.usage_metadata.prompt_token_count = 7168
            first_response.usage_metadata.total_token_count = 7171
            expected_prompt_tokens = 7168
            expected_normal_total = 7171
        normal_model = scripted_model([first_response])
        summary_model = scripted_model(
            [structured_result("Bounded terminal summary")],
            model="worker-summary-model",
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            tools,
            normal_model,
            summary_model=summary_model,
            max_output_tokens=1024,
            context_window_tokens=8192,
            cumulative_budget=10 if trigger == "cumulative-budget" else None,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        assert completion.result.result == "Bounded terminal summary"
        assert completion.result.summarized is True
        assert completion.result.observations.tools["read_artifact"].calls == 1
        assert len(normal_model.requests) == 1
        assert len(summary_model.requests) == 1
        summary_request = summary_model.requests[0]
        assert summary_request["model"] == "worker-summary-model"
        assert summary_request["maxOutputTokens"] == 2048
        assert summary_request["temperature"] == 0.0
        assert summary_request["toolNames"] == []
        assert summary_request["hasResponseSchema"] is True
        assert "read_artifact" in summary_request["contentText"]
        assert SECRET not in summary_request["contentText"]
        assert client.calls == ["read_artifact"]

        snapshot = await state.snapshot()
        finished = snapshot["lastCompletedInvocation"]
        assert finished["summarizer"]["phase"] == "succeeded"
        assert finished["summarizer"]["modelCalls"] == 1
        assert finished["summarizer"]["totalTokens"] == 10
        assert finished["metrics"]["modelCalls"] == 1
        assert finished["metrics"]["latestPromptTokens"] == expected_prompt_tokens
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.model_calls == 1
        assert report.metrics.total_tokens == expected_normal_total
        assert report.metrics.summarizer is not None
        assert report.metrics.summarizer.attempts == 1
        assert report.metrics.summarizer.succeeded == 1
        assert report.metrics.summarizer.model_calls == 1
        assert report.metrics.summarizer.total_tokens == 10
        assert report.metrics.summarizer.failure_codes == {}
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_keeps_a_valid_normal_result_at_the_soft_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        normal_model = scripted_model([terminal_text("Normal result wins")])
        summary_model = scripted_model([structured_result("Must not run")])
        runtime = await create_runtime(
            tmp_path,
            state,
            {},
            normal_model,
            summary_model=summary_model,
            cumulative_budget=10,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        assert completion.result.result == "Normal result wins"
        assert completion.result.summarized is False
        assert len(normal_model.requests) == 2
        assert summary_model.requests == []
        snapshot = await state.snapshot()
        assert snapshot["lastCompletedInvocation"]["summarizer"]["phase"] == "not_requested"
        assert (
            state.metrics.build_report(report_id="worker-report", duration_ms=1).metrics.summarizer
            is None
        )
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_guess_missing_usage_for_summarization(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        first = tool_call("probe", {}, call_id="missing-usage-probe")
        first.usage_metadata = None
        state = WorkerState()
        normal_model = scripted_model([first, terminal_text("Completed normally")])
        summary_model = scripted_model([structured_result("Must not run")])
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            cumulative_budget=1,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        assert completion.result.summarized is False
        assert len(normal_model.requests) == 3
        assert summary_model.requests == []
        snapshot = await state.snapshot()
        assert snapshot["lastCompletedInvocation"]["metrics"]["latestPromptTokens"] == 7
        assert snapshot["lastCompletedInvocation"]["metrics"]["tokenUsageUnavailable"] == 1
        assert snapshot["lastCompletedInvocation"]["summarizer"]["phase"] == "not_requested"
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_does_not_trigger_from_inconsistent_provider_usage(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        first = tool_call("probe", {}, call_id="inconsistent-usage-probe")
        assert first.usage_metadata is not None
        first.usage_metadata.prompt_token_count = 7168
        first.usage_metadata.candidates_token_count = 3
        first.usage_metadata.total_token_count = 100
        state = WorkerState()
        normal_model = scripted_model([first, terminal_text("Completed normally")])
        summary_model = scripted_model([structured_result("Must not run")])
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            max_output_tokens=1024,
            context_window_tokens=8192,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.result is not None
        assert completion.result.summarized is False
        assert len(normal_model.requests) == 3
        assert summary_model.requests == []
        snapshot = await state.snapshot()
        invocation = snapshot["lastCompletedInvocation"]
        assert invocation["metrics"]["totalTokens"] == 20
        assert invocation["metrics"]["tokenUsageUnavailable"] == 1
        assert invocation["summarizer"]["phase"] == "not_requested"
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.observed_total_tokens == 20
        assert budget.token_usage_unavailable == 1
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_reports_invalid_terminal_summary_as_one_safe_failure(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        state = WorkerState()
        normal_model = scripted_model([tool_call("probe", {}, call_id="probe-1")])
        summary_model = scripted_model([text_result("not structured")])
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            cumulative_budget=10,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.failure is not None
        assert completion.failure.code == "worker_summarization_failed"
        assert completion.failure.retryable is True
        assert "result_invalid" in completion.failure.message
        assert len(normal_model.requests) == 1
        assert len(summary_model.requests) == 1
        snapshot = await state.snapshot()
        summary = snapshot["lastCompletedInvocation"]["summarizer"]
        assert summary["phase"] == "failed"
        assert summary["failureCode"] == "result_invalid"
        report_summary = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert report_summary is not None
        assert report_summary.attempts == 1
        assert report_summary.failed == 1
        assert report_summary.failure_codes == {"result_invalid": 1}
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_terminal_summarizer_maps_provider_timeout_to_one_safe_failure(
    tmp_path: Path,
) -> None:
    class TimeoutSummaryModel(LiteLlm):
        async def generate_content_async(self, _request: LlmRequest, stream: bool = False) -> Any:
            del stream
            if False:
                yield None
            raise GatewayModelError("TimeoutError")

    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        state = WorkerState()
        normal_model = scripted_model([tool_call("probe", {}, call_id="timeout-probe")])
        summary_model = TimeoutSummaryModel(
            model="openai/worker-summary-model",
            api_base="https://gateway.invalid/v1",
            api_key=SECRET,
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            cumulative_budget=10,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.failure is not None
        assert completion.failure.code == "worker_summarization_failed"
        summary = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert summary is not None
        assert summary.attempts == 1
        assert summary.failed == 1
        assert summary.model_calls == 1
        assert summary.token_usage_unavailable == 1
        assert summary.failure_codes == {"gateway_unavailable": 1}
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_terminal_summarizer_enforces_its_independent_total_budget(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        state = WorkerState()
        normal_model = scripted_model([tool_call("probe", {}, call_id="budget-probe")])
        summary_response = structured_result("Summary beyond its own budget")
        assert summary_response.usage_metadata is not None
        summary_response.usage_metadata.total_token_count = 11
        summary_model = scripted_model([summary_response], model="worker-summary-model")
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            summary_max_total_tokens=10,
            cumulative_budget=10,
        )

        completion = await runtime.invoke(stage_request())

        assert completion.failure is not None
        assert completion.failure.code == "worker_summarization_failed"
        summary = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert summary is not None
        assert summary.model_calls == 1
        assert summary.total_tokens == 11
        assert summary.failure_codes == {"budget_exhausted": 1}
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_finalize_and_abort_never_start_an_idle_terminal_summarizer(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        for index, operation in enumerate(("finalize", "abort")):
            normal_model = scripted_model([])
            summary_model = scripted_model(
                [structured_result("Must not run")], model="worker-summary-model"
            )
            runtime = await create_runtime(
                tmp_path / operation,
                WorkerState(),
                {},
                normal_model,
                summary_model=summary_model,
                cumulative_budget=10,
            )
            deadline = datetime.now(UTC) + timedelta(seconds=1)
            if index == 0:
                await runtime.finalize(deadline)
            else:
                await runtime.abort(deadline)
            assert normal_model.requests == []
            assert summary_model.requests == []

    asyncio.run(scenario())


def test_abort_cancels_terminal_summarizer_and_records_one_failed_attempt(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async def probe() -> dict[str, bool]:
            return {"ok": True}

        state = WorkerState()
        normal_model = scripted_model([tool_call("probe", {}, call_id="probe-before-abort")])
        summary_model = scripted_model(
            [structured_result("Too late")],
            block=True,
            model="worker-summary-model",
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            normal_model,
            summary_model=summary_model,
            cumulative_budget=10,
        )
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(summary_model.started.wait(), timeout=1)

        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert len(normal_model.requests) == 1
        assert len(summary_model.requests) == 1
        snapshot = await state.snapshot()
        completed = snapshot["lastCompletedInvocation"]
        assert completed["phase"] == "cancelled"
        assert completed["summarizer"]["phase"] == "failed"
        assert completed["summarizer"]["failureCode"] == "cancelled"
        summary = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert summary is not None
        assert summary.attempts == 1
        assert summary.failed == 1
        assert summary.model_calls == 1
        assert summary.failure_codes == {"cancelled": 1}

    asyncio.run(scenario())


def test_abort_cancels_long_running_adk_invocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = scripted_model(
            [terminal_text("Too late")],
            block=True,
        )
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {}, model)
        invocation = asyncio.create_task(runtime.invoke(stage_request()))
        await asyncio.wait_for(model.started.wait(), timeout=1)

        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert state.metrics.final_outcome == "cancelled"
        assert state.metrics.counters["llm_calls"] == 1
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_gateway_adapter_discards_secret_bearing_provider_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def leaking_provider(_model: LiteLlm, _request: LlmRequest, stream: bool = False) -> Any:
        del stream
        if False:
            yield None
        raise RuntimeError(f"provider rejected Authorization: Bearer {SECRET}")

    monkeypatch.setattr(LiteLlm, "generate_content_async", leaking_provider)
    model = GatewayLiteLlm(
        model="openai/worker-model",
        api_base="https://llm.example/v1",
        api_key=SECRET,
    )

    async def scenario() -> None:
        with pytest.raises(GatewayModelError) as captured:
            async for _response in model.generate_content_async(LlmRequest()):
                pass
        assert SECRET not in repr(captured.value)
        assert captured.value.__context__ is None
        assert captured.value.provider_error_type == "RuntimeError"

    asyncio.run(scenario())
    model.clear_credentials()


def test_gateway_model_disables_hidden_provider_retries(tmp_path: Path) -> None:
    model = gateway_model(build_context(tmp_path, WorkerState(), {}))

    assert isinstance(model, GatewayLiteLlm)
    assert model._additional_args["num_retries"] == 0
    model.clear_credentials()


async def create_runtime(
    tmp_path: Path,
    state: WorkerState,
    tools: dict[str, object],
    model: object,
    *,
    max_output_tokens: int = 4096,
    max_model_calls: int = 8,
    max_tool_calls: int = 16,
    max_total_tokens: int = 32768,
    instrumentation: RuntimeInstrumentation | None = None,
    project_workspace: Any = None,
    summary_model: object | None = None,
    summary_max_total_tokens: int | None = None,
    cumulative_budget: int | None = None,
    context_window_tokens: int = 131_072,
    context_window_ratio: float = 0.9,
) -> AdkWorkerRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = build_context(tmp_path, state, tools)
    if instrumentation is not None:
        context = replace(
            context,
            adapter_handles=AdapterHandles(instrumentation=instrumentation),
        )
    if project_workspace is not None:
        context = replace(context, project_workspace=project_workspace)
    context = replace(
        context,
        model_policy=context.model_policy.model_copy(
            update={
                "context_window_tokens": (
                    context_window_tokens if summary_model is not None else None
                ),
                "max_output_tokens": max_output_tokens,
                "max_model_calls": max_model_calls,
                "max_tool_calls": max_tool_calls,
                "max_total_tokens": max_total_tokens,
            }
        ),
    )
    if summary_model is not None:
        summary_policy = ResolvedModelPolicy(
            ref=ModelPolicyRef(
                policyId="worker_summarizer",
                version="1",
                digest="sha256:" + "9" * 64,
            ),
            model="worker-summary-model",
            contextWindowTokens=131_072,
            maxOutputTokens=2048,
            maxModelCalls=1,
            maxTotalTokens=summary_max_total_tokens,
            temperature=0.0,
        )
        context = replace(
            context,
            summarizer=WorkerSummarizerConfig(
                modelPolicy=summary_policy,
                contextWindowRatio=context_window_ratio,
                cumulativeBudget=cumulative_budget,
            ),
        )

    def select_model(build: WorkerBuildContext) -> object:
        if build.model_policy.model == "worker-summary-model":
            if summary_model is None:
                raise RuntimeError("summary model was not supplied")
            return summary_model
        return model

    factory = AdkWorkerRuntimeFactory(select_model)  # type: ignore[arg-type]
    runtime = await factory.create(context)
    assert isinstance(runtime, AdkWorkerRuntime)
    return runtime


async def selected_tools(
    tmp_path: Path, client: FakeArtifactClient, state: WorkerState
) -> dict[str, object]:
    settings = runtime_settings()
    factory = RunArtifactsToolsetFactory(lambda _allocation, _settings: client)  # type: ignore[arg-type]
    result = await factory.create_selected(
        selected=["read_artifact", "write_artifact"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="builder",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        state=state,
    )
    return dict(result)


async def selected_memory_tools(
    tmp_path: Path, client: object, state: WorkerState
) -> dict[str, object]:
    factory = MemoryToolsetFactory(lambda _allocation, _settings: client)  # type: ignore[arg-type,return-value]
    result = await factory.create_selected(
        selected=["write_memory"],
        allocation_id="allocation-1",
        run_id="run-1",
        namespace="builder",
        runtime_settings=runtime_settings(),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        state=state,
    )
    return dict(result)


def build_context(
    tmp_path: Path, state: WorkerState, tools: dict[str, object]
) -> WorkerBuildContext:
    template = agent_template()
    return WorkerBuildContext(
        allocation_id="allocation-1",
        run_id="run-1",
        stage_execution_id="stage-execution-1",
        logical_agent_name="builder",
        namespace="builder",
        description=template.description,
        instruction=template.instructions.text,
        card_version=template.ref.version,
        model_policy=template.model_policy.model_copy(deep=True),
        workspace=AllocationWorkspace(root=tmp_path.parent, path=tmp_path),
        tools=tools,  # type: ignore[arg-type]
        state=state,
        a2a_base_url="https://runtime.example",
        runtime_settings=runtime_settings(),
    )


class RecordingInstrumentation:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: dict[str, TelemetryAttribute] | None = None,
    ) -> RuntimeSpan:
        span = RecordingSpan(name, attributes or {})
        self.spans.append(span)
        return span


class RecordingSpan:
    def __init__(self, name: str, attributes: dict[str, TelemetryAttribute]) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.outcome: str | None = None

    def end(
        self,
        *,
        outcome: str,
        attributes: dict[str, TelemetryAttribute] | None = None,
    ) -> None:
        self.outcome = outcome
        if attributes is not None:
            self.attributes.update(attributes)

    def __repr__(self) -> str:
        return (
            f"RecordingSpan(name={self.name!r}, outcome={self.outcome!r}, "
            f"attribute_keys={tuple(sorted(self.attributes))!r})"
        )


def agent_template() -> ResolvedAgentTemplate:
    return ResolvedAgentTemplate(
        ref=AgentTemplateRef(
            templateId="artifact_builder", version="1", digest="sha256:" + "0" * 64
        ),
        description="Build artifacts",
        runtime=WorkerRuntimeRef(runtimeId="adk", version="1"),
        instructions=ResolvedInstructions(
            ref="instructions/worker.md", digest="sha256:" + "1" * 64, text="Use tools."
        ),
        modelPolicy=ResolvedModelPolicy(
            ref=ModelPolicyRef(policyId="worker", version="1", digest="sha256:" + "2" * 64),
            model="worker-model",
            maxOutputTokens=4096,
            maxModelCalls=8,
            maxToolCalls=16,
            maxTotalTokens=32768,
            temperature=0.1,
        ),
        toolsets=[
            ToolsetSelection(
                ref=ToolsetRef(toolsetId="run-artifacts", version="1"),
                tools=["read_artifact", "write_artifact"],
            )
        ],
        sandboxProfile=SandboxProfileRef(sandboxProfileId="local-workdir", version="1"),
    )


def runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://llm.example/v1",
        llmGatewayToken=SECRET,
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )


def stage_request() -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        subtaskId="0",
        objective="Build a report",
        instructions="Read the input and write the report.",
        parameters={"format": "json"},
        artifacts={"source": ArtifactRef(namespace="inputs", name="source", revision="input-r1")},
        resultArtifacts={"report": ArtifactRef(namespace="builder", name="report")},
    )


class FakeArtifactClient:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._known: dict[tuple[str, str, str], ArtifactRef] = {}
        self._observed: list[ArtifactRef] = []

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(self._known.values())

    @property
    def observation_cursor(self) -> int:
        return len(self._observed)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return tuple(self._observed[cursor:])

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        self.calls.append("read_artifact")
        exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision="read-r1")
        self._remember(exact)
        return ArtifactValue(
            artifact=exact,
            media_type="text/plain",
            data=b"source",
            binding_created_at=datetime.now(UTC),
            revision_created_at=datetime.now(UTC),
        )

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult:
        del expected_revision
        self.calls.append("write_artifact")
        exact = ArtifactRef(namespace=target.namespace, name=target.name, revision="write-r1")
        self._remember(exact)
        return ArtifactWriteResult(
            apiVersion=API_VERSION,
            artifact=exact,
            mediaType=media_type,
            size=len(data),
        )

    def _remember(self, ref: ArtifactRef) -> None:
        assert ref.revision is not None
        self._known[(ref.namespace, ref.name, ref.revision)] = ref
        self._observed.append(ref)


class NoArtifactAccessClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_artifacts(self, namespace: str | None = None) -> list[ArtifactRef]:
        del namespace
        self.calls.append("list_artifacts")
        raise AssertionError("malformed Memory arguments reached ArtifactClient")

    async def read_artifact(self, ref: ArtifactRef) -> ArtifactValue:
        del ref
        self.calls.append("read_artifact")
        raise AssertionError("malformed Memory arguments reached ArtifactClient")

    async def write_artifact(self, *args: object, **kwargs: object) -> ArtifactWriteResult:
        del args, kwargs
        self.calls.append("write_artifact")
        raise AssertionError("malformed Memory arguments reached ArtifactClient")


class RefExposingTool:
    def __init__(self, ref: ArtifactRef) -> None:
        self.__name__ = "ref_probe"
        self.__doc__ = "Expose trusted test provenance."
        self._ref = ref
        self._observations = 0

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return (self._ref,)

    @property
    def artifact_observation_cursor(self) -> int:
        return self._observations

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return (self._ref,) if cursor < self._observations else ()

    async def __call__(self) -> dict[str, bool]:
        self._observations += 1
        return {"ok": True}
