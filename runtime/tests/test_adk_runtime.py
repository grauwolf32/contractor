from __future__ import annotations

import asyncio
import base64
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
)
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.toolsets.memory import MemoryToolsetFactory
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-adk-gateway-token"


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
                text_result("Report created"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)
        assert runtime._agent is not None
        assert runtime._agent.output_schema is None
        assert not hasattr(runtime, "_finalizer_agent")

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.artifacts["report"].revision == "write-r1"
        assert client.calls == ["read_artifact", "write_artifact"]
        assert state.metrics.counters == {
            "input_tokens": 21,
            "llm_calls": 3,
            "output_tokens": 9,
            "outcomes.succeeded": 1,
            "tool_calls": 2,
            "tool_calls.read_artifact": 1,
            "tool_calls.write_artifact": 1,
            "total_tokens": 30,
        }
        budget = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.worker_budget
        assert budget is not None
        assert budget.max_model_calls == 8
        assert budget.max_tool_calls == 16
        assert budget.max_total_tokens == 32768
        assert budget.observed_model_calls == 3
        assert budget.observed_tool_calls == 2
        assert budget.observed_total_tokens == 30
        assert budget.token_usage_unavailable == 0
        assert budget.exhausted is None
        assert all(request["maxOutputTokens"] == 4096 for request in model.requests)
        assert all(request["temperature"] == 0.1 for request in model.requests)
        assert all(
            request["toolNames"] == ["read_artifact", "write_artifact"]
            for request in model.requests
        )
        session = await runtime._session_service.get_session(
            app_name=runtime._app_name,
            user_id=runtime._user_id,
            session_id=runtime._session_id,
        )
        assert session is not None
        assert session.state["metrics"]["finalOutcome"] == "succeeded"
        assert session.state["metrics"]["counters"]["tool_calls"] == 2
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
                text_result("Telemetry-safe result"),
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

        assert result.outcome.value == "succeeded"
        assert [span.name for span in instrumentation.spans] == [
            "contractor.worker.a2a_task",
            "contractor.worker.model",
            "contractor.worker.tool",
            "contractor.worker.model",
        ]
        assert [span.outcome for span in instrumentation.spans] == [
            "succeeded",
            "succeeded",
            "succeeded",
            "succeeded",
        ]
        model_spans = [
            span for span in instrumentation.spans if span.name == "contractor.worker.model"
        ]
        assert [span.attributes["tokens.total"] for span in model_spans] == [10, 10]
        assert instrumentation.spans[2].attributes["tool.name"] == "read_artifact"
        assert instrumentation.spans[0].attributes["counts.model_calls"] == 2
        assert instrumentation.spans[0].attributes["counts.tool_calls"] == 1
        rendered = repr([span.attributes for span in instrumentation.spans])
        for forbidden in (SECRET, "inputs", "source", "input-r1", "Telemetry-safe result"):
            assert forbidden not in rendered
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_treats_protocol_shaped_text_as_summary_and_blocks_secrets(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        shaped = '{"outcome":"succeeded","artifacts":{"report":{"revision":"guessed"}}}'
        model = scripted_model([text_result(shaped)])
        state = WorkerState()
        runtime = await create_runtime(
            tmp_path / "shaped",
            state,
            {},
            model,
        )
        result = await runtime.invoke(stage_request())
        assert result.outcome.value == "succeeded"
        assert result.summary == shaped
        assert result.artifacts == {}
        assert len(model.requests) == 1
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

        secret_bearing = scripted_model([text_result(SECRET)])
        secret_state = WorkerState()
        secret_runtime = await create_runtime(tmp_path / "secret", secret_state, {}, secret_bearing)
        secret_result = await secret_runtime.invoke(stage_request())
        assert secret_result.error is not None
        assert secret_result.error.code == "unsafe_worker_result"
        assert SECRET not in secret_result.model_dump_json(by_alias=True)
        assert SECRET not in repr(secret_state.metrics.snapshot())
        await secret_runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_maps_only_declared_runtime_result_bindings(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        hidden = ArtifactRef(namespace="builder", name="memory.hidden", revision="memory-r1")
        hidden_model = scripted_model(
            [
                tool_call("ref_probe", {}, call_id="hidden-ref"),
                text_result("Notebook update completed"),
            ]
        )
        first = await create_runtime(
            tmp_path / "hidden",
            WorkerState(),
            {"ref_probe": RefExposingTool(hidden)},
            hidden_model,
        )
        hidden_result = await first.invoke(stage_request())
        assert hidden_result.outcome.value == "succeeded"
        assert hidden_result.artifacts == {}
        await first.finalize(datetime.now(UTC) + timedelta(seconds=1))

        allowed = ArtifactRef(namespace="inputs", name="memory.workflow_input", revision="input-r1")
        allowed_model = scripted_model(
            [
                tool_call("ref_probe", {}, call_id="allowed-ref"),
                text_result("Purpose input selected"),
            ]
        )
        second = await create_runtime(
            tmp_path / "allowed",
            WorkerState(),
            {"ref_probe": RefExposingTool(allowed)},
            allowed_model,
        )
        request = stage_request().model_copy(
            update={
                "result_artifacts": {
                    "report": ArtifactRef(namespace="inputs", name="memory.workflow_input")
                }
            }
        )
        accepted = await second.invoke(request)
        assert accepted.outcome.value == "succeeded"
        assert accepted.artifacts["report"] == allowed
        await second.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_builds_result_from_plain_summary_and_trusted_artifacts(tmp_path: Path) -> None:
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
                text_result("Report created"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.artifacts["report"].revision == "write-r1"
        assert result.summary == "Report created"
        assert len(model.requests) == 2
        assert model.requests[1]["hasResponseSchema"] is False
        assert all(request["hasResponseSchema"] is False for request in model.requests)
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
                text_result("Could not update the report"),
            ]
        )
        runtime = await create_runtime(
            tmp_path,
            WorkerState(),
            {"ref_probe": FailingRefTool(ref)},
            model,
        )

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.artifacts == {}
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
                text_result("First task completed"),
                text_result("Second task completed without touching the result"),
            ]
        )
        runtime = await create_runtime(tmp_path, WorkerState(), tools, model)

        first = await runtime.invoke(stage_request())
        second = await runtime.invoke(stage_request())

        assert first.artifacts["report"].revision == "write-r1"
        assert second.outcome.value == "succeeded"
        assert second.artifacts == {}
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_plain_summary_needs_no_recovery_model_call(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([text_result("not-json")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_model_calls=1)

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert result.summary == "not-json"
        assert len(model.requests) == 1
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.exhausted is None
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
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Recovered from optional absence",
                        "artifacts": {},
                    }
                ),
            ]
        )
        runtime = await create_runtime(
            tmp_path, WorkerState(), {"load_optional": load_optional}, model
        )

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        assert len(model.requests) == 2
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
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Handled bounded memory failure",
                        "artifacts": {},
                    }
                ),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
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


def test_adk_worker_does_not_start_a_serializer_when_final_text_is_missing(
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

        assert result.error is not None
        assert result.error.code == "worker_result_missing"
        assert len(model.requests) == 2
        assert model.requests[0]["toolNames"] == ["read_artifact", "write_artifact"]
        assert model.requests[1]["toolNames"] == ["read_artifact", "write_artifact"]
        assert all("result_slot" not in request["contentText"] for request in model.requests)
        assert all("StageContentResult" not in request["contentText"] for request in model.requests)
        assert SECRET not in repr(state.metrics.snapshot())
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_missing_summary_fails_without_an_extra_model_turn(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model(
            [
                thought_result("The task is complete."),
            ]
        )
        runtime = await create_runtime(tmp_path, state, {}, model)

        result = await runtime.invoke(stage_request())

        assert result.error is not None
        assert result.error.code == "worker_result_missing"
        assert result.error.retryable
        assert len(model.requests) == 1
        assert model.requests[0]["toolNames"] == []
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_missing_summary_does_not_exhaust_model_budget(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model([thought_result("Task complete")])
        runtime = await create_runtime(tmp_path, state, {}, model, max_model_calls=1)

        result = await runtime.invoke(stage_request())

        assert result.error is not None
        assert result.error.code == "worker_result_missing"
        assert result.error.retryable
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

        assert result.error is not None
        assert result.error.code == "worker_budget_exhausted"
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

        assert result.error is not None
        assert result.error.code == "worker_budget_exhausted"
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
        response = json_result(
            {
                "apiVersion": API_VERSION,
                "outcome": "succeeded",
                "summary": "Done without provider token usage",
                "artifacts": {},
            }
        )
        response.usage_metadata = None
        state = WorkerState()
        model = scripted_model([response])
        runtime = await create_runtime(
            tmp_path, state, {}, model, max_model_calls=1, max_total_tokens=1
        )

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_model_calls == 1
        assert report.metrics.worker_budget.observed_total_tokens == 0
        assert report.metrics.worker_budget.token_usage_unavailable == 1
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_adk_worker_accepts_final_text_exactly_at_token_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = WorkerState()
        model = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Finished at the exact token ceiling",
                        "artifacts": {},
                    }
                )
            ]
        )
        runtime = await create_runtime(tmp_path, state, {}, model, max_total_tokens=10)

        result = await runtime.invoke(stage_request())

        assert result.outcome.value == "succeeded"
        report = state.metrics.build_report(report_id="worker-report", duration_ms=1)
        assert report.metrics.worker_budget is not None
        assert report.metrics.worker_budget.observed_total_tokens == 10
        assert report.metrics.worker_budget.exhausted is None
        await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


def test_abort_cancels_long_running_adk_invocation(tmp_path: Path) -> None:
    async def scenario() -> None:
        model = scripted_model(
            [
                json_result(
                    {
                        "apiVersion": API_VERSION,
                        "outcome": "succeeded",
                        "summary": "Too late",
                        "artifacts": {},
                    }
                )
            ],
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


async def create_runtime(
    tmp_path: Path,
    state: WorkerState,
    tools: dict[str, object],
    model: object,
    *,
    max_model_calls: int = 8,
    max_tool_calls: int = 16,
    max_total_tokens: int = 32768,
    instrumentation: RuntimeInstrumentation | None = None,
) -> AdkWorkerRuntime:
    tmp_path.mkdir(parents=True, exist_ok=True)
    context = build_context(tmp_path, state, tools)
    if instrumentation is not None:
        context = replace(
            context,
            adapter_handles=AdapterHandles(instrumentation=instrumentation),
        )
    context = replace(
        context,
        model_policy=context.model_policy.model_copy(
            update={
                "max_model_calls": max_model_calls,
                "max_tool_calls": max_tool_calls,
                "max_total_tokens": max_total_tokens,
            }
        ),
    )
    factory = AdkWorkerRuntimeFactory(lambda _: model)  # type: ignore[arg-type,return-value]
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
