"""Exercise registered invocation ceilings through native Runner dispatch."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from fakes.model import scripted_model, text_result, tool_call
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types
from shared_config_defaults import MAX_AGENT_STATE_SNAPSHOT_BYTES
from test_adk_runtime import create_runtime, stage_request

from contractor_runtime.contracts import WorkerSessionMode
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer
from contractor_runtime.telemetry.metrics import MAX_METRIC_TOOL_CALLS, MAX_REPORT_JSON_BYTES
from contractor_runtime.worker.sessions import (
    MAX_CARRIED_STATE_BYTES,
    WorkerSessionLifecycle,
    WorkerSessionLifecycleError,
)
from contractor_runtime.worker.state import WorkerStateStore as WorkerState


@pytest.mark.parametrize("proposed", [10_000, 10_001])
def test_registered_tool_ceiling_keeps_exact_effects_and_bounded_detail(tmp_path, proposed):
    async def scenario():
        effects = []

        async def scan(index: int) -> dict:
            """Record a deterministic local effect for one admitted integer."""
            effects.append(index)
            return {"index": index}

        response = tool_call("scan", {"index": 0}, call_id="scale-00000")
        # One native batch crosses the containing call bound without inventing
        # a ModelPolicy above the supported maximum.
        response.content.parts = [
            types.Part(
                function_call=types.FunctionCall(
                    id=f"scale-{index:05d}", name="scan", args={"index": index}
                )
            )
            for index in range(proposed)
        ]
        state = WorkerState()
        model = scripted_model([response, text_result("All admitted effects completed")])
        runtime = await create_runtime(
            tmp_path, state, {"scan": scan}, model, max_model_calls=3, max_tool_calls=10_000
        )
        try:
            result = await runtime.invoke(stage_request())
            snapshot = await runtime.agent_state_snapshot()
            report = state.metrics.build_report(report_id="tool-ceiling", duration_ms=0)
            if proposed == 10_000:
                assert result.result is not None and result.failure is None
                assert sorted(effects) == list(range(10_000))
                assert len(model.requests) == 3
                assert report.metrics.tools["scan"].calls == 10_000
                assert report.metrics.worker_budget.observed_tool_calls == 10_000
                assert len(state.metrics.tool_calls) == MAX_METRIC_TOOL_CALLS
                assert state.metrics.tool_calls[0].call_id == "tool-00009001"
                assert state.metrics.tool_calls[-1].call_id == "tool-00010000"
                assert report.truncated
            else:
                # The paired-history ledger rejects the oversized containing
                # batch before any tool admission, preserving its owning failure.
                assert result.failure.code == "worker_session_lifecycle_failed"
                assert result.failure.retryable is False
                assert effects == [] and len(model.requests) == 1
                assert report.metrics.worker_budget.observed_tool_calls == 0
                assert not runtime._accepting
                next_request = stage_request().model_copy(update={"subtask_id": "1"})
                assert (await runtime.invoke(next_request)).failure.code == "worker_draining"
                assert len(model.requests) == 1 and effects == []
            assert len(snapshot.model_dump_json(by_alias=True).encode()) <= (
                MAX_AGENT_STATE_SNAPSHOT_BYTES
            )
            assert len(report.model_dump_json(by_alias=True).encode()) <= MAX_REPORT_JSON_BYTES
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("over", [False, True])
def test_registered_token_ceiling_keeps_known_usage_and_no_recovery_dispatch(tmp_path, over):
    async def scenario():
        response = text_result("Completed at the registered token ceiling")
        # The ordinary finalizer contributes its known ten tokens.
        response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=99_999_980 + int(over),
            candidates_token_count=10,
            total_token_count=99_999_990 + int(over),
        )
        state = WorkerState()
        model = scripted_model([response])
        runtime = await create_runtime(
            tmp_path, state, {}, model, max_model_calls=2, max_total_tokens=100_000_000
        )
        try:
            result = await runtime.invoke(stage_request())
            assert len(model.requests) == 2
            report = state.metrics.build_report(report_id="token-ceiling", duration_ms=0)
            assert report.metrics.total_tokens == 100_000_000 + int(over)
            if over:
                assert result.failure.code == "worker_budget_exhausted"
                assert report.metrics.worker_budget.exhausted == "total_tokens"
            else:
                assert result.result is not None and result.failure is None
                assert report.metrics.worker_budget.exhausted is None
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("over", [False, True])
def test_registered_carried_state_byte_ceiling_and_deletion(over):
    async def scenario():
        service = InMemorySessionService()
        lifecycle = WorkerSessionLifecycle(
            mode=WorkerSessionMode.ISOLATED,
            app_name="scale-state",
            user_id="scale-owner",
            service=service,
        )
        # Compact UTF-8 JSON includes the key, quotes and braces.
        payload = {"payload": "x" * (MAX_CARRIED_STATE_BYTES - len(b'{"payload":""}') + over)}
        assert len(json.dumps(payload, separators=(",", ":")).encode()) == (
            MAX_CARRIED_STATE_BYTES + over
        )
        session_id = await lifecycle.begin_invocation({"stateRevision": 1})
        session = await lifecycle.active_session()
        await service.append_event(
            session,
            Event(
                invocation_id="state-limit",
                author="worker",
                actions=EventActions(state_delta=payload),
            ),
        )
        if over:
            with pytest.raises(WorkerSessionLifecycleError, match="state_too_large"):
                await lifecycle.finish_invocation()
            assert lifecycle.failed
        else:
            await lifecycle.finish_invocation()
            next_id = await lifecycle.begin_invocation({"stateRevision": 2})
            assert next_id != session_id
            carried = await lifecycle.active_session()
            assert carried.state["payload"] == payload["payload"]
            carried.state["payload"] = "mutated detached snapshot"
            assert (await lifecycle.active_session()).state["payload"] == payload["payload"]
            await lifecycle.finish_invocation()
        assert (
            await service.list_sessions(app_name="scale-state", user_id="scale-owner")
        ).sessions == []
        await lifecycle.close()
        assert "scale-state" not in service.app_state
        assert "scale-state" not in service.user_state

    asyncio.run(scenario())


def test_registered_tool_name_ceiling_keeps_all_aggregates_when_detail_is_full():
    async def scenario():
        reducer = InvocationMetricsReducer()
        state = WorkerState()
        await state.begin_invocation(
            invocation_id="scale-names", subtask_id="0", metrics=reducer.snapshot()
        )
        for index in range(257):
            reducer.record_tool_call(f"selected_{index:03d}", failed=index % 2 == 0)
        detail = reducer.snapshot()
        assert detail["toolCalls"] == 257 and detail["toolErrors"] == 129
        assert len(detail["tools"]) == 256 and detail["truncated"]
        assert detail["tools"]["selected_255"] == {"calls": 1, "failures": 0}
        assert "selected_256" not in detail["tools"]
        completed = await state.complete_invocation(
            invocation_id="scale-names", phase="succeeded", metrics=detail
        )
        assert completed["lastCompletedInvocation"]["metrics"] == detail
        # A consumer cannot mutate the projection or any nested stored counter.
        completed["lastCompletedInvocation"]["metrics"]["tools"]["selected_000"]["calls"] = 0
        assert (await state.snapshot())["lastCompletedInvocation"]["metrics"] == detail

    asyncio.run(scenario())
