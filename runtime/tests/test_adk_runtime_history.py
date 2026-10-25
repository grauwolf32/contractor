"""The actual Worker seals truthful interrupted history before another invocation."""

from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.model import text_result, tool_call
from test_adk_runtime import (
    PAST_STAGE_DEADLINE_SECONDS,
    CapturingScriptedLlm,
    RecordingInstrumentation,
    assert_balanced_tool_calls,
    assert_slot_released,
    create_runtime,
    stage_request,
)

import contractor_runtime.worker.runtime as runtime_module
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import WorkerSessionMode
from contractor_runtime.worker.history import ToolHistory, ToolHistoryError
from contractor_runtime.worker.one_shot import OneShotSession


def responses(payload):
    return {
        message["tool_call_id"]: json.loads(message["content"])
        for message in payload["messages"]
        if message["role"] == "tool"
    }


async def assert_session_responses_once(runtime, expected):
    session_id = runtime._session_lifecycle.shared_session_id
    assert session_id is not None
    session = await runtime._session_service.get_session(
        app_name=runtime._app_name, user_id=runtime._user_id, session_id=session_id
    )
    actual = [
        response.id for event in session.events for response in event.get_function_responses()
    ]
    assert sorted(actual) == sorted(expected)


def test_started_tool_cancellation_keeps_paired_truthful_history_for_next_shared_invocation(
    tmp_path,
):
    async def scenario():
        entered, finished = asyncio.Event(), asyncio.Event()
        effects = []

        async def wait_for_release() -> dict:
            """Start a controlled operation whose cancellation cannot prove absence of effects."""
            effects.append("started")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
            return {"impossible": "completed"}

        model = CapturingScriptedLlm(
            model="scripted",
            responses=[
                tool_call("wait_for_release", {}, call_id="cancelled-call"),
                text_result("Recovered on the next subtask"),
            ],
        )
        state = WorkerState()
        runtime = await create_runtime(
            tmp_path,
            state,
            {"wait_for_release": wait_for_release},
            model,
            session_mode=WorkerSessionMode.SHARED,
        )
        try:
            task = asyncio.create_task(runtime.invoke(stage_request()))
            await asyncio.wait_for(entered.wait(), timeout=10)
            runtime.cancel_active(task)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            assert effects == ["started"]
            assert not runtime._session_lifecycle.failed
            assert runtime._active_task is None
            assert runtime._active_budget is None
            assert not runtime._invoke_lock.locked()
            snapshot = await runtime._worker_state.snapshot()
            assert snapshot["lastCompletedInvocation"]["phase"] == "cancelled"
            payload_count = len(model.payloads)
            result = await runtime.invoke(stage_request().model_copy(update={"subtask_id": "1"}))
            assert result.result is not None
            payload = model.payloads[payload_count]
            assert_balanced_tool_calls(payload)
            assert responses(payload)["cancelled-call"]["error"]["code"] == "tool_call_cancelled"
            assert effects == ["started"]
            await assert_session_responses_once(runtime, ["cancelled-call"])
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


@pytest.mark.parametrize("cancelled", [False, True])
def test_parallel_interruption_keeps_real_completed_sibling_and_no_repeated_effects(
    tmp_path: Path, cancelled: bool
):
    async def scenario():
        entered, finished = asyncio.Event(), asyncio.Event()
        effects = []

        async def probe(tag: str) -> dict:
            """Return exact evidence once, or wait for a cancellation barrier."""
            effects.append(tag)
            if tag == "waiting":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()
            return {"ok": True, "effect": tag, "ordinal": len(effects)}

        selected = tool_call("probe", {"tag": "completed"}, call_id="completed-sibling")
        selected.content.parts.extend(
            tool_call(
                "probe",
                {"tag": "waiting" if cancelled else "unadmitted"},
                call_id="interrupted-sibling",
            ).content.parts
        )
        model = CapturingScriptedLlm(
            model="scripted", responses=[selected, text_result("Recovered next subtask")]
        )
        state = WorkerState()
        runtime = await create_runtime(
            tmp_path,
            state,
            {"probe": probe},
            model,
            max_tool_calls=2 if cancelled else 1,
            session_mode=WorkerSessionMode.SHARED,
        )
        try:
            if cancelled:
                task = asyncio.create_task(runtime.invoke(stage_request()))
                await asyncio.wait_for(entered.wait(), timeout=10)
                runtime.cancel_active(task)
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert finished.is_set()
            else:
                first = await runtime.invoke(stage_request())
                assert first.failure is not None
                assert first.failure.code == "worker_budget_exhausted"
                assert first.failure.retryable
            assert not runtime._session_lifecycle.failed
            payload_count = len(model.payloads)
            second = await runtime.invoke(stage_request().model_copy(update={"subtask_id": "1"}))
            assert second.result is not None
            payload = model.payloads[payload_count]
            assert_balanced_tool_calls(payload)
            result_by_id = responses(payload)
            assert result_by_id["completed-sibling"] == {
                "ok": True,
                "effect": "completed",
                "ordinal": 1,
            }
            assert result_by_id["interrupted-sibling"]["error"]["code"] == (
                "tool_call_cancelled" if cancelled else "tool_call_not_executed"
            )
            assert effects == (["completed", "waiting"] if cancelled else ["completed"])
            await assert_session_responses_once(
                runtime, ["completed-sibling", "interrupted-sibling"]
            )
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_swallowed_after_tool_history_overflow_fences_completion_and_future_calls(
    tmp_path, monkeypatch
):
    class SmallHistory(ToolHistory):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **(kwargs | {"max_retained_bytes": 512}))

    monkeypatch.setattr(runtime_module, "ToolHistory", SmallHistory)

    async def scenario():
        effects, caught = [], []

        async def large_result() -> dict:
            """Produce one real completed result that exceeds the controlled pending cap."""
            effects.append("executed")
            return {"payload": "x" * 1000}

        model = CapturingScriptedLlm(
            model="scripted",
            responses=[
                tool_call("large_result", {}, call_id="large-call"),
                text_result("false success"),
            ],
        )
        runtime = await create_runtime(
            tmp_path, WorkerState(), {"large_result": large_result}, model
        )
        after_tool = runtime._plugin.after_tool_callback

        async def swallow(**kwargs):
            try:
                await after_tool(**kwargs)
            except ToolHistoryError as error:
                caught.append(error.code)

        monkeypatch.setattr(runtime._plugin, "after_tool_callback", swallow)
        try:
            completion = await runtime.invoke(stage_request())
            assert completion.result is None
            assert completion.failure is not None
            assert completion.failure.code == "worker_session_lifecycle_failed"
            assert not completion.failure.retryable
            assert caught == ["tool_history_limit_exceeded"]
            assert effects == ["executed"]
            assert len(model.payloads) == 1
            assert runtime._session_lifecycle.failed
            next_completion = await runtime.invoke(
                stage_request().model_copy(update={"subtask_id": "1"})
            )
            assert next_completion.result is None
            assert next_completion.failure.code == "worker_draining"
            assert len(model.payloads) == 1
            assert effects == ["executed"]
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_cancelled_invocation_joins_its_shielded_history_append_before_return(
    tmp_path, monkeypatch
):
    async def scenario():
        tool_entered, seal_entered, seal_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def blocked() -> dict:
            """Wait until cancellation of the owning Worker invocation."""
            tool_entered.set()
            await asyncio.Event().wait()
            return {"unexpected": "success"}

        model = CapturingScriptedLlm(
            model="scripted", responses=[tool_call("blocked", {}, call_id="cancelled-call")]
        )
        runtime = await create_runtime(tmp_path, WorkerState(), {"blocked": blocked}, model)
        append = runtime._session_service.append_event

        async def delayed_append(session, event):
            if event.id.startswith("worker-history-"):
                seal_entered.set()
                await seal_release.wait()
            return await append(session=session, event=event)

        monkeypatch.setattr(runtime._session_service, "append_event", delayed_append)
        task = asyncio.create_task(runtime.invoke(stage_request()))
        try:
            await asyncio.wait_for(tool_entered.wait(), timeout=10)
            runtime.cancel_active(task)
            await asyncio.wait_for(seal_entered.wait(), timeout=10)
            assert not task.done()
            assert runtime._invoke_lock.locked()
            assert runtime._active_task is task
            seal_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime._active_task is None
            assert not runtime._invoke_lock.locked()
            assert len(model.payloads) == 1
            await assert_session_responses_once(runtime, ["cancelled-call"])
        finally:
            seal_release.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_sealing_uses_bounded_public_reads_without_copying_full_history(tmp_path, monkeypatch):
    async def scenario():
        async def probe(tag: str) -> dict:
            """Complete one exact result before the next call exceeds its budget."""
            return {"ok": True, "tag": tag}

        selected = tool_call("probe", {"tag": "completed"}, call_id="completed-call")
        selected.content.parts.extend(
            tool_call("probe", {"tag": "unadmitted"}, call_id="unadmitted-call").content.parts
        )
        model = CapturingScriptedLlm(model="scripted", responses=[selected])
        runtime = await create_runtime(
            tmp_path, WorkerState(), {"probe": probe}, model, max_tool_calls=1
        )
        inside_seal = ContextVar("test_inside_history_seal", default=False)
        seal_reads, ordinary_reads = [], []
        original_seal = runtime._seal_tool_history
        original_get = runtime._session_service.get_session

        async def observed_seal():
            token = inside_seal.set(True)
            try:
                await original_seal()
            finally:
                inside_seal.reset(token)

        async def observed_get(**kwargs):
            config = kwargs.get("config")
            if inside_seal.get():
                seal_reads.append(config.num_recent_events if config is not None else None)
            else:
                ordinary_reads.append(config)
            return await original_get(**kwargs)

        monkeypatch.setattr(runtime, "_seal_tool_history", observed_seal)
        monkeypatch.setattr(runtime._session_service, "get_session", observed_get)
        try:
            completion = await runtime.invoke(stage_request())
            assert completion.failure.code == "worker_budget_exhausted"
            assert seal_reads and all(limit is not None and 0 < limit <= 7 for limit in seal_reads)
            assert any(config is None for config in ordinary_reads)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())


def test_seal_failure_during_cancel_preserves_cancel_and_fences_shared_reuse(
    tmp_path, monkeypatch, caplog
):
    async def scenario():
        entered = asyncio.Event()

        async def blocked() -> dict:
            """Remain in flight until the invocation owner is cancelled."""
            entered.set()
            await asyncio.Event().wait()
            return {"unexpected": "success"}

        model = CapturingScriptedLlm(
            model="scripted", responses=[tool_call("blocked", {}, call_id="in-flight-call")]
        )
        state = WorkerState()
        runtime = await create_runtime(tmp_path, state, {"blocked": blocked}, model)
        append = runtime._session_service.append_event

        async def unavailable_append(session, event):
            if event.id.startswith("worker-history-"):
                raise RuntimeError("private-session-transport-canary")
            return await append(session=session, event=event)

        monkeypatch.setattr(runtime._session_service, "append_event", unavailable_append)
        try:
            task = asyncio.create_task(runtime.invoke(stage_request()))
            await asyncio.wait_for(entered.wait(), timeout=10)
            runtime.cancel_active(task)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runtime._session_lifecycle.failed
            assert runtime._active_task is None
            assert not runtime._invoke_lock.locked()
            snapshot = await runtime._worker_state.snapshot()
            assert snapshot["lastCompletedInvocation"]["phase"] == "cancelled"
            completion = await runtime.invoke(
                stage_request().model_copy(update={"subtask_id": "1"})
            )
            assert completion.result is None
            assert completion.failure.code == "worker_draining"
            assert len(model.payloads) == 1
            assert "private-session-transport-canary" not in repr(state.metrics.snapshot())
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    asyncio.run(scenario())
    assert "private-session-transport-canary" not in caplog.text


@pytest.mark.parametrize("expired", [False, True])
def test_finalizer_sdk_admission_respects_owner_deadline_and_nested_timeout_classification(
    tmp_path, monkeypatch, expired
):
    original_run = OneShotSession.run
    boundaries = []

    async def before_sdk_run(session, *, prompt, invocation_id):
        if session._session_id.startswith("result-finalizer-"):
            boundaries.append(invocation_id)
            if expired:
                # Cross the absolute deadline after serializer setup but before
                # ADK actually admits the finalizer's model callback.
                advance_clock(PAST_STAGE_DEADLINE_SECONDS)
            else:
                raise TimeoutError("controlled nested serializer timeout")
        return await original_run(session, prompt=prompt, invocation_id=invocation_id)

    monkeypatch.setattr(OneShotSession, "run", before_sdk_run)

    async def scenario():
        model = CapturingScriptedLlm(model="scripted", responses=[text_result("main finished")])
        state = WorkerState()
        instrumentation = RecordingInstrumentation()
        runtime = await create_runtime(tmp_path, state, {}, model, instrumentation=instrumentation)
        try:
            completion = await asyncio.wait_for(
                runtime.invoke(stage_request()), timeout=PAST_STAGE_DEADLINE_SECONDS + 30
            )
            assert completion.result is None
            assert completion.failure.code == (
                "worker_timeout" if expired else "worker_result_finalizer_failed"
            )
            assert len(boundaries) == 1
            assert len(model.payloads) == 1
            assert state.metrics.counters["llm_calls"] == 1
            model_spans = [
                span for span in instrumentation.spans if span.name == "contractor.worker.model"
            ]
            assert len(model_spans) == 1
            assert model_spans[0].outcome == "succeeded"
            assert all(span.end_calls == 1 for span in instrumentation.spans)
            assert_slot_released(runtime)
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))

    run_with_advanceable_clock(scenario())
