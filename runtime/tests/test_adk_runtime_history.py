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
from fakes.worker_state import WorkerState
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


@pytest.mark.parametrize("mode", list(WorkerSessionMode))
@pytest.mark.parametrize("parallel", [False, True])
def test_child_tool_cancellation_cannot_become_missing_result(tmp_path, mode, parallel):
    async def scenario():
        sibling_settled = asyncio.Event()
        effects = []

        async def known_sibling() -> dict:
            """Return a real completed sibling exactly once."""
            effects.append("known")
            return {"known": "exact sibling value"}

        async def child_cancel() -> dict:
            """Cancel this native child after its sibling has fully settled."""
            if parallel:
                await sibling_settled.wait()
            effects.append("child")
            raise asyncio.CancelledError

        response = tool_call("child_cancel", {}, call_id="self-cancelled")
        if parallel:
            selected = tool_call("known_sibling", {}, call_id="known-sibling")
            selected.content.parts.extend(response.content.parts)
            response = selected
        state = WorkerState()
        model = CapturingScriptedLlm(
            model="deterministic-child-cancel",
            responses=[response, text_result("A later healthy invocation")],
        )
        runtime = await create_runtime(
            tmp_path,
            state,
            {"known_sibling": known_sibling, "child_cancel": child_cancel},
            model,
            session_mode=mode,
        )
        original_after = runtime._plugin.after_tool_callback

        async def after_tool(**kwargs):
            await original_after(**kwargs)
            if kwargs["tool"].name == "known_sibling":
                sibling_settled.set()

        runtime._plugin.after_tool_callback = after_tool
        with pytest.raises(asyncio.CancelledError):
            await runtime.invoke(stage_request())
        completed = (await state.snapshot())["lastCompletedInvocation"]
        assert completed["phase"] == "cancelled"
        assert completed["metrics"]["modelCalls"] == 1
        assert completed["metrics"]["toolCalls"] == (2 if parallel else 1)
        assert completed["metrics"]["toolErrors"] == 1
        assert len(model.requests) == 1
        expected = ["known", "child"] if parallel else ["child"]
        assert effects == expected
        next_payload = len(model.payloads)
        second = await runtime.invoke(stage_request().model_copy(update={"subtask_id": "1"}))
        assert second.result is not None and second.failure is None
        assert second.invocation_id != completed["invocationId"]
        assert effects == expected
        carried = model.payloads[next_payload]
        assert_balanced_tool_calls(carried)
        returned = responses(carried)
        if mode == WorkerSessionMode.SHARED:
            assert returned["self-cancelled"]["error"]["code"] == "tool_call_cancelled"
            if parallel:
                assert returned["known-sibling"] == {"known": "exact sibling value"}
        else:
            assert returned == {}
        assert len(model.requests) == 3
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


def test_child_cancellation_keeps_authority_when_its_history_seal_fails(tmp_path, monkeypatch):
    async def scenario():
        async def child_cancel() -> dict:
            """Cancel the native child without cancelling its owning task."""
            raise asyncio.CancelledError

        state = WorkerState()
        model = CapturingScriptedLlm(
            model="deterministic-child-cancel",
            responses=[tool_call("child_cancel", {}, call_id="child-before-bad-seal")],
        )
        runtime = await create_runtime(tmp_path, state, {"child_cancel": child_cancel}, model)

        async def unavailable(*_args, **_kwargs):
            raise OSError("seal-failure-content-canary")

        monkeypatch.setattr(runtime._tool_history, "seal", unavailable)
        with pytest.raises(asyncio.CancelledError):
            await runtime.invoke(stage_request())
        assert (await state.snapshot())["lastCompletedInvocation"]["phase"] == "cancelled"
        assert runtime._session_lifecycle.failed
        assert not runtime._accepting
        later = await runtime.invoke(stage_request())
        assert later.failure.code == "worker_draining"
        assert len(model.requests) == 1
        assert "seal-failure-content-canary" not in json.dumps(await state.snapshot())
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("prior_history", [True, False], ids=["history-first", "child-first"])
def test_child_cancel_keeps_the_first_history_or_cancellation_outcome(
    tmp_path, monkeypatch, prior_history
):
    async def scenario():
        known_settled = asyncio.Event()

        async def known() -> dict:
            """Return a known sibling before the controlled child cancellation."""
            return {"known": True}

        async def child_cancel() -> dict:
            """Cancel only after the earlier sibling boundary has settled."""
            await known_settled.wait()
            raise asyncio.CancelledError

        batch = tool_call("known", {}, call_id="known-before-order")
        batch.content.parts.extend(
            tool_call("child_cancel", {}, call_id="ordered-child").content.parts
        )
        state = WorkerState()
        model = CapturingScriptedLlm(model="order-probe", responses=[batch])
        runtime = await create_runtime(
            tmp_path, state, {"known": known, "child_cancel": child_cancel}, model
        )
        original = runtime._plugin.after_tool_callback

        async def after_tool(**kwargs):
            try:
                await original(**kwargs)
                if prior_history:
                    runtime._tool_history._fail("invalid_tool_history")
            finally:
                if kwargs["tool"].name == "known":
                    known_settled.set()

        monkeypatch.setattr(runtime._plugin, "after_tool_callback", after_tool)
        if not prior_history:

            async def failed_after_cancel(*_args, **_kwargs):
                runtime._tool_history._fail("tool_history_limit_exceeded")

            monkeypatch.setattr(runtime._tool_history, "failed", failed_after_cancel)
        try:
            if prior_history:
                result = await runtime.invoke(stage_request())
                assert result.failure is not None
                assert result.failure.code == "worker_session_lifecycle_failed"
                assert not result.failure.retryable
                assert (await state.snapshot())["lastCompletedInvocation"]["phase"] == "failed"
            else:
                with pytest.raises(asyncio.CancelledError):
                    await runtime.invoke(stage_request())
                assert (await state.snapshot())["lastCompletedInvocation"]["phase"] == "cancelled"
            assert not runtime._accepting
            assert runtime._session_lifecycle.failed
            assert len(model.requests) == 1
        finally:
            await runtime.abort(datetime.now(UTC) + timedelta(seconds=2))

    asyncio.run(scenario())
