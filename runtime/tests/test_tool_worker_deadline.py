from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from fakes.clock import advance_clock, run_with_advanceable_clock
from google.protobuf.json_format import ParseDict
from test_a2a_server import RecordingQueue, data_request, result_from_message
from test_tool_worker import ArtifactStore, FixtureTool, request, tool_spec, worker

from contractor_runtime.a2a_server import ContractorAgentExecutor


def deadline_request(spec, seconds=10):
    return request(spec).model_copy(
        update={"deadline": datetime.now(UTC) + timedelta(seconds=seconds)}
    )


async def through_a2a(runtime, spec, stage_request):
    envelope = data_request(spec.allocation_id)
    ParseDict(
        stage_request.model_dump(mode="json", by_alias=True, exclude_none=True),
        envelope.message.parts[0].data,
    )
    context = RequestContext(
        call_context=ServerCallContext(tenant=spec.allocation_id),
        request=envelope,
        task_id="deadline-tool",
        context_id="deadline-context",
    )
    queue = RecordingQueue()
    await ContractorAgentExecutor(runtime).execute(context, queue)
    return result_from_message(queue.terminal().status.message)


def assert_terminal(snapshot, phase="failed", calls=0):
    assert snapshot.state.current_invocation is None
    completed = snapshot.state.last_completed_invocation
    assert completed.phase == phase
    assert completed.metrics.tool_calls == calls
    assert completed.metrics.model_calls == 0


def test_shared_tool_stage_deadline_expired_a2a_request_starts_no_scan(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await through_a2a(runtime, spec, deadline_request(spec, -1))
        assert result["failure"]["code"] == "tool_timeout"
        assert result["failure"]["retryable"] is False
        assert tool.calls == [] and store.values == {}
        assert_terminal(await runtime.agent_state_snapshot())

    asyncio.run(scenario())


@pytest.mark.parametrize("boundary", ["lookup", "start_ack"])
def test_shared_tool_stage_deadline_passing_in_receipt_io_starts_no_scan(tmp_path, boundary):
    class ExpiringStore(ArtifactStore):
        async def request(self, method, path, **kwargs):
            result = await super().request(method, path, **kwargs)
            if (boundary == "lookup" and method == "GET") or (
                boundary == "start_ack" and method == "PUT" and self.writes == 1
            ):
                # The response wins before asyncio runs the deadline timer.
                advance_clock(11)
            return result

    async def scenario():
        store, tool, spec = ExpiringStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await through_a2a(runtime, spec, deadline_request(spec))
        assert result["failure"]["code"] == "tool_timeout"
        assert tool.calls == []
        assert (spec.namespace, "report") not in store.values
        assert_terminal(await runtime.agent_state_snapshot())
        if boundary == "start_ack":
            saved = json.loads(next(iter(store.values.values()))[0])
            assert saved["phase"] == "failed" and saved["errorCode"] == "tool_timeout"

    run_with_advanceable_clock(scenario())


def test_shared_tool_stage_deadline_during_ambiguous_start_keeps_no_rescan_fence(tmp_path):
    class LostStartStore(ArtifactStore):
        def __init__(self):
            super().__init__()
            self.committed = asyncio.Event()
            self.lose_ack = True

        async def request(self, method, path, **kwargs):
            result = await super().request(method, path, **kwargs)
            if method == "PUT" and self.lose_ack:
                self.committed.set()
                await asyncio.Event().wait()
            return result

    async def scenario():
        store, tool, spec = LostStartStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(deadline_request(spec)))
        await asyncio.wait_for(store.committed.wait(), 2)
        advance_clock(11)
        completion = await asyncio.wait_for(task, 2)
        assert completion.failure.code == "tool_timeout"
        assert tool.calls == []
        assert json.loads(next(iter(store.values.values()))[0])["phase"] == "started"
        assert_terminal(await runtime.agent_state_snapshot())
        store.lose_ack = False
        replacement = await worker(tmp_path, store, tool, spec=spec)
        replay = await replacement.invoke(deadline_request(spec, -1))
        assert replay.failure.code == "tool_outcome_unknown"
        assert tool.calls == []

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("ending", ["stage", "configured", "cancel"])
def test_shared_tool_stage_deadline_joins_real_child_before_completion(tmp_path, ending):
    class ChildTool(FixtureTool):
        def __init__(self):
            super().__init__()
            self.stopping = asyncio.Event()
            self.release_cleanup = asyncio.Event()
            self.child = None

        async def __call__(self, url: str, rate_limit: int = 10) -> dict:
            self.calls.append((url, rate_limit))
            self.child = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "import time; time.sleep(60)"
            )
            self.started.set()
            try:
                await self.child.wait()
                return self.result
            finally:
                self.child.terminate()
                await self.child.wait()
                self.stopping.set()
                await self.release_cleanup.wait()
                self.cancelled.set()

    async def scenario():
        store, tool, spec = ArtifactStore(), ChildTool(), tool_spec()
        spec.agent_template.execution.timeout_seconds = 10 if ending == "configured" else 60
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(
            runtime.invoke(deadline_request(spec, 60 if ending == "configured" else 10))
        )
        await asyncio.wait_for(tool.started.wait(), 2)
        if ending == "cancel":
            runtime.cancel_active(task)
        else:
            advance_clock(11)
        try:
            await asyncio.wait_for(tool.stopping.wait(), 2)
        except BaseException:
            tool.release_cleanup.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        assert tool.child.returncode is not None
        assert not task.done() and runtime._lock.locked()
        if ending == "cancel":
            # Cancellation wins before the absolute deadline, even if cleanup
            # crosses it. It must not be reclassified as a timeout.
            advance_clock(11)
            for _ in range(4):
                await asyncio.sleep(0)
            assert not task.done() and not tool.cancelled.is_set()
        tool.release_cleanup.set()
        if ending == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.failure.code == "tool_timeout"
            assert result.failure.retryable is False
        assert tool.cancelled.is_set() and not runtime._lock.locked()
        assert_terminal(
            await runtime.agent_state_snapshot(),
            "cancelled" if ending == "cancel" else "failed",
            calls=1,
        )
        replacement = await worker(tmp_path, store, tool, spec=spec)
        assert (await replacement.invoke(request(spec))).failure is not None
        assert len(tool.calls) == 1

    run_with_advanceable_clock(scenario())


def test_shared_tool_stage_deadline_does_not_change_completed_replay_or_input_identity(tmp_path):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        completion = await runtime.invoke(request(spec))
        expired = deadline_request(spec, -1)
        assert await runtime.invoke(expired) == completion
        replacement = await worker(tmp_path, store, tool, spec=spec)
        replay = await replacement.invoke(expired)
        assert replay.result.artifacts == completion.result.artifacts
        assert_terminal(await replacement.agent_state_snapshot(), "succeeded")
        expired.parameters = {"target": "https://different.invalid"}
        assert (await replacement.invoke(expired)).failure.code == "tool_input_conflict"
        other = await worker(tmp_path, store, tool, spec=spec)
        assert (await other.invoke(expired)).failure.code == "tool_input_conflict"
        assert len(tool.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["report", "terminal"])
@pytest.mark.parametrize("expire_io", [False, True])
def test_shared_tool_stage_deadline_publication_has_a_separate_budget(tmp_path, phase, expire_io):
    class PausingStore(ArtifactStore):
        def __init__(self):
            super().__init__()
            self.publishing = asyncio.Event()
            self.release = asyncio.Event()
            self.pause = True

        async def request(self, method, path, **kwargs):
            name = urlsplit(path).path.rsplit("/", 1)[-1]
            current_phase = "report" if name == "report" else "terminal" if self.writes else "start"
            if method == "PUT" and current_phase == phase and self.pause:
                self.publishing.set()
                await self.release.wait()
            return await super().request(method, path, **kwargs)

    async def scenario():
        store, tool, spec = PausingStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(deadline_request(spec)))
        await asyncio.wait_for(store.publishing.wait(), 2)
        # Semantic work already completed. Only the publication request budget
        # may stop this phase; the Stage deadline cannot reset or restart work.
        advance_clock(spec.runtime_settings.request_timeout_seconds + 1 if expire_io else 11)
        if not expire_io:
            store.release.set()
        result = await asyncio.wait_for(task, 2)
        if expire_io:
            assert result.failure.code == (
                "tool_report_failed" if phase == "report" else "tool_outcome_unknown"
            )
        else:
            assert result.result is not None
        assert_terminal(
            await runtime.agent_state_snapshot(), "failed" if expire_io else "succeeded", calls=1
        )
        store.pause = False
        replacement = await worker(tmp_path, store, tool, spec=spec)
        replay = await replacement.invoke(request(spec))
        assert (replay.failure is not None) == expire_io
        assert len(tool.calls) == 1

    run_with_advanceable_clock(scenario())


def test_shared_tool_stage_deadline_rejects_a_late_non_yielding_result(tmp_path):
    class LateTool(FixtureTool):
        async def __call__(self, url: str, rate_limit: int = 10) -> dict:
            result = await super().__call__(url, rate_limit)
            advance_clock(11)
            return result

    async def scenario():
        store, tool, spec = ArtifactStore(), LateTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        result = await runtime.invoke(deadline_request(spec))
        assert result.failure.code == "tool_timeout"
        assert (spec.namespace, "report") not in store.values
        assert_terminal(await runtime.agent_state_snapshot(), calls=1)

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("expire_cleanup", [False, True])
def test_shared_tool_stage_deadline_state_settlement_has_its_own_budget(tmp_path, expire_cleanup):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        runtime = await worker(tmp_path, store, tool, spec=spec)
        tool.wait = True
        finishing = asyncio.Event()
        release = asyncio.Event()
        original = runtime._state.complete_invocation

        async def delayed_finish(**kwargs):
            finishing.set()
            await release.wait()
            return await original(**kwargs)

        runtime._state.complete_invocation = delayed_finish
        task = asyncio.create_task(runtime.invoke(deadline_request(spec)))
        try:
            await asyncio.wait_for(tool.started.wait(), 2)
            advance_clock(11)
            await asyncio.wait_for(finishing.wait(), 2)
            # The Stage timer already fired. A fresh State-only budget owns
            # publication and cannot revive the cancelled tool.
            advance_clock(6 if expire_cleanup else 4)
            if expire_cleanup:
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(task), 2)
                assert not runtime._accepting
                assert (await runtime.invoke(request(spec))).failure.code == "worker_draining"
            else:
                release.set()
                assert (await task).failure.code == "tool_timeout"
                assert_terminal(await runtime.agent_state_snapshot(), calls=1)
            assert len(tool.calls) == 1 and tool.cancelled.is_set()
            assert not runtime._lock.locked()
        finally:
            # A failed assertion on the unfixed runtime must not strand the
            # terminal publication while asyncio.run drains its tasks.
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("failure", ["transport", "timeout"])
def test_shared_tool_stage_deadline_terminal_receipt_failure_keeps_fence(tmp_path, failure):
    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.wait = True
        store.fail_terminal = failure == "transport"
        store.block_write = "terminal" if failure == "timeout" else None
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(deadline_request(spec)))
        await asyncio.wait_for(tool.started.wait(), 2)
        advance_clock(11)
        if failure == "timeout":
            await asyncio.wait_for(store.write_started.wait(), 2)
            advance_clock(spec.runtime_settings.request_timeout_seconds + 1)
        result = await asyncio.wait_for(task, 2)
        assert result.failure.code == "tool_outcome_unknown"
        assert result.failure.retryable is False
        assert tool.cancelled.is_set()
        assert_terminal(await runtime.agent_state_snapshot(), calls=1)
        assert json.loads(next(iter(store.values.values()))[0])["phase"] == "started"
        store.fail_terminal = False
        store.block_write = None
        replacement = await worker(tmp_path, store, tool, spec=spec)
        assert (await replacement.invoke(request(spec))).failure.code == "tool_outcome_unknown"
        assert len(tool.calls) == 1

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("boundary", ["admission", "settlement"])
@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout"])
def test_shared_tool_state_joins_repeated_executor_cancellation(tmp_path, boundary, outcome):
    from contractor_runtime.worker.state import WorkerStateError

    async def scenario():
        store, tool, spec = ArtifactStore(), FixtureTool(), tool_spec()
        tool.wait = boundary == "settlement"
        spec.agent_template.execution.timeout_seconds = 60
        runtime = await worker(tmp_path, store, tool, spec=spec)
        executor = ContractorAgentExecutor(runtime)
        entered, release, interrupted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        operation = "begin_invocation" if boundary == "admission" else "complete_invocation"
        original = getattr(runtime._state, operation)
        state_task = None
        committed_while_owned = False

        async def gated(**kwargs):
            nonlocal state_task, committed_while_owned
            state_task = asyncio.current_task()
            entered.set()
            try:
                await release.wait()
                if outcome == "failed":
                    raise WorkerStateError("controlled State publication failure")
                result = await original(**kwargs)
                committed_while_owned = not owner.done() and runtime._lock.locked()
                return result
            except asyncio.CancelledError:
                interrupted.set()
                raise

        setattr(runtime._state, operation, gated)
        envelope = data_request(spec.allocation_id)
        ParseDict(
            deadline_request(spec).model_dump(mode="json", by_alias=True, exclude_none=True),
            envelope.message.parts[0].data,
        )
        context = RequestContext(
            call_context=ServerCallContext(tenant=spec.allocation_id),
            request=envelope,
            task_id="repeated-cancel-tool",
            context_id="repeated-cancel-context",
        )
        owner = asyncio.create_task(executor.execute(context, RecordingQueue()))
        try:
            if boundary == "settlement":
                await asyncio.wait_for(tool.started.wait(), 2)
                advance_clock(11)
            await asyncio.wait_for(entered.wait(), 2)
            if boundary == "settlement":
                assert tool.cancelled.is_set()
                receipt = json.loads(next(iter(store.values.values()))[0])
                assert receipt["phase"] == "failed" and receipt["errorCode"] == "tool_timeout"
            # Route both cancellations through the real A2A ownership map.
            # Yield between them so the second reaches the cleanup join after
            # the first has already interrupted the invocation's normal await.
            await executor.cancel(context, RecordingQueue())
            await asyncio.sleep(0)
            await executor.cancel(context, RecordingQueue())
            for _ in range(4):
                await asyncio.sleep(0)
            assert not owner.done() and runtime._lock.locked()
            assert not interrupted.is_set()
            assert state_task is not None and not state_task.done()
            if outcome == "timeout":
                advance_clock(6)
            else:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(owner), 2)
            assert state_task.done() and not runtime._lock.locked()
            assert tool.calls == (
                [] if boundary == "admission" else [(request(spec).parameters["target"], 10)]
            )
            if outcome == "completed":
                assert committed_while_owned
                assert runtime._accepting
                assert_terminal(
                    await runtime.agent_state_snapshot(),
                    "cancelled" if boundary == "admission" else "failed",
                    calls=0 if boundary == "admission" else 1,
                )
            else:
                assert not runtime._accepting
                assert (await runtime.invoke(request(spec))).failure.code == "worker_draining"
                assert not committed_while_owned
            if boundary == "settlement":
                replacement = await worker(tmp_path, store, tool, spec=spec)
                assert (await replacement.invoke(request(spec))).failure.code == "tool_timeout"
                assert len(tool.calls) == 1
            elif outcome == "completed":
                # An admission cancelled before scanning is not a poisoned
                # slot: an explicitly delivered next invocation can run.
                setattr(runtime._state, operation, original)
                next_request = request(spec).model_copy(update={"subtask_id": "next"})
                assert (await runtime.invoke(next_request)).result is not None
                assert len(tool.calls) == 1
        finally:
            release.set()
            owner.cancel()
            await asyncio.gather(owner, return_exceptions=True)
            if state_task is not None:
                await asyncio.gather(state_task, return_exceptions=True)

    run_with_advanceable_clock(scenario())
