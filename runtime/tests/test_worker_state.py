from __future__ import annotations

import asyncio
import copy
import gc
import json
import threading
import weakref
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from shared_config_defaults import MAX_AGENT_STATE_SNAPSHOT_BYTES
from test_allocation import NOW, TrackedMetricsState, make_service, make_spec

import contractor_runtime.worker.state as state_module
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    ExecutionError,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    ToolCallRecord,
)
from contractor_runtime.observations import WorkspaceToolObservation
from contractor_runtime.projectfs import WorkspaceObservationMetadata
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer
from contractor_runtime.telemetry.metrics import (
    MAX_METRIC_ERRORS,
    MAX_METRIC_TOOL_CALLS,
    MetricsState,
)
from contractor_runtime.worker.observations import WorkspaceObservationReducer
from contractor_runtime.worker.state import (
    WorkerStateError,
    WorkerStateStore,
)

SECRET = "worker-state-secret-canary"
_DUMP_ARGUMENTS: dict[str, Any] = {"mode": "json", "by_alias": True, "exclude_none": True}


def empty_invocation_metrics() -> dict[str, object]:
    return InvocationMetricsReducer().snapshot()


def encoded_envelope(state: dict[str, object]) -> bytes:
    return json.dumps(
        {"apiVersion": API_VERSION, "state": state},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def assert_complete_state(state: dict[str, object]) -> None:
    assert set(state) == {
        "schemaVersion",
        "stateRevision",
        "metrics",
        "currentInvocation",
        "lastCompletedInvocation",
    }
    assert state["schemaVersion"] == 2
    assert type(state["stateRevision"]) is int
    assert state["stateRevision"] > 0
    assert len(encoded_envelope(state)) <= MAX_AGENT_STATE_SNAPSHOT_BYTES
    current = state["currentInvocation"]
    completed = state["lastCompletedInvocation"]
    assert current is None or current["phase"] == "running"  # type: ignore[index]
    assert completed is None or completed["phase"] in {  # type: ignore[index]
        "succeeded",
        "failed",
        "cancelled",
    }


def test_state_tracks_sequential_invocations_with_immutable_snapshots() -> None:
    async def scenario() -> None:
        allocation_metrics = MetricsState()
        state = WorkerStateStore(allocation_metrics)
        revisions = [(await state.snapshot())["stateRevision"]]

        first_metrics = InvocationMetricsReducer()
        first = await state.begin_invocation(
            invocation_id="worker-first",
            subtask_id="0",
            metrics=first_metrics.snapshot(),
        )
        revisions.append(first["stateRevision"])
        assert first["currentInvocation"]["subtaskId"] == "0"
        first["currentInvocation"]["subtaskId"] = "tampered"
        assert (await state.snapshot())["currentInvocation"]["subtaskId"] == "0"

        allocation_metrics.record_model_call()
        first_metrics.record_model_call()
        published = await state.publish_invocation_metrics(
            invocation_id="worker-first",
            metrics=first_metrics.snapshot(),
        )
        revisions.append(published["stateRevision"])
        completed_first = await state.complete_invocation(
            invocation_id="worker-first",
            phase="succeeded",
            metrics=first_metrics.snapshot(),
        )
        revisions.append(completed_first["stateRevision"])
        retained_first = completed_first["lastCompletedInvocation"]
        assert completed_first["currentInvocation"] is None
        assert retained_first == {
            "invocationId": "worker-first",
            "subtaskId": "0",
            "phase": "succeeded",
            "metrics": first_metrics.snapshot(),
            "summarizer": {
                "phase": "disabled",
                "modelCalls": 0,
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
                "tokenUsageUnavailable": 0,
            },
            "workspace": None,
        }

        allocation_metrics.record_model_call()
        second_metrics = InvocationMetricsReducer()
        second_metrics.record_model_call()
        begun_second = await state.begin_invocation(
            invocation_id="worker-second",
            subtask_id="1",
            metrics=second_metrics.snapshot(),
        )
        revisions.append(begun_second["stateRevision"])
        assert begun_second["lastCompletedInvocation"] == retained_first
        assert begun_second["currentInvocation"]["subtaskId"] == "1"
        completed_second = await state.complete_invocation(
            invocation_id="worker-second",
            phase="failed",
            metrics=second_metrics.snapshot(),
        )
        revisions.append(completed_second["stateRevision"])

        assert revisions == sorted(set(revisions))
        assert completed_second["metrics"]["counters"]["llm_calls"] == 2
        assert completed_second["lastCompletedInvocation"]["subtaskId"] == "1"
        assert completed_second["lastCompletedInvocation"]["phase"] == "failed"
        assert_complete_state(completed_second)

    asyncio.run(scenario())


def test_state_rejects_untyped_or_content_bearing_invocation_facts() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        metrics = empty_invocation_metrics()
        metrics["prompt"] = SECRET
        with pytest.raises(WorkerStateError, match="fields"):
            await state.begin_invocation(
                invocation_id="worker-secret",
                subtask_id="0",
                metrics=metrics,
            )

        invalid_tool = empty_invocation_metrics()
        invalid_tool["toolCalls"] = 1
        invalid_tool["tools"] = {f"https://{SECRET}.example": {"calls": 1, "failures": 0}}
        with pytest.raises(WorkerStateError, match="name"):
            await state.begin_invocation(
                invocation_id="worker-secret",
                subtask_id="0",
                metrics=invalid_tool,
            )
        assert SECRET not in json.dumps(await state.snapshot())

    asyncio.run(scenario())


def test_state_admission_bounds_full_envelope_without_losing_aggregates() -> None:
    async def scenario() -> None:
        metrics = MetricsState()
        for index in range(1000):
            metrics.record_tool_call(
                "large_probe",
                arguments={"index": index, "safe": "x" * 4050},
                duration_ms=index,
            )
        state = WorkerStateStore(metrics)
        snapshot = await state.sync_metrics()

        assert_complete_state(snapshot)
        assert snapshot["metrics"]["counters"]["tool_calls"] == 1000
        assert snapshot["metrics"]["truncated"] is True
        assert len(snapshot["metrics"]["toolCalls"]) < 1000
        assert await state.encoded_snapshot_size() <= MAX_AGENT_STATE_SNAPSHOT_BYTES

    asyncio.run(scenario())


def test_snapshot_race_returns_only_complete_revisions_under_asyncio_debug() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        reducer = InvocationMetricsReducer()
        await state.begin_invocation(
            invocation_id="worker-race",
            subtask_id="1.2",
            metrics=reducer.snapshot(),
        )
        start_completion = asyncio.Event()
        heartbeat = 0
        observed: list[list[int]] = [[], [], []]

        async def publish() -> None:
            for index in range(100):
                reducer.record_tool_call("read_file", failed=index % 11 == 0)
                try:
                    await state.publish_invocation_metrics(
                        invocation_id="worker-race",
                        metrics=reducer.snapshot(),
                    )
                except WorkerStateError:
                    return
                if index == 50:
                    start_completion.set()
                await asyncio.sleep(0)

        async def complete() -> None:
            await start_completion.wait()
            await state.complete_invocation(
                invocation_id="worker-race",
                phase="cancelled",
                metrics=reducer.snapshot(),
            )

        async def read_snapshots(slot: int) -> None:
            nonlocal heartbeat
            for _ in range(150):
                snapshot = await state.snapshot()
                assert_complete_state(snapshot)
                observed[slot].append(snapshot["stateRevision"])
                heartbeat += 1
                await asyncio.sleep(0)

        await asyncio.gather(
            publish(),
            complete(),
            *(read_snapshots(slot) for slot in range(len(observed))),
        )
        final = await state.snapshot()
        assert final["currentInvocation"] is None
        assert final["lastCompletedInvocation"]["phase"] == "cancelled"
        assert heartbeat == 450
        assert all(values == sorted(values) for values in observed)

    asyncio.run(scenario(), debug=True)


@pytest.mark.parametrize("phase", ["failed", "cancelled"])
def test_completion_closes_a_pending_summarization_request(phase: str) -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        await state.begin_invocation(
            invocation_id="worker-cut-off",
            subtask_id="2.1",
            metrics=empty_invocation_metrics(),
            summarizer_enabled=True,
        )
        requested = await state.request_summarization(invocation_id="worker-cut-off")
        request_revision = requested["currentInvocation"]["summarizer"]["requestStateRevision"]

        # The invocation ends before the summarizer records an outcome.
        completed = await state.complete_invocation(
            invocation_id="worker-cut-off",
            phase=phase,
            metrics=empty_invocation_metrics(),
        )

        assert completed["lastCompletedInvocation"]["summarizer"] == {
            "phase": "failed",
            "requestStateRevision": request_revision,
            "modelCalls": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
            "tokenUsageUnavailable": 0,
            "failureCode": "cancelled",
        }
        summary_metrics = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert summary_metrics is not None
        assert summary_metrics.model_dump(by_alias=True) == {
            "attempts": 1,
            "succeeded": 0,
            "failed": 1,
            "modelCalls": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
            "tokenUsageUnavailable": 0,
            "failureCodes": {"cancelled": 1},
        }
        # The Server's and Runtime's State contracts accept the terminal revision.
        snapshot = await state.agent_state_snapshot()
        assert snapshot.state.last_completed_invocation is not None
        assert snapshot.state.last_completed_invocation.summarizer.phase == "failed"

        # An invocation without a request is completed as it is.
        await state.begin_invocation(
            invocation_id="worker-unrequested",
            subtask_id="2.2",
            metrics=empty_invocation_metrics(),
            summarizer_enabled=True,
        )
        unrequested = await state.complete_invocation(
            invocation_id="worker-unrequested",
            phase=phase,
            metrics=empty_invocation_metrics(),
        )
        assert unrequested["lastCompletedInvocation"]["summarizer"]["phase"] == "not_requested"
        summary_metrics = state.metrics.build_report(
            report_id="worker-report", duration_ms=1
        ).metrics.summarizer
        assert summary_metrics is not None and summary_metrics.attempts == 1

    asyncio.run(scenario())


def test_state_records_one_closed_terminal_summarizer_attempt() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        metrics = empty_invocation_metrics()
        begun = await state.begin_invocation(
            invocation_id="worker-summary",
            subtask_id="2.1",
            metrics=metrics,
            summarizer_enabled=True,
        )
        assert begun["currentInvocation"]["summarizer"]["phase"] == "not_requested"

        requested = await state.request_summarization(invocation_id="worker-summary")
        request_revision = requested["currentInvocation"]["summarizer"]["requestStateRevision"]
        assert request_revision == begun["stateRevision"]

        finished = await state.complete_summarization(
            invocation_id="worker-summary",
            succeeded=True,
            model_calls=1,
            input_tokens=80,
            output_tokens=20,
            total_tokens=100,
            token_usage_unavailable=0,
        )
        assert finished["currentInvocation"]["summarizer"] == {
            "phase": "succeeded",
            "requestStateRevision": request_revision,
            "modelCalls": 1,
            "inputTokens": 80,
            "outputTokens": 20,
            "totalTokens": 100,
            "tokenUsageUnavailable": 0,
        }
        with pytest.raises(WorkerStateError, match="not requested"):
            await state.complete_summarization(
                invocation_id="worker-summary",
                succeeded=False,
                model_calls=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                token_usage_unavailable=0,
                failure_code="duplicate",
            )

    asyncio.run(scenario())


def count_record_serializations(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count retained-record serializations without changing their output."""

    counts: Counter[str] = Counter()
    for model in (ToolCallRecord, ExecutionError):
        original = model.model_dump

        def counted(
            self: Any,
            *args: Any,
            _original: Any = original,
            _name: str = model.__name__,
            **kwargs: Any,
        ) -> dict[str, Any]:
            counts[_name] += 1
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(model, "model_dump", counted)
    return counts


def test_state_publication_serializes_each_retained_record_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dump_call = ToolCallRecord.model_dump
    dump_error = ExecutionError.model_dump
    counts = count_record_serializations(monkeypatch)

    async def scenario() -> None:
        metrics = MetricsState()
        state = WorkerStateStore(metrics)
        reducer = InvocationMetricsReducer()
        await state.begin_invocation(
            invocation_id="worker-cost",
            subtask_id="0",
            metrics=reducer.snapshot(),
        )
        failures = 0
        # Tool callbacks publish State before and after every call.
        for index in range(MAX_METRIC_TOOL_CALLS):
            await state.publish_invocation_metrics(
                invocation_id="worker-cost", metrics=reducer.snapshot()
            )
            failed = index % 5 == 0
            failures += failed
            metrics.record_tool_call(
                "read_file",
                arguments={},
                error=RuntimeError("probe") if failed else None,
                duration_ms=index,
            )
            reducer.record_tool_call("read_file", failed=failed)
            await state.publish_invocation_metrics(
                invocation_id="worker-cost", metrics=reducer.snapshot()
            )

        assert counts == {"ToolCallRecord": MAX_METRIC_TOOL_CALLS, "ExecutionError": failures}
        assert failures > MAX_METRIC_ERRORS

        published = dict(counts)
        for _ in range(3):
            await state.sync_metrics()
            await state.publish_invocation_metrics(
                invocation_id="worker-cost", metrics=reducer.snapshot()
            )
            await state.snapshot()
            await state.agent_state_snapshot()
        await state.complete_invocation(
            invocation_id="worker-cost", phase="succeeded", metrics=reducer.snapshot()
        )
        await state.begin_invocation(
            invocation_id="worker-cost-next",
            subtask_id="1",
            metrics=empty_invocation_metrics(),
        )
        assert counts == published

        # The reused projection is exactly the record serialization it replaces.
        snapshot = await state.snapshot()
        assert snapshot["metrics"]["toolCalls"] == [
            dump_call(record, **_DUMP_ARGUMENTS) for record in metrics.tool_calls
        ]
        assert snapshot["metrics"]["errors"] == [
            dump_error(error, **_DUMP_ARGUMENTS) for error in metrics.errors
        ]
        assert snapshot["metrics"]["toolCalls"][0]["callId"] == "tool-00000001"
        assert len(snapshot["metrics"]["errors"]) == MAX_METRIC_ERRORS
        assert snapshot["metrics"]["truncated"] is True
        assert_complete_state(snapshot)

    asyncio.run(scenario())


def tamper(snapshot: dict[str, Any]) -> None:
    detail = snapshot["metrics"]
    detail["toolCalls"][0]["arguments"]["nested"]["kind"] = "tampered"
    detail["toolCalls"][0]["tool"] = "tampered"
    detail["toolCalls"].append({"tampered": True})
    detail["errors"][0]["message"] = "tampered"
    detail["counters"]["tool_calls"] = 0
    for field_name in ("currentInvocation", "lastCompletedInvocation"):
        invocation = snapshot[field_name]
        if invocation is not None:
            invocation["metrics"]["tools"]["tampered"] = {"calls": 1, "failures": 1}
            invocation["summarizer"]["phase"] = "tampered"
            invocation["subtaskId"] = "tampered"


def test_returned_snapshots_never_alias_the_store_or_shared_projections() -> None:
    async def scenario() -> None:
        metrics = MetricsState()
        state = WorkerStateStore(metrics)
        reducer = InvocationMetricsReducer()
        metrics.record_tool_call(
            "read_file", arguments={"path": "a.py", "nested": {"kind": "source"}}, duration_ms=1
        )
        metrics.record_tool_call("read_file", arguments={}, error=RuntimeError("probe"))
        reducer.record_tool_call("read_file", failed=False)
        reducer.record_tool_call("read_file", failed=True)
        begun = await state.begin_invocation(
            invocation_id="worker-alias", subtask_id="0", metrics=reducer.snapshot()
        )
        published = await state.publish_invocation_metrics(
            invocation_id="worker-alias", metrics=reducer.snapshot()
        )
        completed = await state.complete_invocation(
            invocation_id="worker-alias", phase="succeeded", metrics=reducer.snapshot()
        )
        synced = await state.sync_metrics()
        read = await state.snapshot()
        canonical = copy.deepcopy(read)
        projection = copy.deepcopy(metrics.snapshot())

        for returned in (begun, published, completed, synced, read):
            tamper(returned)

        assert await state.snapshot() == canonical
        assert metrics.snapshot() == projection
        # The previous completed invocation survives the next one unchanged.
        following = await state.begin_invocation(
            invocation_id="worker-alias-next", subtask_id="1", metrics=empty_invocation_metrics()
        )
        assert following["lastCompletedInvocation"] == canonical["lastCompletedInvocation"]
        assert following["metrics"]["toolCalls"] == canonical["metrics"]["toolCalls"]
        assert following["metrics"]["errors"] == canonical["metrics"]["errors"]

    asyncio.run(scenario())


def test_overflow_admission_keeps_the_oldest_prefix_of_unchanged_projections() -> None:
    async def scenario() -> None:
        metrics = MetricsState()
        for index in range(MAX_METRIC_TOOL_CALLS):
            metrics.record_tool_call(
                "large_probe",
                arguments={"index": index, "safe": "x" * 4050},
                error=RuntimeError("probe") if index % 10 == 0 else None,
                duration_ms=index,
            )
        projection = copy.deepcopy(metrics.snapshot())
        state = WorkerStateStore(metrics)
        snapshot = await state.sync_metrics()

        admitted = snapshot["metrics"]["toolCalls"]
        assert snapshot["metrics"]["errors"] == projection["errors"]
        assert 0 < len(admitted) < MAX_METRIC_TOOL_CALLS
        assert admitted == projection["toolCalls"][: len(admitted)]
        assert snapshot["metrics"]["truncated"] is True
        assert len(encoded_envelope(snapshot)) <= MAX_AGENT_STATE_SNAPSHOT_BYTES
        widened = copy.deepcopy(snapshot)
        widened["metrics"]["toolCalls"] = projection["toolCalls"][: len(admitted) + 1]
        assert len(encoded_envelope(widened)) > MAX_AGENT_STATE_SNAPSHOT_BYTES
        # Admission re-slices, and never edits, the reducer's projections.
        assert metrics.snapshot() == projection
        assert await state.snapshot() == snapshot

    asyncio.run(scenario())


def large_workspace_observation() -> dict[str, Any]:
    paths = tuple(f"{index:05d}-{'x' * 180}.py" for index in range(9_000))
    reducer = WorkspaceObservationReducer.from_metadata(
        WorkspaceObservationMetadata(digest="sha256:" + "0" * 64, managed_text_paths=paths)
    )
    reducer.record(WorkspaceToolObservation(read=paths), ordinal=1)
    observation = reducer.snapshot()
    assert observation["scopeComplete"] is True
    assert observation["detailComplete"] is True
    return observation


def test_conflicting_revision_retries_admission_from_the_validated_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = large_workspace_observation()
    submitted = copy.deepcopy(observation)
    metrics = empty_invocation_metrics()

    async def scenario() -> None:
        reference = await WorkerStateStore().begin_invocation(
            invocation_id="worker-large", subtask_id="1", metrics=metrics, workspace=observation
        )
        fitted = reference["currentInvocation"]["workspace"]
        assert not (fitted["scopeComplete"] and fitted["detailComplete"])

        state = WorkerStateStore()
        admit = state_module._fit_snapshot
        loop = asyncio.get_running_loop()
        held = asyncio.Event()
        release = threading.Event()
        entered: list[dict[str, Any] | None] = []

        def held_admission(candidate: dict[str, Any]) -> dict[str, Any]:
            current = candidate["currentInvocation"]
            entered.append(copy.deepcopy(current["workspace"]) if current is not None else None)
            if len(entered) == 1:
                loop.call_soon_threadsafe(held.set)
                if not release.wait(timeout=60):
                    raise RuntimeError("held admission was never released")
            return admit(candidate)

        monkeypatch.setattr(state_module, "_fit_snapshot", held_admission)
        begin = asyncio.create_task(
            state.begin_invocation(
                invocation_id="worker-large",
                subtask_id="1",
                metrics=metrics,
                workspace=observation,
            )
        )
        try:
            await held.wait()
            # A concurrent lifecycle publication commits while admission runs.
            concurrent = await state.sync_metrics()
        finally:
            release.set()
        begun = await begin

        assert concurrent["stateRevision"] == 2
        assert begun["stateRevision"] == 3
        assert len(entered) == 3
        # The retried transform starts from the untrimmed validated observation.
        assert entered[0] == observation
        assert entered[1] is None
        assert entered[2] == observation
        assert begun["currentInvocation"]["workspace"] == fitted
        assert observation == submitted
        assert_complete_state(begun)

    asyncio.run(scenario())


def test_released_allocation_retains_no_worker_state_projection(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state_module, "MetricsState", TrackedMetricsState)

    async def scenario() -> None:
        _, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        assert service._context is not None
        worker_state = service._context.worker_state
        assert worker_state is not None
        assert type(worker_state.metrics) is TrackedMetricsState
        worker_state.metrics.record_tool_call(
            "read_artifact", arguments={"path": "safe.txt"}, duration_ms=1
        )
        assert (await worker_state.sync_metrics())["metrics"]["toolCalls"]
        retained_state = weakref.ref(worker_state)
        retained_metrics = weakref.ref(worker_state.metrics)
        del worker_state

        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="state-release",
                deadline=NOW + timedelta(seconds=30),
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        gc.collect()
        assert retained_state() is None
        # The record projections belong to the allocation's metrics reducer,
        # and release leaves nothing that holds it, ADK's caches included.
        assert retained_metrics() is None

    asyncio.run(scenario())


def test_validation_finishing_after_a_mutation_cannot_validate_the_new_revision(monkeypatch):
    async def scenario():
        state = WorkerStateStore()
        loop = asyncio.get_running_loop()
        started, unblock = asyncio.Event(), threading.Event()
        original = state_module._agent_state_snapshot

        def gated(snapshot):
            loop.call_soon_threadsafe(started.set)
            unblock.wait()
            return original(snapshot)

        monkeypatch.setattr(state_module, "_agent_state_snapshot", gated)
        assert await state.validated_state_revision() is None
        read = asyncio.create_task(state.agent_state_snapshot())
        await started.wait()
        try:
            changed = await state.sync_metrics()
        finally:
            unblock.set()
        old = await read
        assert old.state.state_revision < changed["stateRevision"]
        assert await state.validated_state_revision() is None
        current = await state.agent_state_snapshot()
        assert await state.validated_state_revision() == current.state.state_revision
        await state.sync_metrics()
        assert await state.validated_state_revision() is None

    asyncio.run(scenario())


def test_compaction_revisions_measurement_and_terminal_failure() -> None:
    async def scenario() -> None:
        state = WorkerStateStore()
        await state.begin_invocation(
            invocation_id="worker-1",
            subtask_id="1",
            metrics=empty_invocation_metrics(),
            compaction_enabled=True,
        )

        async def validated() -> int:
            assert await state.validated_state_revision() is None
            snapshot = await state.agent_state_snapshot()
            revision = snapshot.state.state_revision
            assert await state.validated_state_revision() == revision
            return revision

        revision = await validated()
        await state.request_compaction(invocation_id="worker-1", generation=0, prompt_tokens=600)
        assert await validated() > revision
        state.metrics.record_model_call(phase="compaction")
        await state.start_compaction(
            invocation_id="worker-1",
            input_bytes=100,
            reasoning_available=2,
            reasoning_included=1,
            reasoning_truncated=1,
        )
        await validated()
        state.metrics.record_model_usage(None, phase="compaction")
        state.metrics.record_model_duration("compaction", 25)
        await state.apply_compaction(invocation_id="worker-1", generation=1, summary_bytes=20)
        snapshot = await state.agent_state_snapshot()
        progress = snapshot.state.current_invocation.compaction
        assert progress.phase == "applied" and progress.after_prompt.prompt_tokens is None
        assert (
            state.metrics.model_phase_usage("compaction").duration_ms == 25
            and state.metrics.model_phase_usage("compaction").token_usage_unavailable == 1
        )
        await state.publish_invocation_metrics(
            invocation_id="worker-1", metrics=empty_invocation_metrics(), compaction_prompt=(1, 80)
        )
        await validated()
        # Duplicate/late observations cannot rewrite the first measured prompt.
        await state.publish_invocation_metrics(
            invocation_id="worker-1", metrics=empty_invocation_metrics(), compaction_prompt=(1, 90)
        )
        assert (
            await state.agent_state_snapshot()
        ).state.current_invocation.compaction.after_prompt.prompt_tokens == 80
        await state.request_compaction(invocation_id="worker-1", generation=1, prompt_tokens=700)
        await state.complete_invocation(
            invocation_id="worker-1", phase="cancelled", metrics=empty_invocation_metrics()
        )
        snapshot = await state.agent_state_snapshot()
        progress = snapshot.state.last_completed_invocation.compaction
        assert (progress.phase, progress.generation, progress.failures, progress.last_cause) == (
            "failed",
            1,
            1,
            "cancelled",
        )
        assert progress.after_prompt is None
        assert snapshot.state.last_completed_invocation.summarizer.phase == "disabled"

    asyncio.run(scenario())
