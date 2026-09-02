from __future__ import annotations

import asyncio
import json

import pytest

from contractor_runtime.contracts import API_VERSION
from contractor_runtime.instrumentation import InvocationMetricsReducer
from contractor_runtime.metrics import MetricsState
from contractor_runtime.worker_state import (
    MAX_AGENT_STATE_SNAPSHOT_BYTES,
    WorkerStateError,
    WorkerStateStore,
)

SECRET = "worker-state-secret-canary"


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
    assert state["schemaVersion"] == 1
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
