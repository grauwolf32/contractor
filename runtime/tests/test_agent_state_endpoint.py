from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from fakes.allocation import allocation_context
from fakes.state_rpc import state_client
from test_allocation import NOW, make_service, make_spec

from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AgentStateSnapshot,
    FinalizeAllocationRequest,
)
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer

SECRET = "recognizable-agent-state-secret"


def test_agent_state_endpoint_returns_exact_snapshot_and_etag(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        assert allocation_context(service) is not None
        worker_state = allocation_context(service).worker_state
        assert worker_state is not None
        worker_state.metrics.record_tool_call(
            "read_artifact",
            arguments={"token": SECRET, "path": "safe.txt"},
            duration_ms=1,
        )
        invocation = InvocationMetricsReducer()
        await worker_state.begin_invocation(
            invocation_id="worker-state-test",
            subtask_id="1",
            metrics=invocation.snapshot(),
        )
        invocation.record_tool_call("read_artifact", failed=False)
        completed = await worker_state.complete_invocation(
            invocation_id="worker-state-test",
            phase="succeeded",
            metrics=invocation.snapshot(),
        )

        async with state_client(state, service) as client:
            first = await client.read(request_id="state-read-1")
            assert first.status == grpc.StatusCode.OK

            assert first.revision == (completed["stateRevision"])
            assert first.request_id == "state-read-1"
            snapshot = AgentStateSnapshot.model_validate_json(first.document)
            assert snapshot.state.state_revision == completed["stateRevision"]
            assert snapshot.state.last_completed_invocation is not None
            assert snapshot.state.last_completed_invocation.subtask_id == "1"
            assert "allocation-1" not in first.document.decode()
            assert SECRET not in first.document.decode()

            unchanged = await client.read(revision=first.revision)
            assert unchanged.not_modified
            assert unchanged.document == b""
            assert unchanged.revision == first.revision

            nonmatching = await client.read(revision=1)
            assert nonmatching.status == grpc.StatusCode.OK

    asyncio.run(scenario())


def test_agent_state_endpoint_is_read_only_and_lifecycle_bound(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        async with state_client(state, service) as client:
            wrong = await client.read("another-allocation")
            assert wrong.status == grpc.StatusCode.NOT_FOUND
            assert wrong.error.code == "allocation_not_found"

            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="state-finalize",
                    deadline=NOW + timedelta(seconds=30),
                )
            )
            stopped = await client.read()
            assert stopped.status == grpc.StatusCode.FAILED_PRECONDITION
            assert stopped.error.code == "agent_state_unavailable"
            assert stopped.error.retryable is True

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["invalid", "oversized", "infrastructure"])
def test_agent_state_endpoint_classifies_snapshot_failures(
    tmp_path, runtime_capabilities, monkeypatch, failure
):
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        if failure == "invalid":
            allocation_context(service).worker_state._state["stateRevision"] = 0
        elif failure == "oversized":
            # Exercise the wire-size guard after the model has validated the snapshot.
            monkeypatch.setattr(
                "contractor_runtime.grpc_server.get_shared_config",
                lambda: SimpleNamespace(worker_state_max_bytes=1),
            )
        else:

            async def unavailable():
                raise RuntimeError(SECRET)

            monkeypatch.setattr(
                allocation_context(service).worker, "agent_state_snapshot", unavailable
            )
        async with state_client(state, service) as client:
            for _ in range(2):
                response = await client.read()
                assert response.status == grpc.StatusCode.UNAVAILABLE
                assert response.error.code == "agent_state_unavailable"
                assert response.error.retryable is (failure == "infrastructure")

                assert SECRET not in str(response.error) + response.document.decode()

    asyncio.run(scenario())


def test_agent_state_snapshot_races_remain_revision_consistent_and_nonblocking(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        assert (
            allocation_context(service) is not None
            and allocation_context(service).worker_state is not None
        )
        worker_state = allocation_context(service).worker_state
        metrics = InvocationMetricsReducer()
        await worker_state.begin_invocation(
            invocation_id="worker-state-race",
            subtask_id="2",
            metrics=metrics.snapshot(),
        )
        heartbeat = 0
        stop_heartbeat = asyncio.Event()

        async def publish() -> None:
            for index in range(100):
                metrics.record_tool_call("read_file", failed=index % 17 == 0)
                await worker_state.publish_invocation_metrics(
                    invocation_id="worker-state-race",
                    metrics=metrics.snapshot(),
                )
                await asyncio.sleep(0)

        async def tick() -> None:
            nonlocal heartbeat
            while not stop_heartbeat.is_set():
                heartbeat += 1
                await asyncio.sleep(0)

        async with state_client(state, service) as client:
            ticker = asyncio.create_task(tick())

            async def read_many() -> list[int]:
                revisions: list[int] = []
                for _ in range(40):
                    response = await client.read()
                    assert response.status == grpc.StatusCode.OK
                    snapshot = AgentStateSnapshot.model_validate_json(response.document)
                    revisions.append(snapshot.state.state_revision)
                    assert len(response.document) <= 4 * 1024 * 1024
                return revisions

            observed = await asyncio.gather(publish(), read_many(), read_many(), read_many())
            stop_heartbeat.set()
            await ticker

        for revisions in observed[1:]:
            assert revisions == sorted(revisions)
        assert heartbeat > 0
        retained = json.dumps(await worker_state.snapshot())
        assert SECRET not in retained

    asyncio.run(scenario(), debug=True)


def test_large_unchanged_state_revalidation_runs_no_full_snapshot_work(
    tmp_path, runtime_capabilities, monkeypatch
):
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        store = allocation_context(service).worker_state
        for index in range(1000):
            store.metrics.record_tool_call(
                "read_file", arguments={"index": index, "safe": "x" * 3500}, duration_ms=index
            )
        await store.sync_metrics()
        async with state_client(state, service) as client:
            first = await client.read()
            assert first.status == grpc.StatusCode.OK
            assert len(first.document) > 3_000_000

            def unexpected(*args, **kwargs):
                raise RuntimeError("unchanged revision repeated full snapshot work")

            # Reject each costly boundary; the 304 must not enter any of them.
            with monkeypatch.context() as patch:
                patch.setattr(store, "snapshot", unexpected)
                patch.setattr("contractor_runtime.worker.state._agent_state_snapshot", unexpected)
                patch.setattr(AgentStateSnapshot, "model_dump_json", unexpected)
                for _ in range(3):
                    response = await client.read(revision=first.revision)
                    assert response.not_modified
                    assert response.document == b""
                    assert response.revision == first.revision

            await store.sync_metrics()
            changed = await client.read(revision=first.revision)
            assert changed.status == grpc.StatusCode.OK
            assert changed.revision != first.revision

    asyncio.run(scenario())


def test_fresh_and_changed_conditional_revisions_must_pass_validation(
    tmp_path, runtime_capabilities, monkeypatch
):
    from contractor_runtime.worker import state as state_module

    original = state_module._agent_state_snapshot
    validations = []

    def counted(snapshot):
        validations.append(snapshot["stateRevision"])
        return original(snapshot)

    monkeypatch.setattr(state_module, "_agent_state_snapshot", counted)

    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        store = allocation_context(service).worker_state
        async with state_client(state, service) as client:
            guessed = 1
            first = await client.read(revision=guessed)
            assert first.not_modified
            assert validations == [1]
            repeated = await client.read(revision=guessed)
            assert repeated.not_modified
            assert validations == [1]

            changed = await store.sync_metrics()
            revision = changed["stateRevision"]
            # Fault injection into a new revision must never be hidden by the
            # preceding revision's successful validation, even with a guessed ETag.
            store._state["currentInvocation"] = {"unknown": True}
            invalid = await client.read(revision=revision)
            assert invalid.status == grpc.StatusCode.UNAVAILABLE
            assert invalid.error.retryable is False
            assert validations == [1, revision]
            assert await store.validated_state_revision() is None

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", ["draining", "stopped", "released", "replaced"])
def test_conditional_state_reuse_is_bound_to_the_live_allocation(
    tmp_path, runtime_capabilities, monkeypatch, terminal
):
    from contractor_runtime.worker import state as state_module

    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        async with state_client(state, service) as client:
            first = await client.read()
            assert first.status == grpc.StatusCode.OK
            if terminal == "draining":
                await state.begin_draining("allocation-1")
            else:
                await service.expire_control_lease(10)
                if terminal in {"released", "replaced"}:
                    from contractor_runtime.contracts import ReleaseAllocationRequest

                    await service.release(
                        ReleaseAllocationRequest(
                            apiVersion=API_VERSION, allocationId="allocation-1"
                        )
                    )
                    await service.confirm_release("allocation-1")
                if terminal == "replaced":
                    await state.confirm_release(None)
                    await service.prepare(make_spec())

            calls = []
            original = state_module._agent_state_snapshot

            def counted(snapshot):
                calls.append(snapshot["stateRevision"])
                return original(snapshot)

            monkeypatch.setattr(state_module, "_agent_state_snapshot", counted)
            response = await client.read(revision=first.revision)
            if terminal == "replaced":
                assert response.not_modified
                assert calls == [1]  # Same revision number, a different owner: validate again.
            else:
                assert response.status == (
                    grpc.StatusCode.NOT_FOUND
                    if terminal == "released"
                    else grpc.StatusCode.FAILED_PRECONDITION
                )
                assert calls == []

    asyncio.run(scenario())
