from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_allocation import NOW, make_service, make_spec

from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AgentStateSnapshot,
    FinalizeAllocationRequest,
)
from contractor_runtime.server import create_app
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer

STATE_PATH = "/private/v1/allocations/allocation-1/agent-state"
SECRET = "recognizable-agent-state-secret"


def test_agent_state_endpoint_returns_exact_snapshot_and_etag(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        assert service._context is not None
        worker_state = service._context.worker_state
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

        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://runtime.example",
        ) as client:
            first = await client.get(
                STATE_PATH,
                headers={"X-Request-ID": "state-read-1"},
            )
            assert first.status_code == 200
            assert first.headers["cache-control"] == "private, no-cache"
            assert first.headers["etag"] == (
                f'"contractor-agent-state-v1-{completed["stateRevision"]}"'
            )
            assert first.headers["x-request-id"] == "state-read-1"
            snapshot = AgentStateSnapshot.model_validate_json(first.content)
            assert snapshot.state.state_revision == completed["stateRevision"]
            assert snapshot.state.last_completed_invocation is not None
            assert snapshot.state.last_completed_invocation.subtask_id == "1"
            assert "allocation-1" not in first.text
            assert SECRET not in first.text

            unchanged = await client.get(
                STATE_PATH,
                headers={"If-None-Match": first.headers["etag"]},
            )
            assert unchanged.status_code == 304
            assert unchanged.content == b""
            assert unchanged.headers["etag"] == first.headers["etag"]
            assert unchanged.headers["cache-control"] == "private, no-cache"

            nonmatching = await client.get(
                STATE_PATH,
                headers={"If-None-Match": '"contractor-agent-state-v1-1"'},
            )
            assert nonmatching.status_code == 200

    asyncio.run(scenario())


def test_agent_state_endpoint_is_read_only_and_lifecycle_bound(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        await service.prepare(spec)
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://runtime.example",
        ) as client:
            wrong = await client.get("/private/v1/allocations/another-allocation/agent-state")
            assert wrong.status_code == 404
            assert wrong.json()["code"] == "allocation_not_found"

            with_body = await client.request("GET", STATE_PATH, content=b"{}")
            assert with_body.status_code == 422
            assert with_body.json()["code"] == "invalid_request"

            for method in ("HEAD", "POST", "PUT", "PATCH", "DELETE"):
                rejected = await client.request(method, STATE_PATH, content=b"{}")
                assert rejected.status_code == 405, (method, rejected.text)

            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="state-finalize",
                    deadline=NOW + timedelta(seconds=30),
                )
            )
            stopped = await client.get(STATE_PATH)
            assert stopped.status_code == 409
            assert stopped.json()["code"] == "agent_state_unavailable"
            assert stopped.json()["retryable"] is True
            assert stopped.headers["cache-control"] == "no-store"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["invalid", "oversized", "infrastructure"])
def test_agent_state_endpoint_classifies_snapshot_failures(
    tmp_path, runtime_capabilities, monkeypatch, failure
):
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        if failure == "invalid":
            service._context.worker_state._state["stateRevision"] = 0
        elif failure == "oversized":
            # Exercise the wire-size guard after the model has validated the snapshot.
            monkeypatch.setattr(
                "contractor_runtime.server.get_shared_config",
                lambda: SimpleNamespace(worker_state_max_bytes=1),
            )
        else:

            async def unavailable():
                raise RuntimeError(SECRET)

            monkeypatch.setattr(service._context.worker, "agent_state_snapshot", unavailable)
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            for _ in range(2):
                response = await client.get(STATE_PATH)
                assert response.status_code == 503
                assert response.json()["code"] == "agent_state_unavailable"
                assert response.json()["retryable"] is (failure == "infrastructure")
                assert response.headers["cache-control"] == "no-store"
                assert SECRET not in response.text

    asyncio.run(scenario())


def test_agent_state_snapshot_races_remain_revision_consistent_and_nonblocking(
    tmp_path: Path,
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        assert service._context is not None and service._context.worker_state is not None
        worker_state = service._context.worker_state
        metrics = InvocationMetricsReducer()
        await worker_state.begin_invocation(
            invocation_id="worker-state-race",
            subtask_id="2",
            metrics=metrics.snapshot(),
        )
        app = create_app(state, allocation_service=service, require_verified_peer=False)
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

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://runtime.example",
        ) as client:
            ticker = asyncio.create_task(tick())

            async def read_many() -> list[int]:
                revisions: list[int] = []
                for _ in range(40):
                    response = await client.get(STATE_PATH)
                    assert response.status_code == 200
                    snapshot = AgentStateSnapshot.model_validate_json(response.content)
                    revisions.append(snapshot.state.state_revision)
                    assert len(response.content) <= 4 * 1024 * 1024
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
        store = service._context.worker_state
        for index in range(1000):
            store.metrics.record_tool_call(
                "read_file", arguments={"index": index, "safe": "x" * 3500}, duration_ms=index
            )
        await store.sync_metrics()
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            first = await client.get(STATE_PATH)
            assert first.status_code == 200
            assert len(first.content) > 3_000_000

            def unexpected(*args, **kwargs):
                raise RuntimeError("unchanged revision repeated full snapshot work")

            # Reject each costly boundary; the 304 must not enter any of them.
            with monkeypatch.context() as patch:
                patch.setattr(store, "snapshot", unexpected)
                patch.setattr("contractor_runtime.worker.state._agent_state_snapshot", unexpected)
                patch.setattr(AgentStateSnapshot, "model_dump_json", unexpected)
                for _ in range(3):
                    response = await client.get(
                        STATE_PATH, headers={"If-None-Match": first.headers["etag"]}
                    )
                    assert response.status_code == 304
                    assert response.content == b""
                    assert response.headers["etag"] == first.headers["etag"]
                    assert response.headers["cache-control"] == "private, no-cache"

            await store.sync_metrics()
            changed = await client.get(STATE_PATH, headers={"If-None-Match": first.headers["etag"]})
            assert changed.status_code == 200
            assert changed.headers["etag"] != first.headers["etag"]

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
        store = service._context.worker_state
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            guessed = '"contractor-agent-state-v1-1"'
            first = await client.get(STATE_PATH, headers={"If-None-Match": guessed})
            assert first.status_code == 304
            assert validations == [1]
            repeated = await client.get(STATE_PATH, headers={"If-None-Match": guessed})
            assert repeated.status_code == 304
            assert validations == [1]

            changed = await store.sync_metrics()
            revision = changed["stateRevision"]
            # Fault injection into a new revision must never be hidden by the
            # preceding revision's successful validation, even with a guessed ETag.
            store._state["currentInvocation"] = {"unknown": True}
            invalid = await client.get(
                STATE_PATH, headers={"If-None-Match": f'"contractor-agent-state-v1-{revision}"'}
            )
            assert invalid.status_code == 503
            assert invalid.json()["retryable"] is False
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
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            first = await client.get(STATE_PATH)
            assert first.status_code == 200
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
                    await service.prepare(make_spec())

            calls = []
            original = state_module._agent_state_snapshot

            def counted(snapshot):
                calls.append(snapshot["stateRevision"])
                return original(snapshot)

            monkeypatch.setattr(state_module, "_agent_state_snapshot", counted)
            response = await client.get(
                STATE_PATH, headers={"If-None-Match": first.headers["etag"]}
            )
            if terminal == "replaced":
                assert response.status_code == 304
                assert calls == [1]  # Same revision number, a different owner: validate again.
            else:
                assert response.status_code == (404 if terminal == "released" else 409)
                assert calls == []

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["duplicate", "weak", "wildcard"])
def test_early_revalidation_preserves_exact_single_etag_matching(
    tmp_path, runtime_capabilities, kind
):
    async def scenario():
        state, service = await make_service(tmp_path, runtime_capabilities)
        await service.prepare(make_spec())
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            first = await client.get(STATE_PATH)
            tag = first.headers["etag"]
            values = {
                "duplicate": [("If-None-Match", tag), ("If-None-Match", tag)],
                "weak": [("If-None-Match", f"W/{tag}")],
                "wildcard": [("If-None-Match", "*")],
            }
            response = await client.get(STATE_PATH, headers=values[kind])
            assert response.status_code == 200
            assert response.content == first.content

    asyncio.run(scenario())
