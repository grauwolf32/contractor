from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import httpx
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
            assert stopped.headers["cache-control"] == "no-store"

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
