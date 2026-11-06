from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.engagement import preflight_spec
from fakes.model import scripted_model
from fakes.spec import allocation_spec
from test_adk_runtime import FakeArtifactClient

from contractor_runtime.adapters.http_transport import HTTPTransportFactory
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import API_VERSION, ReleaseAllocationRequest
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.server import create_app
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.toolsets.run_artifacts.tools import RunArtifactsToolsetFactory
from contractor_runtime.worker import state as state_module
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from contractor_runtime.worker.runtime import AdkWorkerRuntime
from contractor_runtime.workspace import LocalWorkdirFactory

STATE_PATH = "/private/v1/allocations/allocation-1/agent-state"


async def make_adk_service(root: Path, *, controlled_http: bool = False):
    client = FakeArtifactClient()
    registry = FactoryRegistry(
        worker_runtimes={
            "adk@1": AdkWorkerRuntimeFactory(model_factory=lambda _: scripted_model([]))
        },
        toolsets={"run-artifacts@1": RunArtifactsToolsetFactory(lambda *_: client)},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(root)},
        http_transport_factory=HTTPTransportFactory() if controlled_http else None,
    )
    capabilities = CapabilitySnapshot.create(
        runtimes=registry.worker_runtimes,
        toolsets={key: factory.exported_tools for key, factory in registry.toolsets.items()},
        sandbox_profiles=registry.sandbox_profiles,
        runtime_adapters=["http-target@1"] if controlled_http else [],
    )
    state = RuntimeState(instance_id="stop-regression", capabilities=capabilities)
    await state.mark_registered()
    exits = []
    service = AllocationService(
        state,
        registry,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=exits.append,
    )
    await service.prepare(preflight_spec() if controlled_http else allocation_spec())
    assert isinstance(service._context.worker, AdkWorkerRuntime)
    return state, service, exits


async def release(service):
    await service.release(
        ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId="allocation-1")
    )
    await service.confirm_release("allocation-1")


def test_queued_real_adk_state_read_cannot_hold_up_lease_stop(tmp_path):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        started, unblock = asyncio.Event(), threading.Event()

        def occupy_executor():
            loop.call_soon_threadsafe(started.set)
            unblock.wait()

        blocker = asyncio.create_task(asyncio.to_thread(occupy_executor))
        await started.wait()
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            reader = asyncio.create_task(client.get(STATE_PATH))
            await asyncio.sleep(0)
            try:
                # A guard against a broken test, not a wall-clock performance assertion.
                await asyncio.wait_for(service.expire_control_lease(10), timeout=2)
                assert (await state.snapshot()).process_state is ProcessState.FENCED
                assert not reader.done()
                assert exits == []
            finally:
                unblock.set()
                await blocker
                response = await reader
            assert response.status_code == 409
            assert response.json()["code"] == "agent_state_unavailable"
        await release(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("reconcile", [False, True])
def test_real_http_join_and_worker_abort_share_one_deadline(tmp_path, reconcile):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path, controlled_http=True)
        now = [datetime.now(UTC)]
        service._now = lambda: now[0]
        start = now[0]
        transport = service._context.http_transport
        transport.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
        started, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def in_flight_http():
            added = transport.own_invocation()
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
            finally:
                transport.finish_invocation(added)

        io_task = asyncio.create_task(in_flight_http())
        await started.wait()
        deadlines = []
        worker = service._context.worker
        original_abort = worker.abort

        async def observe_abort(deadline):
            deadlines.append(deadline)
            await original_abort(deadline)

        worker.abort = observe_abort
        stop = asyncio.create_task(
            service.reconcile_drain("allocation-1", 10)
            if reconcile
            else service.expire_control_lease(10)
        )
        await cancelled.wait()
        assert (await state.snapshot()).process_state is ProcessState.DRAINING
        now[0] += timedelta(seconds=8)
        finish.set()
        await stop
        await io_task
        assert deadlines == [start + timedelta(seconds=10)]
        assert exits == []
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        await release(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("reconcile", [False, True])
def test_stop_cannot_wait_for_the_lifecycle_lock_beyond_its_grace(tmp_path, reconcile):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path)
        await service._lock.acquire()
        stop = asyncio.create_task(
            service.reconcile_drain("allocation-1", 10)
            if reconcile
            else service.expire_control_lease(10)
        )
        await asyncio.sleep(0)
        advance_clock(11)
        try:
            with pytest.raises(AllocationError) as error:
                await asyncio.wait_for(stop, timeout=2)
            assert error.value.code == "worker_stop_unconfirmed"
            assert exits == [70]
            assert (await state.snapshot()).process_state is ProcessState.FENCED
        finally:
            service._lock.release()
        # The injected exit returns; explicitly finish the test's owned resources.
        await service.expire_control_lease(30)
        await release(service)

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("replace_allocation", [False, True])
def test_detached_snapshot_cannot_escape_release_or_same_id_replacement(
    tmp_path, monkeypatch, replace_allocation
):
    async def scenario():
        _, service, exits = await make_adk_service(tmp_path)
        loop = asyncio.get_running_loop()
        started, unblock = asyncio.Event(), threading.Event()
        original = state_module._agent_state_snapshot

        def gated(snapshot):
            loop.call_soon_threadsafe(started.set)
            unblock.wait()
            return original(snapshot)

        monkeypatch.setattr(state_module, "_agent_state_snapshot", gated)
        reader = asyncio.create_task(service.agent_state_snapshot("allocation-1"))
        await started.wait()
        try:
            await asyncio.wait_for(service.expire_control_lease(10), timeout=2)
            await release(service)
            if replace_allocation:
                await service.prepare(allocation_spec())
        finally:
            unblock.set()
        with pytest.raises(AllocationError) as error:
            await reader
        assert error.value.status_code == (409 if replace_allocation else 404)
        assert exits == []
        if replace_allocation:
            await service.expire_control_lease(10)
            await release(service)

    asyncio.run(scenario())
