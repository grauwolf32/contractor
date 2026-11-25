from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fakes.allocation import allocation_context, allocation_owner, allocation_snapshot
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


async def make_adk_service(root: Path, *, controlled_http: bool = False, target=None):
    from fakes.engagement_socket import resolver_for, socket_spec

    client = FakeArtifactClient()
    registry = FactoryRegistry(
        worker_runtimes={
            "adk@1": AdkWorkerRuntimeFactory(model_factory=lambda _: scripted_model([]))
        },
        toolsets={"run-artifacts@1": RunArtifactsToolsetFactory(lambda *_: client)},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(root)},
        http_transport_factory=(
            HTTPTransportFactory(resolver=resolver_for(target).resolver if target else None)
            if controlled_http
            else None
        ),
    )
    capabilities = CapabilitySnapshot.create(
        runtimes=registry.worker_runtimes,
        toolsets={key: factory.exported_tools for key, factory in registry.toolsets.items()},
        sandbox_profiles=registry.sandbox_profiles,
        runtime_adapters=["http-target@1"] if controlled_http else [],
    )
    state = RuntimeState(
        allocation_slots=1, instance_id="stop-regression", capabilities=capabilities
    )
    await state.mark_registered()
    exits = []
    service = AllocationService(
        state,
        registry,
        capabilities,
        a2a_base_url="https://runtime.example",
        force_exit=exits.append,
    )
    await service.prepare(
        socket_spec(target)
        if target
        else preflight_spec()
        if controlled_http
        else allocation_spec()
    )
    assert isinstance(allocation_context(service).worker, AdkWorkerRuntime)
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
                assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
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
        allocation_owner(service)._now = lambda: now[0]
        start = now[0]
        transport = allocation_context(service).http_transport
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
        worker = allocation_context(service).worker
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
        assert (await state.snapshot("allocation-1")).process_state is (
            ProcessState.DRAINING if reconcile else ProcessState.FENCED
        )
        now[0] += timedelta(seconds=8)
        finish.set()
        await stop
        await io_task
        assert deadlines == [start + timedelta(seconds=10)]
        assert exits == []
        assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
        await release(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("reconcile", [False, True])
def test_stop_cannot_wait_for_the_lifecycle_lock_beyond_its_grace(tmp_path, reconcile):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path)
        await allocation_owner(service)._lock.acquire()
        stop = asyncio.create_task(
            service.reconcile_drain("allocation-1", 10)
            if reconcile
            else service.expire_control_lease(10)
        )
        while not allocation_owner(service)._lock._waiters:
            await asyncio.sleep(0)
        advance_clock(11)
        try:
            with pytest.raises(AllocationError) as error:
                await asyncio.wait_for(stop, timeout=2)
            assert error.value.code == "worker_stop_unconfirmed"
            assert exits == [70]
            assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
        finally:
            allocation_owner(service)._lock.release()
        # The injected exit returns; explicitly finish the test's owned resources.
        await service.expire_control_lease(30)
        await release(service)

    run_with_advanceable_clock(scenario())


@pytest.mark.parametrize("replace_allocation", [False, True])
def test_detached_snapshot_cannot_escape_release_or_same_id_replacement(
    tmp_path, monkeypatch, replace_allocation
):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path)
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
        finally:
            unblock.set()
        with pytest.raises(AllocationError) as error:
            await reader
        assert error.value.status_code == 409
        await release(service)
        assert exits == []
        if replace_allocation:
            await state.confirm_release(None)
            await service.prepare(allocation_spec())
            assert await service.agent_state_snapshot("allocation-1") is not None
            await service.expire_control_lease(10)
            await release(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("expire_release", [False, True])
@pytest.mark.parametrize("cancel_reader", [False, True])
def test_release_joins_admitted_http_state_reads_and_erases_the_real_state(
    tmp_path, monkeypatch, expire_release, cancel_reader
):
    async def scenario():
        state, service, exits = await make_adk_service(tmp_path)
        retained = weakref.ref(allocation_context(service).worker_state)
        loop = asyncio.get_running_loop()
        started, unblock = asyncio.Event(), threading.Event()
        release_started, scratch_cleanup_started = asyncio.Event(), asyncio.Event()
        original_snapshot = state_module._agent_state_snapshot
        original_release = allocation_owner(service)._cleanup.release
        original_cleanup = allocation_context(service).sandbox.cleanup

        def gated(snapshot):
            loop.call_soon_threadsafe(started.set)
            unblock.wait()
            return original_snapshot(snapshot)

        async def observed_release(context, deadline):
            release_started.set()
            await original_release(context, deadline)

        async def observed_cleanup(workspace):
            scratch_cleanup_started.set()
            await original_cleanup(workspace)

        monkeypatch.setattr(state_module, "_agent_state_snapshot", gated)
        monkeypatch.setattr(allocation_owner(service)._cleanup, "release", observed_release)
        monkeypatch.setattr(allocation_context(service).sandbox, "cleanup", observed_cleanup)
        app = create_app(state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            reader = asyncio.create_task(client.get(STATE_PATH))
            await started.wait()
            if cancel_reader:
                reader.cancel()
                await asyncio.sleep(0)
            await service.expire_control_lease(10)
            if expire_release:
                allocation_context(service).runtime_settings.request_timeout_seconds = 1
            releasing = asyncio.create_task(release(service))
            await release_started.wait()
            try:
                for _ in range(10):
                    await asyncio.sleep(0)
                assert not scratch_cleanup_started.is_set()
                assert not releasing.done()
                assert retained() is not None
                if expire_release:
                    advance_clock(2)
                    with pytest.raises(AllocationError) as error:
                        await asyncio.wait_for(releasing, timeout=2)
                    assert error.value.retryable
                    assert not allocation_context(service).release_prepared
                    assert (
                        await state.snapshot("allocation-1")
                    ).process_state is ProcessState.FENCED
            finally:
                unblock.set()
                if cancel_reader:
                    with pytest.raises(asyncio.CancelledError):
                        await reader
                else:
                    response = await reader
                    assert response.status_code == 409
            if expire_release:
                await release(service)
            else:
                await releasing
        gc.collect()
        assert scratch_cleanup_started.is_set()
        assert retained() is None
        assert exits == []

    run_with_advanceable_clock(scenario())


def test_release_queued_before_the_final_state_check_cannot_deadlock_or_retain_state(
    tmp_path, monkeypatch
):
    async def scenario():
        _, service, exits = await make_adk_service(tmp_path, controlled_http=True)
        context = allocation_context(service)
        retained = weakref.ref(context.worker_state)
        transport = context.http_transport
        transport.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
        loop = asyncio.get_running_loop()
        parse_started, unblock_parse = asyncio.Event(), threading.Event()
        http_started, http_cancelled, finish_http = (
            asyncio.Event(),
            asyncio.Event(),
            asyncio.Event(),
        )
        original = state_module._agent_state_snapshot

        def gated(snapshot):
            loop.call_soon_threadsafe(parse_started.set)
            unblock_parse.wait()
            return original(snapshot)

        async def http_in_flight():
            added = transport.own_invocation()
            http_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                http_cancelled.set()
                await finish_http.wait()
            finally:
                transport.finish_invocation(added)

        monkeypatch.setattr(state_module, "_agent_state_snapshot", gated)
        io_task = asyncio.create_task(http_in_flight())
        await http_started.wait()
        app = create_app(service._state, allocation_service=service, require_verified_peer=False)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://runtime.example"
        ) as client:
            reader = asyncio.create_task(client.get(STATE_PATH))
            await parse_started.wait()
            stopping = asyncio.create_task(service.expire_control_lease(10))
            releasing = None
            try:
                await asyncio.wait_for(http_cancelled.wait(), timeout=2)
                releasing = asyncio.create_task(release(service))
                await asyncio.sleep(
                    0
                )  # Queue release behind stop, ahead of the read's final check.
                unblock_parse.set()
                await asyncio.wait_for(context.state_reads_drained.wait(), timeout=2)
                assert not reader.done() and not releasing.done()
                finish_http.set()
                await asyncio.wait_for(stopping, timeout=2)
                await asyncio.wait_for(releasing, timeout=2)
                response = await reader
                assert response.status_code in {404, 409}
                gc.collect()
                assert retained() is None
                assert exits == []
            finally:
                unblock_parse.set()
                finish_http.set()
                for task in (reader, stopping, releasing):
                    if task is not None and not task.done():
                        task.cancel()
                await asyncio.gather(
                    *(task for task in (reader, stopping, releasing, io_task) if task is not None),
                    return_exceptions=True,
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["finalize", "abort", "lease"])
def test_peer_reset_preserves_worker_report_and_next_allocation(tmp_path, stop):
    from fakes.bridge_reset import EarlyDenial
    from fakes.engagement_socket import SocketTarget, socket_spec

    from contractor_runtime.contracts import (
        AbortAllocationRequest,
        FinalizeAllocationRequest,
        TerminationError,
    )

    async def scenario():
        async with EarlyDenial() as target:
            state, service, exits = await make_adk_service(
                tmp_path, controlled_http=True, target=target
            )
            context = allocation_context(service)
            transport = context.http_transport
            transport.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
            outcome, response = await transport.request(
                target.origin + "/allowed", method="POST", body=b'{"amount":1}'
            )
            assert outcome.kind == "completed" and response.status == 404
            await target.task
            sessions = tuple(transport._sessions)
            assert any(session.cookie_summary()["cookie_count"] == 1 for session in sessions)
            context.worker_state.metrics.record_worker_budget_exhausted("tool_calls")
            completion = await context.worker.failure_completion(
                "worker_budget_exhausted", "Worker tool-call budget exhausted", retryable=True
            )
            assert completion.failure.code == "worker_budget_exhausted"
            metrics = context.worker_state.metrics
            expected_metrics = dict(metrics.counters)
            engine = transport._engine
            directory = Path(engine.directory.name)
            deadline = datetime.now(UTC) + timedelta(seconds=30)
            if stop == "finalize":
                request = FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId="allocation-1",
                    finalizationId="finalize-1",
                    deadline=deadline,
                )
                result = await service.finalize(request)
                assert await service.finalize(request) == result
            else:
                if stop == "lease":
                    await service.expire_control_lease(30)
                    assert (
                        await state.snapshot("allocation-1")
                    ).process_state is ProcessState.FENCED
                request = AbortAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId="allocation-1",
                    abortId="abort-1",
                    deadline=deadline,
                    reason=TerminationError(
                        code="run_cancelled",
                        message="Cancelled",
                        retryable=False,
                    ),
                )
                result = await service.abort(request)
                assert await service.abort(request) == result
            assert result.report.worker.complete and result.report.runtime.complete
            assert metrics.counters == expected_metrics
            assert metrics.final_outcome == "failed"
            assert completion.failure.code == "worker_budget_exhausted"
            assert result.report.allocation_id == "allocation-1"
            assert any(
                error.code == "worker_budget_exhausted" for error in result.report.worker.errors
            )
            assert engine.closed and engine.runtime.master is not None and not directory.exists()
            assert transport._settings is None and transport._sessions == []
            assert all(not session.cookies for session in sessions)
            assert exits == []
            await release(service)
            if (await state.snapshot()).process_state is ProcessState.FENCED:
                await state.confirm_release(None)
            assert (await state.snapshot("allocation-1")).process_state is ProcessState.IDLE
            assert await allocation_snapshot(service) is None
        async with SocketTarget() as target:
            await service.prepare(socket_spec(target))
            transport = allocation_context(service).http_transport
            transport.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
            outcome, response = await transport.request(target.origin + "/allowed")
            assert outcome.kind == "completed" and response.body == b"ok"
            assert (
                allocation_context(service).worker_state.metrics.counters.get("outcomes.failed", 0)
                == 0
            )
            assert b"cookie:" not in target.written.lower()
            await service.expire_control_lease(30)
            await release(service)
            assert exits == []
            await service._factories.http_transport_factory.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["writer", "files", "callback"])
def test_real_http_cleanup_defects_still_fence_and_request_exit(tmp_path, monkeypatch, failure):
    from fakes.engagement_socket import SocketTarget

    async def scenario():
        async with SocketTarget() as target:
            state, service, exits = await make_adk_service(
                tmp_path, controlled_http=True, target=target
            )
            transport = allocation_context(service).http_transport
            transport.configure(authority_deadline=lambda: asyncio.get_running_loop().time() + 60)
            await transport.request(target.origin + "/allowed")
            engine = transport._engine
            pool = next(iter(engine.pools.values()))
            directory = Path(engine.directory.name)
            with monkeypatch.context() as patch:
                if failure == "writer":
                    writer = next(io.writer.inner for io in pool.transports.values() if io.writer)

                    async def reject():
                        raise RuntimeError("injected writer defect")

                    patch.setattr(writer, "wait_closed", reject)
                elif failure == "files":

                    def reject():
                        raise OSError("injected file-removal defect")

                    patch.setattr(engine.directory, "cleanup", reject)
                else:
                    original = pool._release_connection

                    async def reject(connection):
                        await original(connection)
                        raise OSError("injected retirement callback defect")

                    patch.setattr(pool, "_release_connection", reject)
                with pytest.raises((RuntimeError, OSError), match="injected"):
                    await service.expire_control_lease(30)
                assert exits == [70]
                assert (await state.snapshot("allocation-1")).process_state is ProcessState.FENCED
                assert (
                    engine.runtime.master is not None and directory.exists() and not engine.closed
                )
                if failure == "writer":
                    assert pool.writers
            # force_exit is captured by the test. Join all remaining test-owned
            # resources using the ordinary cleanup retry after removing the fault.
            await service.expire_control_lease(30)
            await release(service)
            await service._factories.http_transport_factory.close()
            assert not directory.exists()

    asyncio.run(scenario())
