from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes.spec import allocation_spec

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    TerminationError,
)
from contractor_runtime.control_client import ControlClient
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
    StubWorkerRuntime,
    WorkerBuildContext,
)
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import LocalWorkdirFactory


def test_response_partition_expires_worker_and_late_ack_cannot_revive(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = FakeMonotonic()
        state, service = await make_service(tmp_path)
        spec = allocation_spec()
        await service.prepare(spec)
        watchdog = LeaseWatchdog(lambda: service.expire_control_lease(1), monotonic=clock)
        await watchdog.arm(60)

        assert await watchdog.acknowledge(1, 60)
        clock.advance(59.9)
        assert not await watchdog.expire_if_due()
        clock.advance(0.1)
        assert await watchdog.expire_if_due()

        snapshot = await state.snapshot()
        assert snapshot.process_state is ProcessState.FENCED
        assert snapshot.allocation_id == spec.allocation_id
        assert not (await service.snapshot()).has_worker  # type: ignore[union-attr]
        assert not await watchdog.acknowledge(2, 60)
        with pytest.raises(AllocationError, match="already owns another allocation"):
            await service.prepare(allocation_spec(allocation_id="allocation-2"))

        # The Scheduler can bind the report produced by local self-termination
        # to its durable abort operation, then perform two-phase release.
        response = await service.abort(
            AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                abortId="abort-after-lease-loss",
                reason=TerminationError(
                    code="control_lease_expired",
                    message="Control Plane confirmed lease expired",
                    retryable=True,
                ),
                deadline=spec.lease_expires_at,
            )
        )
        assert response.report.errors[-1].code == "control_lease_expired"
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        await service.confirm_release(spec.allocation_id)
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_idle_lease_loss_fences_until_new_registration_generation() -> None:
    async def scenario() -> None:
        clock = FakeMonotonic()
        state = RuntimeState(instance_id="runtime-idle")
        await state.mark_registered()

        async def expire() -> None:
            await state.fence_control_lease()

        watchdog = LeaseWatchdog(expire, monotonic=clock)
        await watchdog.arm(60)
        clock.advance(60)
        assert await watchdog.expire_if_due()
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert (await state.snapshot()).allocation_id is None
        assert not await watchdog.acknowledge(1, 60)

        # A successful re-registration is a new authority generation.
        await state.confirm_release(None)
        await watchdog.arm(60)
        assert await watchdog.acknowledge(2, 60)
        assert not watchdog.expired
        assert (await state.snapshot()).process_state is ProcessState.IDLE

    asyncio.run(scenario())


def test_lost_release_response_keeps_slot_fenced_until_repeated_action(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path)
        spec = allocation_spec()
        await service.prepare(spec)
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="finalization-before-release",
                deadline=spec.lease_expires_at,
            )
        )
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        transport = ScriptedTransport(
            [
                TimeoutError("first release response was lost"),
                {
                    "apiVersion": API_VERSION,
                    "ackSeq": 2,
                    "action": "release",
                    "allocationId": spec.allocation_id,
                },
            ]
        )
        client = ControlClient(make_settings(tmp_path), state, transport, reconciliation=service)

        with pytest.raises(TimeoutError):
            await client.heartbeat_once()
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        assert await service.snapshot() is not None

        await client.heartbeat_once()
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert await service.snapshot() is None

    asyncio.run(scenario())


def test_replayed_ack_does_not_extend_monotonic_deadline() -> None:
    async def scenario() -> None:
        clock = FakeMonotonic()
        expirations = 0

        async def expire() -> None:
            nonlocal expirations
            expirations += 1

        watchdog = LeaseWatchdog(expire, monotonic=clock)
        await watchdog.arm(60)
        clock.advance(10)
        assert await watchdog.acknowledge(4, 60)
        clock.advance(50)
        assert not await watchdog.acknowledge(4, 60)
        clock.advance(10)
        assert await watchdog.expire_if_due()
        assert not await watchdog.expire_if_due()
        assert expirations == 1

    asyncio.run(scenario())


def test_unconfirmed_worker_stop_requests_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        clock = FakeMonotonic()
        state = RuntimeState(instance_id="runtime-failed-stop")
        await state.mark_registered()
        exits: list[int] = []
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": FailingStopRuntimeFactory()},
            toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
        )
        service = AllocationService(
            state,
            registry,
            a2a_base_url="https://runtime.example",
            force_exit=exits.append,
        )
        await service.prepare(allocation_spec())
        watchdog = LeaseWatchdog(lambda: service.expire_control_lease(0.1), monotonic=clock)
        await watchdog.arm(60)
        clock.advance(60)

        with pytest.raises(AllocationError) as error:
            await watchdog.expire_if_due()
        assert error.value.code == "worker_stop_unconfirmed"
        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


async def make_service(tmp_path: Path) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-lease-test")
    await state.mark_registered()
    registry = FactoryRegistry(
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
    )
    return state, AllocationService(
        state,
        registry,
        a2a_base_url="https://runtime.example",
        force_exit=lambda _: None,
    )


class FakeMonotonic:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FailingStopRuntimeFactory:
    ref = "adk@1"

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        return FailingStopRuntime(context)


class FailingStopRuntime(StubWorkerRuntime):
    async def abort(self, deadline: datetime) -> None:
        raise RuntimeError("synthetic stop failure")


class ScriptedTransport:
    def __init__(self, responses: list[Mapping[str, Any] | BaseException]) -> None:
        self.responses = list(responses)

    async def post_json(self, _: str, __: Mapping[str, Any]) -> Mapping[str, Any]:
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def make_settings(tmp_path: Path) -> Settings:
    placeholder = tmp_path / "unused-pki"
    return Settings(
        control_plane_url="https://control.example",
        advertised_control_url="https://runtime.example:9443",
        advertised_a2a_url="https://runtime.example:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
    )
