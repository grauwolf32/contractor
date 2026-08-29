from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.spec import allocation_spec

from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.contracts import API_VERSION, AbortAllocationRequest, TerminationError
from contractor_runtime.factories import (
    FactoryRegistry,
    RunArtifactsToolsetFactory,
    StubADKWorkerRuntimeFactory,
    StubWorkerRuntime,
    WorkerBuildContext,
)
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import LocalWorkdirFactory


def test_abort_is_idempotent_and_returns_cached_metrics(tmp_path: Path) -> None:
    async def scenario() -> None:
        state, service = await make_service(tmp_path)
        spec = allocation_spec()
        await service.prepare(spec)
        assert service._context is not None
        assert service._context.worker_state is not None
        service._context.worker_state.metrics.record_model_call()
        request = abort_request(spec.allocation_id, "abort-1")

        first = await service.abort(request)
        second = await service.abort(request)

        assert second == first
        assert first.report.worker.complete
        assert first.report.runtime.complete
        assert first.report.worker.metrics.model_calls == 1
        assert [error.code for error in first.report.worker.errors] == ["user_cancelled"]
        assert (await state.snapshot()).process_state is ProcessState.DRAINING
        assert not (await service.snapshot()).has_worker  # type: ignore[union-attr]

        with pytest.raises(AllocationError) as different_id:
            await service.abort(abort_request(spec.allocation_id, "abort-2"))
        assert different_id.value.code == "allocation_conflict"

    asyncio.run(scenario())


def test_abort_deadline_fences_slot_and_requests_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        state, service = await make_service(
            tmp_path,
            runtime_factory=BlockingRuntimeFactory(),
            force_exit=exits.append,
        )
        spec = allocation_spec()
        await service.prepare(spec)
        request = abort_request(
            spec.allocation_id,
            "abort-deadline",
            deadline=datetime.now(UTC) + timedelta(milliseconds=10),
        )

        with pytest.raises(AllocationError) as unconfirmed:
            await service.abort(request)

        assert unconfirmed.value.code == "worker_stop_unconfirmed"
        assert not unconfirmed.value.retryable
        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED
        with pytest.raises(AllocationError, match="still incomplete") as repeated:
            await service.abort(request)
        assert repeated.value.code == "allocation_conflict"

    asyncio.run(scenario())


def test_cancelled_abort_request_still_fences_and_requests_process_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        exits: list[int] = []
        runtime_factory = BlockingRuntimeFactory()
        state, service = await make_service(
            tmp_path,
            runtime_factory=runtime_factory,
            force_exit=exits.append,
        )
        spec = allocation_spec()
        await service.prepare(spec)
        task = asyncio.create_task(
            service.abort(
                abort_request(
                    spec.allocation_id,
                    "abort-disconnected",
                    deadline=datetime.now(UTC) + timedelta(seconds=30),
                )
            )
        )
        await asyncio.wait_for(runtime_factory.started.wait(), timeout=1)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert exits == [70]
        assert (await state.snapshot()).process_state is ProcessState.FENCED

    asyncio.run(scenario())


async def make_service(
    tmp_path: Path,
    *,
    runtime_factory: StubADKWorkerRuntimeFactory | BlockingRuntimeFactory | None = None,
    force_exit: object | None = None,
) -> tuple[RuntimeState, AllocationService]:
    state = RuntimeState(instance_id="runtime-abort-test")
    await state.mark_registered()
    selected_runtime = runtime_factory or StubADKWorkerRuntimeFactory()
    registry = FactoryRegistry(
        worker_runtimes={"adk@1": selected_runtime},
        toolsets={"run-artifacts@1": RunArtifactsToolsetFactory()},
        sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path)},
    )
    options: dict[str, object] = {}
    if force_exit is not None:
        options["force_exit"] = force_exit
    service = AllocationService(
        state,
        registry,
        a2a_base_url="https://runtime.example",
        **options,  # type: ignore[arg-type]
    )
    return state, service


def abort_request(
    allocation_id: str,
    abort_id: str,
    *,
    deadline: datetime | None = None,
) -> AbortAllocationRequest:
    return AbortAllocationRequest(
        apiVersion=API_VERSION,
        allocationId=allocation_id,
        abortId=abort_id,
        reason=TerminationError(
            code="user_cancelled",
            message="WorkflowRun cancellation was requested",
            retryable=False,
        ),
        deadline=deadline or datetime.now(UTC) + timedelta(seconds=5),
    )


class BlockingRuntimeFactory:
    ref = "adk@1"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def create(self, context: WorkerBuildContext) -> StubWorkerRuntime:
        return BlockingRuntime(context, self.started)


class BlockingRuntime(StubWorkerRuntime):
    def __init__(self, context: WorkerBuildContext, started: asyncio.Event) -> None:
        super().__init__(context)
        self._started = started

    async def abort(self, deadline: datetime) -> None:
        del deadline
        self._started.set()
        await asyncio.Event().wait()
