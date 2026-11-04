"""Allocation-owned preflight execution, control identity and shutdown proofs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fakes.engagement import preflight_spec
from fakes.engagement_engine import transport as HTTPTransport
from fakes.worker_runtime import StubADKWorkerRuntimeFactory
from test_engagement_target import Results, policy_config

from contractor_runtime.allocation import AllocationService
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import (
    API_VERSION,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.state import RuntimeState


class Factory:
    ref = "http-target@1"

    def __init__(self, results):
        self.results, self.created = (results, [])

    def create(self, spec):
        resource = HTTPTransport(spec, resolver=policy_config().resolver, engine=self.results)
        self.created.append(resource)
        return resource


async def service_for(tmp_path, capabilities, factory):
    state = RuntimeState(instance_id="runtime-engagement")
    await state.mark_registered()
    factories = replace(
        built_in_factories(tmp_path),
        worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
        http_transport_factory=factory,
    )
    return (
        state,
        AllocationService(
            state,
            factories,
            replace(capabilities, runtime_adapters=("http-target@1",)),
            a2a_base_url="https://runtime.example",
            confirmed_control_deadline=lambda: time.monotonic() + 60,
        ),
    )


@pytest.mark.parametrize("operation", ["finalize", "lease", "reconcile"])
def test_lifecycle_cancels_and_joins_target_before_worker_resources(
    tmp_path, runtime_capabilities, operation
):

    async def scenario():
        factory = Factory(Results(hold=True))
        _, service = await service_for(tmp_path, runtime_capabilities, factory)
        spec = preflight_spec()
        await service.prepare(spec)
        task = asyncio.create_task(factory.created[0].request("http://target.example/explicit"))
        while not factory.results.attempts:
            await asyncio.sleep(0)
        await factory.results.attempts[0].entered.wait()
        if operation == "finalize":
            await service.finalize(
                FinalizeAllocationRequest(
                    apiVersion=API_VERSION,
                    allocationId=spec.allocation_id,
                    finalizationId="final-1",
                    deadline=datetime.now(UTC) + timedelta(seconds=10),
                )
            )
        elif operation == "lease":
            await service.expire_control_lease(10)
        else:
            await service.reconcile_drain(spec.allocation_id, 10)
        assert factory.results.attempts[0].finished
        assert not factory.created[0]._tasks
        with pytest.raises(asyncio.CancelledError):
            await task
        await service.release(
            ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
        )
        await service.confirm_release(spec.allocation_id)

    asyncio.run(scenario())


def test_capability_is_advertised_only_with_private_client_factory(tmp_path):

    async def scenario():
        without = built_in_factories(tmp_path / "without")
        with_client = built_in_factories(tmp_path / "with", artifact_client_factory=lambda *_: None)
        assert "http-target@1" not in (await discover_capabilities(without)).runtime_adapters
        assert "http-target@1" in (await discover_capabilities(with_client)).runtime_adapters

    asyncio.run(scenario())
