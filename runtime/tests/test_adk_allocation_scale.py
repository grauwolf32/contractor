"""Cross the SDK cache capacity through actual allocation release, never eviction."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fakes.allocation import allocation_context
from fakes.model import scripted_model, text_result, tool_call
from pydantic import SecretStr
from test_adk_runtime import stage_request
from test_allocation import (
    TrackedArtifactClients,
    TrackedMetricsState,
    allocation_references,
    finalize_and_release,
    make_service,
    make_spec,
)

import contractor_runtime.worker.adk_caches as adk_caches
import contractor_runtime.worker.state as state_module
from contractor_runtime.factories import WorkerBuildContext
from contractor_runtime.state import ProcessState
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory

FIXTURE = json.loads((Path(__file__).parent / "testdata/adk213-scale-workloads.json").read_text())


def test_1025_real_allocations_release_every_declared_closure_graph(
    tmp_path, runtime_capabilities, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(state_module, "MetricsState", TrackedMetricsState)
    clients = TrackedArtifactClients()
    closures = []
    capacities = [cache.cache_info().maxsize for cache in adk_caches._TOOL_CALLABLE_CACHES]
    assert all(value == 1024 for value in capacities)
    assert FIXTURE["allocation_cycles"] > max(capacities)

    def model(context: WorkerBuildContext):
        return scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {
                        "namespace": "inputs",
                        "name": "source",
                        "revision": "input-r1",
                    },
                    call_id=f"read-{context.allocation_id}",
                ),
                text_result(f"Completed {context.allocation_id}"),
            ]
        )

    class ClosureFactory(AdkWorkerRuntimeFactory):
        async def create(self, context: WorkerBuildContext) -> Any:
            original = context.tools["read_artifact"]

            async def read_artifact(
                namespace: str,
                name: str,
                revision: str | None = None,
                offset: int = 0,
                length: int = 262144,
            ) -> dict[str, Any]:
                return await original(namespace, name, revision, offset, length)

            read_artifact.__doc__ = f"Read the exact input for {context.allocation_id}."
            closures.append(weakref.ref(read_artifact))
            return await super().create(replace(context, tools={"read_artifact": read_artifact}))

    async def scenario():
        state, service = await make_service(
            tmp_path, runtime_capabilities, artifact_client_factory=clients, model_factory=model
        )
        service._factories = replace(
            service._factories,
            worker_runtimes={
                "adk@1": ClosureFactory(model_factory=model, artifact_client_factory=clients),
            },
        )
        pending = []
        for index in range(FIXTURE["allocation_cycles"]):
            spec = make_spec(allocation_id=f"allocation-scale-{index:04d}", tools=["read_artifact"])
            spec.runtime_settings.llm_gateway_token = SecretStr(
                f"allocation-secret-{index:04d}-fixture"
            )
            await service.prepare(spec)
            pending.append(allocation_references(service))
            assert (
                allocation_context(service) is not None
                and allocation_context(service).worker is not None
            )
            result = await allocation_context(service).worker.invoke(stage_request())
            assert result.failure is None and result.result is not None
            assert result.result.result == f"Completed {spec.allocation_id}"
            del result
            assert all(
                cache.cache_info().currsize > 0 for cache in adk_caches._TOOL_CALLABLE_CACHES
            )
            await finalize_and_release(service, spec.allocation_id)
            assert all(
                cache.cache_info().currsize == 0 for cache in adk_caches._TOOL_CALLABLE_CACHES
            )
            await service.confirm_release(spec.allocation_id)
            assert (await state.snapshot()).process_state is ProcessState.IDLE
            assert allocation_context(service) is None
            # At most 32 weak-reference groups exist between collection checks;
            # there is no growing strong owner and no test-side cache_clear.
            if (index + 1) % 32 == 0 or index + 1 == FIXTURE["allocation_cycles"]:
                gc.collect()
                assert all(ref() is None for graph in pending for ref in graph.values())
                assert clients.live() == 0
                assert all(ref() is None for ref in closures)
                pending.clear()
        assert len(clients.created) == FIXTURE["allocation_cycles"]
        assert len(closures) == FIXTURE["allocation_cycles"]

    asyncio.run(scenario())
