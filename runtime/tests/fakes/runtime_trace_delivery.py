"""Actual allocation, ADK Worker and OTLP exporter in a disposable process."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from test_adk_runtime import FakeArtifactClient, stage_request
from test_otlp_adapter import allocation_service, configured_allocation

from contractor_runtime import shared_config
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    ReleaseAllocationRequest,
    SharedConfig,
    TerminationError,
    decode_private,
)
from contractor_runtime.digests import _model_policy_digest
from contractor_runtime.log import configure_logging
from contractor_runtime.toolsets.run_artifacts.tools import RunArtifactsToolsetFactory
from contractor_runtime.worker.factory import AdkWorkerRuntimeFactory
from fakes.allocation import allocation_context
from fakes.model import scripted_model, tool_call

SECRET = "cleanup-secret-canary"


async def run(endpoint: str, root: Path, operation: str, fault: str) -> None:
    configure_logging("error")
    shared_config._snapshot = decode_private(
        SharedConfig, (Path(__file__).parents[1] / "testdata/shared-config.json").read_bytes()
    ).snapshot()
    state, service = await allocation_service(root, OTLPHTTPAdapterFactory())
    model = scripted_model(
        [
            tool_call(
                "read_artifact",
                {"namespace": "inputs", "name": "source", "revision": "input-r1"},
                call_id=f"read-{number}",
            )
            for number in range(2)
        ]
    )
    service._factories.worker_runtimes["adk@1"] = AdkWorkerRuntimeFactory(lambda _: model)
    service._factories.toolsets["run-artifacts@1"] = RunArtifactsToolsetFactory(
        lambda *_: FakeArtifactClient()
    )
    service._exit_process = os._exit
    spec = configured_allocation()
    spec.runtime_settings.telemetry = spec.runtime_settings.telemetry.model_copy(
        update={"endpoint": endpoint, "headers": {}}
    )
    spec.model_policy.max_tool_calls = 1
    spec.model_policy.ref.digest = _model_policy_digest(spec.model_policy)
    await service.prepare(spec)
    context = allocation_context(service)
    completion = await context.worker.invoke(stage_request())
    assert completion.failure.code == "worker_budget_exhausted"
    instrumentation = context.adapter_host.handles.instrumentation
    pending = instrumentation.pending_spans
    assert pending > 0
    assert instrumentation.pending_bytes < spec.runtime_settings.telemetry.export.batch_size_bytes
    print(json.dumps({"failure": completion.failure.code, "queued": pending}), flush=True)

    async def fail(*_args, **_kwargs):
        raise RuntimeError(f"{SECRET}: https://private.invalid/secret-response")

    if fault == "worker":
        context.worker.finalize = fail
        context.worker.abort = fail
    elif fault == "http":
        context.http_transport = SimpleNamespace(close=fail)
    elif fault == "tool":
        next(iter(context.tools.values())).close = fail
    elif fault == "execution":
        context.execution = SimpleNamespace(reject=lambda: None, stop=fail, remove=fail)
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    if operation == "finalize":
        await service.finalize(
            FinalizeAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                finalizationId="trace-check",
                deadline=deadline,
            )
        )
    elif operation == "abort":
        await service.abort(
            AbortAllocationRequest(
                apiVersion=API_VERSION,
                allocationId=spec.allocation_id,
                abortId="trace-check",
                deadline=deadline,
                reason=TerminationError(code="run_cancelled", message="cancelled", retryable=False),
            )
        )
    elif operation == "lease":
        await service.expire_control_lease(5)
    else:
        await service.reconcile_drain(spec.allocation_id, 5)
    await service.release(
        ReleaseAllocationRequest(apiVersion=API_VERSION, allocationId=spec.allocation_id)
    )
    await service.confirm_release(spec.allocation_id)
    print(json.dumps({"released": (await state.snapshot()).process_state.value}), flush=True)


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
