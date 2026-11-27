"""Native lifecycle admission, validation and safe failure outcomes."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager

import grpc
import pytest
from fakes.spec import allocation_spec

from contractor_runtime._grpc_client import _error
from contractor_runtime.contracts import API_VERSION, PrepareAllocationRequest
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.grpc_server import LifecycleHandler
from contractor_runtime.state import RuntimeState


@asynccontextmanager
async def lifecycle_client(service):
    state = RuntimeState(instance_id="runtime-test")
    with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as directory:
        target = f"unix:{directory}/lifecycle.sock"
        server = grpc.aio.server()
        runtime_pb2_grpc.add_RuntimeLifecycleServicer_to_server(
            LifecycleHandler(state, service), server
        )
        server.add_insecure_port(target)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(
                target, options=(("grpc.enable_retries", 0),)
            ) as channel:
                yield runtime_pb2_grpc.RuntimeLifecycleStub(channel), state
        finally:
            await server.stop(0)
            await server.wait_for_termination()


def request_value():
    return PrepareAllocationRequest(apiVersion=API_VERSION, spec=allocation_spec()).model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )


@pytest.mark.parametrize(
    "variant",
    [
        "missing-provenance",
        "unknown-root",
        "duplicate-secret",
        "unknown-document",
        "api-version",
        "nan",
        "missing-labels",
        "invalid-label-name",
        "invalid-label-value",
    ],
)
def test_lifecycle_rejects_invalid_wire_before_dispatch(variant):
    class Service:
        calls = 0

        async def prepare(self, spec):
            self.calls += 1
            raise AssertionError("invalid request dispatched")

    async def scenario():
        service = Service()
        request = codec.encode(request_value(), pb.PrepareAllocationRequest())
        if variant == "missing-provenance":
            request.spec.ClearField("resolved_runtime_config_provenance")
        elif variant == "unknown-root":
            request.MergeFromString(b"\xf8\x07\x01")
        elif variant in {"duplicate-secret", "unknown-document", "nan"}:
            raw = request.spec.runtime_settings.json
            prefix = {
                "duplicate-secret": b'{"llmGatewayToken":"attacker-token",',
                "unknown-document": b'{"unknown":true,',
                "nan": b'{"unknown":NaN,',
            }[variant]
            request.spec.runtime_settings.json = prefix + raw[1:]
        elif variant == "api-version":
            request.api_version = "contractor.dev/v0"
        elif variant == "missing-labels":
            request.spec.ClearField("run_metadata_labels")
        elif variant == "invalid-label-name":
            request.spec.run_metadata_labels.values["Eval.ID"] = "eval_01"
        else:
            request.spec.run_metadata_labels.values["purpose"] = "before\0after"
        async with lifecycle_client(service) as (client, state):
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await client.Prepare(
                    request, metadata=(("x-request-id", "strict-request"),), timeout=3
                )
            detail = _error(caught.value, "fallback")
            assert detail.code == "invalid_request" and detail.permanent
            assert detail.request_id == "strict-request"
            assert caught.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert service.calls == 0
            assert (await state.snapshot()).route_dispatches == 1

    asyncio.run(scenario())


def test_lifecycle_internal_failure_is_correlated_without_secret_logs(caplog):
    injected = "secret-do-not-log"

    class Service:
        async def prepare(self, spec):
            # Proves protobuf timestamps survived the strict model parser.
            assert spec.lease_expires_at.tzinfo is not None
            raise RuntimeError(injected)

    async def scenario():
        request = codec.encode(request_value(), pb.PrepareAllocationRequest())
        async with lifecycle_client(Service()) as (client, _):
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await client.Prepare(
                    request, metadata=(("x-request-id", "failure-request-1"),), timeout=3
                )
            detail = _error(caught.value, "fallback")
            assert detail.code == "internal_error" and detail.retryable
            assert detail.request_id == "failure-request-1"
            assert caught.value.code() == grpc.StatusCode.INTERNAL
            assert injected not in str(caught.value)

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert "failure-request-1" in caplog.text
    assert injected not in caplog.text
