"""Native lifecycle admission, validation and safe failure outcomes."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import timedelta

import grpc
import pytest
from fakes.spec import allocation_spec
from test_allocation import NOW, make_service, make_spec, resign_template, sha256_digest

from contractor_runtime._grpc_client import _error
from contractor_runtime.contracts import API_VERSION, PrepareAllocationRequest
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.grpc_server import LifecycleHandler, RPCAdmission
from contractor_runtime.state import RuntimeState


@asynccontextmanager
async def lifecycle_client(service):
    state = RuntimeState(instance_id="runtime-test")
    with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as directory:
        target = f"unix:{directory}/lifecycle.sock"
        server = grpc.aio.server(interceptors=(RPCAdmission(),))
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


@pytest.mark.parametrize(
    "fault",
    ["missing-allocation", "empty-allocation", "invalid-allocation", "zero-revision", "unknown"],
)
def test_state_rejects_invalid_request_before_snapshot(fault):
    class Service:
        async def agent_state_snapshot(self, *args, **kwargs):
            raise AssertionError("invalid State read dispatched")

    async def scenario():
        request = pb.StateReadRequest(allocation_id="allocation-1")
        if fault == "missing-allocation":
            request.ClearField("allocation_id")
        elif fault == "empty-allocation":
            request.allocation_id = ""
        elif fault == "invalid-allocation":
            request.allocation_id = "../foreign"
        elif fault == "zero-revision":
            request.if_revision = 0
        else:
            request.MergeFromString(b"\xf8\x07\x01")
        async with lifecycle_client(Service()) as (client, _):
            with pytest.raises(grpc.aio.AioRpcError) as caught:
                await client.ReadState(request, timeout=3)
            error = _error(caught.value, "fallback")
            assert error.code == "invalid_request" and error.permanent

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["finalize", "abort"])
def test_cancelled_terminal_rpc_preserves_siblings_and_cached_report(
    tmp_path, runtime_capabilities, monkeypatch, operation
):
    async def scenario():
        _, service = await make_service(tmp_path, runtime_capabilities, allocation_slots=4)
        specs = [make_spec(allocation_id=f"allocation-{index}") for index in range(4)]
        for spec in specs:
            await service.prepare(spec)
        exits = []
        service._exit_process = exits.append
        worker = service._allocations[specs[0].allocation_id]._context.worker
        original = getattr(worker, operation)
        entered, resume = asyncio.Event(), asyncio.Event()
        stops = 0

        async def stop(deadline):
            nonlocal stops
            stops += 1
            entered.set()
            await resume.wait()
            await original(deadline)

        monkeypatch.setattr(worker, operation, stop)
        value = dict(
            api_version=API_VERSION,
            allocation_id=specs[0].allocation_id,
            deadline=(NOW + timedelta(seconds=30)).isoformat(),
        )
        if operation == "finalize":
            request = pb.FinalizeAllocationRequest(finalization_id="terminal-1", **value)
        else:
            request = pb.AbortAllocationRequest(
                abort_id="terminal-1",
                reason=pb.TerminationError(
                    code="user_cancelled", message="Run cancelled", retryable=False
                ),
                **value,
            )
        try:
            async with lifecycle_client(service) as (client, _):
                method = getattr(client, operation.capitalize())
                call = method(request, timeout=10)
                await entered.wait()
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                # A later stream makes progress on the same native connection
                # while the original terminal operation still owns its Worker.
                sibling = codec.encode(
                    PrepareAllocationRequest(apiVersion=API_VERSION, spec=specs[1]).model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    pb.PrepareAllocationRequest(),
                )
                await client.Prepare(sibling, timeout=10)
                retry = method(request, timeout=10)
                resume.set()
                report = await retry
                assert await method(request, timeout=10) == report
                assert stops == 1 and exits == []
                different = type(request)()
                different.CopyFrom(request)
                setattr(
                    different, "finalization_id" if operation == "finalize" else "abort_id", "other"
                )
                with pytest.raises(grpc.aio.AioRpcError) as caught:
                    await method(different, timeout=10)
                assert _error(caught.value, "fallback").code == "allocation_conflict"
                for spec in specs[1:]:
                    assert service._allocations[spec.allocation_id]._context.worker is not None
        finally:
            resume.set()
            await service.close(30)

    asyncio.run(scenario())


def test_unicode_prepare_reaches_allocation_service_with_verified_digests(
    tmp_path, runtime_capabilities
):
    async def scenario():
        _, service = await make_service(tmp_path, runtime_capabilities)
        spec = make_spec()
        spec.agent_template.instructions.text = "я" * 180000
        spec.agent_template.instructions.digest = sha256_digest(
            spec.agent_template.instructions.text.encode()
        )
        resign_template(spec.agent_template)
        request = codec.encode(
            PrepareAllocationRequest(apiVersion=API_VERSION, spec=spec).model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            pb.PrepareAllocationRequest(),
        )
        try:
            async with lifecycle_client(service) as (client, _):
                response = await client.Prepare(request, timeout=10)
                assert response.worker_handle.allocation_id == spec.allocation_id
        finally:
            await service.close(30)

    asyncio.run(scenario())


def test_runtime_admission_is_typed_before_decode_and_releases_every_outcome():
    from contractor_runtime.contracts.grpc_wire.limits import MAX_CONCURRENT_RPCS

    async def scenario():
        entered = asyncio.Queue()
        release = asyncio.Event()
        decoded = 0

        def parse(raw):
            nonlocal decoded
            decoded += 1
            if raw != b"hold":
                raise ValueError("invalid probe")
            return raw

        async def hold(request, context):
            await entered.put(None)
            await release.wait()
            return b"ok"

        with tempfile.TemporaryDirectory(prefix="ct-admission-", dir="/tmp") as directory:
            target = f"unix:{directory}/admission.sock"
            server = grpc.aio.server(interceptors=(RPCAdmission(),))
            server.add_generic_rpc_handlers(
                (
                    grpc.method_handlers_generic_handler(
                        "probe.Admission",
                        {
                            "Hold": grpc.unary_unary_rpc_method_handler(
                                hold,
                                request_deserializer=parse,
                                response_serializer=lambda value: value,
                            )
                        },
                    ),
                )
            )
            server.add_insecure_port(target)
            await server.start()
            calls = []
            try:
                async with grpc.aio.insecure_channel(target) as channel:
                    method = channel.unary_unary("/probe.Admission/Hold")
                    for _ in range(MAX_CONCURRENT_RPCS):
                        calls.append(method(b"hold", timeout=10))
                        await entered.get()
                    with pytest.raises(grpc.aio.AioRpcError) as caught:
                        await method(b"malformed", timeout=10)
                    failure = _error(caught.value, "fallback")
                    assert failure.status_code is grpc.StatusCode.RESOURCE_EXHAUSTED
                    assert failure.code == "control_rpc_capacity" and failure.retryable
                    assert decoded == MAX_CONCURRENT_RPCS
                    release.set()
                    assert await asyncio.gather(*calls) == [b"ok"] * MAX_CONCURRENT_RPCS
                    # Decode failures also return admission, before any service method runs.
                    for _ in range(MAX_CONCURRENT_RPCS + 1):
                        with pytest.raises(grpc.aio.AioRpcError) as malformed:
                            await method(b"malformed", timeout=10)
                        assert malformed.value.code() is not grpc.StatusCode.RESOURCE_EXHAUSTED
                    assert await method(b"hold", timeout=10) == b"ok"
                    # Transport cancellation returns its permit as well.
                    release.clear()
                    call = method(b"hold", timeout=10)
                    await entered.get()  # drain successful reuse
                    await entered.get()  # new blocked call
                    call.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await call
                    release.set()
                    assert await method(b"hold", timeout=10) == b"ok"
            finally:
                release.set()
                for call in calls:
                    call.cancel()
                await server.stop(0)
                await server.wait_for_termination()

    asyncio.run(scenario())
