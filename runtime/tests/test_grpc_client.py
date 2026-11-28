"""Production control client against the Go Control Plane handler."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import grpc
import pytest
from google.protobuf.any_pb2 import Any
from google.rpc.status_pb2 import Status
from test_grpc_interop import go_client as go_client
from test_grpc_interop import pki as pki
from test_grpc_interop import runtime_context

from contractor_runtime._grpc_client import ControlRPC, ControlRPCError, _error
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb


@pytest.mark.parametrize("mode", ["success", "wrong_role", "invalid_request"])
def test_shared_registration_uses_production_go_handler(pki, go_client, mode):
    async def scenario():
        config = {
            "CA": str(pki.ca),
            "Certificate": str(pki.control_plane_certificate),
            "Key": str(pki.control_plane_key),
            "Mode": "control",
        }
        if mode == "wrong_role":
            config.update(Certificate=str(pki.agent_certificate), Key=str(pki.agent_key))
        process = await asyncio.create_subprocess_exec(
            str(go_client),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        owner = None
        try:
            process.stdin.write(json.dumps(config).encode() + b"\n")
            await process.stdin.drain()
            async with asyncio.timeout(10):
                startup = await process.stdout.readline()
            assert startup, (await process.stderr.read()).decode()
            port = json.loads(startup)["port"]
            now = [0.0]
            owner = ControlRPC(
                f"https://localhost:{port}", runtime_context(pki), 3, clock=lambda: now[0]
            )
            payload = json.loads(
                (
                    Path(__file__).parents[2]
                    / "api/testdata/v1alpha1/valid/agent-registration.json"
                ).read_bytes()
            )
            if mode == "invalid_request":
                del payload["supportedRuntimeAdapters"]
            if mode == "success":
                bootstrap = await owner.bootstrap()
                assert bootstrap["digest"].startswith("sha256:")
                first = owner._channel
                result = await owner.register(payload)
                assert owner._channel is first
                assert result == {
                    "apiVersion": "contractor/v1alpha1",
                    "heartbeatIntervalSeconds": 10,
                    "confirmedLeaseSeconds": 60,
                }
                for sequence in range(1, 10):
                    heartbeat = await owner.heartbeat(
                        {
                            "apiVersion": "contractor/v1alpha1",
                            "instanceId": payload["instanceId"],
                            "heartbeatSeq": sequence,
                            "echoedAckSeq": sequence - 1,
                            "observedState": "idle",
                            "allocations": [],
                        }
                    )
                    assert heartbeat["ackSeq"] == sequence
                    assert owner._channel is first
                assert await owner.register(payload) == result
                assert owner._channel is first
                now[0] = 300
                assert await owner.register(payload) == result
                assert owner._channel is not first
            else:
                with pytest.raises(ControlRPCError) as caught:
                    await owner.register(payload)
                if mode == "invalid_request":
                    assert caught.value.code == "invalid_request"
                    assert caught.value.permanent
                else:
                    assert caught.value.status_code == grpc.StatusCode.UNAUTHENTICATED
                    assert caught.value.permanent
            directory = owner._directory.name if owner._directory else None
            await owner.aclose()
            assert owner._channel is None and owner._forwarder is None
            if directory:
                assert not Path(directory).exists()
            process.stdin.close()
            async with asyncio.timeout(10):
                out, error = await process.communicate()
            assert process.returncode == 0, error.decode()
            assert json.loads(out)["calls"] == (3 if mode == "success" else 0)
        finally:
            if owner is not None:
                await owner.aclose()
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(scenario())


@pytest.mark.parametrize("raw", [b"\xff", b"", b"x" * 4097])
def test_malformed_status_details_fail_closed(raw):
    error = grpc.aio.AioRpcError(
        grpc.StatusCode.UNAVAILABLE,
        (),
        (("grpc-status-details-bin", raw),),
        "must-not-escape",
        "secret-debug",
    )
    result = _error(error, "request_test")
    assert result.code == "invalid_error_response" and result.permanent
    assert "must-not-escape" not in str(result) and "secret" not in str(result)


def test_status_details_preserve_domain_retry_verdict():
    packed = Any()
    packed.Pack(pb.ErrorDetail(code="conflict", retryable=True, request_id="request_1"))
    raw = Status(
        code=grpc.StatusCode.FAILED_PRECONDITION.value[0], details=[packed]
    ).SerializeToString()
    error = grpc.aio.AioRpcError(
        grpc.StatusCode.FAILED_PRECONDITION, (), (("grpc-status-details-bin", raw),)
    )
    result = _error(error, "request_test")
    assert result.retryable and result.code == "conflict" and result.request_id == "request_1"


def test_shared_private_rpc_error_classification():
    cases = json.loads(
        (Path(__file__).parents[2] / "api/testdata/v1alpha1/grpc-error-cases.json").read_bytes()
    )["cases"]
    statuses = {value.value[0]: value for value in grpc.StatusCode}
    for case in cases:
        trailers = ()
        if case["detail"] is not None:
            detail = pb.ErrorDetail(
                **{
                    {"requestId": "request_id"}.get(key, key): value
                    for key, value in case["detail"].items()
                }
            )
            if case.get("unknownDetail"):
                detail.MergeFromString(b"\xf8\x07\x01")
            packed = Any()
            packed.Pack(detail)
            trailers = (
                (
                    "grpc-status-details-bin",
                    Status(code=case["status"], details=[packed]).SerializeToString(),
                ),
            )
        error = grpc.aio.AioRpcError(
            statuses[case["status"]], (), trailers, "untrusted diagnostics"
        )
        failure = _error(error, "request_fallback")
        assert (failure.code, failure.retryable) == (case["code"], case["retryable"]), case["name"]
