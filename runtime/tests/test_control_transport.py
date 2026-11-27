"""Control RPC presence, byte bounds, deadline and cancellation ownership."""

from __future__ import annotations

import asyncio
import json
import ssl
import tempfile
from pathlib import Path

import grpc
import pytest

from contractor_runtime._grpc_client import ControlRPC, ControlRPCError
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire.limits import MAX_CONTROL_BYTES


@pytest.mark.parametrize("fault", ["missing", "unknown", "digest", "oversized", "ack"])
def test_control_rejects_malformed_responses_without_replaying(fault, monkeypatch):
    class Peer(runtime_pb2_grpc.ControlPlaneServicer):
        calls = 0

        async def Bootstrap(self, request, context):
            self.calls += 1
            value = json.loads((Path(__file__).parent / "testdata/shared-config.json").read_bytes())
            response = codec.encode(value, pb.SharedConfig())
            if fault == "missing":
                response.ClearField("limits")
            elif fault == "unknown":
                response.MergeFromString(b"\xf8\x07\x01")
            elif fault == "digest":
                response.digest = "sha256:" + "0" * 64
            else:
                response.digest = "x" * (MAX_CONTROL_BYTES + 1)
            return response

        async def Heartbeat(self, request, context):
            self.calls += 1
            return pb.HeartbeatResponse(
                api_version="contractor/v1alpha1",
                ack_seq=2,
                action="continue",
                allocation_actions=pb.ListOfAllocationAction(),
            )

    async def scenario():
        peer = Peer()
        with tempfile.TemporaryDirectory(prefix="ct-control-", dir="/tmp") as directory:
            target = f"unix:{directory}/rpc.sock"
            server = grpc.aio.server()
            runtime_pb2_grpc.add_ControlPlaneServicer_to_server(peer, server)
            server.add_insecure_port(target)
            await server.start()
            owner = ControlRPC("https://control.test", ssl.create_default_context(), 3)

            async def connect():
                owner._channel = grpc.aio.insecure_channel(
                    target,
                    options=(
                        ("grpc.max_receive_message_length", MAX_CONTROL_BYTES),
                        ("grpc.enable_retries", 0),
                    ),
                )

            monkeypatch.setattr(owner, "_connect", connect)
            try:
                with pytest.raises(ControlRPCError) as caught:
                    if fault == "ack":
                        await owner.heartbeat(
                            {
                                "apiVersion": "contractor/v1alpha1",
                                "instanceId": "runtime_1",
                                "heartbeatSeq": 1,
                                "echoedAckSeq": 0,
                                "observedState": "idle",
                                "allocations": [],
                            }
                        )
                    else:
                        await owner.bootstrap()
                assert caught.value.request_id.startswith("request_")
                if fault == "oversized":
                    assert caught.value.status_code == grpc.StatusCode.RESOURCE_EXHAUSTED
                assert caught.value.code == "invalid_response" and caught.value.permanent
                assert peer.calls == 1
                assert owner._channel is None
            finally:
                await owner.aclose()
                await server.stop(0)
                await server.wait_for_termination()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_control_deadline_or_cancellation_joins_connection_setup(cancel, monkeypatch):
    async def scenario():
        owner = ControlRPC(
            "https://control.test", ssl.create_default_context(), 0.02 if not cancel else 3
        )
        entered, joined = asyncio.Event(), asyncio.Event()

        async def connect():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

        monkeypatch.setattr(owner, "_connect", connect)
        task = asyncio.create_task(owner.bootstrap())
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert joined.is_set()
        assert not owner._lock.locked()
        await owner.aclose()

    asyncio.run(scenario())
