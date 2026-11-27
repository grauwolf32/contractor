"""Native State RPC test peer; retains protobuf results and typed failures."""

from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass

import grpc

from contractor_runtime._grpc_client import ControlRPCError, _error
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire import runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire.limits import MAX_STATE_MESSAGE_BYTES
from contractor_runtime.grpc_server import LifecycleHandler


@dataclass
class StateReply:
    result: pb.StateReadResponse | None
    error: ControlRPCError | None
    request_id: str

    @property
    def status(self):
        return self.error.status_code if self.error else grpc.StatusCode.OK

    @property
    def revision(self):
        return self.result.revision if self.result else None

    @property
    def not_modified(self):
        return self.result is not None and self.result.WhichOneof("result") == "not_modified"

    @property
    def document(self):
        return self.result.snapshot.json if self.result else b""

    def value(self):
        return json.loads(self.document)


class StateClient:
    def __init__(self, stub):
        self.stub = stub

    async def read(self, allocation="allocation-1", *, revision=None, request_id="state_test"):
        request = pb.StateReadRequest(allocation_id=allocation)
        if revision is not None:
            request.if_revision = revision
        call = self.stub.ReadState(request, metadata=(("x-request-id", request_id),), timeout=10)
        try:
            result = await call
        except grpc.aio.AioRpcError as error:
            return StateReply(None, _error(error, request_id), request_id)
        metadata = dict(await call.initial_metadata())
        return StateReply(result, None, metadata["x-request-id"])


@asynccontextmanager
async def state_client(state, service):
    with tempfile.TemporaryDirectory(prefix="ct-state-", dir="/tmp") as directory:
        target = f"unix:{directory}/rpc.sock"
        options = (
            ("grpc.max_receive_message_length", MAX_STATE_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_STATE_MESSAGE_BYTES),
            ("grpc.enable_retries", 0),
        )
        server = grpc.aio.server(options=options)
        runtime_pb2_grpc.add_RuntimeLifecycleServicer_to_server(
            LifecycleHandler(state, service), server
        )
        server.add_insecure_port(target)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(target, options=options) as channel:
                yield StateClient(runtime_pb2_grpc.RuntimeLifecycleStub(channel))
        finally:
            await server.stop(0)
            await server.wait_for_termination()
