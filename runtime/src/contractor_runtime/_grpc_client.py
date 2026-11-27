"""Process-owned, deadline-bounded registration RPC over verified TLS."""

from __future__ import annotations

import asyncio
import json
import math
import ssl
import tempfile
import time
import uuid
from pathlib import Path

import grpc
from google.protobuf.message import DecodeError
from google.rpc.status_pb2 import Status

from contractor_runtime._grpc_tls import TLSForwarder
from contractor_runtime._https import decode_private_error
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire.limits import MAX_CONTROL_BYTES, MAX_METADATA_BYTES
from contractor_runtime.core.async_tasks import join_owned_task

IDLE_SECONDS = 30.0
LIFETIME_SECONDS = 300.0


class ControlRPCError(Exception):
    """Only safe status, domain code and correlation identity cross this boundary."""

    def __init__(self, status_code, code, retryable, request_id):
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        super().__init__(f"Control Plane gRPC status {status_code.name} ({code})")

    @property
    def permanent(self):
        return not self.retryable


class RegistrationRPC:
    def __init__(
        self, origin: str, context: ssl.SSLContext, timeout: float, *, clock=time.monotonic
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Control Plane RPC timeout must be finite and positive")
        self._clock = clock
        self._origin = origin
        self._context = context
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._channel = None
        self._forwarder = None
        self._directory = None
        self._created = 0.0
        self._last_used = 0.0
        self._closed = False

    async def register(self, value):
        start = self._clock()
        correlation = "request_" + uuid.uuid4().hex
        try:
            request = codec.encode(value, runtime_pb2.AgentRegistration())
        except codec.InvalidWireValue:
            raise ControlRPCError(
                grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, correlation
            ) from None
        if request.ByteSize() > MAX_CONTROL_BYTES:
            raise ControlRPCError(
                grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, correlation
            )
        async with asyncio.timeout(self._timeout), self._lock:
            if self._closed:
                raise RuntimeError("Control Plane connections are closed")
            try:
                await self._connect()
                remaining = self._timeout - (self._clock() - start)
                if remaining <= 0:
                    raise TimeoutError
                response = await runtime_pb2_grpc.ControlPlaneStub(self._channel).Register(
                    request, timeout=remaining, metadata=(("x-request-id", correlation),)
                )
                try:
                    return codec.decode(response)
                except codec.InvalidWireValue:
                    raise ControlRPCError(
                        grpc.StatusCode.INTERNAL, "invalid_response", False, correlation
                    ) from None
            except grpc.aio.AioRpcError as error:
                rejected = self._forwarder is not None and self._forwarder.peer_rejected
                await self._reset()
                if rejected:
                    raise ControlRPCError(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "invalid_control_plane_identity",
                        False,
                        correlation,
                    ) from None
                raise _error(error, correlation) from None
            except BaseException:
                await self._reset()
                raise
            finally:
                self._last_used = self._clock()

    async def _connect(self):
        now = self._clock()
        if self._channel is not None:
            if now - self._created < LIFETIME_SECONDS and now - self._last_used < IDLE_SECONDS:
                return
            await self._reset()
        self._directory = tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp")
        path = Path(self._directory.name) / "control.sock"
        self._forwarder = TLSForwarder()
        await self._forwarder.connect(path, self._origin, self._context)
        self._channel = grpc.aio.insecure_channel(
            f"unix:{path}",
            options=(
                ("grpc.enable_retries", 0),
                ("grpc.max_send_message_length", MAX_CONTROL_BYTES),
                ("grpc.max_receive_message_length", MAX_CONTROL_BYTES),
                ("grpc.max_metadata_size", MAX_METADATA_BYTES),
            ),
        )
        self._created = now

    async def _reset(self):
        channel, self._channel = self._channel, None
        forwarder, self._forwarder = self._forwarder, None
        directory, self._directory = self._directory, None

        async def cleanup():
            try:
                if channel is not None:
                    await channel.close()
            finally:
                try:
                    if forwarder is not None:
                        await forwarder.close()
                finally:
                    if directory is not None:
                        directory.cleanup()

        owner = asyncio.create_task(cleanup(), name="grpc-control-retire")
        interrupted = await join_owned_task(owner)
        owner.result()
        if interrupted:
            raise asyncio.CancelledError

    async def aclose(self):
        self._closed = True
        async with self._lock:
            await self._reset()


def _error(error, correlation):
    status_code = error.code()
    retryable = status_code in {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
    }
    code = "transport_unavailable"
    details = [
        value
        for key, value in (error.trailing_metadata() or ())
        if key == "grpc-status-details-bin"
    ]
    if len(details) == 1 and isinstance(details[0], bytes) and len(details[0]) <= 4096:
        try:
            status = Status.FromString(details[0])
            if status.code != status_code.value[0] or len(status.details) != 1:
                raise ValueError
            detail = runtime_pb2.ErrorDetail()
            if not status.details[0].Unpack(detail):
                raise ValueError
            value = codec.decode(detail)
            # Reuse the private error grammar; it validates code, retry class
            # and request ID and never propagates the remote raw error text.
            parsed = decode_private_error(json.dumps({**value, "message": "RPC failed"}).encode())
            if parsed is None:
                raise ValueError
            code, retryable, correlation = parsed.code, parsed.retryable, parsed.request_id
        except (ValueError, TypeError, DecodeError):
            code, retryable = "invalid_error_response", False
    elif details:
        code, retryable = "invalid_error_response", False
    return ControlRPCError(status_code, code, retryable, correlation)
