"""Process-owned, deadline-bounded control RPC over verified TLS."""

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
from google.rpc.error_details_pb2 import RetryInfo
from google.rpc.status_pb2 import Status

from contractor_runtime._grpc_metadata import error_detail
from contractor_runtime._grpc_tls import TLSForwarder
from contractor_runtime.contracts import (
    AgentRegistrationResponse,
    HeartbeatResponse,
    PrivateProtocolDecodeError,
    SharedConfig,
    decode_private,
)
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire.limits import (
    IDLE_SECONDS,
    LIFETIME_SECONDS,
    MAX_CONTROL_BYTES,
    METADATA_OPTIONS,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.threads import to_thread_until_done


class ControlRPCError(Exception):
    """Only safe status, domain code and correlation identity cross this boundary."""

    def __init__(self, status_code, code, retryable, request_id):
        self.status_code = status_code
        self.code = code
        self.retryable = retryable
        self.request_id = request_id
        self.retry_after_seconds = None
        super().__init__(f"Control Plane gRPC status {status_code.name} ({code})")

    @property
    def permanent(self):
        return not self.retryable


class ControlRPC:
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
        return await self._call(
            "Register", value, runtime_pb2.AgentRegistration, AgentRegistrationResponse
        )

    async def bootstrap(self):
        return await self._call("Bootstrap", {}, runtime_pb2.BootstrapRequest, SharedConfig)

    async def heartbeat(self, value):
        return await self._call("Heartbeat", value, runtime_pb2.AgentHeartbeat, HeartbeatResponse)

    async def _call(self, operation, value, request_type, response_model):
        start = self._clock()
        correlation = "request_" + uuid.uuid4().hex
        async with asyncio.timeout(self._timeout), self._lock:
            if self._closed:
                raise RuntimeError("Control Plane connections are closed")
            try:
                try:
                    request = await to_thread_until_done(
                        _encode_request, value, request_type, name="control-protobuf-encode"
                    )
                except codec.InvalidWireValue:
                    raise ControlRPCError(
                        grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, correlation
                    ) from None
                await self._connect()
                remaining = self._timeout - (self._clock() - start)
                if remaining <= 0:
                    raise TimeoutError
                method = getattr(runtime_pb2_grpc.ControlPlaneStub(self._channel), operation)
                response = await method(
                    request, timeout=remaining, metadata=(("x-request-id", correlation),)
                )
                try:
                    result = await to_thread_until_done(
                        _decode_response, response, response_model, name="control-protobuf-decode"
                    )
                    if operation == "Heartbeat" and result["ackSeq"] != value["heartbeatSeq"]:
                        raise codec.InvalidWireValue
                    return result
                except (codec.InvalidWireValue, PrivateProtocolDecodeError, ValueError):
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
                failure = _error(error, correlation)
                raise failure from None
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
                *METADATA_OPTIONS,
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


def _encode_request(value, request_type):
    request = codec.encode(value, request_type())
    if request.ByteSize() > MAX_CONTROL_BYTES:
        raise codec.InvalidWireValue
    return request


def _decode_response(response, model):
    raw = json.dumps(
        codec.decode(response), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    if len(raw.encode()) > MAX_CONTROL_BYTES:
        raise codec.InvalidWireValue
    value = decode_private(model, raw)
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


def _error(error, correlation, *, allow_retry_info=False):
    retry_after = None
    status_code = error.code()
    retryable = status_code in {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.ABORTED,
    }
    # Capacity rejections are explicitly typed. An untyped status may be a
    # local receive limit and cannot safely become a retry loop.
    code = (
        "invalid_response"
        if status_code == grpc.StatusCode.RESOURCE_EXHAUSTED
        else "transport_unavailable"
    )
    details = [
        value
        for key, value in (error.trailing_metadata() or ())
        if key == "grpc-status-details-bin"
    ]
    if len(details) == 1 and isinstance(details[0], bytes) and len(details[0]) <= 4096:
        try:
            status = Status.FromString(details[0])
            known_status = Status()
            known_status.CopyFrom(status)
            known_status.DiscardUnknownFields()
            if known_status != status:
                raise ValueError
            if status.code != status_code.value[0] or len(status.details) not in (
                {1, 2} if allow_retry_info else {1}
            ):
                raise ValueError
            if len(status.details) == 2:
                retry_info = RetryInfo()
                if not status.details[1].Unpack(retry_info) or not retry_info.HasField(
                    "retry_delay"
                ):
                    raise ValueError
                known = RetryInfo()
                known.CopyFrom(retry_info)
                known.DiscardUnknownFields()
                duration = retry_info.retry_delay
                if (
                    known != retry_info
                    or duration.seconds < 0
                    or not 0 <= duration.nanos < 1_000_000_000
                ):
                    raise ValueError
                retry_after = duration.seconds + duration.nanos / 1_000_000_000
                if not 0 < retry_after <= 300:
                    raise ValueError
            detail = runtime_pb2.ErrorDetail()
            if not status.details[0].Unpack(detail):
                raise ValueError
            value = codec.decode(detail)
            code, retryable, correlation = error_detail(value)

        except (ValueError, TypeError, DecodeError):
            code, retryable, retry_after = "invalid_error_response", False, None
    elif details:
        code, retryable = "invalid_error_response", False
    result = ControlRPCError(status_code, code, retryable, correlation)
    result.retry_after_seconds = retry_after
    return result
