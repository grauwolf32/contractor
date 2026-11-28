"""Bounded leased private RPC channels shared by independent connection owners."""

from __future__ import annotations

import asyncio
import math
import ssl
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import grpc

from contractor_runtime._allocation_errors import ArtifactAPIError, ArtifactTransportError
from contractor_runtime._grpc_client import IDLE_SECONDS, LIFETIME_SECONDS, _error
from contractor_runtime._grpc_tls import TLSForwarder
from contractor_runtime.contracts.base import parse_private_rpc_origin
from contractor_runtime.contracts.grpc_wire.limits import BULK_LOOKAHEAD_BYTES, METADATA_OPTIONS
from contractor_runtime.core.async_tasks import join_owned_task

MAX_CHANNELS = 4
MAX_PENDING_CALLS = 32


@dataclass(eq=False)
class _Connection:
    key: tuple[str, int]
    channel: grpc.aio.Channel
    forwarder: TLSForwarder
    directory: tempfile.TemporaryDirectory
    created: float
    last_used: float
    busy: bool = True

    async def close(self):
        async def cleanup():
            try:
                await self.channel.close()
            finally:
                try:
                    await self.forwarder.close()
                finally:
                    self.directory.cleanup()

        owner = asyncio.create_task(cleanup(), name="allocation-channel-retire")
        interrupted = await join_owned_task(owner)
        owner.result()
        if interrupted:
            raise asyncio.CancelledError


class RPCConnections:
    """Lease at most four exclusive channels and admit at most 32 calls.

    Each process owns separate instances for allocation-domain and bulk RPCs.
    A leased channel retires only after its operation and cleanup have joined.
    This owner reconnects channels but never replays an operation.
    """

    def __init__(self, context: ssl.SSLContext, *, clock=time.monotonic, bulk=False):
        self._context = context
        self._clock = clock
        self._bulk = bulk
        self._permits = asyncio.Semaphore(MAX_CHANNELS)
        self._lock = asyncio.Lock()
        self._connections: list[_Connection] = []
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    async def execute(self, origin, operation, limit, timeout):
        if self._closed:
            raise ArtifactTransportError("allocation RPC connections are closed")
        if len(self._tasks) >= MAX_PENDING_CALLS:
            raise ArtifactAPIError(
                grpc.StatusCode.RESOURCE_EXHAUSTED, "allocation_rpc_capacity", True
            )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("allocation RPC timeout must be finite and positive")
        parse_private_rpc_origin("allocation RPC origin", origin)
        task = asyncio.create_task(
            self._execute(origin, operation, limit, timeout), name="private-allocation-rpc"
        )
        self._tasks.add(task)
        try:
            return await task
        finally:
            if not task.done():
                task.cancel()
            interrupted = await join_owned_task(task)
            self._tasks.discard(task)
            if interrupted:
                raise asyncio.CancelledError

    async def _execute(self, origin, operation, limit, timeout):
        correlation = "request_" + uuid.uuid4().hex
        start = self._clock()

        def remaining():
            value = timeout - (self._clock() - start)
            if value <= 0:
                raise TimeoutError
            return value

        try:
            async with asyncio.timeout(timeout), self._permits:
                connection = await self._borrow(origin, limit)
                healthy = False
                try:
                    value = await operation(connection.channel, remaining, correlation)
                    healthy = True
                    return value
                except grpc.aio.AioRpcError as error:
                    if connection.forwarder.peer_rejected:
                        raise ArtifactAPIError(
                            grpc.StatusCode.UNAUTHENTICATED,
                            "invalid_control_plane_identity",
                            False,
                            correlation,
                        ) from None
                    failure = _error(error, correlation, allow_retry_info=True)
                    if failure.code == "transport_unavailable" and (
                        failure.retryable
                        or failure.status_code
                        in {
                            grpc.StatusCode.UNKNOWN,
                            grpc.StatusCode.INTERNAL,
                            grpc.StatusCode.DATA_LOSS,
                            grpc.StatusCode.CANCELLED,
                        }
                    ):
                        if failure.status_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                            raise ArtifactAPIError(
                                grpc.StatusCode.INTERNAL, "invalid_response", False, correlation
                            ) from None
                        raise ArtifactTransportError("allocation RPC transport failed") from None
                    healthy = failure.code != "invalid_error_response"
                    rejected = ArtifactAPIError(
                        failure.status_code, failure.code, failure.retryable, failure.request_id
                    )
                    rejected.retry_after_seconds = failure.retry_after_seconds
                    raise rejected from None
                finally:
                    cleanup = asyncio.create_task(self._release(connection, healthy))
                    interrupted = await join_owned_task(cleanup)
                    cleanup.result()
                    if interrupted:
                        raise asyncio.CancelledError
        except (OSError, TimeoutError):
            raise ArtifactTransportError("allocation RPC transport failed") from None

    async def _borrow(self, origin, limit):
        async with self._lock:
            if self._closed:
                raise ArtifactTransportError("allocation RPC connections are closed")
            now = self._clock()
            key = (origin, limit)
            for connection in self._connections:
                if connection.busy or connection.key != key:
                    continue
                if (
                    now - connection.created < LIFETIME_SECONDS
                    and now - connection.last_used < IDLE_SECONDS
                ):
                    connection.busy = True
                    return connection
                self._connections.remove(connection)
                await connection.close()
                break
            if len(self._connections) == MAX_CHANNELS:
                idle = next((item for item in self._connections if not item.busy), None)
                if idle is None:
                    raise RuntimeError("allocation RPC admission invariant failed")
                self._connections.remove(idle)
                await idle.close()
            directory = tempfile.TemporaryDirectory(
                prefix="ct-bulk-rpc-" if self._bulk else "ct-allocation-rpc-", dir="/tmp"
            )
            forwarder = TLSForwarder()
            try:
                path = Path(directory.name) / "rpc.sock"
                await forwarder.connect(path, origin, self._context)
                channel = grpc.aio.insecure_channel(
                    f"unix:{path}",
                    options=(
                        ("grpc.enable_retries", 0),
                        ("grpc.max_send_message_length", limit),
                        ("grpc.max_receive_message_length", limit),
                        *METADATA_OPTIONS,
                        *(
                            (
                                ("grpc.http2.bdp_probe", 0),
                                ("grpc.http2.lookahead_bytes", BULK_LOOKAHEAD_BYTES),
                            )
                            if self._bulk
                            else ()
                        ),
                    ),
                )
            except BaseException:
                cleanup = asyncio.create_task(forwarder.close())
                try:
                    await join_owned_task(cleanup)
                    cleanup.result()
                finally:
                    directory.cleanup()
                raise
            connection = _Connection(key, channel, forwarder, directory, now, now)
            self._connections.append(connection)
            return connection

    async def _release(self, connection, healthy):
        async with self._lock:
            connection.busy = False
            connection.last_used = self._clock()
            if not healthy:
                self._connections.remove(connection)
                await connection.close()

    async def aclose(self):
        self._closed = True

        async def cleanup():
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            async with self._lock:
                connections, self._connections = self._connections, []
                outcomes = await asyncio.gather(
                    *(item.close() for item in connections), return_exceptions=True
                )
                for outcome in outcomes:
                    if isinstance(outcome, BaseException):
                        raise outcome

        owner = asyncio.create_task(cleanup(), name="allocation-rpc-close")
        interrupted = await join_owned_task(owner)
        owner.result()
        if interrupted:
            raise asyncio.CancelledError
