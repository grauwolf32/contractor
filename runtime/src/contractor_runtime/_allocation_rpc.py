"""Bounded process-owned allocation RPC channels over the verified TLS adapter."""

from __future__ import annotations

import asyncio
import json
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
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire.limits import MAX_METADATA_BYTES
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.threads import to_thread_until_done

ENVELOPE_BYTES = 1024
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


class AllocationConnections:
    """At most four leased channels; calls never replay at the transport layer.

    An active call keeps its channel until it joins. Idle/aged channels retire
    before the next call, so rotation cannot cancel another allocation's work.
    Process control and bulk payloads have separate connection owners.
    """

    def __init__(self, context: ssl.SSLContext, *, clock=time.monotonic):
        self._context = context
        self._clock = clock
        self._permits = asyncio.Semaphore(MAX_CHANNELS)
        self._lock = asyncio.Lock()
        self._connections: list[_Connection] = []
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    async def call(
        self,
        origin,
        instance_id,
        allocation_id,
        method,
        value,
        request_type,
        limit,
        timeout,
        response_limit=None,
    ):
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
            self._call(
                origin,
                instance_id,
                allocation_id,
                method,
                value,
                request_type,
                limit,
                timeout,
                response_limit or limit,
            ),
            name="allocation-rpc",
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

    async def _call(
        self,
        origin,
        instance_id,
        allocation_id,
        method,
        value,
        request_type,
        limit,
        timeout,
        response_limit,
    ):
        correlation = "request_" + uuid.uuid4().hex
        start = self._clock()
        try:
            async with asyncio.timeout(timeout), self._permits:
                try:
                    request = await to_thread_until_done(
                        _encode, value, request_type, limit, name="allocation-protobuf-encode"
                    )
                except codec.InvalidWireValue:
                    raise ArtifactAPIError(
                        grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, correlation
                    ) from None
                connection = await self._borrow(origin, limit + ENVELOPE_BYTES)
                healthy = False
                try:
                    remaining = timeout - (self._clock() - start)
                    if remaining <= 0:
                        raise TimeoutError
                    rpc = getattr(runtime_pb2_grpc.AllocationStub(connection.channel), method)
                    response = await rpc(
                        request,
                        timeout=remaining,
                        metadata=(
                            ("x-request-id", correlation),
                            ("x-contractor-runtime-instance-id", instance_id),
                            ("x-contractor-allocation-id", allocation_id),
                        ),
                    )
                    try:
                        value = await to_thread_until_done(
                            _decode, response, response_limit, name="allocation-protobuf-decode"
                        )
                    except codec.InvalidWireValue:
                        raise ArtifactAPIError(
                            grpc.StatusCode.INTERNAL, "invalid_response", False, correlation
                        ) from None
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
                    if failure.code == "transport_unavailable" and failure.retryable:
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
            directory = tempfile.TemporaryDirectory(prefix="ct-allocation-rpc-", dir="/tmp")
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
                        ("grpc.max_metadata_size", MAX_METADATA_BYTES),
                    ),
                )
            except BaseException:
                cleanup = asyncio.create_task(forwarder.close())
                await join_owned_task(cleanup)
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
                await asyncio.gather(*(item.close() for item in connections))

        owner = asyncio.create_task(cleanup(), name="allocation-rpc-close")
        interrupted = await join_owned_task(owner)
        owner.result()
        if interrupted:
            raise asyncio.CancelledError


def _encode(value, request_type, limit):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, OverflowError):
        raise codec.InvalidWireValue from None
    domain_size = len(raw)
    if request_type in (
        runtime_pb2.SitemapMutationRequest,
        runtime_pb2.SitemapSnapshotRequest,
        runtime_pb2.SitemapCollectionRequest,
    ):
        body = {key: item for key, item in value.items() if key != "idempotencyKey"}
        domain_size = len(
            json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        )
    if domain_size > limit:
        raise codec.InvalidWireValue
    request = codec.encode(value, request_type())
    if request.ByteSize() > limit + ENVELOPE_BYTES:
        raise codec.InvalidWireValue
    return request


def _decode(response, limit):
    if response.ByteSize() > limit + ENVELOPE_BYTES:
        raise codec.InvalidWireValue
    value = codec.decode(response)
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError, OverflowError):
        raise codec.InvalidWireValue from None
    if len(raw) > limit:
        raise codec.InvalidWireValue
    return value


class AllocationRPC:
    """Named operations bound to the advertised Server and Runtime process."""

    def __init__(self, owner, origin, instance_id, timeout):
        self._owner = owner
        self._origin = origin
        self._instance_id = instance_id
        self._timeout = timeout

    async def planner_operation(self, allocation_id, value, *, timeout_seconds):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "PlannerOperation",
            value,
            runtime_pb2.PlannerOperationRequest,
            4 << 20,
            timeout_seconds,
        )

    async def gateway_recovery(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "GatewayRecovery",
            value,
            runtime_pb2.GatewayRecoveryRequest,
            4096,
            self._timeout,
        )

    async def submit_audit_evidence(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "SubmitAuditEvidence",
            value,
            runtime_pb2.AuditEvidenceRequest,
            8 << 20,
            self._timeout,
            1 << 20,
        )

    async def submit_audit_judgments(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "SubmitAuditJudgments",
            value,
            runtime_pb2.AuditJudgmentsRequest,
            8 << 20,
            self._timeout,
            1 << 20,
        )

    async def mutate_sitemap(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "MutateSitemap",
            value,
            runtime_pb2.SitemapMutationRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def snapshot_sitemap(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "SnapshotSitemap",
            value,
            runtime_pb2.SitemapSnapshotRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def collect_sitemap(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "CollectSitemap",
            value,
            runtime_pb2.SitemapCollectionRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def list_artifacts(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "ListArtifacts",
            value,
            runtime_pb2.ArtifactListRequest,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def read_artifact_metadata(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "ReadArtifactMetadata",
            value,
            runtime_pb2.ArtifactReadTarget,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def check_artifact_access(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "CheckArtifactAccess",
            value,
            runtime_pb2.ArtifactReadTarget,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def list_findings(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "ListFindings",
            value,
            runtime_pb2.FindingListRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )

    async def read_finding(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "ReadFinding",
            value,
            runtime_pb2.FindingReadRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )

    async def select_findings(self, allocation_id, value):
        return await self._owner.call(
            self._origin,
            self._instance_id,
            allocation_id,
            "SelectFindings",
            value,
            runtime_pb2.FindingSelectionRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )
