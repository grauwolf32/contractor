"""Named allocation RPC adapters over the shared leased channel owner."""

import json

import grpc

from contractor_runtime._allocation_errors import ArtifactAPIError
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2, runtime_pb2_grpc
from contractor_runtime.threads import to_thread_until_done

ENVELOPE_BYTES = 1024


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

    async def _call(
        self, allocation_id, method, value, request_type, limit, timeout, response_limit=None
    ):
        async def operation(channel, remaining, correlation):
            try:
                request = await to_thread_until_done(
                    _encode, value, request_type, limit, name="allocation-protobuf-encode"
                )
            except codec.InvalidWireValue:
                raise ArtifactAPIError(
                    grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, correlation
                ) from None
            rpc = getattr(runtime_pb2_grpc.AllocationStub(channel), method)
            response = await rpc(
                request,
                timeout=remaining(),
                metadata=(
                    ("x-request-id", correlation),
                    ("x-contractor-runtime-instance-id", self._instance_id),
                    ("x-contractor-allocation-id", allocation_id),
                ),
            )
            try:
                return await to_thread_until_done(
                    _decode, response, response_limit or limit, name="allocation-protobuf-decode"
                )
            except codec.InvalidWireValue:
                raise ArtifactAPIError(
                    grpc.StatusCode.INTERNAL, "invalid_response", False, correlation
                ) from None

        return await self._owner.execute(self._origin, operation, limit + ENVELOPE_BYTES, timeout)

    async def planner_operation(self, allocation_id, value, *, timeout_seconds):
        return await self._call(
            allocation_id,
            "PlannerOperation",
            value,
            runtime_pb2.PlannerOperationRequest,
            4 << 20,
            timeout_seconds,
        )

    async def gateway_recovery(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "GatewayRecovery",
            value,
            runtime_pb2.GatewayRecoveryRequest,
            4096,
            self._timeout,
        )

    async def submit_audit_evidence(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "SubmitAuditEvidence",
            value,
            runtime_pb2.AuditEvidenceRequest,
            8 << 20,
            self._timeout,
            1 << 20,
        )

    async def submit_audit_judgments(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "SubmitAuditJudgments",
            value,
            runtime_pb2.AuditJudgmentsRequest,
            8 << 20,
            self._timeout,
            1 << 20,
        )

    async def mutate_sitemap(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "MutateSitemap",
            value,
            runtime_pb2.SitemapMutationRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def snapshot_sitemap(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "SnapshotSitemap",
            value,
            runtime_pb2.SitemapSnapshotRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def collect_sitemap(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "CollectSitemap",
            value,
            runtime_pb2.SitemapCollectionRequest,
            256 << 10,
            self._timeout,
            16 << 10,
        )

    async def list_artifacts(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "ListArtifacts",
            value,
            runtime_pb2.ArtifactListRequest,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def read_artifact_metadata(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "ReadArtifactMetadata",
            value,
            runtime_pb2.ArtifactReadTarget,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def check_artifact_access(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "CheckArtifactAccess",
            value,
            runtime_pb2.ArtifactReadTarget,
            1 << 20,
            self._timeout,
            1 << 20,
        )

    async def list_findings(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "ListFindings",
            value,
            runtime_pb2.FindingListRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )

    async def read_finding(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "ReadFinding",
            value,
            runtime_pb2.FindingReadRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )

    async def select_findings(self, allocation_id, value):
        return await self._call(
            allocation_id,
            "SelectFindings",
            value,
            runtime_pb2.FindingSelectionRequest,
            1 << 20,
            self._timeout,
            256 << 10,
        )
