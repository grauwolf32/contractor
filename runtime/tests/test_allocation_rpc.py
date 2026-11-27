"""Native allocation channels preserve PKI, admission and joined lifetimes."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc
import pytest
from google.protobuf.any_pb2 import Any
from google.rpc.error_details_pb2 import RetryInfo
from google.rpc.status_pb2 import Status
from test_grpc_interop import pki as pki
from test_grpc_interop import runtime_context

from contractor_runtime._allocation_rpc import AllocationConnections, AllocationRPC
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire import runtime_pb2_grpc as rpc
from contractor_runtime.contracts.grpc_wire.limits import MAX_METADATA_BYTES
from contractor_runtime.planner.server_host import ServerPlannerHost


def test_untyped_stream_loss_reconciles_the_original_planner_sequence(pki):
    class LostReply(Operations):
        def __init__(self):
            super().__init__()
            self.requests = []

        async def PlannerOperation(self, request, context):
            self.requests.append(request.SerializeToString())
            if len(self.requests) == 1:
                # An HTTP/2 stream reset can surface as untyped INTERNAL. It
                # establishes no domain rejection and proves no non-commit.
                await context.abort(grpc.StatusCode.INTERNAL, "untrusted reset diagnostic")
            return await super().PlannerOperation(request, context)

    async def scenario():
        service = LostReply()
        server, origin = await serve(pki, service)
        owner = AllocationConnections(runtime_context(pki))
        try:
            host = ServerPlannerHost(
                "allocation-1",
                AllocationRPC(owner, origin, "runtime-1", 3),
                deadline=datetime.now(UTC) + timedelta(seconds=10),
            )
            result = await host.execute("before_model", {})
            assert not result.stop
            assert len(service.requests) == 2 and service.requests[0] == service.requests[1]
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_native_metadata_ceiling_remains_hard_above_the_soft_threshold(pki, monkeypatch):
    class MetadataReply(Operations):
        async def PlannerOperation(self, request, context):
            await context.send_initial_metadata((("x-padding", "x" * (MAX_METADATA_BYTES + 1024)),))
            return await super().PlannerOperation(request, context)

    channel_factory = grpc.aio.insecure_channel

    def relaxed_soft_threshold(target, *, options):
        # Distinguish a hard cap from grpcio's probabilistic soft rejection:
        # without the hard cap this native peer response is always accepted.
        options = {**dict(options), "grpc.max_metadata_size": 64 << 10}
        return channel_factory(target, options=tuple(options.items()))

    monkeypatch.setattr(grpc.aio, "insecure_channel", relaxed_soft_threshold)

    async def scenario():
        server, origin = await serve(pki, MetadataReply())
        owner = AllocationConnections(runtime_context(pki))
        try:
            with pytest.raises(ArtifactAPIError) as failure:
                await AllocationRPC(owner, origin, "runtime-1", 3).planner_operation(
                    "allocation-1", operation(), timeout_seconds=3
                )
            assert failure.value.code == "invalid_response"
            assert not failure.value.retryable and not owner._connections
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


class Operations(rpc.AllocationServicer):
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.full = asyncio.Event()
        self.release = asyncio.Event()
        self.joined = asyncio.Event()

    async def PlannerOperation(self, request, context):
        metadata = dict(context.invocation_metadata())
        assert metadata["x-contractor-runtime-instance-id"] == "runtime-1"
        assert metadata["x-contractor-allocation-id"] == "allocation-1"
        self.calls += 1
        self.active += 1
        self.peak = max(self.active, self.peak)
        if self.active == 4:
            self.full.set()
        try:
            if request.name == "block":
                await self.release.wait()
            return pb.PlannerOperationResponse(
                api_version="contractor/v1alpha1",
                result=pb.PlannerResultDocument(json=b"{}"),
                stop=False,
                escalate=False,
            )
        finally:
            self.active -= 1
            if self.active == 0:
                self.joined.set()

    async def MutateSitemap(self, request, context):
        domain = Any()
        domain.Pack(
            pb.ErrorDetail(
                code="artifact_history_maintenance_pending",
                retryable=True,
                request_id="request_retry",
            )
        )
        retry = Any()
        delay = RetryInfo()
        delay.retry_delay.seconds = 1
        retry.Pack(delay)
        details = Status(code=grpc.StatusCode.UNAVAILABLE.value[0], details=[domain, retry])
        await context.abort(
            grpc.StatusCode.UNAVAILABLE,
            "untrusted text",
            trailing_metadata=(("grpc-status-details-bin", details.SerializeToString()),),
        )


async def serve(pki, service, *, wrong_role=False):
    server = grpc.aio.server()
    rpc.add_AllocationServicer_to_server(service, server)
    certificate = pki.agent_certificate if wrong_role else pki.control_plane_certificate
    key = pki.agent_key if wrong_role else pki.control_plane_key
    credentials = grpc.ssl_server_credentials(
        ((Path(key).read_bytes(), Path(certificate).read_bytes()),),
        root_certificates=Path(pki.ca).read_bytes(),
        require_client_auth=True,
    )
    port = server.add_secure_port("127.0.0.1:0", credentials)
    await server.start()
    return server, f"https://localhost:{port}"


def operation(name="before_model"):
    return {"apiVersion": "contractor/v1alpha1", "sequence": 1, "name": name, "arguments": {}}


def test_pool_bounds_admission_rotates_idle_channels_and_joins_shutdown(pki):
    async def scenario():
        service = Operations()
        server, origin = await serve(pki, service)
        clock = [0.0]
        owner = AllocationConnections(runtime_context(pki), clock=lambda: clock[0])
        client = AllocationRPC(owner, origin, "runtime-1", 3)
        try:
            calls = [
                asyncio.create_task(
                    client.planner_operation("allocation-1", operation("block"), timeout_seconds=10)
                )
                for _ in range(4)
            ]
            await asyncio.wait_for(service.full.wait(), 5)
            assert len(owner._connections) == service.peak == 4
            with pytest.raises(ArtifactTransportError):
                await client.planner_operation("allocation-1", operation(), timeout_seconds=0.01)
            assert service.calls == 4
            service.release.set()
            await asyncio.gather(*calls)
            previous = set(owner._connections)
            clock[0] = 300
            response = await client.planner_operation(
                "allocation-1", operation(), timeout_seconds=3
            )
            assert response["stop"] is False
            assert len(set(owner._connections) - previous) == 1
            service.release.clear()
            service.joined.clear()
            service.full.clear()
            calls = [
                asyncio.create_task(
                    client.planner_operation("allocation-1", operation("block"), timeout_seconds=10)
                )
                for _ in range(4)
            ]
            await asyncio.wait_for(service.full.wait(), 5)
            queued = [
                asyncio.create_task(
                    client.planner_operation("allocation-1", operation("block"), timeout_seconds=10)
                )
                for _ in range(28)
            ]
            await asyncio.sleep(0)
            with pytest.raises(ArtifactAPIError) as capacity:
                await client.planner_operation("allocation-1", operation(), timeout_seconds=3)
            assert capacity.value.code == "allocation_rpc_capacity" and capacity.value.retryable
            assert service.calls == 9  # Four initial, one after rotation, four active.
            calls.extend(queued)
            directories = [connection.directory.name for connection in owner._connections]
            await owner.aclose()
            outcomes = await asyncio.gather(*calls, return_exceptions=True)
            assert all(isinstance(item, asyncio.CancelledError) for item in outcomes)
            await asyncio.wait_for(service.joined.wait(), 5)
            assert not owner._connections and not owner._tasks
            assert all(not Path(directory).exists() for directory in directories)
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_wrong_control_plane_role_is_permanent_and_never_calls_domain(pki):
    async def scenario():
        service = Operations()
        server, origin = await serve(pki, service, wrong_role=True)
        owner = AllocationConnections(runtime_context(pki))
        try:
            with pytest.raises(ArtifactAPIError) as caught:
                await AllocationRPC(owner, origin, "runtime-1", 3).planner_operation(
                    "allocation-1", operation(), timeout_seconds=3
                )
            assert caught.value.code == "invalid_control_plane_identity"
            assert not caught.value.retryable and service.calls == 0
            assert not owner._connections
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_typed_domain_retry_delay_survives_native_status(pki):
    async def scenario():
        server, origin = await serve(pki, Operations())
        owner = AllocationConnections(runtime_context(pki))
        try:
            client = AllocationRPC(owner, origin, "runtime-1", 3)
            for _ in range(2):
                with pytest.raises(ArtifactAPIError) as failure:
                    await client.mutate_sitemap(
                        "allocation-1",
                        {
                            "apiVersion": "contractor/v1alpha1",
                            "idempotencyKey": "mutation-1",
                            "mutation": {
                                "operation": "declare",
                                "expectedRevision": "absent",
                                "url": "https://example.test/",
                                "method": "GET",
                            },
                        },
                    )
                assert failure.value.code == "artifact_history_maintenance_pending"
                assert failure.value.retryable and failure.value.retry_after_seconds == 1
                assert "untrusted text" not in str(failure.value)
                assert len(owner._connections) == 1
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_malformed_committed_receipt_reconciles_same_native_mutation(pki):
    from types import SimpleNamespace

    from test_audit_evidence import evidence_receipt

    from contractor_runtime.audit_evidence import AuditEvidenceClient
    from contractor_runtime.contracts.grpc_wire import codec

    source, receipt = evidence_receipt()

    class EvidenceMutation(Operations):
        def __init__(self):
            super().__init__()
            self.requests = []

        async def SubmitAuditEvidence(self, request, context):
            self.requests.append(request.SerializeToString())
            response = codec.encode(receipt, pb.AuditEvidenceReceipt())
            if len(self.requests) == 1:
                response.MergeFromString(b"\xf8\x07\x01")
            return response

    async def scenario():
        service = EvidenceMutation()
        server, origin = await serve(pki, service)
        owner = AllocationConnections(runtime_context(pki))
        try:
            client = AuditEvidenceClient(
                SimpleNamespace(
                    transport=AllocationRPC(owner, origin, "runtime-1", 3),
                    allocation_id="allocation-1",
                ),
                run_id="run-1",
                audit_id="audit-1",
            )
            value = {
                "invocationId": "invocation-1",
                "submissionId": "tool-call-1",
                "itemKey": "item-00",
                "sources": [source.model_dump()],
                "interpretation": "Scoped observation",
                "testedConditions": [],
                "limitations": [],
            }
            assert (await client.submit(value))["replayed"]
            assert len(service.requests) == 2 and service.requests[0] == service.requests[1]
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())
