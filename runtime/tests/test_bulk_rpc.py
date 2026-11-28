"""Native streaming framing, bounded ownership and mTLS bulk isolation."""

import asyncio
from pathlib import Path

import pytest
from test_allocation_rpc import Operations, operation, serve
from test_grpc_interop import pki as pki
from test_grpc_interop import runtime_context

from contractor_runtime._allocation_rpc import AllocationRPC
from contractor_runtime._bulk_rpc import BulkRPC, prepare_finding_submission
from contractor_runtime._rpc_connections import RPCConnections
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire import runtime_pb2_grpc as rpc
from contractor_runtime.contracts.grpc_wire.limits import BULK_CHUNK_BYTES


def metadata(size):
    return pb.ArtifactPayloadMetadata(
        api_version="contractor/v1alpha1",
        artifact=pb.ArtifactRef(namespace="analysis", name="report", revision="revision-1"),
        media_type="application/octet-stream",
        size=size,
        binding_created_at="2026-11-28T00:00:00Z",
        revision_created_at="2026-11-28T00:00:00.123456789Z",
    )


def header(size):
    return {
        "artifact": {"namespace": "analysis", "name": "report"},
        "mediaType": "application/octet-stream",
        "size": size,
        "createOnly": True,
    }


TARGET = {"artifact": {"namespace": "analysis", "name": "report"}}


class Payloads(rpc.ArtifactServicer):
    def __init__(self, payload=b"", damage=None):
        self.payload, self.damage = payload, damage
        self.writes, self.submissions = [], []
        self.entered, self.release, self.joined = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def Read(self, request, context):
        authority = dict(context.invocation_metadata())
        assert authority["x-contractor-runtime-instance-id"] == "runtime-1"
        assert authority["x-contractor-allocation-id"] == "allocation-1"
        self.entered.set()
        try:
            if self.damage == "block":
                await self.release.wait()
            value = metadata(len(self.payload))
            if self.damage == "metadata":
                value.binding_created_at = "secret-invalid-metadata"
            yield pb.ArtifactReadFrame(header=value)
            if self.damage == "header":
                yield pb.ArtifactReadFrame(header=value)
                return
            if self.damage == "short":
                return
            for offset in range(0, len(self.payload), BULK_CHUNK_BYTES):
                chunk = pb.ArtifactReadFrame(chunk=self.payload[offset : offset + BULK_CHUNK_BYTES])
                if self.damage == "chunk":
                    chunk.chunk = chunk.chunk[:-1]
                if self.damage == "unknown":
                    chunk.MergeFromString(b"\xf8\x07\x01")
                yield chunk
            if self.damage == "extra":
                yield pb.ArtifactReadFrame(chunk=b"x")
        finally:
            self.joined.set()

    async def Write(self, frames, context):
        messages = [frame async for frame in frames]
        assert messages[0].WhichOneof("frame") == "header"
        request = messages[0].header
        chunks = [frame.chunk for frame in messages[1:]]
        assert all(
            len(chunk) == min(BULK_CHUNK_BYTES, request.size - i * BULK_CHUNK_BYTES)
            for i, chunk in enumerate(chunks)
        )
        self.writes.append((request, b"".join(chunks)))
        return metadata(request.size)

    async def SubmitFinding(self, frames, context):
        self.submissions.append([frame async for frame in frames])
        return pb.FindingSubmissionReceipt(api_version="contractor/v1alpha1", replayed=False)


@pytest.mark.parametrize("size", [0, 1, BULK_CHUNK_BYTES, 3 * BULK_CHUNK_BYTES + 13])
def test_native_bulk_roundtrip_has_exact_chunks_and_typed_precondition(pki, size):
    async def scenario():
        payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
        service = Payloads(payload)
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        client = BulkRPC(owner, origin, "runtime-1", 5)
        try:
            reply = await client.write_artifact("allocation-1", header(size), payload)
            result = await client.read_artifact("allocation-1", TARGET, max_response_bytes=size)
            assert result.data == payload and reply == result.metadata
            assert service.writes[0][1] == payload
            assert service.writes[0][0].WhichOneof("precondition") == "create_only"
            assert not owner._tasks
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["metadata", "header", "short", "chunk", "extra", "unknown"])
def test_native_bulk_rejects_malformed_frames_without_disclosing_metadata(pki, damage):
    async def scenario():
        service = Payloads(b"payload", damage)
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        try:
            with pytest.raises(ArtifactTransportError) as failure:
                await BulkRPC(owner, origin, "runtime-1", 5).read_artifact(
                    "allocation-1", TARGET, max_response_bytes=7
                )
            assert "secret-invalid-metadata" not in str(failure.value)
            assert not owner._tasks
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_bulk_read_limit_precedes_buffer_and_cancellation_joins_owners(pki):
    async def scenario():
        service = Payloads(b"payload")
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        try:
            client = BulkRPC(owner, origin, "runtime-1", 5)
            with pytest.raises(ArtifactResponseLimitError):
                await client.read_artifact("allocation-1", TARGET, max_response_bytes=6)
            service.damage = "block"
            service.entered.clear()
            service.joined.clear()
            task = asyncio.create_task(
                client.read_artifact("allocation-1", TARGET, max_response_bytes=7)
            )
            await asyncio.wait_for(service.entered.wait(), 5)
            directories = [item.directory.name for item in owner._connections]
            await owner.aclose()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(service.joined.wait(), 5)
            assert not owner._tasks and not owner._connections
            assert all(not Path(directory).exists() for directory in directories)
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_bulk_saturation_does_not_queue_allocation_control_calls(pki):
    async def scenario():
        service = Payloads(b"x", "block")
        server, origin = await serve(pki, Operations(), bulk=service)
        bulk = RPCConnections(runtime_context(pki), bulk=True)
        unary = RPCConnections(runtime_context(pki))
        calls = []
        try:
            client = BulkRPC(bulk, origin, "runtime-1", 5)
            calls = [
                asyncio.create_task(
                    client.read_artifact("allocation-1", TARGET, max_response_bytes=1)
                )
                for _ in range(32)
            ]
            # Wait for admission, without wall-clock assertions or examining progress.
            while len(bulk._tasks) < 32:
                await asyncio.sleep(0)
            with pytest.raises(ArtifactAPIError) as failure:
                await client.read_artifact("allocation-1", TARGET, max_response_bytes=1)
            assert failure.value.code == "allocation_rpc_capacity"
            reply = await AllocationRPC(unary, origin, "runtime-1", 3).planner_operation(
                "allocation-1", operation(), timeout_seconds=3
            )
            assert not reply["stop"]
        finally:
            service.release.set()
            await bulk.aclose()
            await asyncio.gather(*calls, return_exceptions=True)
            await unary.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_finding_stream_freezes_documents_and_uses_named_order(pki):
    async def scenario():
        source = {
            "apiVersion": "contractor/v1alpha1",
            "invocationId": "invocation-1",
            "submissionId": "submission-1",
            "evidenceRefs": [],
            "proposal": {"description": "x" * (BULK_CHUNK_BYTES + 3)},
            "derived_anchors": {"anchors": []},
            "httpEvidence": [
                {"invocationId": "invocation-1", "capture": {"source": "http", "request_id": "1"}},
                {
                    "invocationId": "earlier-invocation",
                    "capture": {"source": "http", "request_id": "2"},
                },
            ],
        }
        prepared = prepare_finding_submission(source, 64 << 20, 8 << 20)
        source["proposal"]["description"] = "changed"
        service = Payloads()
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        try:
            await BulkRPC(owner, origin, "runtime-1", 5).submit_finding("allocation-1", prepared)
            frames = service.submissions[0]
            assert frames[0].header.proposal_size == len(prepared.documents[0][1])
            kinds = [frame.chunk.document for frame in frames[1:]]
            assert kinds == [pb.FINDING_DOCUMENT_PROPOSAL] * 2 + [
                pb.FINDING_DOCUMENT_DERIVED_ANCHORS,
                pb.FINDING_DOCUMENT_HTTP_EVIDENCE,
            ]
            assert b"changed" not in b"".join(frame.chunk.data for frame in frames[1:])
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_bulk_early_typed_rejection_stops_upload_without_replay_or_peer_text(pki):
    import grpc
    from google.protobuf.any_pb2 import Any
    from google.rpc.status_pb2 import Status

    class Rejected(Payloads):
        calls = 0

        async def Write(self, frames, context):
            self.calls += 1
            detail = Any()
            detail.Pack(
                pb.ErrorDetail(
                    code="allocation_write_fenced", retryable=False, request_id="request_1"
                )
            )
            status = Status(code=grpc.StatusCode.FAILED_PRECONDITION.value[0], details=[detail])
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "PRIVATE_PEER_DIAGNOSTIC",
                trailing_metadata=(("grpc-status-details-bin", status.SerializeToString()),),
            )

    async def scenario():
        service = Rejected()
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        try:
            payload = b"x" * (2 * BULK_CHUNK_BYTES + 7)
            with pytest.raises((ArtifactAPIError, ArtifactTransportError)) as failure:
                await BulkRPC(owner, origin, "runtime-1", 5).write_artifact(
                    "allocation-1", header(len(payload)), payload
                )
            if isinstance(failure.value, ArtifactAPIError):
                assert failure.value.code == "allocation_write_fenced"
                assert not failure.value.retryable
            else:
                # A reset can lose the peer trailers entirely. That is an
                # ambiguous transport failure, never evidence of a rejection
                # or of non-commit. Available final status is pinned separately
                # by the deterministic upload-step tests below.
                assert str(failure.value) == "allocation RPC transport failed"
            assert "PRIVATE_PEER_DIAGNOSTIC" not in str(failure.value)
            assert service.calls == 1 and not owner._tasks
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


def test_native_finding_lost_reply_reuses_capture_and_original_receipt(pki):
    import json

    import grpc
    from test_http_capture import capture
    from test_security_findings_toolset import _receipt_response

    from contractor_runtime.contracts.grpc_wire import codec
    from contractor_runtime.finding_proposals import FindingProposalClient

    request = {
        "apiVersion": "contractor/v1alpha1",
        "invocationId": "worker-invocation-1",
        "submissionId": "finding-" + "a" * 64,
        "proposal": {"client_key": "candidate-1"},
        "evidenceRefs": [],
        "httpEvidence": [{"invocationId": "worker-invocation-1", "capture": capture()}],
    }

    class LostReceipt(Payloads):
        async def SubmitFinding(self, frames, context):
            self.submissions.append([frame.SerializeToString() async for frame in frames])
            if len(self.submissions) == 1:
                await context.abort(grpc.StatusCode.INTERNAL, "untrusted post-commit reset")
            return codec.encode(
                json.loads(_receipt_response(200, replayed=True, request=request).body),
                pb.FindingSubmissionReceipt(),
            )

    async def scenario():
        service = LostReceipt()
        server, origin = await serve(pki, Operations(), bulk=service)
        owner = RPCConnections(runtime_context(pki), bulk=True)
        try:
            client = FindingProposalClient("allocation-1", BulkRPC(owner, origin, "runtime-1", 5))
            receipt = await client.submit(request)
            assert receipt["receiptId"] == "receipt-1" and receipt["replayed"]
            assert receipt["evidence"][0]["ref"]["namespace"] == "http-exchanges"
            assert (
                len(service.submissions) == 2 and service.submissions[0] == service.submissions[1]
            )
            assert not owner._tasks
        finally:
            await owner.aclose()
            await server.stop(0)

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["frame", "half_close"])
@pytest.mark.parametrize("fault", ["already_finished", "native_write_error"])
@pytest.mark.parametrize("peer_success", [False, True])
def test_bulk_upload_joins_final_peer_status_after_local_send_error(
    monkeypatch, phase, fault, peer_success
):
    from types import SimpleNamespace

    import grpc

    async def scenario():
        final = grpc.aio.AioRpcError(grpc.StatusCode.FAILED_PRECONDITION, (), ())
        local = (
            asyncio.InvalidStateError("finished")
            if fault == "already_finished"
            else grpc.aio.AioRpcError(grpc.StatusCode.INTERNAL, (), ())
        )

        class Call:
            writes = 0
            half_closes = 0
            status_reads = 0

            async def write(self, _frame):
                self.writes += 1
                if phase == "frame":
                    raise local

            async def done_writing(self):
                self.half_closes += 1
                raise local

            def __await__(self):
                async def status():
                    self.status_reads += 1
                    if peer_success:
                        return b"unexpected early success"
                    raise final

                return status().__await__()

            def done(self):
                return True

            async def code(self):
                return final.code()

        call = Call()
        starts = []

        def start(**_kwargs):
            starts.append(None)
            return call

        monkeypatch.setattr(rpc, "ArtifactStub", lambda _channel: SimpleNamespace(Write=start))

        class Owner:
            async def execute(self, _origin, operation, _limit, _timeout):
                return await operation(None, lambda: 5, "request_1")

        expected = ArtifactTransportError if peer_success else grpc.aio.AioRpcError
        with pytest.raises(expected) as failure:
            await BulkRPC(Owner(), "https://control.invalid", "runtime-1", 5).write_artifact(
                "allocation-1", header(0), b""
            )
        if not peer_success:
            assert failure.value is final
        assert len(starts) == call.writes == 1
        assert call.half_closes == (1 if phase == "half_close" else 0)
        assert call.status_reads == 2  # status normalization plus existing cleanup join

    asyncio.run(scenario())
