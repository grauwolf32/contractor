"""Bounded allocation payload streams; domain owners retain CAS and receipts."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import grpc
from google.protobuf.message import Message
from pydantic import ValidationError

from contractor_runtime._allocation_errors import (
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime._allocation_rpc import _decode, _encode
from contractor_runtime.contracts.grpc_wire import codec
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire import runtime_pb2_grpc as rpc
from contractor_runtime.contracts.grpc_wire.bulk import ArtifactPayloadMetadata
from contractor_runtime.contracts.grpc_wire.limits import (
    BULK_CHUNK_BYTES,
    BULK_FRAME_BYTES,
    BULK_HEADER_BYTES,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.threads import to_thread_until_done


@dataclass(frozen=True, slots=True)
class PayloadDownload:
    metadata: Mapping = field(repr=False)
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class FindingSubmission:
    header: Mapping = field(repr=False)
    documents: tuple[tuple[int, bytes], ...] = field(repr=False)


def _json(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def prepare_finding_submission(value: Mapping, maximum: int, proposal_limit: int):
    documents = []
    fields = {
        "proposal": (pb.FINDING_DOCUMENT_PROPOSAL, "proposalSize"),
        "derived_anchors": (pb.FINDING_DOCUMENT_DERIVED_ANCHORS, "derivedAnchorsSize"),
        "httpEvidence": (pb.FINDING_DOCUMENT_HTTP_EVIDENCE, "httpEvidenceSize"),
    }
    allowed = {
        "apiVersion",
        "invocationId",
        "submissionId",
        "incidental",
        "itemKey",
        "evidenceRefs",
        "httpEvidenceInvocationId",
    }
    if set(value) - allowed - fields.keys() or "proposal" not in value:
        raise ValueError("invalid finding submission")
    header = {key: item for key, item in value.items() if key in allowed}
    # Freeze typed selections as well as document bytes before the first call.
    header = json.loads(_json(header))
    logical = len(_json(header))
    for name, (kind, size_field) in fields.items():
        if name not in value:
            continue
        if not isinstance(value[name], Mapping):
            raise ValueError("invalid finding document")
        data = _json(value[name])
        if name == "proposal" and len(data) > proposal_limit:
            raise ValueError("finding proposal exceeds its size limit")
        logical += len(f',"{name}":') + len(data)
        documents.append((kind, data))
        header[size_field] = len(data)
    if logical > maximum:
        raise ValueError("finding proposal request exceeds its size limit")
    _encode(header, pb.FindingSubmissionHeader, BULK_HEADER_BYTES)
    return FindingSubmission(header, tuple(documents))


def _metadata(message):
    if not isinstance(message, Message):
        raise ArtifactTransportError("bulk RPC returned invalid payload metadata")
    try:
        value = _decode(message, BULK_HEADER_BYTES)
        metadata = ArtifactPayloadMetadata.model_validate(value)
    except (codec.InvalidWireValue, ValidationError, ValueError):
        raise ArtifactTransportError("bulk RPC returned invalid payload metadata") from None
    return value, metadata


def _read_chunk(frame, remaining):
    if not isinstance(frame, pb.ArtifactReadFrame):
        raise ArtifactTransportError("bulk RPC returned an invalid frame")
    known = pb.ArtifactReadFrame()
    known.CopyFrom(frame)
    known.DiscardUnknownFields()
    if (
        known != frame
        or frame.WhichOneof("frame") != "chunk"
        or len(frame.chunk) != min(BULK_CHUNK_BYTES, remaining)
    ):
        raise ArtifactTransportError("bulk RPC returned an invalid chunk sequence")
    return frame.chunk


async def _write_frame(call, message):
    await _upload_step(call, lambda: call.write(message))


async def _upload_step(call, send):
    try:
        await send()
    except (asyncio.InvalidStateError, grpc.aio.AioRpcError):
        # Local send completion may race the final peer status. Join the native
        # call before classifying it; never retry a frame or overwrite its verdict.
        await call
        raise ArtifactTransportError("bulk RPC finished before upload EOF") from None


async def _finish_call(call, *, unary_response):
    async def cleanup():
        if not call.done():
            call.cancel()
        with contextlib.suppress(asyncio.CancelledError, grpc.aio.AioRpcError):
            if unary_response:
                await call
            else:
                await call.wait_for_connection()
        await call.code()

    owner = asyncio.create_task(cleanup(), name="bulk-rpc-join")
    interrupted = await join_owned_task(owner)
    owner.result()
    if interrupted:
        raise asyncio.CancelledError


class BulkRPC:
    def __init__(self, owner, origin, instance_id, timeout):
        self._owner, self._origin = owner, origin
        self._instance_id, self._timeout = instance_id, timeout

    def _authority(self, allocation_id, correlation):
        return (
            ("x-request-id", correlation),
            ("x-contractor-runtime-instance-id", self._instance_id),
            ("x-contractor-allocation-id", allocation_id),
        )

    async def read_artifact(self, allocation_id, value, *, max_response_bytes):
        async def operation(channel, remaining, correlation):
            request = await to_thread_until_done(
                _encode,
                value,
                pb.ArtifactReadTarget,
                BULK_HEADER_BYTES,
                name="bulk-read-header",
            )
            call = rpc.ArtifactStub(channel).Read(
                request, timeout=remaining(), metadata=self._authority(allocation_id, correlation)
            )
            try:
                frame = await call.read()
                if not isinstance(frame, pb.ArtifactReadFrame):
                    raise ArtifactTransportError("bulk RPC omitted its metadata header")
                known = pb.ArtifactReadFrame()
                known.CopyFrom(frame)
                known.DiscardUnknownFields()
                if known != frame or frame.WhichOneof("frame") != "header":
                    raise ArtifactTransportError("bulk RPC returned an invalid metadata header")
                raw, metadata = await to_thread_until_done(
                    _metadata, frame.header, name="bulk-read-metadata"
                )
                requested = value["artifact"]
                if (
                    metadata.artifact.namespace != requested["namespace"]
                    or metadata.artifact.name != requested["name"]
                    or (
                        "revision" in requested
                        and metadata.artifact.revision != requested["revision"]
                    )
                ):
                    raise ArtifactTransportError("bulk RPC returned another Artifact")
                if metadata.size > max_response_bytes:
                    raise ArtifactResponseLimitError("Artifact payload exceeds the read limit")
                buffer = await to_thread_until_done(
                    lambda: io.BytesIO(bytes(metadata.size)), name="bulk-read-buffer"
                )
                try:
                    position = 0
                    while position < metadata.size:
                        frame = await call.read()
                        data = await to_thread_until_done(
                            _read_chunk, frame, metadata.size - position, name="bulk-read-chunk"
                        )
                        await to_thread_until_done(buffer.write, data, name="bulk-read-copy")
                        position += len(data)
                    if await call.read() is not grpc.aio.EOF:
                        raise ArtifactTransportError("bulk RPC returned extra payload frames")
                    data = await to_thread_until_done(buffer.getvalue, name="bulk-read-result")
                    return PayloadDownload(raw, data)
                finally:
                    buffer.close()
            finally:
                await _finish_call(call, unary_response=False)

        return await self._owner.execute(self._origin, operation, BULK_FRAME_BYTES, self._timeout)

    async def write_artifact(self, allocation_id, value, data):
        async def operation(channel, remaining, correlation):
            header = await to_thread_until_done(
                _encode,
                value,
                pb.ArtifactWriteHeader,
                BULK_HEADER_BYTES,
                name="bulk-write-header",
            )
            call = rpc.ArtifactStub(channel).Write(
                timeout=remaining(), metadata=self._authority(allocation_id, correlation)
            )
            try:
                await _write_frame(call, pb.ArtifactWriteFrame(header=header))
                for offset in range(0, len(data), BULK_CHUNK_BYTES):
                    frame = await to_thread_until_done(
                        lambda offset=offset: pb.ArtifactWriteFrame(
                            chunk=data[offset : offset + BULK_CHUNK_BYTES]
                        ),
                        name="bulk-write-chunk",
                    )
                    await _write_frame(call, frame)
                await _upload_step(call, call.done_writing)
                response = await call
                raw, _ = await to_thread_until_done(_metadata, response, name="bulk-write-receipt")
                return raw
            finally:
                await _finish_call(call, unary_response=True)

        return await self._owner.execute(self._origin, operation, BULK_FRAME_BYTES, self._timeout)

    async def submit_finding(self, allocation_id, submission: FindingSubmission):
        async def operation(channel, remaining, correlation):
            header = await to_thread_until_done(
                _encode,
                submission.header,
                pb.FindingSubmissionHeader,
                BULK_HEADER_BYTES,
                name="finding-stream-header",
            )
            call = rpc.ArtifactStub(channel).SubmitFinding(
                timeout=remaining(), metadata=self._authority(allocation_id, correlation)
            )
            try:
                await _write_frame(call, pb.FindingSubmissionFrame(header=header))
                for kind, data in submission.documents:
                    for offset in range(0, len(data), BULK_CHUNK_BYTES):
                        frame = await to_thread_until_done(
                            lambda offset=offset, kind=kind, data=data: pb.FindingSubmissionFrame(
                                chunk=pb.FindingDocumentChunk(
                                    document=kind, data=data[offset : offset + BULK_CHUNK_BYTES]
                                )
                            ),
                            name="finding-stream-chunk",
                        )
                        await _write_frame(call, frame)
                await _upload_step(call, call.done_writing)
                response = await call
                if not isinstance(response, Message):
                    raise ArtifactTransportError("finding RPC returned an invalid receipt")
                try:
                    return await to_thread_until_done(
                        _decode, response, BULK_HEADER_BYTES, name="finding-stream-receipt"
                    )
                except codec.InvalidWireValue:
                    raise ArtifactTransportError(
                        "finding RPC returned an invalid receipt"
                    ) from None
            finally:
                await _finish_call(call, unary_response=True)

        return await self._owner.execute(self._origin, operation, BULK_FRAME_BYTES, self._timeout)
