"""Native typed allocation RPCs, behind the verified private TLS boundary."""

from __future__ import annotations

import json
import logging

import grpc
from google.protobuf.any_pb2 import Any
from google.rpc.status_pb2 import Status

from contractor_runtime._grpc_metadata import response_request_id
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.contracts import (
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    PrepareAllocationRequest,
    PrivateProtocolDecodeError,
    ReleaseAllocationRequest,
)
from contractor_runtime.contracts.base import RESOURCE_ID_PATTERN
from contractor_runtime.contracts.codec import decode_private
from contractor_runtime.contracts.grpc_wire import codec, runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire import runtime_pb2 as pb
from contractor_runtime.contracts.grpc_wire.limits import MAX_CONTROL_BYTES, MAX_STATE_MESSAGE_BYTES
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.state import RuntimeState
from contractor_runtime.threads import to_thread_until_done

logger = logging.getLogger(__name__)
MAX_LIFECYCLE_BYTES = MAX_CONTROL_BYTES
HTTP_STATUS = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.UNAUTHENTICATED,
    403: grpc.StatusCode.PERMISSION_DENIED,
    404: grpc.StatusCode.NOT_FOUND,
    409: grpc.StatusCode.FAILED_PRECONDITION,
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    422: grpc.StatusCode.INVALID_ARGUMENT,
    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
    500: grpc.StatusCode.INTERNAL,
    503: grpc.StatusCode.UNAVAILABLE,
    504: grpc.StatusCode.DEADLINE_EXCEEDED,
}


class LifecycleHandler(runtime_pb2_grpc.RuntimeLifecycleServicer):
    def __init__(self, state: RuntimeState, service: AllocationService):
        self.state = state
        self.service = service

    async def Prepare(self, request, context):
        return await self._call(
            request, context, PrepareAllocationRequest, "prepare", pb.PrepareAllocationResponse
        )

    async def Finalize(self, request, context):
        return await self._call(
            request, context, FinalizeAllocationRequest, "finalize", pb.AllocationFinalResponse
        )

    async def Abort(self, request, context):
        return await self._call(
            request, context, AbortAllocationRequest, "abort", pb.AllocationFinalResponse
        )

    async def Release(self, request, context):
        return await self._call(
            request, context, ReleaseAllocationRequest, "release", pb.ReleaseAllocationResponse
        )

    async def ReadState(self, request, context):
        request_id = await response_request_id(context)
        await self.state.record_route_dispatch()
        try:
            if request.ByteSize() > MAX_CONTROL_BYTES:
                raise codec.InvalidWireValue
            value = codec.decode(request)
            allocation = value.get("allocationId")
            revision = value.get("ifRevision")
            if (
                not isinstance(allocation, str)
                or RESOURCE_ID_PATTERN.fullmatch(allocation) is None
                or revision == 0
            ):
                raise codec.InvalidWireValue
        except codec.InvalidWireValue:
            return await _abort(
                context, grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, request_id
            )
        try:
            conditional = (
                f'"contractor-agent-state-v1-{revision}"' if revision is not None else None
            )
            snapshot = await self.service.agent_state_snapshot(
                allocation, if_none_match=conditional
            )
            if snapshot is None:
                if revision is None:
                    raise RuntimeError("unconditional State read returned no snapshot")
                return pb.StateReadResponse(revision=revision, not_modified=pb.StateNotModified())
            if revision == snapshot.state.state_revision:
                return pb.StateReadResponse(revision=revision, not_modified=pb.StateNotModified())
            return await to_thread_until_done(_encode_state, snapshot, name="state-protobuf-encode")
        except AllocationError as error:
            return await _abort(
                context,
                HTTP_STATUS.get(error.status_code, grpc.StatusCode.INTERNAL),
                error.code,
                error.retryable,
                request_id,
            )
        except Exception as error:
            logger.error(
                "private RPC failed request_id=%s operation=state error_type=%s",
                request_id,
                type(error).__name__,
            )
            return await _abort(
                context, grpc.StatusCode.INTERNAL, "internal_error", True, request_id
            )

    async def _call(self, request, context, model, operation, response_type):
        request_id = await response_request_id(context)
        await self.state.record_route_dispatch()
        try:
            value = await to_thread_until_done(
                _decode_request, request, model, name="lifecycle-protobuf-decode"
            )
        except (codec.InvalidWireValue, PrivateProtocolDecodeError):
            return await _abort(
                context, grpc.StatusCode.INVALID_ARGUMENT, "invalid_request", False, request_id
            )
        try:
            method = getattr(self.service, operation)
            result = await method(value.spec if operation == "prepare" else value)
            if operation == "release":
                return response_type()
            response = await to_thread_until_done(
                _encode_response, result, response_type, name="lifecycle-protobuf-encode"
            )
            if response.ByteSize() > MAX_LIFECYCLE_BYTES:
                return await _abort(
                    context,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    "response_too_large",
                    False,
                    request_id,
                )
            return response
        except AllocationError as error:
            return await _abort(
                context,
                HTTP_STATUS.get(error.status_code, grpc.StatusCode.INTERNAL),
                error.code,
                error.retryable,
                request_id,
            )
        except Exception as error:
            logger.error(
                "private RPC failed request_id=%s operation=%s error_type=%s",
                request_id,
                operation,
                type(error).__name__,
            )
            return await _abort(
                context, grpc.StatusCode.INTERNAL, "internal_error", True, request_id
            )


async def _abort(context, status, code, retryable, request_id):
    detail = Any()
    detail.Pack(pb.ErrorDetail(code=code, retryable=retryable, request_id=request_id))
    payload = Status(code=status.value[0], message=code, details=[detail]).SerializeToString()
    await context.abort(status, code, trailing_metadata=(("grpc-status-details-bin", payload),))


def _decode_request(request, model):
    if request.ByteSize() > MAX_LIFECYCLE_BYTES:
        raise codec.InvalidWireValue
    document = json.dumps(codec.decode(request), allow_nan=False, separators=(",", ":"))
    if len(document.encode("utf-8")) > MAX_LIFECYCLE_BYTES:
        raise codec.InvalidWireValue
    return decode_private(model, document)


def _encode_response(result, response_type):
    return codec.encode(
        result.model_dump(mode="json", by_alias=True, exclude_none=True), response_type()
    )


def _encode_state(snapshot):
    document = snapshot.model_dump_json(by_alias=True).encode()
    response = pb.StateReadResponse(
        revision=snapshot.state.state_revision, snapshot=pb.AgentStateDocument(json=document)
    )
    if (
        len(document) > get_shared_config().worker_state_max_bytes
        or response.ByteSize() > MAX_STATE_MESSAGE_BYTES
    ):
        raise AllocationError(
            "agent_state_unavailable",
            "allocation Worker State is unavailable",
            retryable=False,
            status_code=503,
        )
    return response
