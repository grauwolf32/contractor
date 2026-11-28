"""Scripted domain-response records; never an installed Runtime transport."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import grpc

from contractor_runtime._grpc_metadata import error_detail
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.strict_json import strict_json_loads


@dataclass(frozen=True, slots=True)
class ArtifactHTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


HTTP_TO_RPC = {
    400: grpc.StatusCode.INVALID_ARGUMENT,
    401: grpc.StatusCode.UNAUTHENTICATED,
    403: grpc.StatusCode.PERMISSION_DENIED,
    404: grpc.StatusCode.NOT_FOUND,
    409: grpc.StatusCode.FAILED_PRECONDITION,
    412: grpc.StatusCode.FAILED_PRECONDITION,
    413: grpc.StatusCode.RESOURCE_EXHAUSTED,
    422: grpc.StatusCode.INVALID_ARGUMENT,
    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
    500: grpc.StatusCode.INTERNAL,
    502: grpc.StatusCode.UNAVAILABLE,
    503: grpc.StatusCode.UNAVAILABLE,
    504: grpc.StatusCode.DEADLINE_EXCEEDED,
}


def recorded_status(status):
    return {
        grpc.StatusCode.INVALID_ARGUMENT: 400,
        grpc.StatusCode.UNAUTHENTICATED: 401,
        grpc.StatusCode.PERMISSION_DENIED: 403,
        grpc.StatusCode.NOT_FOUND: 404,
        grpc.StatusCode.FAILED_PRECONDITION: 409,
        grpc.StatusCode.RESOURCE_EXHAUSTED: 503,
        grpc.StatusCode.INTERNAL: 500,
        grpc.StatusCode.UNKNOWN: 500,
        grpc.StatusCode.UNAVAILABLE: 503,
        grpc.StatusCode.ABORTED: 503,
        grpc.StatusCode.DEADLINE_EXCEEDED: 504,
    }[status]


def raise_for_status(response):
    if 200 <= response.status_code < 300:
        return
    status = HTTP_TO_RPC.get(response.status_code, grpc.StatusCode.UNKNOWN)
    try:
        value = strict_json_loads(response.body)
        if not isinstance(value, dict) or set(value) != {
            "code",
            "message",
            "retryable",
            "requestId",
        }:
            raise ValueError
        code, retryable, request_id = error_detail(
            {key: item for key, item in value.items() if key != "message"}
        )
        if not isinstance(value["message"], str) or not value["message"].strip():
            raise ValueError
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ArtifactAPIError(status, "invalid_error_response", False) from None
    if code == "artifact_transfer_capacity":
        status = grpc.StatusCode.RESOURCE_EXHAUSTED
    elif code == "storage_transaction_conflict":
        status = grpc.StatusCode.ABORTED
    raise ArtifactAPIError(status, code, retryable, request_id)


def require_json(headers):
    if headers.get("content-type") != "application/json":
        raise ArtifactTransportError("scripted response is not application/json")


def _strong_etag(headers):
    value = headers.get("etag", "")
    if (
        len(value) < 3
        or value[0] != '"'
        or value[-1] != '"'
        or re.fullmatch(r"[!#-+\--\[\]-~]+", value[1:-1]) is None
    ):
        raise ArtifactTransportError("scripted response has invalid revision metadata")
    return value[1:-1]
