"""Private mTLS ASGI server and allocation lifecycle routes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import ssl
import uuid
from collections.abc import Generator, Mapping
from typing import Any

import uvicorn
from pydantic import BaseModel, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.protocols.http.h11_impl import H11Protocol

from contractor_runtime.a2a_server import AllocationA2AGateway
from contractor_runtime.allocation import AllocationError, AllocationService
from contractor_runtime.contracts import (
    AbortAllocationRequest,
    FinalizeAllocationRequest,
    PrepareAllocationRequest,
    ReleaseAllocationRequest,
)
from contractor_runtime.mtls import verify_control_plane_peer
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState

logger = logging.getLogger(__name__)

VERIFIED_PEER_EXTENSION = "contractor.mtls.peer_certificate"
MAX_LIFECYCLE_REQUEST_BYTES = 1 << 20
REQUEST_ID_HEADER = b"x-request-id"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class VerifiedMTLSH11Protocol(H11Protocol):
    """Reject non-Control-Plane TLS peers before parsing an HTTP request."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._verified_peer: Mapping[str, Any] | None = None
        super().__init__(*args, **kwargs)
        application = self.app

        async def inject_verified_peer(scope: Scope, receive: Receive, send: Send) -> None:
            if self._verified_peer is None:
                raise RuntimeError("HTTP dispatch attempted without a verified mTLS peer")
            copied = dict(scope)
            extensions = dict(copied.get("extensions", {}))
            extensions[VERIFIED_PEER_EXTENSION] = self._verified_peer
            copied["extensions"] = extensions
            await application(copied, receive, send)

        self.app = inject_verified_peer

    def connection_made(self, transport: asyncio.Transport) -> None:
        super().connection_made(transport)
        ssl_object = transport.get_extra_info("ssl_object")
        try:
            if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                raise ssl.SSLCertVerificationError("private listener requires TLS")
            verify_control_plane_peer(ssl_object)
            certificate = ssl_object.getpeercert()
            if not certificate:
                raise ssl.SSLCertVerificationError("private listener requires a peer certificate")
            self._verified_peer = certificate
        except ssl.SSLError:
            logger.warning("rejected private connection with invalid Control Plane identity")
            transport.abort()


class VerifiedPeerMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and VERIFIED_PEER_EXTENSION not in scope.get("extensions", {}):
            request_id = str(scope.get("state", {}).get("request_id", "request-unavailable"))
            response = JSONResponse(
                {
                    "code": "mtls_required",
                    "message": "verified mTLS is required",
                    "retryable": False,
                    "requestId": request_id,
                },
                status_code=401,
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


class CorrelationIDMiddleware:
    """Propagate one bounded private request ID through responses and logs."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        request_id = _request_id(scope)
        copied = dict(scope)
        state = dict(copied.get("state", {}))
        state["request_id"] = request_id
        copied["state"] = state
        status_code = 500
        response_started = False

        async def send_with_request_id(message: Mapping[str, Any]) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != REQUEST_ID_HEADER
                ]
                headers.append((REQUEST_ID_HEADER, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)  # type: ignore[arg-type]

        try:
            await self._app(copied, receive, send_with_request_id)
        except Exception as error:
            copied["state"]["error_type"] = type(error).__name__
            if response_started:
                raise
            response = JSONResponse(
                {
                    "code": "internal_error",
                    "message": "private request could not be processed",
                    "retryable": True,
                    "requestId": request_id,
                },
                status_code=500,
            )
            await response(copied, receive, send_with_request_id)
        finally:
            if status_code >= 500:
                logger.error(
                    "private HTTP request failed request_id=%s method=%s status=%d error_type=%s",
                    request_id,
                    scope.get("method", ""),
                    status_code,
                    copied["state"].get("error_type", "handled_failure"),
                )


def create_app(
    state: RuntimeState | None = None,
    *,
    allocation_service: AllocationService | None = None,
    require_verified_peer: bool = True,
) -> ASGIApp:
    runtime_state = state or RuntimeState()

    async def health(_: Request) -> JSONResponse:
        snapshot = await runtime_state.snapshot()
        return JSONResponse({"status": "ok", "state": snapshot.process_state.value})

    async def readiness(_: Request) -> JSONResponse:
        snapshot = await runtime_state.snapshot()
        ready = snapshot.process_state in {ProcessState.IDLE, ProcessState.ALLOCATED}
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "state": snapshot.process_state.value},
            status_code=200 if ready else 503,
        )

    async def prepare(request: Request) -> Response:
        return await lifecycle_call(request, PrepareAllocationRequest, "prepare")

    async def finalize(request: Request) -> Response:
        return await lifecycle_call(request, FinalizeAllocationRequest, "finalize")

    async def abort(request: Request) -> Response:
        return await lifecycle_call(request, AbortAllocationRequest, "abort")

    async def release(request: Request) -> Response:
        return await lifecycle_call(request, ReleaseAllocationRequest, "release")

    async def lifecycle_call[RequestModel: BaseModel](
        request: Request,
        model: type[RequestModel],
        operation: str,
    ) -> Response:
        await runtime_state.record_route_dispatch()
        if allocation_service is None:
            return await lifecycle_unavailable_without_dispatch(request)
        try:
            value = await _decode_request(request, model)
            allocation_id = (
                value.spec.allocation_id
                if isinstance(value, PrepareAllocationRequest)
                else value.allocation_id
            )
            if request.path_params["allocation_id"] != allocation_id:
                raise AllocationError(
                    "allocation_id_mismatch",
                    "path allocation ID does not match the request body",
                    retryable=False,
                    status_code=409,
                )
            if operation == "prepare":
                result = await allocation_service.prepare(value.spec)
            elif operation == "finalize":
                result = await allocation_service.finalize(value)
            elif operation == "abort":
                result = await allocation_service.abort(value)
            else:
                await allocation_service.release(value)
                return Response(status_code=204)
            return JSONResponse(result.model_dump(mode="json", by_alias=True, exclude_none=True))
        except ValidationError:
            return JSONResponse(
                {
                    "code": "invalid_request",
                    "message": "request does not match the allocation lifecycle contract",
                    "retryable": False,
                    "requestId": request.state.request_id,
                },
                status_code=422,
            )
        except AllocationError as error:
            return JSONResponse(
                {**error.payload(), "requestId": request.state.request_id},
                status_code=error.status_code,
            )
        except Exception as error:
            request.state.error_type = type(error).__name__
            return JSONResponse(
                {
                    "code": "internal_error",
                    "message": "allocation lifecycle operation failed",
                    "retryable": True,
                    "requestId": request.state.request_id,
                },
                status_code=500,
            )

    async def lifecycle_unavailable_without_dispatch(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "code": "allocation_service_unavailable",
                "message": "allocation lifecycle service is not configured",
                "retryable": True,
                "requestId": request.state.request_id,
            },
            status_code=503,
        )

    application: ASGIApp = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", readiness, methods=["GET"]),
            Route(
                "/private/v1/allocations/{allocation_id}/prepare",
                prepare,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/finalize",
                finalize,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/abort",
                abort,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/release",
                release,
                methods=["POST"],
            ),
        ]
    )
    if allocation_service is not None:
        application = AllocationA2AGateway(application, allocation_service)
    if require_verified_peer:
        application = VerifiedPeerMiddleware(application)
    return CorrelationIDMiddleware(application)


def _request_id(scope: Scope) -> str:
    values: list[str] = []
    for name, value in scope.get("headers", []):
        if name.lower() != REQUEST_ID_HEADER:
            continue
        try:
            values.append(value.decode("ascii"))
        except UnicodeDecodeError:
            return f"request_{uuid.uuid4().hex}"
    if len(values) == 1 and REQUEST_ID_PATTERN.fullmatch(values[0]) is not None:
        return values[0]
    return f"request_{uuid.uuid4().hex}"


async def _decode_request[RequestModel: BaseModel](
    request: Request, model: type[RequestModel]
) -> RequestModel:
    content_types = request.headers.getlist("content-type")
    if content_types != ["application/json"]:
        raise _invalid_request()
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
            if parsed_length < 0 or parsed_length > MAX_LIFECYCLE_REQUEST_BYTES:
                raise _invalid_request()
        except ValueError:
            raise _invalid_request() from None
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_LIFECYCLE_REQUEST_BYTES:
            raise _invalid_request()
    if not body:
        raise _invalid_request()
    return model.model_validate_json(bytes(body))


def _invalid_request() -> AllocationError:
    return AllocationError(
        "invalid_request",
        "request does not match the allocation lifecycle contract",
        retryable=False,
        status_code=422,
    )


def create_server_config(
    settings: Settings,
    application: ASGIApp,
    tls_context: ssl.SSLContext,
) -> uvicorn.Config:
    return uvicorn.Config(
        application,
        host=settings.host,
        port=settings.port,
        http=VerifiedMTLSH11Protocol,
        ws="none",
        lifespan="off",
        access_log=False,
        log_config=None,
        proxy_headers=False,
        timeout_graceful_shutdown=max(1, math.ceil(settings.shutdown_grace_seconds)),
        ssl_context_factory=lambda _config, _default: tls_context,
    )


class RuntimeServer(uvicorn.Server):
    """Uvicorn server whose signals are coordinated by the process runner."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield
