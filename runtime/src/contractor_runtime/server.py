"""Private mTLS ASGI server and allocation lifecycle route shell."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import ssl
from collections.abc import Generator, Mapping
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.protocols.http.h11_impl import H11Protocol

from contractor_runtime.mtls import verify_control_plane_peer
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState

logger = logging.getLogger(__name__)

VERIFIED_PEER_EXTENSION = "contractor.mtls.peer_certificate"


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
            response = JSONResponse(
                {
                    "code": "mtls_required",
                    "message": "verified mTLS is required",
                    "retryable": False,
                },
                status_code=401,
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


def create_app(state: RuntimeState | None = None, *, require_verified_peer: bool = True) -> ASGIApp:
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

    async def lifecycle_not_implemented(_: Request) -> JSONResponse:
        await runtime_state.record_route_dispatch()
        return JSONResponse(
            {
                "code": "not_implemented",
                "message": "allocation lifecycle is added by MVP-010",
                "retryable": False,
            },
            status_code=501,
        )

    application: ASGIApp = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", readiness, methods=["GET"]),
            Route(
                "/private/v1/allocations/{allocation_id}/prepare",
                lifecycle_not_implemented,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/finalize",
                lifecycle_not_implemented,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/abort",
                lifecycle_not_implemented,
                methods=["POST"],
            ),
            Route(
                "/private/v1/allocations/{allocation_id}/release",
                lifecycle_not_implemented,
                methods=["POST"],
            ),
        ]
    )
    if require_verified_peer:
        application = VerifiedPeerMiddleware(application)
    return application


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
