"""Verified private TLS listener for native RPCs and operational HTTP probes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import re
import ssl
import tempfile
import uuid
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from contractor_runtime._grpc_tls import TLSForwarder
from contractor_runtime._https import PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS
from contractor_runtime.a2a_server import MAX_A2A_REQUEST_BYTES, register_a2a_gateway
from contractor_runtime.allocation import AllocationService
from contractor_runtime.contracts.grpc_wire import runtime_pb2_grpc
from contractor_runtime.contracts.grpc_wire.limits import (
    MAX_CONCURRENT_RPCS,
    MAX_STATE_MESSAGE_BYTES,
    METADATA_OPTIONS,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.grpc_server import LifecycleHandler
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState

logger = logging.getLogger(__name__)

VERIFIED_PEER_EXTENSION = "contractor.mtls.peer_certificate"
REQUEST_ID_HEADER = b"x-request-id"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


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
            await send(message)

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

    application: ASGIApp = Starlette(
        routes=[
            Route("/healthz", health, methods=["GET"]),
            Route("/readyz", readiness, methods=["GET"]),
        ]
    )
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


@dataclass(frozen=True)
class ServerConfig:
    settings: Settings
    application: ASGIApp
    tls_context: ssl.SSLContext
    state: RuntimeState
    allocation_service: AllocationService | None
    timeout_keep_alive: float = PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS


def create_server_config(
    settings, application, tls_context, *, state=None, allocation_service=None
):
    return ServerConfig(
        settings, application, tls_context, state or RuntimeState(), allocation_service
    )


class _LocalHTTPServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


class RuntimeServer:
    """Own native grpcio and temporary HTTP families behind one TLS front door."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.started = False
        self._exit = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._should_exit = False
        self._force_exit = False

    @property
    def should_exit(self):
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value):
        self._should_exit = value
        if value:
            self._signal_exit()

    @property
    def force_exit(self):
        return self._force_exit

    @force_exit.setter
    def force_exit(self, value):
        self._force_exit = value
        if value:
            self._signal_exit()

    def _signal_exit(self):
        if self._loop is None:
            self._exit.set()
        else:
            self._loop.call_soon_threadsafe(self._exit.set)

    async def serve(self):
        self._loop = asyncio.get_running_loop()
        config = self.config
        settings = config.settings
        with tempfile.TemporaryDirectory(prefix="ct-grpc-", dir="/tmp") as directory:
            grpc_path = Path(directory) / "runtime.sock"
            http_path = Path(directory) / "http.sock"
            server = grpc.aio.server(
                maximum_concurrent_rpcs=MAX_CONCURRENT_RPCS,
                options=(
                    ("grpc.max_receive_message_length", MAX_A2A_REQUEST_BYTES),
                    ("grpc.max_send_message_length", MAX_STATE_MESSAGE_BYTES),
                    *METADATA_OPTIONS,
                    ("grpc.max_concurrent_streams", MAX_CONCURRENT_RPCS),
                    ("grpc.max_connection_idle_ms", 60_000),
                    ("grpc.server_handshake_timeout_ms", 5_000),
                ),
            )
            if config.allocation_service is not None:
                runtime_pb2_grpc.add_RuntimeLifecycleServicer_to_server(
                    LifecycleHandler(config.state, config.allocation_service), server
                )
                register_a2a_gateway(server, config.allocation_service)
            server.add_insecure_port(f"unix:{grpc_path}")

            # Only this process can enter the local backend. The TLS front door
            # verified the actual peer before opening its Unix connection.
            async def verified_backend(scope, receive, send):
                copied = dict(scope)
                copied["extensions"] = {
                    **scope.get("extensions", {}),
                    VERIFIED_PEER_EXTENSION: True,
                }
                await config.application(copied, receive, send)

            http = _LocalHTTPServer(
                uvicorn.Config(
                    verified_backend,
                    uds=str(http_path),
                    http="h11",
                    ws="none",
                    lifespan="off",
                    access_log=False,
                    log_config=None,
                    proxy_headers=False,
                    timeout_keep_alive=config.timeout_keep_alive,
                    timeout_graceful_shutdown=max(1, math.ceil(settings.shutdown_grace_seconds)),
                )
            )
            forwarder = TLSForwarder()
            http_task = None
            watchers = []
            try:
                await server.start()
                os.chmod(grpc_path, 0o600)
                http_task = asyncio.create_task(http.serve(), name="runtime-local-http")
                while not http.started:
                    if http_task.done():
                        await http_task
                        raise RuntimeError("private HTTP backend stopped before startup")
                    await asyncio.sleep(0)
                os.chmod(http_path, 0o600)
                await forwarder.listen(
                    settings.host,
                    settings.port,
                    config.tls_context,
                    {"h2": grpc_path, "http/1.1": http_path},
                )
                self.started = True
                watchers = [
                    asyncio.create_task(self._exit.wait()),
                    asyncio.create_task(server.wait_for_termination()),
                    asyncio.create_task(forwarder.serve_forever()),
                ]
                done, _ = await asyncio.wait(
                    [*watchers, http_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    await task
                if not self._exit.is_set():
                    raise RuntimeError("private Runtime listener stopped unexpectedly")
            finally:
                self.started = False

                async def cleanup():
                    try:
                        await forwarder.close()
                    finally:
                        try:
                            await server.stop(0)
                            await server.wait_for_termination()
                        finally:
                            http.should_exit = True
                            try:
                                if http_task is not None:
                                    await http_task
                            finally:
                                for task in watchers:
                                    task.cancel()
                                await asyncio.gather(*watchers, return_exceptions=True)

                owner = asyncio.create_task(cleanup(), name="runtime-listener-close")
                interrupted = await join_owned_task(owner)
                owner.result()
                if interrupted:
                    raise asyncio.CancelledError
