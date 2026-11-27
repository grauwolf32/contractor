"""Verified TLS boundary for native gRPC on process-private Unix sockets.

This module forwards bounded byte streams; grpcio owns HTTP/2 and protobuf.
No network application bytes cross the boundary until the actual TLS peer
passes the deployment CA, endpoint (outgoing) and Control Plane role checks.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import ssl
from collections.abc import Mapping
from pathlib import Path

from contractor_runtime.contracts.base import parse_private_rpc_origin
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.mtls import verify_control_plane_peer
from contractor_runtime.threads import consume_task_exception

CONNECTION_LIMIT = 8
BUFFER_BYTES = 64 << 10
HANDSHAKE_SECONDS = 5.0
CLOSE_SECONDS = 0.5


class TLSForwarder:
    """Own one bounded listener and all of its forwarding/handshake tasks."""

    def __init__(self) -> None:
        self._listener: socket.socket | None = None
        self._server: asyncio.Server | None = None
        self._accept_task: asyncio.Task | None = None
        self._connections: set[asyncio.Task] = set()
        self._closed = False
        self.peer_rejected = False
        self._close_task: asyncio.Task | None = None

    async def listen(
        self,
        host: str,
        port: int,
        context: ssl.SSLContext,
        backends: Mapping[str, Path],
    ) -> int:
        """Accept and verify peers before selecting their local ALPN backend.

        During the operation-family migration the HTTP/1.1 backend serves only
        operations not yet migrated. Each migrated operation has one gRPC route.
        Admission happens before the TLS handshake, including unauthenticated
        and stalled clients, rather than just bounding completed handshakes.
        """
        if self._closed or self._listener is not None or not backends or "h2" not in backends:
            raise RuntimeError("invalid TLS listener lifecycle")
        for path in backends.values():
            _private_directory(path.parent)
        context.set_alpn_protocols(list(backends))
        family = socket.AF_INET6 if ":" in host else socket.AF_INET

        addresses = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=family, type=socket.SOCK_STREAM
        )
        address = addresses[0][4]
        # Resolve first, before creating a resource: cancelling a thread which
        # returns an already-bound socket would otherwise abandon that socket.
        self._listener = socket.create_server(address[:2], family=family, backlog=CONNECTION_LIMIT)
        self._listener.setblocking(False)
        self._accept_task = asyncio.create_task(
            self._accept(context, dict(backends)), name="grpc-tls-accept"
        )
        return self._listener.getsockname()[1]

    async def connect(self, path: Path, origin: str, context: ssl.SSLContext) -> None:
        """Expose a private socket forwarding only to one verified Server origin."""
        if self._closed or self._server is not None:
            raise RuntimeError("invalid TLS connector lifecycle")
        _private_directory(path.parent)
        target = parse_private_rpc_origin("Control Plane endpoint", origin)
        port = target.port or 443
        context.set_alpn_protocols(["h2"])

        def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if self._closed or len(self._connections) >= CONNECTION_LIMIT:
                writer.close()
                return
            self._own(self._outgoing(reader, writer, target.hostname, port, context))

        self._server = await asyncio.start_unix_server(
            accept, path=str(path), limit=BUFFER_BYTES, backlog=CONNECTION_LIMIT
        )
        os.chmod(path, 0o600)

    def _own(self, coroutine) -> None:
        task = asyncio.create_task(coroutine, name="grpc-tls-connection")
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)
        task.add_done_callback(consume_task_exception)

    async def _accept(self, context: ssl.SSLContext, backends: Mapping[str, Path]) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            peer, _ = await loop.sock_accept(self._listener)
            if self._closed or len(self._connections) >= CONNECTION_LIMIT:
                peer.close()
                continue
            self._own(self._incoming(peer, context, backends))

    async def _incoming(
        self, peer: socket.socket, context: ssl.SSLContext, backends: Mapping[str, Path]
    ) -> None:
        writer = None
        backend = None
        try:
            loop = asyncio.get_running_loop()
            reader = asyncio.StreamReader(limit=BUFFER_BYTES)
            protocol = asyncio.StreamReaderProtocol(reader)
            transport, _ = await loop.connect_accepted_socket(
                lambda: protocol,
                peer,
                ssl=context,
                ssl_handshake_timeout=HANDSHAKE_SECONDS,
                ssl_shutdown_timeout=CLOSE_SECONDS,
            )
            writer = asyncio.StreamWriter(transport, protocol, reader, loop)
            negotiated = _verify(writer)
            # HTTP/1.1 clients historically need not offer ALPN. This branch
            # disappears with the final unmigrated HTTP operation family.
            selected = negotiated or "http/1.1"
            if selected not in backends:
                raise ssl.SSLError("unsupported private transport")
            async with asyncio.timeout(HANDSHAKE_SECONDS):
                other, backend = await asyncio.open_unix_connection(
                    str(backends[selected]), limit=BUFFER_BYTES
                )
            await _forward(reader, writer, other, backend)
        except (OSError, TimeoutError):
            # No raw peer, certificate, payload or exception text is logged.
            pass
        finally:
            if writer is None:
                peer.close()
            else:
                await _close_writer(writer)
            if backend is not None:
                await _close_writer(backend)

    async def _outgoing(self, reader, writer, host, port, context) -> None:
        remote = None
        try:
            async with asyncio.timeout(HANDSHAKE_SECONDS):
                other, remote = await asyncio.open_connection(
                    host,
                    port,
                    ssl=context,
                    server_hostname=host,
                    ssl_handshake_timeout=HANDSHAKE_SECONDS,
                    ssl_shutdown_timeout=CLOSE_SECONDS,
                    limit=BUFFER_BYTES,
                )
                if _verify(remote) != "h2":
                    raise ssl.SSLError("gRPC requires negotiated h2")
            await _forward(reader, writer, other, remote)
        except ssl.SSLCertVerificationError:
            self.peer_rejected = True
        except (OSError, TimeoutError):
            pass
        finally:
            await _close_writer(writer)
            if remote is not None:
                await _close_writer(remote)

    async def serve_forever(self) -> None:
        if self._accept_task is None:
            raise RuntimeError("TLS listener has not started")
        await asyncio.shield(self._accept_task)

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="grpc-tls-close")
        interrupted = await join_owned_task(self._close_task)
        self._close_task.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        self._closed = True
        if self._listener is not None:
            self._listener.close()
        if self._server is not None:
            self._server.close()
        if self._accept_task is not None:
            self._accept_task.cancel()
            await asyncio.gather(self._accept_task, return_exceptions=True)
        tasks = tuple(self._connections)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._connections.difference_update(tasks)
        if self._server is not None:
            await self._server.wait_closed()


def _private_directory(path: Path) -> None:
    info = path.stat()
    if not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("gRPC socket directory must be owned by this user and private")


def _verify(writer: asyncio.StreamWriter) -> str | None:
    peer = writer.get_extra_info("ssl_object")
    if not isinstance(peer, ssl.SSLObject | ssl.SSLSocket):
        raise ssl.SSLError("private connection requires TLS")
    verify_control_plane_peer(peer)
    return peer.selected_alpn_protocol()


async def _forward(reader, writer, other, remote) -> None:
    async def pump(source, destination):
        while data := await source.read(BUFFER_BYTES):
            destination.write(data)
            await destination.drain()

    pumps = [asyncio.create_task(pump(reader, remote)), asyncio.create_task(pump(other, writer))]
    try:
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError, TimeoutError):
        async with asyncio.timeout(CLOSE_SECONDS):
            await writer.wait_closed()
