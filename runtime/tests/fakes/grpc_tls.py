"""Bounded TLS/Unix-socket prototype for V490-001; not a production transport.

The actual TLS connection is verified before any byte reaches the gRPC backend.
Only authenticated Control Plane traffic crosses the private local socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import ssl
from collections.abc import Awaitable, Callable
from pathlib import Path

from contractor_runtime.mtls import verify_control_plane_peer

Pair = tuple[asyncio.StreamReader, asyncio.StreamWriter]


class TLSAdapter:
    def __init__(self, *, maximum: int = 8) -> None:
        self.maximum = maximum
        self.server: asyncio.Server | None = None
        self.tasks: set[asyncio.Task] = set()
        self.rejections = 0
        self.verified = 0
        self.closed = False

    def accept(self, connect: Callable[[asyncio.StreamWriter], Awaitable[Pair]]):
        def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if self.closed or len(self.tasks) >= self.maximum:
                self.rejections += 1
                writer.close()
                return
            task = asyncio.create_task(self.forward(reader, writer, connect))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        return accepted

    async def inbound(self, path: Path, context: ssl.SSLContext) -> int:
        context.set_alpn_protocols(["h2"])

        async def connect(writer: asyncio.StreamWriter) -> Pair:
            self.verify(writer)
            return await asyncio.open_unix_connection(str(path), limit=65536)

        self.server = await asyncio.start_server(
            self.accept(connect),
            "127.0.0.1",
            0,
            ssl=context,
            ssl_handshake_timeout=3,
            ssl_shutdown_timeout=0.5,
            limit=65536,
            backlog=self.maximum,
        )
        return self.server.sockets[0].getsockname()[1]

    async def outbound(self, path: Path, port: int, context: ssl.SSLContext) -> None:
        if path.parent.stat().st_mode & 0o077:
            raise ValueError("TLS adapter requires a private Unix-socket directory")
        context.set_alpn_protocols(["h2"])

        async def connect(_writer: asyncio.StreamWriter) -> Pair:
            pair = await asyncio.open_connection(
                "localhost",
                port,
                ssl=context,
                server_hostname="localhost",
                ssl_handshake_timeout=3,
                ssl_shutdown_timeout=0.5,
                limit=65536,
            )
            try:
                self.verify(pair[1])
            except BaseException:
                pair[1].close()
                await self.close_writer(pair[1])
                raise
            return pair

        self.server = await asyncio.start_unix_server(
            self.accept(connect), path=str(path), limit=65536
        )
        os.chmod(path, 0o600)

    def verify(self, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("ssl_object")
        if peer is None or peer.selected_alpn_protocol() != "h2":
            raise ssl.SSLError("gRPC requires a verified TLS connection with h2")
        verify_control_plane_peer(peer)
        self.verified += 1

    async def forward(self, reader, writer, connect) -> None:
        remote = None
        pumps: list[asyncio.Task] = []
        try:
            async with asyncio.timeout(3):
                other_reader, remote = await connect(writer)
            pumps = [
                asyncio.create_task(self.pump(reader, remote)),
                asyncio.create_task(self.pump(other_reader, writer)),
            ]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (OSError, TimeoutError):
            self.rejections += 1
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            await self.close_writer(writer)
            if remote is not None:
                await self.close_writer(remote)

    @staticmethod
    async def pump(reader, writer) -> None:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()

    @staticmethod
    async def close_writer(writer) -> None:
        writer.close()
        with contextlib.suppress(OSError, TimeoutError):
            async with asyncio.timeout(0.5):
                await writer.wait_closed()

    async def close(self) -> None:
        self.closed = True
        if self.server is not None:
            self.server.close()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.difference_update(tasks)
        if self.server is not None:
            await self.server.wait_closed()
