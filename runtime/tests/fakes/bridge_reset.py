"""Owned TCP peers for bridge reset regressions, without mocked network I/O."""

import asyncio
import socket
import struct

import h2.config
import h2.connection
import h2.events

from fakes.engagement_socket import private_interface
from fakes.native_http import NativeTarget


def reset_socket(writer):
    writer.get_extra_info("socket").setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    writer.transport.abort()


class EarlyDenial:
    """Close normally after a 404 while the POST body is still in the kernel."""

    async def __aenter__(self):
        self.address = private_interface()
        self.listener = socket.socket()
        self.listener.settimeout(5)
        self.listener.bind((self.address, 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.origin = f"http://target.example:{self.port}"
        self.unread_body = False
        self.task = asyncio.create_task(asyncio.to_thread(self.reject))
        return self

    def reject(self):
        connection, _ = self.listener.accept()
        with connection:
            connection.settimeout(5)
            headers = b""
            while not headers.endswith(b"\r\n\r\n"):
                byte = connection.recv(1)
                assert byte and len(headers) < 65536
                headers += byte
            self.unread_body = bool(connection.recv(1, socket.MSG_PEEK))
            connection.sendall(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\nConnection: close\r\n"
                b"Set-Cookie: previous=allocation-only; Path=/\r\n\r\n{}"
            )

    async def __aexit__(self, *_):
        try:
            await self.task
        finally:
            self.listener.close()


class ResetTarget(NativeTarget):
    """Reset only after the test observes the complete response or its headers."""

    def __init__(self, *, ssl_context=None, partial=False):
        super().__init__(ssl_context=ssl_context)
        self.partial = partial
        self.reset = asyncio.Event()

    async def http1(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        self.requests.append(head)
        writer.write(
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 4\r\n\r\n"
            + (b"no" if self.partial else b"nope")
        )
        await writer.drain()
        await self.reset.wait()
        reset_socket(writer)

    async def http2(self, reader, writer):
        connection = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        connection.initiate_connection()
        writer.write(connection.data_to_send())
        while data := await reader.read(65536):
            for event in connection.receive_data(data):
                if isinstance(event, h2.events.RequestReceived):
                    self.requests.append(event.headers)
                    connection.send_headers(
                        event.stream_id, [(b":status", b"404"), (b"content-length", b"4")]
                    )
                    connection.send_data(
                        event.stream_id,
                        b"no" if self.partial else b"nope",
                        end_stream=not self.partial,
                    )
                    writer.write(connection.data_to_send())
                    await writer.drain()
                    await self.reset.wait()
                    reset_socket(writer)
                    return
            writer.write(connection.data_to_send())
            await writer.drain()
