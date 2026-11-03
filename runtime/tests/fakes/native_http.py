"""Persistent HTTP/1.1 and multiplexed HTTP/2 targets on an owned RFC1918 socket."""

import asyncio
from contextlib import suppress

import h2.config
import h2.connection
import h2.events
import h2.settings

from fakes.engagement_socket import SocketTarget


class NativeTarget(SocketTarget):
    def __init__(self, *, ssl_context=None, max_streams=100, payload=b"ok", headers=()):
        super().__init__(ssl_context=ssl_context)
        self.payload, self.headers = payload, headers
        self.max_streams = max_streams
        self.trace = []
        self.held = asyncio.Event()
        self.release = asyncio.Event()
        self.responders = set()
        self.resets = []

    async def _observe(self, reader, writer):
        self.connections.append(bytearray())
        self.writers.add(writer)
        self.all_closed.clear()
        try:
            async with asyncio.timeout(15):
                tls = writer.get_extra_info("ssl_object")
                if tls and tls.selected_alpn_protocol() == "h2":
                    await self.http2(reader, writer)
                else:
                    await self.http1(reader, writer)
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            for task in tuple(self.responders):
                task.cancel()
            await asyncio.gather(*self.responders, return_exceptions=True)
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            self.writers.discard(writer)
            if not self.writers:
                self.all_closed.set()

    async def http1(self, reader, writer):
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            self.requests.append(head)
            method, path, protocol = head.split(b"\r\n", 1)[0].split(b" ")
            self.trace.append((method, path, protocol, head))
            assert len(self.trace) <= 64
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

    async def http2(self, reader, writer):
        connection = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        connection.local_settings.max_concurrent_streams = self.max_streams
        connection.local_settings.acknowledge()
        connection.initiate_connection()
        writer.write(connection.data_to_send())
        streams = {}

        async def respond(stream, headers):
            if headers[b":path"] == b"/hold":
                self.held.set()
                await self.release.wait()
            connection.send_headers(
                stream,
                [
                    (b":status", b"200"),
                    (b"content-length", str(len(self.payload)).encode()),
                    *self.headers,
                ],
            )
            connection.send_data(stream, self.payload, end_stream=True)
            writer.write(connection.data_to_send())
            await writer.drain()

        while data := await reader.read(65536):
            for event in connection.receive_data(data):
                if isinstance(event, h2.events.RequestReceived):
                    streams[event.stream_id] = dict(event.headers)
                elif isinstance(event, h2.events.DataReceived):
                    connection.acknowledge_received_data(
                        event.flow_controlled_length, event.stream_id
                    )
                elif isinstance(event, h2.events.StreamReset):
                    self.resets.append(event.stream_id)
                elif isinstance(event, h2.events.StreamEnded):
                    headers = streams.pop(event.stream_id)
                    self.trace.append((headers[b":method"], headers[b":path"], b"HTTP/2", headers))
                    assert len(self.trace) <= 64
                    task = asyncio.create_task(respond(event.stream_id, headers))
                    self.responders.add(task)
                    task.add_done_callback(self.responders.discard)
            writer.write(connection.data_to_send())
            await writer.drain()
