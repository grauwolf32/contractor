"""Local first-write boundary; TLS negotiation never authorizes a retry after a write."""

from __future__ import annotations

import httpcore

from contractor_runtime.adapters.http_stream import RequestBoundary
from contractor_runtime.core.http_transport import PolicyNetworkBackend

MAX_RESPONSE_HEADER_BYTES = 32 << 10


class ControlledBackend(PolicyNetworkBackend):
    def __init__(self, policy, boundary, backend=None):
        self._boundary = boundary
        super().__init__(
            policy,
            backend,
            wrap_stream=lambda stream: ControlledStream(stream, boundary),
        )

    async def connect_tcp(self, *args, **kwargs):
        self._boundary.require_time()
        return await super().connect_tcp(*args, **kwargs)


class ControlledStream(httpcore.AsyncNetworkStream):
    def __init__(self, stream, boundary: RequestBoundary):
        self._stream, self._boundary = stream, boundary
        self._header = bytearray()
        self._headers_done = False
        self._header_total = 0
        self._header_count = 0
        self.closed = False
        boundary.transport = self

    async def read(self, max_bytes, timeout=None):
        self._boundary.require_time()
        limit = MAX_RESPONSE_HEADER_BYTES + 1
        if not self._headers_done:
            limit -= self._header_total + len(self._header)
        data = await self._stream.read(min(max_bytes, limit), timeout)
        if not self._headers_done:
            self._header.extend(data)
            while True:
                end = self._header.find(b"\r\n\r\n")
                if end < 0:
                    if self._header_total + len(self._header) > MAX_RESPONSE_HEADER_BYTES:
                        raise httpcore.RemoteProtocolError("response headers exceed limit")
                    break
                end += 4
                self._header_total += end
                if self._header_total > MAX_RESPONSE_HEADER_BYTES:
                    raise httpcore.RemoteProtocolError("response headers exceed limit")
                header = bytes(self._header[:end])
                del self._header[:end]
                self._header_count += header.count(b"\r\n") - 2
                if self._header_count > 64:
                    raise httpcore.RemoteProtocolError("response header count exceeds limit")
                status = header.split(b" ", 2)[1:2]
                if not status or not status[0].startswith(b"1") or status[0] == b"101":
                    self._headers_done = True
                    self._header.clear()
                    break
        return data

    async def write(self, buffer, timeout=None):
        if not buffer:
            return
        self._boundary.require_time()
        if self._boundary.request_started:
            await self._stream.write(buffer, timeout)
        else:
            self._boundary.require_time()
            self._boundary.request_started = True
            await self._stream.write(buffer, timeout)

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._boundary.require_time()
        stream = await self._stream.start_tls(ssl_context, server_hostname, timeout)
        return ControlledStream(stream, self._boundary)

    async def aclose(self):
        if not self.closed:
            await self._stream.aclose()
            self.closed = True

    def get_extra_info(self, info):
        return self._stream.get_extra_info(info)
