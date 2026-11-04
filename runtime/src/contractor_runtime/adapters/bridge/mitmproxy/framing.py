"""Bound native HTTP/1.1 framing without replacing h11's parser."""

import h11
from h11._receivebuffer import ReceiveBuffer
from mitmproxy.proxy import events
from mitmproxy.proxy.layers.http import RequestProtocolError, ResponseProtocolError
from mitmproxy.proxy.layers.http._base import ReceiveHttp

from contractor_runtime.core.http_limits import MAX_HEADERS


class _FramingBuffer:
    def __init__(self, inner, line_limit: int, trailer_limit: int):
        self.inner = inner
        self.line_limit, self.trailer_limit = line_limit, trailer_limit

    def __len__(self):
        return len(self.inner)

    def maybe_extract_at_most(self, count):
        return self.inner.maybe_extract_at_most(count)

    def maybe_extract_next_line(self):
        before = len(self.inner)
        line = self.inner.maybe_extract_next_line()
        size = len(self.inner) if line is None else before - len(self.inner)
        if size > self.line_limit:
            raise h11.RemoteProtocolError("chunk framing bound")
        return line

    def maybe_extract_lines(self):
        before = len(self.inner)
        lines = self.inner.maybe_extract_lines()
        size = len(self.inner) if lines is None else before - len(self.inner)
        if size > self.trailer_limit:
            raise h11.RemoteProtocolError("trailer bound")
        return lines


class _BodyReader:
    def __init__(self, inner, limit: int, header_bytes: int, header_count: int):
        self.inner, self.limit = inner, limit
        self.trailer_limit = limit - header_bytes
        self.trailer_count = MAX_HEADERS - header_count

    def __call__(self, buffer):
        event = self.inner(_FramingBuffer(buffer, self.limit, self.trailer_limit))
        if isinstance(event, h11.EndOfMessage):
            if len(event.headers) > self.trailer_count:
                raise h11.RemoteProtocolError("trailer count bound")
            # Private messages have no trailer field. Validate through h11,
            # then discard metadata instead of promoting it to initial headers
            # or hitting the native HTTP/1 layer's NotImplementedError.
            return h11.EndOfMessage()
        return event

    def read_eof(self):
        return self.inner.read_eof()


class BoundedHttp1Body:
    """Mixin shared by native HTTP/1.1 request and response readers."""

    def stop_input(self):
        self.state = self.done
        self.buf = ReceiveBuffer()

    def _handle_event(self, event):
        # Native CloseConnection can half-close a socket while already queued
        # reads still arrive. A rejected parser must never append those bytes.
        if isinstance(event, events.DataReceived) and self.state == self.done:
            return
        yield from super()._handle_event(event)

    def read_body(self, event):
        if not isinstance(self.body_reader, _BodyReader):
            limit, size, count = self.framing_budget()
            self.body_reader = _BodyReader(self.body_reader, limit, size, count)
        for command in super().read_body(event):
            if isinstance(command, ReceiveHttp) and isinstance(
                command.event, (RequestProtocolError, ResponseProtocolError)
            ):
                self.stop_input()
            yield command
