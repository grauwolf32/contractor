"""Pinned mitmproxy HTTP layers with Contractor admission and byte bounds.

Protocol parsing, framing, HPACK, flow control, TLS and network I/O stay in the
engine. This adapter binds native messages to the existing HTTP authority.
"""

from __future__ import annotations

import h2.events
from mitmproxy.connection import Server
from mitmproxy.net.http import http1
from mitmproxy.proxy import commands, events, layers, tunnel
from mitmproxy.proxy.layers import http as native_http
from mitmproxy.proxy.layers.http import _http1, _http2
from mitmproxy.proxy.layers.http._base import HttpConnection, ReceiveHttp
from mitmproxy.proxy.layers.http._events import ErrorCode, ResponseProtocolError

from contractor_runtime.adapters.bridge.mitmproxy.framing import BoundedHttp1Body
from contractor_runtime.core.http_limits import MAX_CONTROLLED_HEADER_BYTES, MAX_HEADERS

_REQUEST_MESSAGES = (
    native_http.RequestHeaders,
    native_http.RequestData,
    native_http.RequestTrailers,
    native_http.RequestEndOfMessage,
)


def _admitted_exchange(pool, event):
    exchange = pool.exchanges.get(event.stream_id)
    if exchange is None or exchange.closing:
        return None
    exchange.boundary.require_time()
    return exchange


class Submit(events.DataReceived):
    def __init__(self, connection, exchange):
        super().__init__(connection, b"")
        self.exchange = exchange


class Abort(events.Event):
    def __init__(self, exchange):
        self.exchange = exchange


class VirtualClient(HttpConnection):
    def _handle_event(self, event):
        if isinstance(event, Submit):
            exchange = event.exchange
            request = exchange.flow.request
            body = request.raw_content
            stream = exchange.stream_id
            yield ReceiveHttp(native_http.RequestHeaders(stream, request, not body, exchange.flow))
            if body:
                yield ReceiveHttp(native_http.RequestData(stream, body))
            yield ReceiveHttp(native_http.RequestEndOfMessage(stream))


class BoundedHttp1Client(BoundedHttp1Body, _http1.Http1Client):
    """Use the native parser; charge every informational and final header block."""

    header_bytes = 0
    header_count = 0

    def __init__(self, context, pool):
        super().__init__(context)
        self.pool = pool

    def framing_budget(self):
        return MAX_CONTROLLED_HEADER_BYTES, self.header_bytes, self.header_count

    def send(self, event):
        if isinstance(event, _REQUEST_MESSAGES) and _admitted_exchange(self.pool, event) is None:
            return
        if isinstance(event, native_http.RequestHeaders):
            self.header_bytes = self.header_count = 0
        yield from super().send(event)

    def read_headers(self, event):
        if not isinstance(event, events.DataReceived):
            yield from super().read_headers(event)
            return
        if not self.request or self.stream_id is None:
            yield from super().read_headers(event)
            return
        try:
            while True:
                before = len(self.buf)
                lines = self.buf.maybe_extract_lines()
                if lines is None:
                    if self.header_bytes + len(self.buf) > MAX_CONTROLLED_HEADER_BYTES:
                        raise ValueError("header bound")
                    return
                self.header_bytes += before - len(self.buf)
                self.header_count += len(lines) - 1
                if (
                    self.header_bytes > MAX_CONTROLLED_HEADER_BYTES
                    or self.header_count > MAX_HEADERS
                ):
                    raise ValueError("header bound")
                response = http1.read_response_head([bytes(line) for line in lines])
                if response.http_version != "HTTP/1.1":
                    raise ValueError("unsupported HTTP version")
                if 100 <= response.status_code < 200 and response.status_code != 101:
                    continue
                if response.status_code == 101:
                    raise ValueError("protocol upgrade unavailable")
                self.response = response
                size = http1.expected_http_body_size(self.request, response)
                yield ReceiveHttp(native_http.ResponseHeaders(self.stream_id, response, size == 0))
                self.body_reader = _http1.make_body_reader(size)
                self.state = self.read_body
                yield from self.state(event)
                return
        except ValueError:
            self.stop_input()
            yield commands.CloseConnection(self.conn)
            yield ReceiveHttp(
                ResponseProtocolError(
                    self.stream_id, "response_invalid", ErrorCode.GENERIC_SERVER_ERROR
                )
            )


class BoundedHttp2Client(_http2.Http2Client):
    def __init__(self, context, pool):
        super().__init__(context)
        self.pool = pool
        self.header_totals: dict[int, tuple[int, int]] = {}
        self.h2_conn.decoder.max_header_list_size = MAX_CONTROLLED_HEADER_BYTES + MAX_HEADERS * 32

    def _handle_event(self, event):
        if (
            isinstance(event, native_http.RequestProtocolError)
            and event.stream_id not in self.our_stream_id
        ):
            # Native queuing otherwise retains a cancelled RequestHeaders and
            # can send it when a later SETTINGS/WINDOW update arrives.
            self.stream_queue.pop(event.stream_id, None)
            return
        if isinstance(event, _REQUEST_MESSAGES) and _admitted_exchange(self.pool, event) is None:
            return
        yield from super()._handle_event(event)

    def handle_h2_event(self, event):
        if isinstance(
            event,
            (
                h2.events.ResponseReceived,
                h2.events.InformationalResponseReceived,
                h2.events.TrailersReceived,
            ),
        ):
            count, size = self.header_totals.get(event.stream_id, (0, 0))
            count += len([k for k, _ in event.headers if not k.startswith(b":")])
            size += sum(len(k) + len(v) + 4 for k, v in event.headers)
            self.header_totals[event.stream_id] = count, size
            if count > MAX_HEADERS or size > MAX_CONTROLLED_HEADER_BYTES:
                yield from self.protocol_error("response_invalid")
                return True
        return (yield from super().handle_h2_event(event))

    def forget(self, stream_id):
        self.stream_queue.pop(stream_id, None)
        ours = self.our_stream_id.pop(stream_id, None)
        if ours is not None:
            self.their_stream_id.pop(ours, None)
            self.streams.pop(ours, None)
            self.header_totals.pop(ours, None)


class NativeClient(native_http.HttpClient):
    def __init__(self, context, pool):
        super().__init__(context)
        self.pool = pool

    def _handle_event(self, event):
        error = None
        if not self.context.server.connected:
            error = yield commands.OpenConnection(self.context.server)
        if not error:
            parser = (
                BoundedHttp2Client(self.context, self.pool)
                if self.context.server.alpn == b"h2"
                else BoundedHttp1Client(self.context, self.pool)
            )
            self.pool.parsers[self.context.server] = parser
            self.child_layer = parser
            self._handle_event = parser.handle_event
            yield from self._handle_event(event)
        yield native_http.RegisterHttpConnection(self.context.server, error)


class Child:
    """Filter native messages without copying the engine's dispatch loop."""

    def __init__(self, inner, pool):
        self.inner, self.pool = inner, pool

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def handle_event(self, event):
        for command in self.inner.handle_event(event):
            if isinstance(command, commands.OpenConnection):
                exchange = self.pool.connecting.get(command.connection)
                if (
                    exchange is None
                    or exchange.closing
                    or self.pool.exchanges.get(exchange.stream_id) is not exchange
                ):
                    continue
                exchange.boundary.require_time()
            if isinstance(command, native_http.SendHttp) and isinstance(
                command.event, _REQUEST_MESSAGES
            ):
                exchange = _admitted_exchange(self.pool, command.event)
                if exchange is None:
                    continue
                if isinstance(command.event, native_http.RequestHeaders):
                    exchange.connection = command.connection
                    # Set before entering the encoder/write. A queued H2 send
                    # is conservatively possibly sent, never automatically replayed.
                    exchange.boundary.request_started = True
            if isinstance(command, ReceiveHttp) and isinstance(
                command.event, native_http.ResponseData
            ):
                exchange = self.pool.exchanges.get(command.event.stream_id)
                if exchange is not None and exchange.receive(command.event.data):
                    command = ReceiveHttp(
                        native_http.RequestProtocolError(
                            command.event.stream_id, "response truncated", ErrorCode.CANCEL
                        )
                    )
            yield command


class RoutedHttpLayer(native_http.HttpLayer):
    def __init__(self, context, pool):
        super().__init__(context, native_http.HTTPMode.regular)
        self.pool = pool
        self.connections[context.client] = VirtualClient(context.fork(), context.client)

    def _handle_event(self, event):
        if isinstance(event, Abort):
            stream = self.streams.get(event.exchange.stream_id)
            if stream is not None:
                yield from self.event_to_child(
                    stream,
                    native_http.RequestProtocolError(
                        event.exchange.stream_id, "exchange cancelled", ErrorCode.CANCEL
                    ),
                )
            return
        yield from super()._handle_event(event)

    def event_to_child(self, child, event):
        wrapped = child if isinstance(child, Child) else Child(child, self.pool)
        yield from super().event_to_child(wrapped, event)

    def get_connection(self, event, *, reuse=True):
        source = self.command_sources[event]
        exchange = self.pool.engine.flows.get(source.flow.id)
        if exchange is None or exchange.closing:
            self.command_sources.pop(event, None)
            return
        exchange.boundary.require_time()
        address = str(exchange.addresses[0])
        event.address = (address, self.pool.port)
        self.context.server.address = event.address
        self.context.server.sni = self.pool.host
        matches = event.connection_spec_matches

        def permitted(connection):
            return (
                matches(connection)
                and connection not in self.waiting_for_establishment
                and (connection.alpn == b"h2" or not self.pool.busy(connection))
            )

        event.connection_spec_matches = permitted
        if reuse and any(
            connection.connected and not connection.error and permitted(connection)
            for connection in self.connections
        ):
            yield from super().get_connection(event, reuse=True)
            return
        context = self.context.fork()
        context.server = Server(
            address=event.address,
            sni=self.pool.host,
            tls=event.tls,
            alpn_offers=(b"h2", b"http/1.1") if event.tls else (),
        )
        # Establishment belongs to the request before either TCP or a native
        # callback task can start, including cancellation before first execution.
        exchange.connection = context.server
        stack = tunnel.LayerStack()
        if event.tls:
            stack /= layers.ServerTLSLayer(context)
        stack /= NativeClient(context, self.pool)
        self.connections[context.server] = stack[0]
        self.waiting_for_establishment[context.server].append(event)
        self.pool.connecting[context.server] = exchange
        yield from self.event_to_child(stack[0], events.Start())
