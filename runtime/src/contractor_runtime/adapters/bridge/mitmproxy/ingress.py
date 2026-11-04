"""Native scanner ingress. Target exchanges always delegate to HTTPTransport."""

import asyncio
import hmac
import logging
import time
from urllib.parse import urlsplit

from mitmproxy import http
from mitmproxy.proxy import commands, events, layers, server
from mitmproxy.proxy.layers import http as native_http
from mitmproxy.proxy.layers.http import _http1, _http2
from mitmproxy.proxy.mode_specs import ProxyMode

from contractor_runtime.adapters.bridge.mitmproxy.framing import BoundedHttp1Body
from contractor_runtime.adapters.bridge.mitmproxy.lifecycle import OwnedHandler
from contractor_runtime.adapters.bridge.mitmproxy.messages import (
    end_to_end,
    native_url,
    request_target,
    response_message,
)
from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.contracts.engagement import canonical_url
from contractor_runtime.core.http_limits import MAX_HEADERS, request_header_max_bytes
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.threads import to_thread_until_done


class IngressHttp1(BoundedHttp1Body, _http1.Http1Server):
    header_bytes = 0

    def framing_budget(self):
        return request_header_max_bytes(), self.header_bytes, len(self.request.headers.fields)

    def read_headers(self, event):
        if isinstance(event, events.DataReceived):
            buffered = bytes(self.buf)
            end = buffered.find(b"\r\n\r\n")
            size = len(buffered) if end < 0 else end + 4
            self.header_bytes = size
            if size > request_header_max_bytes() or (
                end >= 0
                and (
                    buffered[:end].count(b"\r\n") > MAX_HEADERS
                    or not buffered.split(b"\r\n", 1)[0].endswith(b" HTTP/1.1")
                )
            ):
                yield commands.CloseConnection(self.conn)
                self.stop_input()
                return
        # Native absolute-form parsing uses urlparse, which erases a trailing '?'.
        # Restore only the exact target path after its normal syntax validation.
        target = bytes(self.buf).split(b"\r\n", 1)[0].split(b" ")
        for command in super().read_headers(event):
            if (
                isinstance(command, native_http.ReceiveHttp)
                and isinstance(command.event, native_http.RequestHeaders)
                and len(target) == 3
                and target[1].startswith((b"http://", b"https://"))
            ):
                command.event.request.path = request_target(target[1].decode("ascii")).decode(
                    "ascii"
                )
            yield command


class IngressHttp2(_http2.Http2Server):
    def __init__(self, context):
        super().__init__(context)
        self.h2_conn.decoder.max_header_list_size = request_header_max_bytes() + MAX_HEADERS * 32


class IngressHttp(native_http.HttpLayer):
    def __init__(self, context, mode, handler):
        super().__init__(context, mode)
        self.handler = handler
        parser = IngressHttp2 if context.client.alpn == b"h2" else IngressHttp1
        self.connections[context.client] = parser(context.fork())
        self.body_sizes: dict[int, int] = {}

    def event_to_child(self, child, event):
        if isinstance(event, native_http.RequestHeaders):
            self.body_sizes[event.stream_id] = 0
        if isinstance(event, native_http.RequestData):
            size = self.body_sizes.get(event.stream_id, 0) + len(event.data)
            self.body_sizes[event.stream_id] = size
            if size > get_shared_config().http_request_body_max_bytes:
                self.handler.bridge.error_code = "scan_request_failed"
                event = native_http.RequestProtocolError(
                    event.stream_id, "request bound", native_http.ErrorCode.REQUEST_TOO_LARGE
                )
        if isinstance(event, (native_http.RequestEndOfMessage, native_http.RequestProtocolError)):
            self.body_sizes.pop(event.stream_id, None)
        yield from super().event_to_child(child, event)


class Ingress(OwnedHandler, server.LiveConnectionHandler):
    def __init__(self, bridge, reader, writer, engine):
        self.bridge, self.engine = bridge, engine
        super().__init__(reader, writer, engine.options, ProxyMode.parse("regular"))
        self.layer = layers.modes.HttpProxy(self.layer.context)
        self.hooks: set[asyncio.Task[None]] = set()
        self.writers: set[asyncio.StreamWriter] = set()
        self.received = self.authenticated = False
        self.pending: set[str] = set()
        self.completed = 0

    def log(self, message, level=logging.INFO, exc_info=None):
        if exc_info:
            self.bridge._exception = RuntimeError("native HTTP ingress dispatch failed")
            self.bridge.error_code = "scan_request_failed"
            for io in self.transports.values():
                if io.handler is not None:
                    io.handler.cancel()

    async def server_event(self, event):
        if isinstance(event, events.DataReceived) and event.data:
            self.received = True
        await super().server_event(event)

    async def open_connection(self, command):
        # The frontend is local IPC, never an independent target transport.
        self.bridge.error_code = "scan_request_failed"
        command.connection.error = "target connection unavailable to ingress"
        await self.server_event(events.OpenConnectionCompleted(command, command.connection.error))

    async def handle_hook(self, hook):
        data = None
        try:
            data = hook.args()[0]
            if hook.name == "next_layer":
                self.engine.next_layer.next_layer(data)
                if isinstance(data.layer, native_http.HttpLayer):
                    context, mode = data.layer.context, data.layer.mode
                    context.layers.remove(data.layer)
                    data.layer = IngressHttp(context, mode, self)
            elif hook.name in {"tls_clienthello", "tls_start_client"}:
                if (
                    hook.name == "tls_clienthello"
                    and self.authenticated
                    and b"h2" in data.client_hello.alpn_protocols
                ):
                    data.context.client.alpn = b"h2"
                await to_thread_until_done(
                    getattr(self.engine.tls, hook.name), data, name="http-ingress-tls"
                )
            elif isinstance(data, http.HTTPFlow):
                await self.flow_hook(hook.name, data)
        except asyncio.CancelledError:
            raise
        except (HTTPTransportError, TimeoutError, OSError, ValueError):
            self.bridge.error_code = self.bridge.error_code or "scan_request_failed"
            if isinstance(data, http.HTTPFlow):
                data.kill()
        except Exception:
            self.bridge._exception = RuntimeError("native HTTP ingress callback failed")
            self.bridge.error_code = "scan_request_failed"
            if isinstance(data, http.HTTPFlow):
                data.kill()

    def authenticate(self, flow):
        if self.authenticated:
            return True
        values = flow.request.headers.get_all("proxy-authorization")
        if len(values) != 1 or not hmac.compare_digest(
            values[0].encode(), self.bridge._authorization
        ):
            flow.kill()
            return False
        self.authenticated = True
        return True

    async def flow_hook(self, name, flow):
        request = flow.request
        if name in {"http_connect", "requestheaders"}:
            if not self.authenticate(flow):
                return
            if name == "http_connect":
                canonical_url("https://" + request.authority + "/")
                return
            headers = request.headers.fields
            if (
                request.http_version not in {"HTTP/1.1", "HTTP/2.0"}
                or len(headers) > MAX_HEADERS
                or sum(len(k) + len(v) + 4 for k, v in headers) > request_header_max_bytes()
                or "upgrade" in request.headers
                or request.method == "CONNECT"
            ):
                raise ValueError("unsupported HTTP request")
            length = request.headers.get("content-length")
            if length is not None and int(length) > get_shared_config().http_request_body_max_bytes:
                raise ValueError("request body bound")
            self.pending.add(flow.id)
        elif name == "request":
            if not self.authenticated:
                flow.kill()
                return
            self.bridge.requests += 1
            url = native_url(request)
            if canonical_url(url) != url:
                raise ValueError("noncanonical target")
            parsed = urlsplit(url)
            hosts = request.headers.get_all("host")
            if request.http_version == "HTTP/2.0" and not hosts:
                hosts = [request.authority]
            if len(hosts) != 1 or canonical_url(parsed.scheme + "://" + hosts[0] + "/") != (
                parsed.scheme + "://" + parsed.netloc + "/"
            ):
                raise ValueError("Host differs from target authority")
            headers = [
                (k, v)
                for k, v in end_to_end(request.headers.fields)
                if k.lower() not in {b"host", b"content-length", b"cookie"}
            ]
            outcome, response = await self.bridge._transport.request(
                url,
                credential_id=self.bridge._credential_id,
                method=request.method,
                headers=headers,
                body=request.raw_content or b"",
                follow_redirects=False,
                timeout=max(0.001, min(30, self.bridge._deadline - time.monotonic())),
            )
            if outcome.kind != "completed" or response is None:
                self.bridge.error_code = (
                    "scan_target_denied" if outcome.kind == "denied" else "scan_request_failed"
                )
                flow.kill()
                return
            if response.truncated:
                self.bridge.error_code = self.bridge.error_code or "scan_request_failed"
            # IPC messages close independently; target pools retain keepalive/H2.
            flow.response = response_message(response, request.method)
        elif name == "response":
            self.pending.discard(flow.id)
            self.completed += 1
        elif name == "error":
            self.pending.discard(flow.id)
            self.bridge.error_code = self.bridge.error_code or "scan_request_failed"
