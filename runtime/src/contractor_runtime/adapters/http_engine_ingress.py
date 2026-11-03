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

from contractor_runtime.adapters.http_engine_lifecycle import OwnedHandler
from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.contracts.engagement import canonical_url
from contractor_runtime.core.http_limits import MAX_HEADERS, request_header_max_bytes
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.threads import to_thread_until_done

HOP_HEADERS = {
    b"connection",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


def end_to_end(headers):
    hop = set(HOP_HEADERS)
    for name, value in headers:
        if name.lower() == b"connection":
            hop.update(item.strip().lower() for item in value.split(b","))
    return [(k, v) for k, v in headers if k.lower() not in hop]


class IngressHttp1(_http1.Http1Server):
    def read_headers(self, event):
        if isinstance(event, events.DataReceived):
            buffered = bytes(self.buf)
            end = buffered.find(b"\r\n\r\n")
            size = len(buffered) if end < 0 else end + 4
            if size > request_header_max_bytes() or (
                end >= 0
                and (
                    buffered[:end].count(b"\r\n") > MAX_HEADERS
                    or not buffered.split(b"\r\n", 1)[0].endswith(b" HTTP/1.1")
                )
            ):
                yield commands.CloseConnection(self.conn)
                self.state = self.done
                return
        yield from super().read_headers(event)


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
        self.body_sizes = {}

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
        self.hooks, self.writers = set(), set()
        self.received = self.authenticated = False
        self.pending = set()
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
            url = request.url
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
            headers = [
                (k, v) for k, v in end_to_end(response.headers) if k.lower() != b"content-length"
            ]
            if request.method == "HEAD" or response.status == 304:
                headers += [(k, v) for k, v in response.headers if k.lower() == b"content-length"]
            elif response.status != 204:
                headers.append((b"content-length", str(len(response.body)).encode()))
            # Frontend messages are independent scanner IPC calls; target pools
            # retain HTTP/1.1 keepalive and HTTP/2 multiplexing.
            headers.append((b"connection", b"close"))
            flow.response = http.Response.make(response.status, response.body, headers)
        elif name == "response":
            self.pending.discard(flow.id)
            self.completed += 1
        elif name == "error":
            self.pending.discard(flow.id)
            self.bridge.error_code = self.bridge.error_code or "scan_request_failed"
