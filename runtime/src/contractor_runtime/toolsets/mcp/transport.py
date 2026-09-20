"""Bounded, endpoint-only HTTPX transport underneath the official SDK."""

from __future__ import annotations

import re
import ssl
from collections.abc import AsyncIterator, Sequence
from urllib.parse import urlsplit

import httpx

from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts.toolset_settings import MCPToolsetSettings
from contractor_runtime.toolsets.mcp.errors import MCPError

MAX_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_DISCOVERY_BYTES = 8 * 1024 * 1024


class _BoundedStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, owner: MCPHTTPTransport, sse: bool) -> None:
        self._stream, self._owner, self._sse = stream, owner, sse

    async def __aiter__(self) -> AsyncIterator[bytes]:
        size = 0
        tail = b""
        async for chunk in self._stream:
            self._owner.observe_bytes(len(chunk))
            # Account for event boundaries before passing any bytes to the SSE
            # decoder. A single unterminated line/event cannot grow its buffer.
            if self._sse:
                data = tail + chunk
                size -= len(tail)
                previous = 0
                for boundary in re.finditer(rb"\r\n\r\n|\n\n|\r\r", data):
                    size += boundary.end() - previous
                    if size > MAX_MESSAGE_BYTES:
                        self._owner.reject("message_too_large")
                    previous = boundary.end()
                    size = 0
                size += len(data) - previous
                if size > MAX_MESSAGE_BYTES:
                    self._owner.reject("message_too_large")
                tail = data[max(previous, len(data) - 3) :]
            else:
                size += len(chunk)
                if size > MAX_MESSAGE_BYTES:
                    self._owner.reject("message_too_large")
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class MCPHTTPTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        endpoint: str,
        inner: httpx.AsyncBaseTransport,
        *,
        forbidden_hosts: Sequence[str] = (),
    ) -> None:
        self._endpoint = httpx.URL(endpoint)
        self._inner = inner
        self._forbidden_hosts = frozenset(forbidden_hosts)
        self.discovery = True
        self.discovery_bytes = 0
        self.closed = False
        self.failure: MCPError | None = None

    def observe_bytes(self, count: int) -> None:
        if self.discovery:
            self.discovery_bytes += count
            if self.discovery_bytes > MAX_DISCOVERY_BYTES:
                self.reject("discovery_too_large")

    def reject(self, code: str) -> None:
        self.failure = MCPError(code)
        raise self.failure

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await self._send(request)
        except MCPError as error:
            self.failure = MCPError(error.code)
            raise self.failure from None
        except Exception:
            self.reject("transport_error")

    async def _send(self, request: httpx.Request) -> httpx.Response:
        parsed = urlsplit(str(request.url))
        if (
            self.closed
            or request.url != self._endpoint
            or request.method not in {"POST", "GET", "DELETE"}
            or parsed.hostname in self._forbidden_hosts
            or parsed.netloc in self._forbidden_hosts
        ):
            raise MCPError("destination_forbidden")
        request.headers["Accept-Encoding"] = "identity"
        response = await self._inner.handle_async_request(request)
        try:
            if request.method == "POST" and response.status_code >= 400:
                self.reject(
                    "authentication_failed"
                    if response.status_code in {401, 403}
                    else "protocol_error"
                )
            # The SDK implements redirects itself, even with follow_redirects
            # disabled on HTTPX. Reject here before it can inspect Location.
            if response.is_redirect:
                raise MCPError("redirect_forbidden")
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise MCPError("unsupported_encoding")
            if len(response.headers.get("mcp-session-id", "").encode()) > 2048:
                raise MCPError("invalid_session")
            sse = response.headers.get("content-type", "").lower().startswith("text/event-stream")
            length = response.headers.get("content-length")
            if length is not None and (
                not length.isdecimal() or (not sse and int(length) > MAX_MESSAGE_BYTES)
            ):
                raise MCPError("message_too_large")
            response.stream = _BoundedStream(response.stream, self, sse)
            return response
        except BaseException:
            await response.aclose()
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()
        self.closed = True

    def __repr__(self) -> str:
        return f"MCPHTTPTransport(closed={self.closed!r})"


def create_mcp_http_client(
    settings: MCPToolsetSettings,
    *,
    proxy: ProxyHTTPClient | None = None,
    forbidden_hosts: Sequence[str] = (),
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[httpx.AsyncClient, MCPHTTPTransport]:
    if proxy is not None and transport is not None:
        raise MCPError("invalid_transport")
    if proxy is not None:
        transport = proxy.new_isolated_transport(settings.ca_bundle_pem)
    elif transport is None:
        tls = ssl.create_default_context()
        if settings.ca_bundle_pem is not None:
            tls.load_verify_locations(cadata=settings.ca_bundle_pem)
        transport = httpx.AsyncHTTPTransport(verify=tls, trust_env=False, retries=0)
    bounded = MCPHTTPTransport(settings.endpoint, transport, forbidden_hosts=forbidden_hosts)
    timeout = httpx.Timeout(
        connect=settings.connect_timeout_seconds,
        pool=settings.connect_timeout_seconds,
        read=settings.request_timeout_seconds,
        write=settings.request_timeout_seconds,
    )
    client = httpx.AsyncClient(
        transport=bounded,
        trust_env=False,
        follow_redirects=False,
        timeout=timeout,
        headers={name: value.get_secret_value() for name, value in settings.headers.items()},
    )
    return client, bounded
