"""Private scanner IPC: every target exchange delegates to the allocation HTTP owner.

CONNECT terminates TLS locally; it never opens a target tunnel. The ordinary HTTP
transport independently verifies target TLS, scope, credentials and slot limits.
This listener accepts one request per connection and supports normal HTTP only.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import secrets
import ssl
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import h11
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from contractor_runtime.adapters.http_transport import HTTPTransport, HTTPTransportError
from contractor_runtime.contracts.engagement import canonical_url
from contractor_runtime.core.http_limits import (
    MAX_HEADERS,
    request_header_max_bytes,
)
from contractor_runtime.redaction import redact_private_values
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.threads import consume_task_exception, to_thread_until_done

_HOP_HEADERS = {
    b"connection",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
}


class _EmptyConnection(Exception):
    """A TCP reachability probe supplied no HTTP message."""


def _private_file(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)


def _tls_context(directory: Path) -> ssl.SSLContext:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Runtime HTTP bridge")])
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    certificate_path, key_path = directory / "bridge.crt", directory / "bridge.key"
    _private_file(certificate_path, certificate.public_bytes(serialization.Encoding.PEM))
    _private_file(
        key_path,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate_path, key_path)
    return context


class HTTPProxyBridge:
    def __init__(
        self,
        transport: HTTPTransport,
        directory: Path,
        *,
        credential_id: str | None,
        deadline: float,
    ) -> None:
        self._transport = transport
        self._directory = directory
        self._credential_id = credential_id
        self._deadline = deadline
        self._token = secrets.token_hex(32)
        encoded = base64.b64encode(("runtime:" + self._token).encode()).decode()
        self._authorization = ("Basic " + encoded).encode()
        private_values = (*transport.private_values(), self._token, encoded)
        self._private_values = tuple(
            {
                representation
                for value in private_values
                for representation in (
                    value,
                    json.dumps(value)[1:-1],
                    json.dumps(value, ensure_ascii=False)[1:-1],
                )
            }
        )
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._tls: ssl.SSLContext | None = None
        self.error_code: str | None = None
        self.requests = 0
        self._proxy_url = ""

    async def __aenter__(self):
        # Passive membership/lease check. No target DNS or connection is performed.
        catalog = self._transport.credentials()
        if self._credential_id is not None and not any(
            item["credential_id"] == self._credential_id for item in catalog
        ):
            raise ValueError("credential is not assigned to this allocation")
        self._tls = await to_thread_until_done(
            _tls_context, self._directory, name="http-bridge-tls"
        )
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self._proxy_url = f"http://runtime:{self._token}@127.0.0.1:{port}"
        return self

    @property
    def proxy_url(self) -> str:
        """Private IPC capability; adapters must pass it through a private config file."""
        if not self._proxy_url:
            raise RuntimeError("HTTP bridge is not open")
        return self._proxy_url

    def _accept(self, reader, writer) -> None:
        self._writers.add(writer)
        task = asyncio.create_task(self._handle(reader, writer), name="http-bridge-request")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(consume_task_exception)

    async def __aexit__(self, *_):
        async def cleanup():
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
            tasks = tuple(self._tasks)
            for task in tasks:
                task.cancel()
            for writer in tuple(self._writers):
                writer.close()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._writers.clear()
            self._private_values = ()
            self._token = ""
            self._proxy_url = ""
            self._authorization = b""
            self._tls = None

        task = asyncio.create_task(cleanup(), name="http-bridge-close")
        cancelled = False
        while not task.done():
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError

    def redact(self, data: bytes) -> bytes:
        return redact_private_values(
            data.decode("utf-8", errors="replace"), self._private_values
        ).encode()

    async def _message(self, reader):
        connection = h11.Connection(
            h11.SERVER, max_incomplete_event_size=request_header_max_bytes()
        )
        request, body, received = None, bytearray(), False
        while True:
            event = connection.next_event()
            if event is h11.NEED_DATA:
                data = await reader.read(8192)
                if not data:
                    if not received:
                        raise _EmptyConnection
                    raise ValueError("incomplete HTTP request")
                received = True
                connection.receive_data(data)
            elif isinstance(event, h11.Request):
                request = event
                if (
                    len(event.headers) > MAX_HEADERS
                    or sum(len(k) + len(v) + 4 for k, v in event.headers)
                    > request_header_max_bytes()
                ):
                    raise ValueError("HTTP request headers exceed bound")
            elif isinstance(event, h11.Data):
                body.extend(event.data)
                if len(body) > get_shared_config().http_request_body_max_bytes:
                    raise ValueError("HTTP request body exceeds bound")
            elif isinstance(event, h11.EndOfMessage):
                if request is None or connection.trailing_data[0]:
                    raise ValueError("HTTP pipelining is unsupported")
                return connection, request, bytes(body)
            else:
                raise ValueError("unsupported HTTP message")

    async def _handle(self, reader, writer):
        authenticated = False
        try:
            async with asyncio.timeout_at(min(self._deadline, time.monotonic() + 30)):
                connection, request, body = await self._message(reader)
                values = [v for k, v in request.headers if k == b"proxy-authorization"]
                if len(values) != 1 or not hmac.compare_digest(values[0], self._authorization):
                    return
                authenticated = True
                authority = None
                if request.method == b"CONNECT":
                    if body:
                        raise ValueError("CONNECT body is unsupported")
                    authority = request.target.decode("ascii")
                    # Validate authority as data; no socket is opened here.
                    origin = canonical_url("https://" + authority + "/")
                    authority = urlsplit(origin).netloc
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                    await writer.start_tls(self._tls, ssl_handshake_timeout=5)
                    connection, request, body = await self._message(reader)
                await self._forward(connection, request, body, authority, reader, writer)
        except asyncio.CancelledError:
            if authenticated:
                self.error_code = self.error_code or "scan_request_failed"
            raise
        except _EmptyConnection:
            if authenticated:
                self.error_code = self.error_code or "scan_request_failed"
        except (HTTPTransportError, TimeoutError, OSError, ValueError, h11.ProtocolError):
            self.error_code = self.error_code or "scan_request_failed"
        finally:
            writer.close()
            with suppress(OSError, TimeoutError):
                async with asyncio.timeout(1):
                    await writer.wait_closed()
            self._writers.discard(writer)

    async def _forward(self, connection, request, body, authority, reader, writer):
        self.requests += 1
        method = request.method.decode("ascii")
        target = request.target.decode("ascii")
        if method == "CONNECT" or (authority is not None and not target.startswith("/")):
            raise ValueError("unsupported target")
        url = ("https://" + authority + target) if authority is not None else target
        if canonical_url(url) != url:
            raise ValueError("noncanonical target")
        parsed = urlsplit(url)
        hosts = [value.decode("ascii") for name, value in request.headers if name == b"host"]
        if len(hosts) != 1 or canonical_url(parsed.scheme + "://" + hosts[0] + "/") != (
            parsed.scheme + "://" + parsed.netloc + "/"
        ):
            raise ValueError("Host differs from target authority")
        hop_headers = set(_HOP_HEADERS)
        for name, value in request.headers:
            if name == b"connection":
                hop_headers.update(item.strip().lower() for item in value.split(b","))
        headers = [
            (name, value)
            for name, value in request.headers
            # Scanner cookie jars must not replace the Runtime's exact-origin
            # jars. Explicit Cookie template headers are refused by admission.
            if name not in hop_headers | {b"host", b"content-length", b"cookie"}
        ]
        exchange = asyncio.create_task(
            self._transport.request(
                url,
                credential_id=self._credential_id,
                method=method,
                headers=headers,
                body=body,
                follow_redirects=False,
                timeout=max(0.001, min(30, self._deadline - time.monotonic())),
            )
        )
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait(
                (exchange, disconnected), return_when=asyncio.FIRST_COMPLETED
            )
            if exchange not in done:
                raise ValueError("scanner connection closed before response")
            outcome, response = await exchange
        finally:
            exchange.cancel()
            disconnected.cancel()
            await asyncio.gather(exchange, disconnected, return_exceptions=True)
        if outcome.kind != "completed" or response is None:
            self.error_code = (
                "scan_target_denied" if outcome.kind == "denied" else "scan_request_failed"
            )
            return
        if response.truncated:
            self.error_code = self.error_code or "scan_request_failed"
        if response.status < 200:
            raise ValueError("protocol switch is unsupported")
        response_hop_headers = set(_HOP_HEADERS)
        for name, value in response.headers:
            if name.lower() == b"connection":
                response_hop_headers.update(item.strip().lower() for item in value.split(b","))
        response_headers = [
            (k, v)
            for k, v in response.headers
            if k.lower() not in response_hop_headers | {b"content-length"}
        ]
        if method == "HEAD" or response.status == 304:
            response_headers += [
                (k, v) for k, v in response.headers if k.lower() == b"content-length"
            ]
        elif response.status != 204:
            response_headers.append((b"content-length", str(len(response.body)).encode()))
        response_headers.append((b"connection", b"close"))
        writer.write(
            connection.send(h11.Response(status_code=response.status, headers=response_headers))
        )
        if method != "HEAD" and response.status not in {204, 304} and response.body:
            writer.write(connection.send(h11.Data(data=response.body)))
        writer.write(connection.send(h11.EndOfMessage()))
        await writer.drain()
