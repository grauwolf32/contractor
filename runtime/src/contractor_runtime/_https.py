"""Private HTTPS/1.1 exchanges shared by the Control Plane clients.

A generic HTTP client verifies chain and hostname but does not expose a
portable hook between TLS completion and the first request byte. This module
opens bounded connections, lets the owner's verifier check the peer role on
every new connection, and only then writes a request. A verified connection is
reused for later requests to the same origin within the bounds of CTL-R165 and
closed on any doubt about its state. Response parsing is strict: the status
line, header names and values, and both body framings are validated, and every
read is bounded by the caller's response byte limit.
"""

from __future__ import annotations

import asyncio
import re
import ssl
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

from contractor_runtime.contracts.base import is_blank
from contractor_runtime.strict_json import strict_json_loads

MAX_RESPONSE_HEADERS = 64
MAX_RESPONSE_HEADER_BYTES = 64 * 1024
MAX_RESPONSE_LINE_BYTES = 8192
MAX_PRIVATE_ERROR_BYTES = 1 << 20
CONNECTION_CLOSE_TIMEOUT_SECONDS = 0.5
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# Bounds of reusable private connections (CTL-R165). The Go Server holds the
# same values in internal/mtls and api/testdata/v1alpha1/
# private-connection-reuse.json pins both sides.
# LIM-CTL-13: a private listener closes an idle keep-alive connection after it.
PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS = 60
# LIM-CTL-14: a client reuses an idle connection for at most this long.
PRIVATE_CONNECTION_IDLE_REUSE_SECONDS = 30.0
# LIM-CTL-15: no request starts on a connection this old or older.
PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS = 300.0
# LIM-CTL-16: open and idle connections per origin and owner.
PRIVATE_CONNECTIONS_PER_PEER = 8
PRIVATE_IDLE_CONNECTIONS_PER_PEER = 2
# LIM-CTL-18: Control Plane origins one Runtime owner keeps connections for.
MAX_PRIVATE_ORIGINS = 4
# Interim (1xx) responses skipped before the final response of one exchange.
MAX_INTERIM_RESPONSES = 5

_HTTP_FIELD_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_CHUNK_SIZE = re.compile(rb"[0-9a-fA-F]+")
_PRIVATE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PRIVATE_ERROR_FIELDS = frozenset({"code", "message", "retryable", "requestId"})

type PeerVerifier = Callable[[ssl.SSLSocket | ssl.SSLObject], None]


class HTTPResponseError(Exception):
    """The peer sent a malformed, unsupported or incomplete HTTP response."""


class HTTPResponseTooLargeError(HTTPResponseError):
    """The response body exceeded the caller's byte limit."""


class HTTPResponseNotStartedError(HTTPResponseError):
    """The connection ended before the first byte of a response."""


class PrivateConnectionsClosedError(Exception):
    """A private exchange started after its owner closed the connections."""


class ResponseHead(NamedTuple):
    status_code: int
    headers: dict[str, str]
    # HTTP/1.1 without Connection: close; the connection may carry the next
    # request once the body is complete.
    persistent: bool


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class PrivateError:
    """The Server's private API error envelope, without its message text."""

    code: str
    retryable: bool
    request_id: str


def decode_private_error(body: bytes) -> PrivateError | None:
    """Return the error envelope a non-2xx private Server response carries.

    The envelope is a strict JSON object with exactly code, message, retryable
    and requestId; its retryable flag is the Server's verdict on whether the
    same request may succeed later. Any other body, such as a proxy error page,
    is None, so the caller classifies the response by its status alone. The
    message is dropped: Server text never reaches a log or a model.
    """

    if len(body) > MAX_PRIVATE_ERROR_BYTES:
        return None
    try:
        value = strict_json_loads(body)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return None
    if (
        not isinstance(value, dict)
        or set(value) != _PRIVATE_ERROR_FIELDS
        or not isinstance(value["code"], str)
        or _PRIVATE_ERROR_CODE.fullmatch(value["code"]) is None
        or not isinstance(value["message"], str)
        or is_blank(value["message"])
        or not isinstance(value["retryable"], bool)
        or not isinstance(value["requestId"], str)
        or REQUEST_ID_PATTERN.fullmatch(value["requestId"]) is None
    ):
        return None
    return PrivateError(
        code=value["code"], retryable=value["retryable"], request_id=value["requestId"]
    )


def new_request_id() -> str:
    return f"request_{uuid.uuid4().hex}"


def host_header(host: str, port: int) -> str:
    host_name = f"[{host}]" if ":" in host else host
    return host_name if port == 443 else f"{host_name}:{port}"


def encode_request(method: str, target: str, headers: Mapping[str, str], body: bytes) -> bytes:
    """Encode one request head and body; raises ValueError on unsafe fields."""

    result = bytearray(f"{method} {target} HTTP/1.1\r\n".encode("ascii"))
    for name, value in headers.items():
        if (
            not name
            or any(char in name for char in "\r\n:")
            or any(char in value for char in "\r\n")
        ):
            raise ValueError("invalid private HTTP header")
        result.extend(f"{name}: {value}\r\n".encode("ascii"))
    result.extend(b"\r\n")
    return bytes(result) + body


class ConnectionReader(asyncio.StreamReader):
    """Stream reader that counts received bytes and reports unread ones."""

    def __init__(
        self, limit: int = MAX_RESPONSE_HEADER_BYTES, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        super().__init__(limit=limit, loop=loop)
        self.received = 0

    def feed_data(self, data: bytes) -> None:
        self.received += len(data)
        super().feed_data(data)

    def has_unread_data(self) -> bool:
        return bool(self._buffer)


@dataclass(slots=True, eq=False)
class _Connection:
    reader: ConnectionReader
    writer: asyncio.StreamWriter
    created_at: float
    idle_since: float = 0.0


class _ReuseFailedError(Exception):
    """An idempotent request received no response on a reused connection."""


async def _open_connection(
    host: str, port: int, context: ssl.SSLContext
) -> tuple[ConnectionReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    reader = ConnectionReader(limit=MAX_RESPONSE_HEADER_BYTES, loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    transport, _ = await loop.create_connection(
        lambda: protocol, host, port, ssl=context, server_hostname=host
    )
    return reader, asyncio.StreamWriter(transport, protocol, reader, loop)


class PrivateConnections:
    """Reusable verified connections to Control Plane origins (CTL-R163).

    The Runtime process owns each instance and closes it at shutdown with
    :meth:`aclose` (CTL-R164). Every connection carries the process
    certificate of ``context`` and is checked by ``verify_peer`` before its
    first request byte; the allocation is named in each request, so a
    connection carries no allocation authority. Each origin keeps at most
    PRIVATE_CONNECTIONS_PER_PEER connections, of which at most
    PRIVATE_IDLE_CONNECTIONS_PER_PEER stay idle, and at most
    MAX_PRIVATE_ORIGINS origins are kept. An idle connection past the idle
    bound or the lifetime is never reused; it closes at the next exchange to
    its origin, at aclose, or when the peer listener closes it (LIFE-CTL-5).
    """

    def __init__(
        self,
        context: ssl.SSLContext,
        verify_peer: PeerVerifier,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._context = context
        self._verify_peer = verify_peer
        self._clock = clock
        self._origins: OrderedDict[tuple[str, int], _OriginConnections] = OrderedDict()
        self._closed = False

    async def exchange(
        self,
        host: str,
        port: int,
        request: bytes,
        *,
        idempotent: bool,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HTTPResponse:
        """Send one request and read the whole response.

        The timeout covers the entire exchange: waiting for a connection, TLS
        setup and peer verification of a new connection, and the response. A
        request whose bytes may have reached the peer is never sent again,
        except that an idempotent request which received no response byte on a
        reused connection is sent once more on a new verified connection.
        """

        if self._closed:
            raise PrivateConnectionsClosedError("private connections are closed")
        if _requests_close(request):
            raise ValueError("private requests leave connection management to their owner")
        key = (host, port)
        origin = self._origins.get(key)
        stale: list[asyncio.StreamWriter] = []
        if origin is None:
            origin = _OriginConnections(
                host, port, self._context, self._verify_peer, self._clock, self._is_closed
            )
            self._origins[key] = origin
            while len(self._origins) > MAX_PRIVATE_ORIGINS:
                stale.extend(self._origins.popitem(last=False)[1].detach())
        else:
            self._origins.move_to_end(key)
        # The evicted origin's idle connections start closing now; waiting for
        # them to finish happens after the exchange, outside its timeout.
        for writer in stale:
            writer.close()
        try:
            return await origin.exchange(
                request,
                idempotent=idempotent,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        finally:
            await _close_all(stale, timeout_seconds)

    def _is_closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Close every connection and refuse later exchanges.

        Idle connections close now; a connection in use closes when its
        exchange ends, and an exchange still waiting for a connection fails
        without opening one.
        """

        self._closed = True
        origins = list(self._origins.values())
        self._origins.clear()
        await _close_all(
            [writer for origin in origins for writer in origin.detach()],
            CONNECTION_CLOSE_TIMEOUT_SECONDS,
        )


class _OriginConnections:
    def __init__(
        self,
        host: str,
        port: int,
        context: ssl.SSLContext,
        verify_peer: PeerVerifier,
        clock: Callable[[], float],
        owner_closed: Callable[[], bool],
    ) -> None:
        self._host = host
        self._port = port
        self._context = context
        self._verify_peer = verify_peer
        self._clock = clock
        self._owner_closed = owner_closed
        self._idle: list[_Connection] = []
        self._slots = asyncio.Semaphore(PRIVATE_CONNECTIONS_PER_PEER)
        # A retired origin (evicted or closed) still serves the exchanges that
        # already reached it but keeps no connection for reuse.
        self._retired = False

    async def exchange(
        self,
        request: bytes,
        *,
        idempotent: bool,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HTTPResponse:
        discarded: list[_Connection] = []
        try:
            async with asyncio.timeout(timeout_seconds), self._slots:
                if self._owner_closed():
                    raise PrivateConnectionsClosedError("private connections are closed")
                connection = self._take_idle(discarded)
                if connection is not None:
                    try:
                        return await self._send(
                            connection,
                            request,
                            max_response_bytes,
                            discarded,
                            retry_on_fresh_connection=idempotent,
                        )
                    except _ReuseFailedError:
                        pass
                connection = await self._open(discarded)
                return await self._send(
                    connection,
                    request,
                    max_response_bytes,
                    discarded,
                    retry_on_fresh_connection=False,
                )
        finally:
            # Closing waits outside the exchange timeout, so a completed
            # response is never lost to a slow close.
            await _close_all([connection.writer for connection in discarded], timeout_seconds)

    def _take_idle(self, discarded: list[_Connection]) -> _Connection | None:
        now = self._clock()
        while self._idle:
            connection = self._idle.pop()
            if (
                now - connection.created_at < PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS
                and now - connection.idle_since < PRIVATE_CONNECTION_IDLE_REUSE_SECONDS
                and not connection.writer.is_closing()
                and not connection.reader.at_eof()
                and not connection.reader.has_unread_data()
            ):
                return connection
            discarded.append(connection)
        return None

    async def _open(self, discarded: list[_Connection]) -> _Connection:
        reader, writer = await _open_connection(self._host, self._port, self._context)
        connection = _Connection(reader, writer, created_at=self._clock())
        try:
            ssl_object = writer.get_extra_info("ssl_object")
            if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                raise ssl.SSLCertVerificationError("private connection has no TLS peer")
            self._verify_peer(ssl_object)
        except BaseException:
            discarded.append(connection)
            raise
        return connection

    async def _send(
        self,
        connection: _Connection,
        request: bytes,
        max_response_bytes: int,
        discarded: list[_Connection],
        *,
        retry_on_fresh_connection: bool,
    ) -> HTTPResponse:
        reusable = False
        received_before = connection.reader.received
        try:
            try:
                connection.writer.write(request)
                await connection.writer.drain()
                head = await read_response_head(connection.reader)
            except (HTTPResponseNotStartedError, ConnectionError) as error:
                # Only a request that received no response byte may go again.
                if retry_on_fresh_connection and connection.reader.received == received_before:
                    raise _ReuseFailedError from error
                raise
            interim = 0
            while 100 <= head.status_code < 200:
                # An interim response has no content; the final one follows.
                if head.status_code == 101:
                    raise HTTPResponseError("unexpected HTTP protocol switch")
                interim += 1
                if interim > MAX_INTERIM_RESPONSES:
                    raise HTTPResponseError("too many interim HTTP responses")
                head = await read_response_head(connection.reader)
            if head.status_code in {204, 304}:
                if _declares_content(head.headers):
                    raise HTTPResponseError("HTTP response without content declares a body")
                body = b""
            else:
                body = await read_response_body(connection.reader, head.headers, max_response_bytes)
            # A body delimited by the end of the connection leaves nothing to
            # reuse, bytes beyond the framed body would answer the next request
            # and an interim response is not worth the doubt: each closes the
            # connection.
            reusable = (
                interim == 0
                and head.persistent
                and (
                    head.status_code in {204, 304}
                    or "content-length" in head.headers
                    or "transfer-encoding" in head.headers
                )
                and not connection.reader.has_unread_data()
            )
            return HTTPResponse(head.status_code, head.headers, body)
        finally:
            self._finish(connection, discarded, reusable=reusable)

    def _finish(
        self, connection: _Connection, discarded: list[_Connection], *, reusable: bool
    ) -> None:
        now = self._clock()
        if (
            not reusable
            or self._retired
            or now - connection.created_at >= PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS
        ):
            discarded.append(connection)
            return
        connection.idle_since = now
        self._idle.append(connection)
        while len(self._idle) > PRIVATE_IDLE_CONNECTIONS_PER_PEER:
            discarded.append(self._idle.pop(0))

    def detach(self) -> list[asyncio.StreamWriter]:
        """Stop reuse and hand over the idle connections to close.

        An exchange already waiting here still runs, on a connection of its
        own, unless the owner closed; a connection in use closes when its
        exchange ends.
        """

        self._retired = True
        idle, self._idle = self._idle, []
        return [connection.writer for connection in idle]


def _declares_content(headers: Mapping[str, str]) -> bool:
    length = headers.get("content-length")
    return "transfer-encoding" in headers or (
        length is not None
        and not (length.isascii() and length.isdecimal() and set(length) == {"0"})
    )


def _requests_close(request: bytes) -> bool:
    head = request.partition(b"\r\n\r\n")[0].lower()
    return any(line.startswith(b"connection:") for line in head.split(b"\r\n")[1:])


async def _close_all(writers: list[asyncio.StreamWriter], timeout_seconds: float) -> None:
    """Close every connection; cancellation aborts the ones not yet closed."""

    for index, writer in enumerate(writers):
        try:
            await _close_connection(writer, timeout_seconds)
        except asyncio.CancelledError:
            for remaining in writers[index + 1 :]:
                remaining.close()
                remaining.transport.abort()
            raise


async def _close_connection(writer: asyncio.StreamWriter, timeout_seconds: float) -> None:
    writer.close()
    current_task = asyncio.current_task()
    if current_task is not None and current_task.cancelling():
        writer.transport.abort()
        return
    try:
        await asyncio.wait_for(
            writer.wait_closed(), timeout=min(CONNECTION_CLOSE_TIMEOUT_SECONDS, timeout_seconds)
        )
    except asyncio.CancelledError:
        writer.transport.abort()
        raise
    except Exception:
        writer.transport.abort()


async def _read_line(reader: asyncio.StreamReader, *, limit: int, error: str) -> bytes:
    try:
        line = await reader.readline()
    except ValueError:
        # StreamReader raises ValueError when its own line limit is exceeded.
        raise HTTPResponseError(error) from None
    if len(line) > limit or not line.endswith(b"\r\n"):
        raise HTTPResponseError(error)
    return line


async def _read_exactly(reader: asyncio.StreamReader, length: int) -> bytes:
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise HTTPResponseError("incomplete HTTP response body") from None


async def read_response_head(reader: asyncio.StreamReader) -> ResponseHead:
    try:
        first = await reader.readline()
    except ValueError:
        raise HTTPResponseError("invalid HTTP status line") from None
    if not first:
        raise HTTPResponseNotStartedError("connection closed before an HTTP response")
    if len(first) > MAX_RESPONSE_LINE_BYTES or not first.endswith(b"\r\n"):
        raise HTTPResponseError("invalid HTTP status line")
    status_line = first
    parts = status_line[:-2].split(b" ", 2)
    if len(parts) != 3 or parts[0] not in {b"HTTP/1.0", b"HTTP/1.1"}:
        raise HTTPResponseError("invalid HTTP status line")
    if len(parts[1]) != 3 or not parts[1].isdigit():
        raise HTTPResponseError("invalid HTTP status code")
    status = int(parts[1])
    if not 100 <= status <= 599:
        raise HTTPResponseError("invalid HTTP status code")
    headers: dict[str, str] = {}
    total = len(status_line)
    while True:
        line = await _read_line(
            reader,
            limit=MAX_RESPONSE_HEADER_BYTES - total,
            error="invalid or oversized HTTP response headers",
        )
        total += len(line)
        if line == b"\r\n":
            persistent = parts[0] == b"HTTP/1.1" and "close" not in {
                token.strip().lower() for token in headers.get("connection", "").split(",")
            }
            return ResponseHead(status, headers, persistent)
        if len(headers) >= MAX_RESPONSE_HEADERS:
            raise HTTPResponseError("too many HTTP response headers")
        name, separator, value = line[:-2].partition(b":")
        if not separator or not _HTTP_FIELD_NAME.fullmatch(name):
            raise HTTPResponseError("invalid HTTP response header")
        key = name.decode("ascii").lower()
        if key in headers:
            raise HTTPResponseError("invalid or duplicate HTTP response header")
        if any((byte < 32 and byte != 9) or byte == 127 for byte in value):
            raise HTTPResponseError("invalid HTTP response header value")
        headers[key] = value.decode("latin-1").strip(" \t")


async def read_response_body(
    reader: asyncio.StreamReader, headers: Mapping[str, str], maximum: int
) -> bytes:
    if "transfer-encoding" in headers:
        if "content-length" in headers:
            raise HTTPResponseError("ambiguous HTTP response framing")
        if headers["transfer-encoding"].lower() != "chunked":
            raise HTTPResponseError("unsupported HTTP transfer encoding")
        return await _read_chunked_body(reader, maximum)
    if "content-length" in headers:
        raw_length = headers["content-length"]
        if not raw_length.isascii() or not raw_length.isdecimal():
            raise HTTPResponseError("invalid HTTP content length")
        try:
            length = int(raw_length)
        except ValueError:
            raise HTTPResponseError("invalid HTTP content length") from None
        if length > maximum:
            raise HTTPResponseTooLargeError("HTTP response is too large")
        return await _read_exactly(reader, length)
    result = bytearray()
    while True:
        chunk = await reader.read(min(64 * 1024, maximum + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        if len(result) > maximum:
            raise HTTPResponseTooLargeError("HTTP response is too large")


async def _read_chunked_body(reader: asyncio.StreamReader, maximum: int) -> bytes:
    result = bytearray()
    while True:
        size_line = await _read_line(
            reader, limit=MAX_RESPONSE_LINE_BYTES, error="invalid chunk size"
        )
        raw_size = size_line[:-2].partition(b";")[0].rstrip(b" \t")
        if not _CHUNK_SIZE.fullmatch(raw_size):
            raise HTTPResponseError("invalid chunk size")
        size = int(raw_size, 16)
        if len(result) + size > maximum:
            raise HTTPResponseTooLargeError("HTTP response is too large")
        if size == 0:
            if await _read_exactly(reader, 2) != b"\r\n":
                raise HTTPResponseError("chunked trailers are not supported")
            return bytes(result)
        result.extend(await _read_exactly(reader, size))
        if await _read_exactly(reader, 2) != b"\r\n":
            raise HTTPResponseError("invalid chunk delimiter")
