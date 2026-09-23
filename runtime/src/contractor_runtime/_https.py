"""Private HTTPS/1.1 exchange shared by the Control Plane clients.

A generic HTTP client verifies chain and hostname but does not expose a
portable hook between TLS completion and the first request byte. This module
opens one fresh bounded connection per exchange, lets the caller verify the
peer role, and only then writes the request. Response parsing is strict: the
status line, header names and values, and both body framings are validated,
and every read is bounded by the caller's response byte limit.
"""

from __future__ import annotations

import asyncio
import re
import ssl
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

MAX_RESPONSE_HEADERS = 64
MAX_RESPONSE_HEADER_BYTES = 64 * 1024
MAX_RESPONSE_LINE_BYTES = 8192
CONNECTION_CLOSE_TIMEOUT_SECONDS = 0.5

_HTTP_FIELD_NAME = re.compile(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+")
_CHUNK_SIZE = re.compile(rb"[0-9a-fA-F]+")

type PeerVerifier = Callable[[ssl.SSLSocket | ssl.SSLObject], None]


class HTTPResponseError(Exception):
    """The peer sent a malformed, unsupported or incomplete HTTP response."""


class HTTPResponseTooLargeError(HTTPResponseError):
    """The response body exceeded the caller's byte limit."""


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes = field(repr=False)


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


async def exchange(
    host: str,
    port: int,
    context: ssl.SSLContext,
    request: bytes,
    *,
    timeout_seconds: float,
    max_response_bytes: int,
    verify_peer: PeerVerifier,
) -> HTTPResponse:
    """Send one request on a fresh connection and read the whole response.

    The timeout covers the entire exchange, including TLS setup. The peer is
    verified before the first request byte is written.
    """

    writer: asyncio.StreamWriter | None = None
    try:
        async with asyncio.timeout(timeout_seconds):
            reader, writer = await asyncio.open_connection(
                host,
                port,
                ssl=context,
                server_hostname=host,
                limit=MAX_RESPONSE_HEADER_BYTES,
            )
            ssl_object = writer.get_extra_info("ssl_object")
            if not isinstance(ssl_object, ssl.SSLObject | ssl.SSLSocket):
                raise ssl.SSLCertVerificationError("private connection has no TLS peer")
            verify_peer(ssl_object)
            writer.write(request)
            await writer.drain()
            status, headers = await read_response_head(reader)
            body = await read_response_body(reader, headers, max_response_bytes)
    finally:
        if writer is not None:
            await _close_connection(writer, timeout_seconds)
    return HTTPResponse(status, headers, body)


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


async def read_response_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    status_line = await _read_line(
        reader, limit=MAX_RESPONSE_LINE_BYTES, error="invalid HTTP status line"
    )
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
            return status, headers
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
