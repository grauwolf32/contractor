from __future__ import annotations

import asyncio

import pytest

from contractor_runtime._https import (
    MAX_RESPONSE_HEADER_BYTES,
    MAX_RESPONSE_HEADERS,
    HTTPResponseError,
    HTTPResponseTooLargeError,
    encode_request,
    host_header,
    read_response_body,
    read_response_head,
)

MAXIMUM = 1 << 20


def response_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.parametrize("count", [MAX_RESPONSE_HEADERS, MAX_RESPONSE_HEADERS + 1])
def test_response_header_count_limit(count: int) -> None:
    async def scenario() -> None:
        data = b"HTTP/1.1 200 OK\r\n"
        data += b"".join(f"X-Header-{index}: value\r\n".encode() for index in range(count))
        reader = response_reader(data + b"\r\n")
        if count > MAX_RESPONSE_HEADERS:
            with pytest.raises(HTTPResponseError, match="too many"):
                await read_response_head(reader)
        else:
            status, headers = await read_response_head(reader)
            assert status == 200
            assert len(headers) == count

    asyncio.run(scenario())


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_response_header_byte_limit(extra_bytes: int) -> None:
    async def scenario() -> None:
        prefix = b"HTTP/1.1 200 OK\r\nX-Value: "
        suffix = b"\r\n\r\n"
        value = b"x" * (MAX_RESPONSE_HEADER_BYTES - len(prefix) - len(suffix) + extra_bytes)
        reader = response_reader(prefix + value + suffix)
        if extra_bytes:
            with pytest.raises(HTTPResponseError, match="oversized"):
                await read_response_head(reader)
        else:
            assert await read_response_head(reader) == (200, {"x-value": value.decode()})

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "data",
    [
        b"HTTP/1.1 +200 OK\r\n\r\n",
        b"HTTP/1.1 0200 OK\r\n\r\n",
        b"HTTP/1.1 2_00 OK\r\n\r\n",
        b"HTTP/1.1 \xff OK\r\n\r\n",
        b"HTTP/1.1 200 OK\n\n",
        b"HTTP/1.1 200 OK\r\nContent-Length : 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nBad Header: value\r\n\r\n",
        b"HTTP/1.1 200 OK\r\n\xff: value\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nX-Value: private\x00token\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\ncontent-length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nX-Value: " + b"x" * (64 * 1024) + b"\r\n\r\n",
        b"HTTP/1.1 200 OK\r\n",
    ],
)
def test_invalid_response_headers_are_rejected(data: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPResponseError):
            await read_response_head(response_reader(data))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"content-length": "2"}, b"{}"),
        ({"transfer-encoding": "ChUnKeD"}, b"1;name=value\r\n{\r\n1\r\n}\r\n0\r\n\r\n"),
        ({"content-length": "02"}, b"{}"),
        ({"transfer-encoding": "chunked"}, b"2\r\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"1 ;x\r\n{\r\n1\r\n}\r\n000\r\n\r\n"),
        ({}, b"{}"),
    ],
)
def test_supported_response_framing(headers: dict[str, str], body: bytes) -> None:
    async def scenario() -> None:
        assert await read_response_body(response_reader(body), headers, MAXIMUM) == b"{}"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"content-length": "+2"}, b"{}"),
        ({"content-length": "0_2"}, b"{}"),
        ({"content-length": "-1"}, b"{}"),
        ({"content-length": "\u00b2"}, b"{}"),
        ({"content-length": "9" * 5000}, b"{}"),
        ({"content-length": "3"}, b"{}"),
        ({"transfer-encoding": ""}, b"{}"),
        ({"transfer-encoding": "gzip"}, b"{}"),
        ({"transfer-encoding": "chunked", "content-length": "2"}, b"2\r\n{}\r\n0\r\n\r\n"),
        ({"content-length": "+0"}, b""),
        ({"content-length": "-0"}, b""),
        ({"content-length": " 2"}, b"{}"),
        ({"transfer-encoding": "chunked"}, b"+2\r\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"+1\r\n{\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"-0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"2\r\n{}\r\n+0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b" 2\r\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0x2\r\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0_2\r\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"2\n{}\r\n0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"3\r\n{}"),
        ({"transfer-encoding": "chunked"}, b"2\r\n{}xx0\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0\r\nX-Trailer: value\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0\r\n" + b"x" * (64 * 1024) + b"\r\n\r\n"),
        ({"transfer-encoding": "chunked"}, b"0\r\n"),
        ({"transfer-encoding": "chunked"}, b"2;" + b"x" * (64 * 1024) + b"\r\n{}\r\n0\r\n\r\n"),
    ],
)
def test_invalid_response_body_is_rejected(headers: dict[str, str], body: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPResponseError):
            await read_response_body(response_reader(body), headers, MAXIMUM)

    asyncio.run(scenario())


@pytest.mark.parametrize("framing", ["content-length", "chunked", "close"])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_response_size_limit(framing: str, extra_bytes: int) -> None:
    async def scenario() -> None:
        body = b"x" * (MAXIMUM + extra_bytes)
        headers: dict[str, str] = {}
        encoded = body
        if framing == "content-length":
            headers["content-length"] = str(len(body))
        elif framing == "chunked":
            headers["transfer-encoding"] = "chunked"
            # Split the body to check the cumulative limit across chunks.
            encoded = b"1\r\nx\r\n" + f"{len(body) - 1:x}\r\n".encode()
            encoded += body[1:] + b"\r\n0\r\n\r\n"
        reader = response_reader(encoded)
        if extra_bytes:
            with pytest.raises(HTTPResponseTooLargeError):
                await read_response_body(reader, headers, MAXIMUM)
        else:
            assert await read_response_body(reader, headers, MAXIMUM) == body

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"content-length": "3"}, b"{}x"),
        ({"transfer-encoding": "chunked"}, b"3\r\n{}x\r\n0\r\n\r\n"),
        ({}, b"{}x"),
    ],
)
def test_caller_limit_bounds_every_framing(headers: dict[str, str], body: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPResponseTooLargeError):
            await read_response_body(response_reader(body), headers, 2)
        assert await read_response_body(response_reader(body), headers, 3) == b"{}x"

    asyncio.run(scenario())


def test_status_line_requires_reason_separator() -> None:
    async def scenario() -> None:
        with pytest.raises(HTTPResponseError, match="status line"):
            await read_response_head(response_reader(b"HTTP/1.1 200\r\n\r\n"))
        assert await read_response_head(response_reader(b"HTTP/1.1 200 \r\n\r\n")) == (200, {})

    asyncio.run(scenario())


def test_request_encoding_rejects_header_injection() -> None:
    assert host_header("::1", 8443) == "[::1]:8443"
    assert host_header("localhost", 443) == "localhost"
    assert (
        encode_request("GET", "/x", {"Host": "h"}, b"b") == b"GET /x HTTP/1.1\r\nHost: h\r\n\r\nb"
    )
    for name, value in [("X-A", "v\r\nInjected: 1"), ("X:A", "v"), ("", "v")]:
        with pytest.raises(ValueError):
            encode_request("GET", "/x", {name: value}, b"")
