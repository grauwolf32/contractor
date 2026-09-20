from __future__ import annotations

import asyncio
import ssl
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from contractor_runtime import control_client
from contractor_runtime.control_client import (
    MAX_CONTROL_HEADER_BYTES,
    MAX_CONTROL_HEADERS,
    MAX_CONTROL_RESPONSE_BYTES,
    ControlClientError,
    ControlHTTPError,
    MTLSJSONTransport,
    _read_response_body,
    _read_response_head,
)

type FakeConnection = Callable[[], tuple[asyncio.StreamReader, Mock]]


def response_reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.parametrize("count", [MAX_CONTROL_HEADERS, MAX_CONTROL_HEADERS + 1])
def test_response_header_count_limit(count: int) -> None:
    async def scenario() -> None:
        data = b"HTTP/1.1 200 OK\r\n"
        data += b"".join(f"X-Header-{index}: value\r\n".encode() for index in range(count))
        reader = response_reader(data + b"\r\n")
        if count > MAX_CONTROL_HEADERS:
            with pytest.raises(ControlClientError, match="too many"):
                await _read_response_head(reader)
        else:
            status, headers = await _read_response_head(reader)
            assert status == 200
            assert len(headers) == count

    asyncio.run(scenario())


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_response_header_byte_limit(extra_bytes: int) -> None:
    async def scenario() -> None:
        prefix = b"HTTP/1.1 200 OK\r\nX-Value: "
        suffix = b"\r\n\r\n"
        value = b"x" * (MAX_CONTROL_HEADER_BYTES - len(prefix) - len(suffix) + extra_bytes)
        reader = response_reader(prefix + value + suffix)
        if extra_bytes:
            with pytest.raises(ControlClientError, match="oversized"):
                await _read_response_head(reader)
        else:
            assert await _read_response_head(reader) == (200, {"x-value": value.decode()})

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
def test_invalid_response_headers_raise_control_error(data: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(ControlClientError):
            await _read_response_head(response_reader(data))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"content-length": "2"}, b"{}"),
        ({"transfer-encoding": "ChUnKeD"}, b"1;name=value\r\n{\r\n1\r\n}\r\n0\r\n\r\n"),
        ({}, b"{}"),
    ],
)
def test_supported_response_framing(headers: dict[str, str], body: bytes) -> None:
    async def scenario() -> None:
        assert await _read_response_body(response_reader(body), headers) == b"{}"

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
        ({"transfer-encoding": "chunked"}, b"+2\r\n{}\r\n0\r\n\r\n"),
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
def test_invalid_response_body_raises_control_error(headers: dict[str, str], body: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(ControlClientError):
            await _read_response_body(response_reader(body), headers)

    asyncio.run(scenario())


@pytest.mark.parametrize("framing", ["content-length", "chunked", "close"])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_response_size_limit(framing: str, extra_bytes: int) -> None:
    async def scenario() -> None:
        body = b"x" * (MAX_CONTROL_RESPONSE_BYTES + extra_bytes)
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
            with pytest.raises(ControlClientError, match="too large"):
                await _read_response_body(reader, headers)
        else:
            assert await _read_response_body(reader, headers) == body

    asyncio.run(scenario())


@pytest.fixture
def connection(monkeypatch: pytest.MonkeyPatch) -> FakeConnection:
    def connect() -> tuple[asyncio.StreamReader, Mock]:
        context = ssl.create_default_context()
        ssl_object = context.wrap_bio(ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="localhost")
        reader = asyncio.StreamReader()
        writer = Mock(spec=asyncio.StreamWriter)
        writer.get_extra_info.return_value = ssl_object
        writer.transport = Mock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        monkeypatch.setattr(asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
        monkeypatch.setattr(control_client, "verify_control_plane_peer", Mock())
        return reader, writer

    return connect


@pytest.mark.parametrize("status", [200, 503])
def test_transport_closes_connection_and_preserves_http_status(
    status: int, connection: FakeConnection
) -> None:
    async def scenario() -> None:
        reader, writer = connection()
        reader.feed_data(f"HTTP/1.1 {status} Reason\r\nContent-Length: 2\r\n\r\n{{}}".encode())
        transport = MTLSJSONTransport("https://[::1]:8443/base/", ssl.create_default_context(), 1)
        if status == 200:
            assert await transport.post_json("/heartbeat", {"label": "метка"}) == b"{}"
        else:
            with pytest.raises(ControlHTTPError) as error:
                await transport.post_json("/heartbeat", {})
            assert error.value.status_code == status
        request = writer.write.call_args.args[0]
        assert request.startswith(b"POST /base/heartbeat HTTP/1.1\r\nHost: [::1]:8443\r\n")
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    asyncio.run(scenario())


def test_role_is_verified_before_writing_request(
    connection: FakeConnection, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        _, writer = connection()
        monkeypatch.setattr(
            control_client,
            "verify_control_plane_peer",
            Mock(side_effect=ssl.SSLCertVerificationError("wrong role")),
        )
        transport = MTLSJSONTransport("https://localhost", ssl.create_default_context(), 1)
        with pytest.raises(ssl.SSLCertVerificationError):
            await transport.post_json("/heartbeat", {})
        writer.write.assert_not_called()
        writer.close.assert_called_once()

    asyncio.run(scenario())


def test_cancellation_during_connection_close_aborts_transport(
    connection: FakeConnection,
) -> None:
    async def scenario() -> None:
        reader, writer = connection()
        reader.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        closing = asyncio.Event()

        async def wait_closed() -> None:
            closing.set()
            await asyncio.Event().wait()

        writer.wait_closed.side_effect = wait_closed
        transport = MTLSJSONTransport("https://localhost", ssl.create_default_context(), 1)
        task = asyncio.create_task(transport.post_json("/heartbeat", {}))
        await closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        writer.transport.abort.assert_called_once()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "request_path",
    ["/bad path", "/bad\r\nInjected: value", "/bad\x00", "relative", "/?query", "/#fragment"],
)
def test_invalid_request_target_is_rejected_before_connection(
    request_path: str, connection: FakeConnection
) -> None:
    async def scenario() -> None:
        connection()
        transport = MTLSJSONTransport("https://localhost", ssl.create_default_context(), 1)
        with pytest.raises(ValueError):
            await transport.post_json(request_path, {})
        asyncio.open_connection.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost",
        "https://user:password@localhost",
        "https://localhost/bad\r\npath",
        "https://localhost/bad path",
        "https://localhost/?",
        "https://localhost/#",
    ],
)
def test_invalid_base_url_is_rejected(base_url: str) -> None:
    with pytest.raises(ValueError):
        MTLSJSONTransport(base_url, ssl.create_default_context(), 1)


def test_request_timeout_covers_sending_and_receiving_together(
    connection: FakeConnection,
) -> None:
    async def scenario() -> None:
        reader, writer = connection()

        async def delayed_drain() -> None:
            await asyncio.sleep(0.03)
            reader.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n")
            asyncio.get_running_loop().call_later(0.03, reader.feed_data, b"{}")

        writer.drain.side_effect = delayed_drain
        transport = MTLSJSONTransport("https://localhost", ssl.create_default_context(), 0.05)
        with pytest.raises(TimeoutError):
            await transport.post_json("/heartbeat", {})
        writer.close.assert_called_once()

    asyncio.run(scenario())
