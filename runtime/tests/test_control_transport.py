from __future__ import annotations

import asyncio
import ssl
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from contractor_runtime import control_client
from contractor_runtime.control_client import (
    MAX_CONTROL_RESPONSE_BYTES,
    ControlClientError,
    ControlHTTPError,
    MTLSJSONTransport,
)

type FakeConnection = Callable[[], tuple[asyncio.StreamReader, Mock]]


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


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (b"HTTP/1.1 200 OK\r\nContent-Length: +2\r\n\r\n{}", "content length"),
        (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-0\r\n\r\n", "chunk size"),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: "
            + str(MAX_CONTROL_RESPONSE_BYTES + 1).encode()
            + b"\r\n\r\n",
            "Control Plane response is too large",
        ),
    ],
)
def test_malformed_response_raises_control_error(
    response: bytes, match: str, connection: FakeConnection
) -> None:
    async def scenario() -> None:
        reader, writer = connection()
        reader.feed_data(response)
        reader.feed_eof()
        transport = MTLSJSONTransport("https://localhost", ssl.create_default_context(), 1)
        with pytest.raises(ControlClientError, match=match):
            await transport.post_json("/heartbeat", {})
        writer.close.assert_called_once()

    asyncio.run(scenario())
