from __future__ import annotations

import asyncio
import ssl
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from contractor_runtime import _https
from contractor_runtime._https import ConnectionReader, PrivateConnections
from contractor_runtime.control_client import (
    MAX_CONTROL_RESPONSE_BYTES,
    ControlClientError,
    ControlHTTPError,
    MTLSJSONTransport,
)

type FakeConnection = Callable[[], tuple[ConnectionReader, Mock]]


@pytest.fixture
def connection(monkeypatch: pytest.MonkeyPatch) -> FakeConnection:
    def connect() -> tuple[ConnectionReader, Mock]:
        context = ssl.create_default_context()
        ssl_object = context.wrap_bio(ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="localhost")
        reader = ConnectionReader()
        writer = Mock(spec=asyncio.StreamWriter)
        writer.get_extra_info.return_value = ssl_object
        writer.is_closing.return_value = False
        writer.transport = Mock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        monkeypatch.setattr(_https, "_open_connection", AsyncMock(return_value=(reader, writer)))
        return reader, writer

    return connect


def transport_for(
    base_url: str, timeout_seconds: float = 1, verify_peer: Mock | None = None
) -> tuple[MTLSJSONTransport, PrivateConnections]:
    connections = PrivateConnections(ssl.create_default_context(), verify_peer or Mock())
    return MTLSJSONTransport(base_url, connections, timeout_seconds), connections


@pytest.mark.parametrize("status", [200, 503])
def test_transport_keeps_a_complete_response_connection_and_preserves_http_status(
    status: int, connection: FakeConnection
) -> None:
    async def scenario() -> None:
        reader, writer = connection()
        reader.feed_data(f"HTTP/1.1 {status} Reason\r\nContent-Length: 2\r\n\r\n{{}}".encode())
        transport, connections = transport_for("https://[::1]:8443/base/")
        if status == 200:
            assert await transport.post_json("/heartbeat", {"label": "метка"}) == b"{}"
        else:
            with pytest.raises(ControlHTTPError) as error:
                await transport.post_json("/heartbeat", {})
            assert error.value.status_code == status
        request = writer.write.call_args.args[0]
        assert request.startswith(b"POST /base/heartbeat HTTP/1.1\r\nHost: [::1]:8443\r\n")
        assert b"\r\nConnection:" not in request
        # The connection stays with its owner until the owner closes it.
        writer.close.assert_not_called()
        await connections.aclose()
        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    asyncio.run(scenario())


_ENVELOPE = (
    b'{"code":"invalid_request","message":"server-message-must-not-escape",'
    b'"retryable":false,"requestId":"request_1"}'
)


@pytest.mark.parametrize(
    ("body", "envelope"),
    [
        (_ENVELOPE, ("invalid_request", False, "request_1")),
        (
            b'{"code":"conflict","message":"m","retryable":true,"requestId":"request-2"}',
            ("conflict", True, "request-2"),
        ),
        # Anything but the exact strict envelope is classified by status alone.
        (_ENVELOPE.replace(b'{"code"', b'{"code":"conflict","code"'), None),
        (_ENVELOPE.replace(b"false", b'"false"'), None),
        (_ENVELOPE.replace(b"invalid_request", b"Invalid Labels"), None),
        (_ENVELOPE.replace(b"}", b',"extra":1}'), None),
        (_ENVELOPE.replace(b"server-message-must-not-escape", b" "), None),
        (_ENVELOPE.replace(b"request_1", b"request 1"), None),
        (b"<html>Bad Gateway</html>", None),
        (b"", None),
    ],
)
def test_transport_attaches_only_a_strict_private_error_envelope(
    body: bytes, envelope: tuple[str, bool, str] | None, connection: FakeConnection
) -> None:
    async def scenario() -> None:
        reader, _ = connection()
        reader.feed_data(
            f"HTTP/1.1 400 Bad Request\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        )
        transport, _ = transport_for("https://localhost")
        with pytest.raises(ControlHTTPError) as error:
            await transport.post_json("/private/v1/agents/register", {})
        attached = (error.value.code, error.value.retryable, error.value.request_id)
        assert attached == (envelope or (None, None, None))
        assert error.value.status_code == 400
        assert "server-message-must-not-escape" not in str(error.value)

    asyncio.run(scenario())


def test_role_is_verified_before_writing_request(connection: FakeConnection) -> None:
    async def scenario() -> None:
        _, writer = connection()
        transport, _ = transport_for(
            "https://localhost",
            verify_peer=Mock(side_effect=ssl.SSLCertVerificationError("wrong role")),
        )
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
        # The peer ends the connection, so the exchange closes it.
        reader.feed_data(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
        closing = asyncio.Event()

        async def wait_closed() -> None:
            closing.set()
            await asyncio.Event().wait()

        writer.wait_closed.side_effect = wait_closed
        transport, _ = transport_for("https://localhost")
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
        transport, _ = transport_for("https://localhost")
        with pytest.raises(ValueError):
            await transport.post_json(request_path, {})
        _https._open_connection.assert_not_awaited()

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
        transport_for(base_url)


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
        transport, _ = transport_for("https://localhost", 0.05)
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
        transport, _ = transport_for("https://localhost")
        with pytest.raises(ControlClientError, match=match):
            await transport.post_json("/heartbeat", {})
        writer.close.assert_called_once()

    asyncio.run(scenario())
