from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import ssl
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from contractor_runtime import _https
from contractor_runtime._https import (
    MAX_INTERIM_RESPONSES,
    MAX_PRIVATE_ORIGINS,
    MAX_RESPONSE_HEADER_BYTES,
    MAX_RESPONSE_HEADERS,
    PRIVATE_CONNECTION_IDLE_REUSE_SECONDS,
    PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS,
    PRIVATE_CONNECTIONS_PER_PEER,
    PRIVATE_IDLE_CONNECTIONS_PER_PEER,
    PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS,
    ConnectionReader,
    HTTPResponseError,
    HTTPResponseNotStartedError,
    HTTPResponseTooLargeError,
    PrivateConnections,
    PrivateConnectionsClosedError,
    encode_request,
    host_header,
    read_response_body,
    read_response_head,
)
from contractor_runtime.mtls import runtime_agent_client_context, verify_control_plane_peer

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
            head = await read_response_head(reader)
            assert head.status_code == 200
            assert len(head.headers) == count

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
            assert await read_response_head(reader) == (200, {"x-value": value.decode()}, True)

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
        head = await read_response_head(response_reader(b"HTTP/1.1 200 \r\n\r\n"))
        assert head == (200, {}, True)

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


def test_response_head_reports_whether_the_connection_may_persist() -> None:
    async def scenario() -> None:
        for data, persistent in [
            (b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", True),
            (b"HTTP/1.1 200 OK\r\nConnection: keep-alive, Close\r\n\r\n", False),
            (b"HTTP/1.0 200 OK\r\nContent-Length: 0\r\n\r\n", False),
        ]:
            assert (await read_response_head(response_reader(data))).persistent is persistent
        with pytest.raises(HTTPResponseNotStartedError):
            await read_response_head(response_reader(b""))

    asyncio.run(scenario())


# Reusable private connections over real TLS (CTL-R163 to CTL-R167).


@dataclass(frozen=True, slots=True)
class PrivatePKI:
    ca: Path
    control_plane: tuple[Path, Path]
    renewed_control_plane: tuple[Path, Path]
    agent: tuple[Path, Path]
    foreign_control_plane: tuple[Path, Path]


@pytest.fixture(scope="module")
def private_pki(tmp_path_factory: pytest.TempPathFactory) -> PrivatePKI:
    repository = Path(__file__).resolve().parents[2]
    work = tmp_path_factory.mktemp("private-connections")
    tool = work / "contractor-pki"
    subprocess.run(
        ["go", "build", "-o", str(tool), "./cmd/contractor-pki"],
        cwd=repository,
        check=True,
        capture_output=True,
    )

    def pki(*arguments: str) -> None:
        subprocess.run([str(tool), *arguments], check=True, capture_output=True)

    root, foreign = work / "pki", work / "foreign"
    pki("init-ca", "--root", str(root))
    pki("issue-control-plane", "--root", str(root))
    pki("issue-runtime", "--root", str(root), "--name", "agent-1")
    original = work / "original"
    original.mkdir()
    for name in ("control-plane.crt", "control-plane.key"):
        shutil.copy(root / name, original / name)
    pki("renew-control-plane", "--root", str(root))
    pki("init-ca", "--root", str(foreign))
    pki("issue-control-plane", "--root", str(foreign))
    return PrivatePKI(
        ca=root / "ca.crt",
        control_plane=(original / "control-plane.crt", original / "control-plane.key"),
        renewed_control_plane=(root / "control-plane.crt", root / "control-plane.key"),
        agent=(root / "agents" / "agent-1.crt", root / "agents" / "agent-1.key"),
        foreign_control_plane=(foreign / "control-plane.crt", foreign / "control-plane.key"),
    )


@dataclass(frozen=True, slots=True)
class ServedRequest:
    connection: int
    method: str
    path: str
    headers: dict[str, str]


# A responder returns the raw response, parts of it to send apart (an interim
# response before the final one), or None to close without a response.
type Responder = Callable[[ServedRequest], Awaitable[bytes | list[bytes] | None]]


def keep_alive_echo(request: ServedRequest) -> bytes:
    body = request.path.encode()
    return b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body)


async def echo(request: ServedRequest) -> bytes:
    return keep_alive_echo(request)


class PrivatePeer:
    """A TLS 1.3 HTTP/1.1 peer that presents a switchable certificate.

    It records every connection and request; a responder returns the raw
    response, or None to close the connection without one. A response that
    carries Connection: close ends the connection after it is written.
    """

    def __init__(self, pki: PrivatePKI, responder: Responder = echo) -> None:
        self.contexts = {
            "original": self._context(pki, pki.control_plane),
            "renewed": self._context(pki, pki.renewed_control_plane),
            "agent": self._context(pki, pki.agent),
            "foreign": self._context(pki, pki.foreign_control_plane),
        }
        self.presenting = "original"
        self.responder = responder
        self.accepted = 0
        self.closed = 0
        self.max_open = 0
        self.requests: list[ServedRequest] = []
        self._server: asyncio.Server | None = None

    @staticmethod
    def _context(pki: PrivatePKI, pair: tuple[Path, Path]) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cafile=pki.ca)
        context.load_cert_chain(certfile=pair[0], keyfile=pair[1])
        return context

    @property
    def open(self) -> int:
        return self.accepted - self.closed

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("peer is not started")
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> PrivatePeer:
        listener = self._context_for_listener()
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=listener)
        return self

    def _context_for_listener(self) -> ssl.SSLContext:
        listener = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        listener.minimum_version = ssl.TLSVersion.TLSv1_3

        def select(connection: ssl.SSLObject, _name: str | None, _: ssl.SSLContext) -> None:
            connection.context = self.contexts[self.presenting]

        listener.sni_callback = select  # type: ignore[assignment]
        return listener

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        index = self.accepted
        self.accepted += 1
        self.max_open = max(self.max_open, self.open)
        try:
            while request_line := await reader.readline():
                method, path, _ = request_line.decode().split(" ")
                headers: dict[str, str] = {}
                while (line := await reader.readline()) != b"\r\n":
                    name, _, value = line.decode().partition(":")
                    headers[name.lower()] = value.strip()
                await reader.readexactly(int(headers.get("content-length", "0")))
                request = ServedRequest(index, method, path, headers)
                self.requests.append(request)
                response = await self._respond(reader, request)
                if response is None:
                    return
                parts = [response] if isinstance(response, bytes) else response
                for number, part in enumerate(parts):
                    if number:
                        # Let the client read the previous part on its own.
                        await asyncio.sleep(0.02)
                    writer.write(part)
                    await writer.drain()
                final_head = parts[-1].partition(b"\r\n\r\n")[0] + b"\r\n"
                if b"\r\nConnection: close\r\n" in final_head:
                    return
        except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError):
            return
        finally:
            self.closed += 1
            writer.close()
            with contextlib.suppress(ConnectionError, ssl.SSLError):
                await writer.wait_closed()

    async def _respond(
        self, reader: asyncio.StreamReader, request: ServedRequest
    ) -> bytes | list[bytes] | None:
        """Run the responder while watching for the client to go away."""

        responding = asyncio.ensure_future(self.responder(request))
        watching = asyncio.ensure_future(reader.read(1))
        await asyncio.wait({responding, watching}, return_when=asyncio.FIRST_COMPLETED)
        if not responding.done():
            # The client closed the connection (or pipelined, which the
            # Runtime never does): end it.
            responding.cancel()
            await asyncio.gather(responding, watching, return_exceptions=True)
            return None
        watching.cancel()
        await asyncio.gather(watching, return_exceptions=True)
        if not watching.cancelled():
            return None
        return responding.result()


class VirtualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def eventually(condition: Callable[[], bool], description: str) -> None:
    """Wait until the peer observed an effect; the deadline only bounds failure."""

    try:
        async with asyncio.timeout(5):
            while not condition():
                await asyncio.sleep(0.001)
    except TimeoutError:
        raise AssertionError(description) from None


def runtime_connections(pki: PrivatePKI, clock: VirtualClock | None = None) -> PrivateConnections:
    context = runtime_agent_client_context(
        ca_file=pki.ca, certificate_file=pki.agent[0], private_key_file=pki.agent[1]
    )
    if clock is None:
        return PrivateConnections(context, verify_control_plane_peer)
    return PrivateConnections(context, verify_control_plane_peer, clock=clock)


async def private_call(
    connections: PrivateConnections,
    peer: PrivatePeer,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> bytes:
    request = encode_request(
        method,
        path,
        {"Host": host_header("localhost", peer.port), "Content-Length": "0", **(headers or {})},
        b"",
    )
    response = await connections.exchange(
        "localhost",
        peer.port,
        request,
        idempotent=method == "GET",
        timeout_seconds=2,
        max_response_bytes=1024,
    )
    return response.body


def closing_echo(request: ServedRequest) -> bytes:
    return keep_alive_echo(request).replace(b"\r\n", b"\r\nConnection: close\r\n", 1)


def test_private_connections_reuse_one_verified_connection(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes:
            if request.path.startswith("/fresh/"):
                return closing_echo(request)
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            for index in range(10):
                method = "POST" if index % 2 else "GET"
                assert await private_call(connections, peer, method, f"/r/{index}") == (
                    f"/r/{index}".encode()
                )
            assert peer.accepted == 1
            assert [request.connection for request in peer.requests] == [0] * 10
            assert all("connection" not in request.headers for request in peer.requests)
            # The transports before reuse asked for Connection: close, so each
            # exchange cost one verified connection.
            for index in range(3):
                await private_call(connections, peer, "GET", f"/fresh/{index}")
            assert [request.connection for request in peer.requests[10:]] == [0, 1, 2]
            assert peer.accepted == 3
            # Connection management belongs to the owner, not the caller.
            with pytest.raises(ValueError):
                await private_call(
                    connections, peer, "GET", "/caller-close", headers={"Connection": "close"}
                )

            await connections.aclose()
            await eventually(lambda: peer.open == 0, "aclose left a connection open")
            with pytest.raises(PrivateConnectionsClosedError):
                await private_call(connections, peer, "GET", "/after-close")
            assert peer.accepted == 3
            assert len(peer.requests) == 13
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "response",
    [
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive, close\r\n\r\n{}",
        b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}",
        b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{}",
        # Bytes beyond the framed body would answer the next request.
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
        b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx",
    ],
    ids=["connection-close", "http-1.0", "eof-delimited", "unsolicited-bytes"],
)
def test_private_connections_never_reuse_an_unframed_or_closing_connection(
    private_pki: PrivatePKI, response: bytes
) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes:
            return response if request.path == "/first" else keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            assert await private_call(connections, peer, "GET", "/first") == b"{}"
            assert await private_call(connections, peer, "GET", "/second") == b"/second"
            assert [request.connection for request in peer.requests] == [0, 1]
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


def test_private_connections_reverify_rotated_certificates(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        peer = await PrivatePeer(private_pki).start()
        clock = VirtualClock()
        connections = runtime_connections(private_pki, clock)
        try:
            await private_call(connections, peer, "GET", "/a")
            created = clock.now
            # A reused connection keeps the certificate it was verified with
            # until the idle bound or the lifetime ends.
            peer.presenting = "renewed"
            clock.now += PRIVATE_CONNECTION_IDLE_REUSE_SECONDS - 1
            await private_call(connections, peer, "POST", "/b")
            assert peer.accepted == 1
            while clock.now + PRIVATE_CONNECTION_IDLE_REUSE_SECONDS - 1 < (
                created + PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS
            ):
                clock.now += PRIVATE_CONNECTION_IDLE_REUSE_SECONDS - 1
                await private_call(connections, peer, "GET", "/active")
            assert peer.accepted == 1
            clock.now = created + PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS
            await private_call(connections, peer, "GET", "/after-lifetime")
            assert peer.accepted == 2
            await eventually(lambda: peer.open == 1, "the expired connection stayed open")

            clock.now += PRIVATE_CONNECTION_IDLE_REUSE_SECONDS
            await private_call(connections, peer, "GET", "/after-idle")
            assert peer.accepted == 3

            served = len(peer.requests)
            for presenting in ("agent", "foreign"):
                peer.presenting = presenting
                clock.now += PRIVATE_CONNECTION_IDLE_REUSE_SECONDS
                with pytest.raises(ssl.SSLCertVerificationError):
                    await private_call(connections, peer, "POST", f"/{presenting}")
                assert len(peer.requests) == served
            peer.presenting = "original"
            assert await private_call(connections, peer, "POST", "/valid") == b"/valid"
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


def test_private_connections_bound_concurrent_connections_without_cross_talk(
    private_pki: PrivatePKI,
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def respond(request: ServedRequest) -> bytes:
            await release.wait()
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            calls = [
                asyncio.create_task(private_call(connections, peer, "POST", f"/c/{index}"))
                for index in range(PRIVATE_CONNECTIONS_PER_PEER + 4)
            ]
            await eventually(
                lambda: len(peer.requests) == PRIVATE_CONNECTIONS_PER_PEER,
                "requests did not reach the peer",
            )
            # Give a request beyond the bound every chance to connect.
            await asyncio.sleep(0.05)
            assert peer.accepted == PRIVATE_CONNECTIONS_PER_PEER
            release.set()
            bodies = await asyncio.gather(*calls)
            assert bodies == [f"/c/{index}".encode() for index in range(len(calls))]
            assert peer.max_open <= PRIVATE_CONNECTIONS_PER_PEER
            await eventually(
                lambda: peer.open <= PRIVATE_IDLE_CONNECTIONS_PER_PEER,
                "surplus idle connections stayed open",
            )
        finally:
            await connections.aclose()
            await peer.stop()
        await eventually(lambda: peer.open == 0, "aclose left a connection open")

    asyncio.run(scenario())


def test_private_connections_close_a_cancelled_exchange(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes:
            if request.path == "/slow":
                await asyncio.Event().wait()
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            await private_call(connections, peer, "GET", "/warm")
            slow = asyncio.create_task(private_call(connections, peer, "POST", "/slow"))
            await eventually(lambda: len(peer.requests) == 2, "the slow request did not arrive")
            slow.cancel()
            with pytest.raises(asyncio.CancelledError):
                await slow
            await eventually(lambda: peer.closed == 1, "the cancelled connection stayed open")
            assert await private_call(connections, peer, "GET", "/after") == b"/after"
            assert [(request.connection, request.path) for request in peer.requests] == [
                (0, "/warm"),
                (0, "/slow"),
                (1, "/after"),
            ]
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


def test_private_connections_never_replay_a_write(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        dropped: set[str] = set()

        async def respond(request: ServedRequest) -> bytes | None:
            if request.path.startswith("/drop") and request.path not in dropped:
                dropped.add(request.path)
                return None
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            await private_call(connections, peer, "GET", "/warm")
            with pytest.raises(HTTPResponseNotStartedError):
                await private_call(connections, peer, "POST", "/drop-write")
            await private_call(connections, peer, "GET", "/warm-again")
            # A read that got no response byte on a reused connection is sent
            # once more on a new verified connection.
            assert await private_call(connections, peer, "GET", "/drop-read") == b"/drop-read"
            assert [(request.connection, request.path) for request in peer.requests] == [
                (0, "/warm"),
                (0, "/drop-write"),
                (1, "/warm-again"),
                (1, "/drop-read"),
                (2, "/drop-read"),
            ]
            # A read on a new connection is not repeated.
            await connections.aclose()
            fresh = runtime_connections(private_pki)
            with pytest.raises(HTTPResponseNotStartedError):
                await private_call(fresh, peer, "GET", "/drop-fresh")
            await fresh.aclose()
            assert [request.path for request in peer.requests].count("/drop-fresh") == 1
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


def test_private_connections_bound_the_origins_they_keep(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        peers = [await PrivatePeer(private_pki).start() for _ in range(MAX_PRIVATE_ORIGINS + 1)]
        connections = runtime_connections(private_pki)
        try:
            for peer in peers:
                await private_call(connections, peer, "GET", "/origin")
            await eventually(lambda: peers[0].open == 0, "the oldest origin stayed open")
            assert [peer.open for peer in peers[1:]] == [1] * MAX_PRIVATE_ORIGINS
        finally:
            await connections.aclose()
            for peer in peers:
                await peer.stop()

    asyncio.run(scenario())


INTERIM_EARLY_HINTS = b"HTTP/1.1 103 Early Hints\r\nLink: </style.css>\r\n\r\n"


@pytest.mark.parametrize(
    "interim",
    [
        # The Go Server strips framing from an interim response.
        [INTERIM_EARLY_HINTS],
        # A peer that frames one must not shift the next response.
        [b"HTTP/1.1 103 Early Hints\r\nContent-Length: 0\r\n\r\n"],
        [b"HTTP/1.1 100 Continue\r\n\r\n", INTERIM_EARLY_HINTS],
        [INTERIM_EARLY_HINTS] * MAX_INTERIM_RESPONSES,
    ],
    ids=["unframed-103", "framed-103", "100-then-103", "bounded-interim"],
)
def test_private_connections_skip_interim_responses_and_never_reuse_after_them(
    private_pki: PrivatePKI, interim: list[bytes]
) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes | list[bytes]:
            if request.path == "/first":
                return [*interim, keep_alive_echo(request)]
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            assert await private_call(connections, peer, "GET", "/first") == b"/first"
            assert await private_call(connections, peer, "GET", "/second") == b"/second"
            assert await private_call(connections, peer, "GET", "/third") == b"/third"
            assert [request.connection for request in peer.requests] == [0, 1, 1]
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            [b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: h2c\r\n\r\n"],
            "protocol switch",
        ),
        ([INTERIM_EARLY_HINTS] * (MAX_INTERIM_RESPONSES + 1), "too many interim"),
        ([b"HTTP/1.1 204 No Content\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"], "body"),
        ([b"HTTP/1.1 204 No Content\r\nContent-Length: 2\r\n\r\n{}"], "body"),
        ([b"HTTP/1.1 304 Not Modified\r\nContent-Length: 2\r\n\r\n"], "body"),
    ],
    ids=["101", "too-many-interim", "204-chunked", "204-length", "304-length"],
)
def test_private_connections_reject_switching_and_content_on_empty_responses(
    private_pki: PrivatePKI, response: list[bytes], message: str
) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes | list[bytes]:
            if request.path == "/bad":
                return [*response, keep_alive_echo(request)]
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            await private_call(connections, peer, "GET", "/warm")
            with pytest.raises(HTTPResponseError, match=message):
                await private_call(connections, peer, "GET", "/bad")
            assert await private_call(connections, peer, "GET", "/after") == b"/after"
            assert [request.connection for request in peer.requests] == [0, 0, 1]
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


def test_private_connections_reuse_after_an_empty_response(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        async def respond(request: ServedRequest) -> bytes:
            if request.path == "/empty":
                return b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
            return keep_alive_echo(request)

        peer = await PrivatePeer(private_pki, respond).start()
        connections = runtime_connections(private_pki)
        try:
            assert await private_call(connections, peer, "POST", "/empty") == b""
            assert await private_call(connections, peer, "GET", "/after") == b"/after"
            assert [request.connection for request in peer.requests] == [0, 0]
        finally:
            await connections.aclose()
            await peer.stop()

    asyncio.run(scenario())


class ScriptedConnection:
    """A connection whose response to each request is fed when it is sent."""

    def __init__(self, replies: list[bytes | tuple[bytes, BaseException]]) -> None:
        context = ssl.create_default_context()
        self.reader = ConnectionReader()
        self.writer = Mock(spec=asyncio.StreamWriter)
        self.writer.get_extra_info.return_value = context.wrap_bio(
            ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="localhost"
        )
        self.writer.is_closing.return_value = False
        self.writer.transport = Mock()
        self.writer.wait_closed = AsyncMock()
        self.writer.drain = AsyncMock(side_effect=self._reply)
        self.replies = list(replies)

    async def _reply(self) -> None:
        reply = self.replies.pop(0)
        if isinstance(reply, bytes):
            self.reader.feed_data(reply)
            return
        data, error = reply
        self.reader.feed_data(data)
        self.reader.set_exception(error)


@pytest.mark.parametrize(
    ("received", "resent"), [(b"", True), (b"HTTP/1.1 2", False)], ids=["no-byte", "partial-head"]
)
def test_private_connections_resend_a_read_only_when_no_response_byte_arrived(
    monkeypatch: pytest.MonkeyPatch, received: bytes, resent: bool
) -> None:
    async def scenario() -> None:
        reused = ScriptedConnection(
            [
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}",
                (received, ConnectionResetError("reset by peer")),
            ]
        )
        fresh = ScriptedConnection([b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nfresh"])
        opened = AsyncMock(
            side_effect=[(reused.reader, reused.writer), (fresh.reader, fresh.writer)]
        )
        monkeypatch.setattr(_https, "_open_connection", opened)
        connections = PrivateConnections(ssl.create_default_context(), Mock())
        request = encode_request("GET", "/x", {"Host": "localhost:8443"}, b"")

        async def read() -> bytes:
            response = await connections.exchange(
                "localhost",
                8443,
                request,
                idempotent=True,
                timeout_seconds=1,
                max_response_bytes=16,
            )
            return response.body

        assert await read() == b"{}"
        if resent:
            assert await read() == b"fresh"
        else:
            with pytest.raises(ConnectionResetError):
                await read()
        assert opened.await_count == (2 if resent else 1)
        await connections.aclose()

    asyncio.run(scenario())


def test_private_connections_serve_waiters_of_an_evicted_origin(private_pki: PrivatePKI) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def respond(request: ServedRequest) -> bytes:
            if request.path.startswith("/held/"):
                await release.wait()
            return keep_alive_echo(request)

        busy = await PrivatePeer(private_pki, respond).start()
        others = [await PrivatePeer(private_pki).start() for _ in range(MAX_PRIVATE_ORIGINS)]
        connections = runtime_connections(private_pki)
        try:
            held = [
                asyncio.create_task(private_call(connections, busy, "GET", f"/held/{index}"))
                for index in range(PRIVATE_CONNECTIONS_PER_PEER)
            ]
            await eventually(
                lambda: len(busy.requests) == PRIVATE_CONNECTIONS_PER_PEER,
                "held requests did not arrive",
            )
            waiting = asyncio.create_task(private_call(connections, busy, "GET", "/waiting"))
            await asyncio.sleep(0)
            # Four more origins evict the busy one while a request waits on it.
            for peer in others:
                await private_call(connections, peer, "GET", "/other")
            release.set()
            assert await waiting == b"/waiting"
            await asyncio.gather(*held)
            # The evicted origin keeps nothing for reuse.
            await eventually(lambda: busy.open == 0, "the evicted origin kept a connection")
        finally:
            release.set()
            await connections.aclose()
            await busy.stop()
            for peer in others:
                await peer.stop()

    asyncio.run(scenario())


def test_shared_private_connection_reuse_limits() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "api"
            / "testdata"
            / "v1alpha1"
            / "private-connection-reuse.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["listenerIdleTimeoutSeconds"] == PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS
    assert fixture["clientIdleReuseSeconds"] == PRIVATE_CONNECTION_IDLE_REUSE_SECONDS
    assert fixture["reuseLifetimeSeconds"] == PRIVATE_CONNECTION_REUSE_LIFETIME_SECONDS
    assert fixture["connectionsPerPeer"] == PRIVATE_CONNECTIONS_PER_PEER
    assert fixture["idleConnectionsPerPeer"] == PRIVATE_IDLE_CONNECTIONS_PER_PEER
    # A client gives up an idle connection before the listener closes it.
    assert PRIVATE_CONNECTION_IDLE_REUSE_SECONDS < PRIVATE_LISTENER_IDLE_TIMEOUT_SECONDS
