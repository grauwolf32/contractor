"""Native HTTP/1.1 chunk framing bounds, exercised through real sockets."""

import asyncio
import contextlib
import time
from urllib.parse import urlsplit

import pytest
from fakes.engagement_socket import SocketTarget, preflight
from test_http_proxy_bridge import Owner

from contractor_runtime.adapters.bridge import HTTPProxyBridge
from contractor_runtime.contracts.engagement import HTTPLimits
from contractor_runtime.core.http_limits import MAX_CONTROLLED_HEADER_BYTES

VALID = b"2\r\nok\r\n0\r\nX-Proof: yes\r\nSet-Cookie: ignored=value\r\n\r\n"
LONG = b"a" * (1 << 20)
CASES = [
    ("trailers", VALID),
    ("size", LONG),
    ("extension", b"2;" + LONG),
    ("complete-extension", b"2;" + LONG + b"\r\nok\r\n0\r\n\r\n"),
    ("incomplete-trailer", b"2\r\nok\r\n0\r\nX-Proof: " + LONG),
    ("complete-trailer", b"2\r\nok\r\n0\r\nX-Proof: " + LONG + b"\r\n\r\n"),
    ("many-trailers", b"2\r\nok\r\n0\r\n" + b"X-Proof: v\r\n" * 65 + b"\r\n"),
    ("invalid-size", b"Z\r\n"),
    ("invalid-trailer", b"2\r\nok\r\n0\r\nno-colon\r\n\r\n"),
]


def observe_buffer(monkeypatch):
    from h11._readers import ChunkedReader

    sizes = []
    original = ChunkedReader.__call__

    def observe(reader, buffer):
        sizes.append(len(buffer))
        return original(reader, buffer)

    monkeypatch.setattr(ChunkedReader, "__call__", observe)
    return sizes


def run(scenario):
    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


@pytest.mark.parametrize("name,payload", CASES, ids=[name for name, _ in CASES])
def test_target_chunk_framing_is_bounded_and_trailers_complete(monkeypatch, name, payload):
    sizes = observe_buffer(monkeypatch)

    async def respond(_, reader, writer, __):
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + payload)
        try:
            await writer.drain()
            await reader.read()
        except OSError:
            pass

    async def scenario():
        async with (
            SocketTarget(respond) as target,
            preflight(target, limits=HTTPLimits(responseBodyBytes=3)) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await owner.request(target.origin + "/allowed", timeout=2)
            if name == "trailers":
                assert outcome.kind == "completed" and response.body == b"ok"
                assert not response.truncated
                assert not any(
                    k.lower() in {b"x-proof", b"set-cookie"} for k, _ in response.headers
                )
            else:
                assert outcome.kind == "failed" and outcome.reason == "response_invalid"
                assert response is None
            # The parser can receive one bounded native I/O chunk beyond its
            # framing allowance; it must fail before retaining another one.
            assert sizes and max(sizes) <= MAX_CONTROLLED_HEADER_BYTES + 65536
            assert not owner._engine.flows

    run(scenario)


@pytest.mark.parametrize("name,payload", CASES, ids=[name for name, _ in CASES])
def test_scanner_request_chunk_framing_is_bounded(monkeypatch, name, payload):
    sizes = observe_buffer(monkeypatch)

    async def scenario():
        owner = Owner()
        try:
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", urlsplit(bridge.proxy_url).port
                )
                try:
                    writer.write(
                        b"POST http://nonresolving.invalid/allowed HTTP/1.1\r\n"
                        b"Host: nonresolving.invalid\r\nTransfer-Encoding: chunked\r\n"
                        b"Proxy-Authorization: " + bridge._authorization + b"\r\n\r\n" + payload
                    )
                    await writer.drain()
                    reply = await reader.read()
                except OSError:
                    reply = b""
                finally:
                    writer.close()
                    with contextlib.suppress(OSError):
                        await writer.wait_closed()
                if name == "trailers":
                    assert reply.startswith(b"HTTP/1.1 200") and bridge.error_code is None
                    assert len(owner.calls) == 1 and owner.calls[0][1]["body"] == b"ok"
                    assert not any(
                        k.lower() in {b"x-proof", b"set-cookie"}
                        for k, _ in owner.calls[0][1]["headers"]
                    )
                else:
                    assert not owner.calls and not reply.startswith(b"HTTP/1.1 200")
                assert sizes and max(sizes) <= MAX_CONTROLLED_HEADER_BYTES + 65536
        finally:
            if owner.engine:
                await owner.engine.close()

    run(scenario)
