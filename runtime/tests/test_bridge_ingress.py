"""Original scanner URL admission through a real bridge, owner and target."""

import asyncio
import ssl
import time
from contextlib import suppress

import h2.config
import h2.connection
import h2.events
import pytest
from fakes.engagement_socket import SocketTarget, preflight, tls_contexts

from contractor_runtime.adapters.bridge import HTTPProxyBridge


def run(scenario):
    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


async def raw_request(bridge, request, partial=None):
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", bridge._server.sockets[0].getsockname()[1]
    )
    try:
        if partial is not None:
            writer.write(request[:20])
            await writer.drain()
            await partial.wait()
            request = request[20:]
        writer.write(request)
        await writer.drain()
        return await reader.read()
    except OSError:
        return b""
    finally:
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()


@pytest.mark.parametrize(
    "case",
    [
        "fragment",
        "query-fragment",
        "zero-port",
        "plus-port",
        "host-case",
        "dot",
        "encoded-path",
        "unicode-query",
    ],
)
@pytest.mark.parametrize("split", [False, True], ids=["whole", "fragmented"])
def test_scanner_rejects_original_absolute_target_before_normalization(monkeypatch, case, split):
    from contractor_runtime.adapters.bridge.mitmproxy.ingress import IngressHttp1

    async def scenario():
        partial = asyncio.Event()
        original = IngressHttp1.read_headers

        def observe(parser, event):
            if b"\r\n\r\n" not in bytes(parser.buf):
                partial.set()
            yield from original(parser, event)

        monkeypatch.setattr(IngressHttp1, "read_headers", observe)
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            targets = {
                "fragment": target.origin + "/allowed#forbidden",
                "query-fragment": target.origin + "/allowed?x=1#forbidden",
                "zero-port": target.origin.replace("://target.example:", "://target.example:0")
                + "/allowed",
                "plus-port": target.origin.replace("://target.example:", "://target.example:+")
                + "/allowed",
                "host-case": target.origin.replace("target.example", "TARGET.example") + "/allowed",
                "dot": target.origin + "/allowed/../allowed",
                "encoded-path": target.origin + "/%61llowed",
                "unicode-query": target.origin + "/allowed?x=π",
            }
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reply = await raw_request(
                    bridge,
                    b"GET "
                    + targets[case].encode()
                    + b" HTTP/1.1\r\nHost: target.example:"
                    + str(target.port).encode()
                    + b"\r\nProxy-Authorization: "
                    + bridge._authorization
                    + b"\r\n\r\n",
                    partial if split else None,
                )
                assert not reply.startswith(b"HTTP/1.1 200")
                assert (
                    target.connections == [] and target.written == b"" and owner._request_count == 0
                )
            assert bridge.error_code == "scan_request_failed"
            assert bridge._exception is None

    run(scenario)


@pytest.mark.parametrize("path", ["/allowed;p?", "/allowed?x=%23literal", "/allowed%23literal"])
def test_scanner_forwards_valid_absolute_target_exactly(path):
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reply = await raw_request(
                    bridge,
                    b"GET "
                    + (target.origin + path).encode()
                    + b" HTTP/1.1\r\nHost: target.example:"
                    + str(target.port).encode()
                    + b"\r\nProxy-Authorization: "
                    + bridge._authorization
                    + b"\r\n\r\n",
                )
                assert reply.startswith(b"HTTP/1.1 200")
                assert (
                    target.requests[0].split(b"\r\n", 1)[0]
                    == b"GET " + path.encode() + b" HTTP/1.1"
                )
            assert bridge.error_code is None

    run(scenario)


@pytest.mark.parametrize(
    "protocol,case",
    [("http11", "fragment"), ("h2", "fragment"), ("h2", "zero-port"), ("h2", "valid")],
)
def test_scanner_tunnel_target_siblings(tmp_path, protocol, case):
    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        async with (
            SocketTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", bridge._server.sockets[0].getsockname()[1]
                )
                try:
                    authority = f"target.example:{target.port}".encode()
                    writer.write(
                        b"CONNECT "
                        + authority
                        + b" HTTP/1.1\r\nHost: "
                        + authority
                        + b"\r\nProxy-Authorization: "
                        + bridge._authorization
                        + b"\r\n\r\n"
                    )
                    await writer.drain()
                    assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
                    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    tls.check_hostname = False
                    tls.verify_mode = ssl.CERT_NONE
                    tls.set_alpn_protocols(["h2" if protocol == "h2" else "http/1.1"])
                    await writer.start_tls(tls, server_hostname="target.example")
                    path = b"/allowed;p?" if case == "valid" else b"/allowed#forbidden"
                    if protocol == "h2":
                        connection = h2.connection.H2Connection(
                            h2.config.H2Configuration(client_side=True)
                        )
                        connection.initiate_connection()
                        connection.send_headers(
                            1,
                            [
                                (b":method", b"GET"),
                                (b":scheme", b"https"),
                                (
                                    b":authority",
                                    f"target.example:0{target.port}".encode()
                                    if case == "zero-port"
                                    else authority,
                                ),
                                (b":path", b"/allowed" if case == "zero-port" else path),
                            ],
                            end_stream=True,
                        )
                        writer.write(connection.data_to_send())
                        await writer.drain()
                        while data := await reader.read(65536):
                            events = connection.receive_data(data)
                            if any(
                                isinstance(
                                    event,
                                    (
                                        h2.events.StreamEnded,
                                        h2.events.StreamReset,
                                        h2.events.ConnectionTerminated,
                                    ),
                                )
                                for event in events
                            ):
                                break
                    else:
                        writer.write(
                            b"GET " + path + b" HTTP/1.1\r\nHost: " + authority + b"\r\n\r\n"
                        )
                        await writer.drain()
                        await reader.read()
                finally:
                    writer.close()
                    with suppress(OSError):
                        await writer.wait_closed()
                if case == "valid":
                    assert target.requests[0].startswith(b"GET /allowed;p? HTTP/1.1\r\n")
                else:
                    assert not target.connections and not owner._request_count
            assert bridge._exception is None
            assert bridge.error_code == (None if case == "valid" else "scan_request_failed")

    run(scenario)


def test_scanner_connect_rejects_original_authority():
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                authority = f"target.example:0{target.port}".encode()
                reply = await raw_request(
                    bridge,
                    b"CONNECT "
                    + authority
                    + b" HTTP/1.1\r\nHost: "
                    + authority
                    + b"\r\nProxy-Authorization: "
                    + bridge._authorization
                    + b"\r\n\r\n",
                )
                assert not reply.startswith(b"HTTP/1.1 200") and not target.connections
            assert bridge.error_code == "scan_request_failed" and bridge._exception is None

    run(scenario)


@pytest.mark.parametrize("separator", [b"  ", b"\t"], ids=["repeated-space", "tab"])
@pytest.mark.parametrize("ending", [b"\r\n", b"\n"], ids=["crlf", "lf"])
def test_scanner_request_line_whitespace_cannot_skip_source_admission(separator, ending):
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            raw = target.origin.replace("://target.example:", "://target.example:0") + "/allowed"
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reply = await raw_request(
                    bridge,
                    b"GET"
                    + separator
                    + raw.encode()
                    + b" HTTP/1.1"
                    + ending
                    + b"Host: target.example:"
                    + str(target.port).encode()
                    + ending
                    + b"Proxy-Authorization: "
                    + bridge._authorization
                    + ending
                    + ending,
                )
                assert not reply.startswith(b"HTTP/1.1 200")
                assert not target.connections and not owner._request_count
            assert bridge.error_code == "scan_request_failed" and bridge._exception is None

    run(scenario)


def test_scanner_lf_headers_preserve_an_admitted_empty_query():
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                reply = await raw_request(
                    bridge,
                    b"GET "
                    + (target.origin + "/allowed;p?").encode()
                    + b" HTTP/1.1\nHost: target.example:"
                    + str(target.port).encode()
                    + b"\nProxy-Authorization: "
                    + bridge._authorization
                    + b"\n\n",
                )
                assert reply.startswith(b"HTTP/1.1 200") and bridge.error_code is None
                assert target.requests[0].startswith(b"GET /allowed;p? HTTP/1.1\r\n")

    run(scenario)
