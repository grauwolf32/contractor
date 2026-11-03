"""Independent real-socket checks; process tests own durable receipt evidence."""

from __future__ import annotations

import asyncio
import gzip
import time

import pytest
from fakes.engagement_socket import (
    SocketTarget,
    launch,
    preflight,
    resolver_for,
    socket_spec,
    tls_contexts,
)

from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.contracts.engagement import HTTPLimits


def run(scenario):

    async def bounded():
        async with asyncio.timeout(20):
            await scenario()

    asyncio.run(bounded())


@pytest.mark.parametrize(
    "forbidden",
    [
        "127.0.0.1",
        "169.254.169.254",
        "100.100.100.200",
        "0.0.0.0",
        "224.0.0.1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
    ],
    ids=[
        "loopback",
        "link_local_metadata",
        "cloud_metadata",
        "unspecified",
        "multicast",
        "mapped_loopback",
        "translated_loopback",
    ],
)
def test_socket_mixed_dns_answers_never_write_to_the_permitted_candidate(forbidden):

    async def scenario():
        async with SocketTarget() as target:
            config = resolver_for(target, [target.address, forbidden])
            async with preflight(target, config=config) as (resource, _):
                assert (await launch(resource)).kind in {"denied", "failed"}
        assert target.written == b""

    run(scenario)


def test_socket_dns_pin_has_no_implicit_second_resolution():

    async def scenario():
        async with SocketTarget() as target:
            calls = []
            answers = iter(([target.address], ["127.0.0.1"]))
            config = resolver_for(target, lambda: next(answers), calls=calls)
            async with preflight(target, config=config) as (resource, _):
                assert (await launch(resource)).kind == "completed"
                await resource.close()
                await target.all_closed.wait()
            assert calls.count(("target.example", target.port)) == 1
            assert len(target.requests) == 1

    run(scenario)


@pytest.mark.parametrize("protected", ["hostname", "address_port"])
def test_socket_protected_virtual_host_never_receives_request_bytes(protected):

    async def scenario():
        async with SocketTarget() as target:
            name = "target.example" if protected == "hostname" else "infrastructure.example"

            def change(projection):
                projection["http"]["protectedDestinations"] = [
                    {"hostname": name, "port": target.port, "addresses": [target.address]}
                ]

            config = resolver_for(target, protected={name: [target.address]})
            async with preflight(
                target, spec=socket_spec(target, change=change), config=config
            ) as (resource, _):
                assert (await launch(resource)).kind in {"denied", "failed"}
        assert target.written == b""

    run(scenario)


def test_socket_actual_peer_port_mismatch_closes_before_any_request_byte(monkeypatch):
    original = asyncio.open_connection

    async def scenario():
        async with SocketTarget() as approved, SocketTarget() as other:

            async def misroute(host, port, **options):
                return await original(
                    host, other.port if port == approved.port else port, **options
                )

            monkeypatch.setattr(asyncio, "open_connection", misroute)
            async with preflight(approved) as (resource, _):
                assert (await launch(resource)).kind in {"denied", "failed"}
        assert approved.written == other.written == b""

    run(scenario)


@pytest.mark.parametrize("trusted", [True, False], ids=["trusted", "untrusted"])
def test_socket_tls_preserves_sni_host_and_requires_trust(tmp_path, trusted):

    async def scenario():
        server, client, names = tls_contexts(tmp_path)
        async with SocketTarget(ssl_context=server) as target:
            options = {"ssl_context": client} if trusted else {}
            async with preflight(target, **options) as (resource, _):
                result = await launch(resource)
                if trusted:
                    assert result.kind == "completed"
                    await resource.close()
                    await target.all_closed.wait()
                    assert f"Host: target.example:{target.port}\r\n".encode() in target.written
                    assert names == ["target.example"]
                else:
                    assert result.kind == "failed"
                    assert names == ["target.example", "target.example"]
        if not trusted:
            assert target.written == b""

    run(scenario)


def test_socket_maximum_canonical_url_and_proxy_environment_are_not_reinterpreted(monkeypatch):

    async def scenario():
        async with SocketTarget() as target, SocketTarget() as proxy:
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
                monkeypatch.setenv(name, f"http://{proxy.address}:{proxy.port}")
            path = "/allowed;parameter?" + "x" * (
                8192 - len(target.origin) - len("/allowed;parameter?")
            )
            async with preflight(target, spec=socket_spec(target, path=path)) as (resource, _):
                assert (await launch(resource, url=target.origin + path)).kind == "completed"
                await resource.close()
                await target.all_closed.wait()
            assert target.requests[0].split(b"\r\n", 1)[0] == f"GET {path} HTTP/1.1".encode()
        assert proxy.written == b""

    run(scenario)


@pytest.mark.parametrize(
    "case", ["64_headers", "65_headers", "32768_bytes", "32769_bytes", "interim_header_overflow"]
)
def test_socket_response_header_limits_include_informational_responses(case):
    if "headers" in case:
        count = int(case.split("_")[0])
        response = b"HTTP/1.1 200 OK\r\n" + b"X-Bound: x\r\n" * (count - 1)
        response += b"Content-Length: 0\r\n\r\n"
    elif "bytes" in case:
        maximum = int(case.split("_")[0])
        start = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nX-Bound: "
        response = start + b"x" * (maximum - len(start) - 4) + b"\r\n\r\n"
        assert len(response) == maximum
    else:
        response = b"HTTP/1.1 100 Continue\r\n" + b"X-Bound: x\r\n" * 64 + b"\r\n"
        response += b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

    async def scenario():

        async def respond(_, __, writer, ___):
            writer.write(response)
            await writer.drain()

        async with SocketTarget(respond) as target:
            async with preflight(target) as (resource, _):
                result = await launch(resource)
                if case in {"64_headers", "32768_bytes"}:
                    assert result.kind == "completed"
                else:
                    assert result.kind == "failed"
                await resource.close()
                await target.all_closed.wait()
            assert len(target.requests) == 1

    run(scenario)


@pytest.mark.parametrize("case", ["exact_maximum", "over_maximum", "lower_cap", "compressed"])
def test_socket_response_body_is_bounded_discarded_and_not_decompressed(case):
    limit = 13 if case == "lower_cap" else 1 << 20
    payload = b"body-canary" * ((limit + 11) // 11)
    payload = payload[:limit] if case == "exact_maximum" else payload[:limit] + b"x"
    if case == "compressed":
        payload = gzip.compress(payload)

    async def scenario():

        async def respond(_, __, writer, ___):
            headers = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n"
            if case == "compressed":
                headers += b"Content-Encoding: gzip\r\n"
            writer.write(headers + b"\r\n" + payload)
            await writer.drain()

        async with (
            SocketTarget(respond) as target,
            preflight(target, limits=HTTPLimits(responseBodyBytes=limit)) as (
                resource,
                _,
            ),
        ):
            resource.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await resource.request(target.origin + "/allowed")
            assert outcome.kind == "completed"
            await resource.close()
            await target.all_closed.wait()
            assert len(response.body) == min(limit, len(payload))
            assert response.truncated == (len(payload) > limit)

    run(scenario)


def test_socket_cancel_joins_a_stalled_response_without_retry():

    async def scenario():

        async def respond(_, __, writer, ___):
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial")
            await writer.drain()

        async with SocketTarget(respond) as target, preflight(target) as (resource, _):
            task = launch(resource)
            await target.request_received.wait()
            await resource.close()
            assert task.done() and task.cancelled()
            await resource.close()
            await target.all_closed.wait()

    run(scenario)


@pytest.mark.parametrize(
    "location",
    ["/api/a/../secret", "../secret", "/api//secret"],
    ids=["absolute_dot", "relative_dot", "repeated_separator"],
)
def test_socket_redirect_rejects_raw_ambiguous_paths_before_resolution(location):

    async def scenario():

        async def respond(target, _, writer, __):
            if len(target.requests) == 1:
                writer.write(
                    b"HTTP/1.1 302 Found\r\nContent-Length: 0\r\nLocation: "
                    + location.encode()
                    + b"\r\n\r\n"
                )
            else:
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()

        async with SocketTarget(respond) as target, preflight(target) as (resource, _):
            assert (await launch(resource)).kind == "completed"
            await resource.close()
            await target.all_closed.wait()
            assert len(target.requests) == 1, [
                value.split(b"\r\n", 1)[0] for value in target.requests
            ]

    run(scenario)


def test_socket_failed_close_keeps_ownership_until_the_real_stream_joins(monkeypatch):
    from contractor_runtime.adapters.http_engine import _Writer

    permitted = False

    def close(writer):
        if not permitted:
            raise RuntimeError("injected close refusal")
        writer.inner.close()

    monkeypatch.setattr(_Writer, "close", close, raising=False)

    async def scenario():
        nonlocal permitted
        async with SocketTarget() as target, preflight(target) as (resource, _):
            assert (await launch(resource)).kind == "completed"
            with pytest.raises(RuntimeError, match="injected close refusal"):
                await resource.close()
            assert resource._engine is not None
            assert any(pool.writers for pool in resource._engine.pools.values())
            with pytest.raises(HTTPTransportError):
                await launch(resource)
            permitted = True
            # The first join rethrows the failed native handler after closing
            # the retained writer; the following close confirms passive state.
            try:
                await resource.close()
            except RuntimeError as error:
                assert str(error) == "injected close refusal"
            await resource.close()
            await target.all_closed.wait()

    run(scenario)
