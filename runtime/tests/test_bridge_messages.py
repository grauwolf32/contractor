"""Raw native message conversion regressions on real sockets and scanner clients."""

import asyncio
import gzip
import ipaddress
import time
from types import SimpleNamespace

import httpx
import pytest
from fakes.engagement_socket import SocketTarget, preflight, tls_contexts
from fakes.http_transport import HTTPTransport
from fakes.native_http import NativeTarget
from test_http_proxy_bridge import Owner

from contractor_runtime.adapters.bridge import HTTPProxyBridge
from contractor_runtime.adapters.http_session import PrivateResponse
from contractor_runtime.core.target_policy import TargetPolicyConfig


def run(scenario):
    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
def test_native_and_scanner_preserve_encoded_bytes(tmp_path, http2):
    body = gzip.compress(b"expected-plaintext")

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with (
            NativeTarget(
                ssl_context=server, payload=body, headers=[(b"content-encoding", b"gzip")]
            ) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await owner.request(
                target.origin + "/allowed;p?",
                method="POST",
                body=body,
                headers=[(b"Content-Encoding", b"gzip")],
            )
            assert outcome.kind == "completed" and response.body == body
            assert target.bodies == [body]
            assert target.trace[0][1] == b"/allowed;p?"
            async with (
                HTTPProxyBridge(
                    owner, credential_id=None, deadline=time.monotonic() + 10
                ) as bridge,
                httpx.AsyncClient(
                    proxy=bridge.proxy_url, trust_env=False, verify=False, http2=http2
                ) as client,
            ):
                response = await client.post(
                    target.origin + "/allowed",
                    content=body,
                    headers={"Content-Encoding": "gzip"},
                )
                assert response.content == b"expected-plaintext"
                assert int(response.headers["content-length"]) == len(body)
            assert target.bodies == [body, body]
            assert {row[2] for row in target.trace} == {b"HTTP/2" if http2 else b"HTTP/1.1"}

    run(scenario)


def test_native_preserves_empty_query_delimiter():
    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, _ = await owner.request(target.origin + "/allowed?")
            assert outcome.kind == "completed"
            assert target.requests[0].split(b"\r\n", 1)[0] == b"GET /allowed? HTTP/1.1"

    run(scenario)


@pytest.mark.parametrize("port", [80, 8080])
def test_native_ipv6_components_pass_equality_before_connect(monkeypatch, port):
    from fakes.engagement_socket import socket_spec
    from mitmproxy.net.http import http1

    from contractor_runtime.adapters.bridge.mitmproxy.engine import (
        MitmproxySession as MitmproxyEngine,
    )

    origin = "http://[fd00::1]" + (f":{port}" if port != 80 else "")
    spec = socket_spec(SimpleNamespace(origin=origin, address="10.20.0.8"))
    spec.engagement.http.routes[0].address_ranges = ["fd00::/64"]
    original = MitmproxyEngine.handle_hook
    checks = []

    async def observe(engine, pool, hook):
        await original(engine, pool, hook)
        if hook.name == "requestheaders":
            flow = hook.args()[0]
            checks.append((engine.flows[flow.id].reason, http1.assemble_request_head(flow.request)))
            flow.kill()  # This probe checks the native boundary, not an unowned IPv6 target.

    async def resolver(host, _):
        return (ipaddress.ip_address("fd00::1" if host == "fd00::1" else "10.30.0.1"),)

    async def scenario():
        monkeypatch.setattr(MitmproxyEngine, "handle_hook", observe)
        owner = HTTPTransport(spec, resolver=TargetPolicyConfig(resolver=resolver).resolver)
        owner.configure(authority_deadline=lambda: time.monotonic() + 60)
        try:
            await owner.request(origin + "/allowed?")
            assert checks and checks[0][0] == ""
            assert checks[0][1].startswith(b"GET /allowed? HTTP/1.1\r\n")
            assert (b"Host: " + origin.removeprefix("http://").encode()) in checks[0][1]
        finally:
            await owner.close()

    run(scenario)


@pytest.mark.parametrize(
    "status,method,length", [(200, "HEAD", "123"), (304, "GET", "123"), (204, "GET", None)]
)
def test_scanner_response_keeps_bodyless_framing(status, method, length):
    class BodylessOwner(Owner):
        async def request(self, url, **options):
            outcome, response = await super().request(url, **options)
            response.status, response.body = status, b""
            response.headers = [(b"content-length", b"123")] if length else []
            return outcome, response

    async def scenario():
        owner = BodylessOwner()
        try:
            async with (
                HTTPProxyBridge(
                    owner, credential_id=None, deadline=time.monotonic() + 10
                ) as bridge,
                httpx.AsyncClient(proxy=bridge.proxy_url, trust_env=False) as client,
            ):
                response = await client.request(method, "http://nonresolving.invalid/allowed")
                assert response.status_code == status and response.content == b""
                assert response.headers.get("content-length") == length
        finally:
            if owner.engine:
                await owner.runtime_engine.close()

    run(scenario)


def test_scanner_gzip_response_is_decoded_once():
    body = gzip.compress(b"expected-plaintext")

    class GzipOwner(Owner):
        async def request(self, url, **options):
            outcome, _ = await super().request(url, **options)
            return outcome, PrivateResponse(url, 200, [(b"content-encoding", b"gzip")], body)

    async def scenario():
        owner = GzipOwner()
        try:
            async with (
                HTTPProxyBridge(
                    owner, credential_id=None, deadline=time.monotonic() + 10
                ) as bridge,
                httpx.AsyncClient(proxy=bridge.proxy_url, trust_env=False) as client,
            ):
                response = await client.get("http://nonresolving.invalid/allowed")
                assert response.content == b"expected-plaintext"
                assert int(response.headers["content-length"]) == len(body)
        finally:
            if owner.engine:
                await owner.runtime_engine.close()

    run(scenario)


def test_native_component_tampering_is_denied_before_target_bytes(monkeypatch):
    from contractor_runtime.adapters.bridge.mitmproxy.engine import (
        MitmproxySession as MitmproxyEngine,
    )

    original = MitmproxyEngine.handle_hook

    async def change(engine, pool, hook):
        if hook.name == "requestheaders":
            hook.args()[0].request.path = "/unauthorized"
        await original(engine, pool, hook)

    async def scenario():
        monkeypatch.setattr(MitmproxyEngine, "handle_hook", change)
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, _ = await owner.request(target.origin + "/allowed")
            assert outcome.reason == "policy_denied" and target.written == b""

    run(scenario)


def test_ingress_preserves_absolute_empty_query_and_rejects_http10():
    from urllib.parse import urlsplit

    async def scenario():
        owner = Owner()
        try:
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 10
            ) as bridge:
                for version in (b"HTTP/1.1", b"HTTP/1.0"):
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1", urlsplit(bridge.proxy_url).port
                    )
                    writer.write(
                        b"GET http://nonresolving.invalid/allowed;p? "
                        + version
                        + b"\r\nHost: nonresolving.invalid\r\nProxy-Authorization: "
                        + bridge._authorization
                        + b"\r\n\r\n"
                    )
                    await writer.drain()
                    await reader.read()
                    writer.close()
                    await writer.wait_closed()
                assert len(owner.calls) == 1
                assert owner.calls[0][0] == "http://nonresolving.invalid/allowed;p?"
        finally:
            if owner.engine:
                await owner.runtime_engine.close()

    run(scenario)
