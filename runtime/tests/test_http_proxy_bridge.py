"""Real HTTP/TLS IPC delegates to its owner and never opens a target tunnel."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from shared_config_defaults import MAX_REQUEST_BODY_BYTES

from contractor_runtime.adapters.bridge import HTTPProxyBridge
from contractor_runtime.adapters.http_session import PrivateResponse
from contractor_runtime.adapters.http_transport import HTTPOutcome


class Owner:
    def __init__(self):
        self.calls = []
        self.engine = None

    async def open_proxy(self):
        from contractor_runtime.adapters.bridge.mitmproxy.engine import MitmproxyEngine

        self.engine = MitmproxyEngine(require_authority=lambda: None)
        await self.engine.start()
        return self.engine

    def credentials(self):
        return []

    def private_values(self):
        return ("private-assigned-credential-canary",)

    async def request(self, url, **options):
        self.calls.append((url, options))
        return HTTPOutcome("completed"), PrivateResponse(
            url,
            200,
            [
                (b"content-type", b"text/plain"),
                (b"set-cookie", b"learned=native-jar; Path=/"),
                (b"connection", b"x-local-hop"),
                (b"x-local-hop", b"private-hop-value"),
            ],
            b"from-owner",
        )


def test_failed_invocation_refuses_a_scanner_retry_on_a_new_proxy_connection():
    async def scenario():
        owner = Owner()

        async def fail(url, **options):
            owner.calls.append((url, options))
            return HTTPOutcome("failed", reason="transport_failed"), None

        owner.request = fail
        try:
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 15
            ) as bridge:
                for _ in range(2):
                    # Keep the process alive deliberately: stopping the scanner
                    # eventually must not be the target-authority fence.
                    async with httpx.AsyncClient(proxy=bridge.proxy_url, trust_env=False) as client:
                        with pytest.raises(httpx.HTTPError):
                            await client.get("http://nonresolving.invalid/allowed")
                assert len(owner.calls) == 1
                assert bridge.error_code == "scan_request_failed"
        finally:
            if owner.engine is not None:
                await owner.engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("expired", [False, True], ids=["request-budget", "scan-deadline"])
@pytest.mark.parametrize("failure", ["timeout", "connect_failed", "denied"])
def test_bridge_classifies_delegated_timeout_by_scan_deadline(monkeypatch, expired, failure):
    from contractor_runtime.adapters.bridge import proxy as module

    async def scenario():
        now = time.monotonic()
        clock = SimpleNamespace(monotonic=lambda: now)
        monkeypatch.setattr(module, "time", clock)
        owner = Owner()

        async def request(*_args, **_options):
            # Settle the inner HTTP outcome before the outer scanner timer fires.
            clock.monotonic = lambda: now + (16 if expired else 1)
            return HTTPOutcome("denied" if failure == "denied" else "failed", reason=failure), None

        owner.request = request
        try:
            async with (
                HTTPProxyBridge(owner, credential_id=None, deadline=now + 15) as bridge,
                httpx.AsyncClient(proxy=bridge.proxy_url, trust_env=False) as client,
            ):
                with pytest.raises(httpx.HTTPError):
                    await client.get("http://nonresolving.invalid/allowed")
            expected = (
                "scan_target_denied"
                if failure == "denied"
                else "scan_timeout"
                if expired and failure == "timeout"
                else "scan_request_failed"
            )
            assert bridge.error_code == expected
            assert not bridge._tasks and not bridge._writers
        finally:
            if owner.engine is not None:
                await owner.engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("expired", [False, True], ids=["request-budget", "scan-deadline"])
@pytest.mark.parametrize("failure", [TimeoutError, OSError])
def test_bridge_classifies_ipc_timeout_and_joins_handler(monkeypatch, expired, failure):
    from contractor_runtime.adapters.bridge import proxy as module

    async def scenario():
        now = time.monotonic()
        clock = SimpleNamespace(monotonic=lambda: now)
        monkeypatch.setattr(module, "time", clock)
        bridge = HTTPProxyBridge(Owner(), credential_id=None, deadline=now + 15)
        joined = False

        class Handler:
            received = True
            pending = frozenset({"flow"})
            completed = 0

            async def handle_client(self):
                clock.monotonic = lambda: now + (16 if expired else 1)
                raise failure("injected IPC failure")

            async def join(self):
                nonlocal joined
                joined = True

        handler = Handler()
        writer = object()
        bridge._writers.add(writer)
        await bridge._handle(handler, writer)
        assert bridge.error_code == (
            "scan_timeout" if expired and failure is TimeoutError else "scan_request_failed"
        )
        assert joined and not bridge._writers

    asyncio.run(scenario())


@pytest.mark.parametrize("scheme,http2", [("http", False), ("https", False), ("https", True)])
def test_bridge_delegates_plaintext_and_tls_without_target_resolution(
    tmp_path, scheme, http2, monkeypatch
):
    from mitmproxy.net import tls

    writes = []
    monkeypatch.setattr(tls, "log_master_secret", lambda *_: writes.append(True))

    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, credential_id=None, deadline=time.monotonic() + 15
        ) as bridge:
            proxy = bridge.proxy_url
            async with httpx.AsyncClient(
                proxy=proxy, verify=False, trust_env=False, http2=http2
            ) as client:
                response = await client.post(
                    scheme + "://nonresolving.invalid/allowed", content=b"payload"
                )
            assert response.status_code == 200 and response.content == b"from-owner"
            assert response.http_version == ("HTTP/2" if http2 else "HTTP/1.1")
            assert "x-local-hop" not in response.headers
            assert len(owner.calls) == 1
            url, options = owner.calls[0]
            assert url == scheme + "://nonresolving.invalid/allowed"
            assert options["body"] == b"payload"
            assert options["method"] == "POST"
            assert options["follow_redirects"] is False
            assert not any(key == b"proxy-authorization" for key, _ in options["headers"])
            assert bridge.error_code is None
            assert bridge._token.encode() not in bridge.redact(bridge._token.encode())
        assert not writes
        assert not bridge._private_values
        assert not bridge._writers and not bridge._tasks

    async def joined():
        try:
            await scenario()
        finally:
            from contractor_runtime.adapters.bridge.mitmproxy import engine as mitmproxy_engine

            if mitmproxy_engine._active is not None:
                await mitmproxy_engine._active.close()

    asyncio.run(joined())


def test_bridge_uses_runtime_cookie_authority_and_keeps_partial_failures(tmp_path):
    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, credential_id=None, deadline=time.monotonic() + 15
        ) as bridge:
            proxy = bridge.proxy_url
            async with httpx.AsyncClient(proxy=proxy, trust_env=False) as client:
                await client.get("http://nonresolving.invalid/allowed")
                assert client.cookies["learned"] == "native-jar"
                await client.get("http://nonresolving.invalid/allowed")
                assert not any(k == b"cookie" for k, _ in owner.calls[-1][1]["headers"])
                with pytest.raises(httpx.HTTPError):
                    await client.post(
                        "http://nonresolving.invalid/allowed",
                        content=b"x" * (MAX_REQUEST_BODY_BYTES + 1),
                    )
            assert len(owner.calls) == 2
            assert bridge.error_code == "scan_request_failed"

    async def joined():
        try:
            await scenario()
        finally:
            from contractor_runtime.adapters.bridge.mitmproxy import engine as mitmproxy_engine

            if mitmproxy_engine._active is not None:
                await mitmproxy_engine._active.close()

    asyncio.run(joined())


def test_bridge_refuses_unauthenticated_local_client_before_forwarding(tmp_path):
    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, credential_id=None, deadline=time.monotonic() + 15
        ) as bridge:
            port = bridge._server.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(
                b"GET http://nonresolving.invalid/ HTTP/1.1\r\nHost: nonresolving.invalid\r\n\r\n"
            )
            await writer.drain()
            assert await reader.read() == b""
            writer.close()
            await writer.wait_closed()
            assert owner.calls == []

    async def joined():
        try:
            await scenario()
        finally:
            from contractor_runtime.adapters.bridge.mitmproxy import engine as mitmproxy_engine

            if mitmproxy_engine._active is not None:
                await mitmproxy_engine._active.close()

    asyncio.run(joined())


def test_bridge_early_engine_exit_joins_and_reports_unfinished_exchange(tmp_path):
    async def scenario():
        entered, joined = asyncio.Event(), asyncio.Event()

        class PendingOwner(Owner):
            async def request(self, *_, **__):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    joined.set()

        async with (
            HTTPProxyBridge(
                PendingOwner(), credential_id=None, deadline=time.monotonic() + 15
            ) as bridge,
            httpx.AsyncClient(proxy=bridge.proxy_url, trust_env=False) as client,
        ):
            pending = asyncio.create_task(client.get("http://nonresolving.invalid/allowed"))
            await entered.wait()
            # Model an engine stopping while its target exchange is pending.
            await bridge.__aexit__()
            with pytest.raises(httpx.HTTPError):
                await pending
        assert joined.is_set()
        assert bridge.error_code == "scan_request_failed"
        assert not bridge._tasks and not bridge._writers

    async def joined():
        try:
            await scenario()
        finally:
            from contractor_runtime.adapters.bridge.mitmproxy import engine as mitmproxy_engine

            if mitmproxy_engine._active is not None:
                await mitmproxy_engine._active.close()

    asyncio.run(joined())


def test_bridge_refuses_http10_before_owner_call(tmp_path):
    async def scenario():
        owner = Owner()
        try:
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 15
            ) as bridge:
                port = bridge._server.sockets[0].getsockname()[1]
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(
                    b"GET http://nonresolving.invalid/ HTTP/1.0\r\n"
                    b"Host: nonresolving.invalid\r\nProxy-Authorization: "
                    + bridge._authorization
                    + b"\r\n\r\n"
                )
                await writer.drain()
                assert await reader.read() == b""
                writer.close()
                await writer.wait_closed()
                assert owner.calls == []
        finally:
            if owner.engine is not None:
                await owner.engine.close()

    asyncio.run(scenario())
