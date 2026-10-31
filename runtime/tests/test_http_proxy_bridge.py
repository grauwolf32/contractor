"""Real HTTP/TLS IPC delegates to its owner and never opens a target tunnel."""

import asyncio
import time

import httpx
import pytest
from shared_config_defaults import MAX_REQUEST_BODY_BYTES

from contractor_runtime.adapters.http_proxy_bridge import HTTPProxyBridge
from contractor_runtime.adapters.http_session import PrivateResponse
from contractor_runtime.adapters.http_transport import HTTPOutcome


class Owner:
    def __init__(self):
        self.calls = []

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


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_bridge_delegates_plaintext_and_tls_without_target_resolution(tmp_path, scheme):
    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, tmp_path, credential_id=None, deadline=time.monotonic() + 15
        ) as bridge:
            proxy = bridge.proxy_url
            async with httpx.AsyncClient(proxy=proxy, verify=False, trust_env=False) as client:
                response = await client.post(
                    scheme + "://nonresolving.invalid/allowed", content=b"payload"
                )
            assert response.status_code == 200 and response.content == b"from-owner"
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
        assert not bridge._private_values
        assert not bridge._writers and not bridge._tasks

    asyncio.run(scenario())


def test_bridge_uses_runtime_cookie_authority_and_keeps_partial_failures(tmp_path):
    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, tmp_path, credential_id=None, deadline=time.monotonic() + 15
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

    asyncio.run(scenario())


def test_bridge_refuses_unauthenticated_local_client_before_forwarding(tmp_path):
    async def scenario():
        owner = Owner()
        async with HTTPProxyBridge(
            owner, tmp_path, credential_id=None, deadline=time.monotonic() + 15
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

    asyncio.run(scenario())


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
                PendingOwner(), tmp_path, credential_id=None, deadline=time.monotonic() + 15
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

    asyncio.run(scenario())
