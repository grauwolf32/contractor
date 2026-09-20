from __future__ import annotations

import asyncio
import logging
import ssl
from pathlib import Path

import httpx
import pytest
from fakes.mcp import MCPFixture, remote_tool
from test_http_proxy_adapter import adapter_context, issue_local_certificate, proxy_settings

from contractor_runtime.adapters.http_proxy import HTTPProxyAdapter
from contractor_runtime.contracts.toolset_settings import MCPToolsetSettings
from contractor_runtime.toolsets.mcp.errors import MCPError
from contractor_runtime.toolsets.mcp.session import MCPSessionOwner

SECRET = "MCP-real-transport-header-private-canary"


def configuration(url: str, **extra) -> MCPToolsetSettings:
    return MCPToolsetSettings(
        endpoint=url,
        headers={"Authorization": SECRET},
        connect_timeout_seconds=2,
        request_timeout_seconds=2,
        **extra,
    )


def deadline(seconds: float = 3) -> float:
    return asyncio.get_running_loop().time() + seconds


@pytest.mark.parametrize("sse", [False, True])
@pytest.mark.parametrize("stateful", [False, True])
def test_real_sdk_initializes_pages_calls_and_closes_without_sensitive_logs(
    sse, stateful, caplog
) -> None:
    caplog.set_level(logging.DEBUG)

    async def scenario() -> None:
        async with MCPFixture(sse=sse, stateful=stateful) as server:
            owner = MCPSessionOwner(configuration(server.url), ["search", "fetch"])
            await owner.prepare(deadline=deadline())
            assert owner.definition("search").input_schema["type"] == "object"
            assert owner.schema_digest.startswith("sha256:")
            assert len([r for r in server.requests if r["rpc"] == "tools/list"]) == 2
            result = await owner.call("search", {"query": "test"}, deadline=deadline())
            assert result.structuredContent == {"found": True} and not result.isError
            await owner.close(deadline=deadline())
            await owner.close(deadline=deadline())
            assert owner.closed and owner._task.done()
            assert len(server.calls) == 1
            assert server.deleted == server.sessions
            initialized = next(r for r in server.requests if r["rpc"] == "initialize")
            capabilities = initialized["body"]["params"]["capabilities"]
            assert not any(key in capabilities for key in ("sampling", "roots", "elicitation"))
            assert all(r["headers"]["authorization"] == SECRET for r in server.requests)
            assert SECRET not in repr(owner)
            assert "private-mcp-session" not in caplog.text and SECRET not in caplog.text

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "limit", ["pages", "tools", "discovery_bytes", "description", "declarations"]
)
def test_inventory_and_declaration_limits_are_applied_before_worker_creation(limit) -> None:
    async def scenario() -> None:
        async with MCPFixture() as server:
            selected = ["search"]
            if limit == "pages":
                server.pages = [[remote_tool("search")]] + [
                    [remote_tool(f"tool{i}")] for i in range(16)
                ]
            elif limit == "tools":
                server.pages = [
                    [remote_tool("search"), *[remote_tool(f"tool{i}") for i in range(1024)]]
                ]
            elif limit == "discovery_bytes":
                server.pages = [
                    [
                        remote_tool(
                            f"tool{page}_{i}",
                            description="d" * 4096,
                            inputSchema={"type": "object", "default": {"padding": "x" * 12000}},
                        )
                        for i in range(64)
                    ]
                    for page in range(10)
                ]
                server.pages[0][0]["name"] = "search"
            elif limit == "description":
                server.pages = [[remote_tool("search", description="d" * 4097)]]
            elif limit == "declarations":
                selected = [f"tool{i}" for i in range(64)]
                server.pages = [[remote_tool(name, description="d" * 4096) for name in selected]]
            owner = MCPSessionOwner(configuration(server.url), selected)
            with pytest.raises(MCPError):
                await owner.prepare(deadline=deadline(5))
            await owner.close(deadline=deadline())
            assert owner.closed and not server.calls

    asyncio.run(scenario())


def test_independent_sessions_freeze_inventory_and_do_not_share_cookies() -> None:
    async def scenario() -> None:
        async with MCPFixture() as server:
            first = MCPSessionOwner(configuration(server.url), ["search"])
            second = MCPSessionOwner(configuration(server.url), ["fetch"])
            await asyncio.gather(
                first.prepare(deadline=deadline()), second.prepare(deadline=deadline())
            )
            assert len(server.sessions) == 2 and server.sessions[0] != server.sessions[1]
            initial = [r for r in server.requests if r["rpc"] == "initialize"]
            assert all("cookie" not in r["headers"] for r in initial)
            digest = first.schema_digest
            server.pages = [[remote_tool("delete_everything")]]
            await first.call("search", {"query": "first"}, deadline=deadline())
            await second.call("fetch", {"query": "second"}, deadline=deadline())
            assert first.schema_digest == digest
            assert len([r for r in server.requests if r["rpc"] == "tools/list"]) == 4
            with pytest.raises(MCPError, match="unselected_tool"):
                await first.call("delete_everything", {}, deadline=deadline())
            await asyncio.gather(
                first.close(deadline=deadline()), second.close(deadline=deadline())
            )
            assert set(server.deleted) == set(server.sessions)

    asyncio.run(scenario())


def test_actual_tls_verification_and_proxy_route_preserve_borrowed_client(
    tmp_path: Path, monkeypatch
) -> None:
    certificate, key = issue_local_certificate(tmp_path)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, key)
    bundle = certificate.read_text()
    monkeypatch.setenv("HTTPS_PROXY", "http://unreachable.invalid:9")
    monkeypatch.setenv("HTTP_PROXY", "http://unreachable.invalid:9")

    async def scenario() -> None:
        async with MCPFixture(tls=tls) as server:
            endpoint = server.url.replace("127.0.0.1", "localhost")
            untrusted = MCPSessionOwner(configuration(endpoint), ["search"])
            with pytest.raises(MCPError):
                await untrusted.prepare(deadline=deadline())
            await untrusted.close(deadline=deadline())
            assert not server.requests
            trusted = MCPSessionOwner(configuration(endpoint, ca_bundle_pem=bundle), ["search"])
            await trusted.prepare(deadline=deadline())
            await trusted.close(deadline=deadline())
        # This fixture terminates an actual authenticated TLS forward-proxy
        # connection and records the absolute target URI sent by HTTPX.
        async with MCPFixture(tls=tls, stateful=False) as proxy:
            proxy_url = proxy.url.replace("127.0.0.1", "localhost").removesuffix("/mcp")
            adapter = HTTPProxyAdapter(
                adapter_context(),
                proxy_settings(proxy_url, targets=["tool-http"], ca_bundle=bundle),
            )
            borrowed = adapter.handles.tool_http
            owner = MCPSessionOwner(
                configuration("http://knowledge.example/mcp"), ["search"], proxy=borrowed
            )
            await owner.prepare(deadline=deadline())
            await owner.close(deadline=deadline())
            assert not borrowed.async_client.is_closed and not borrowed.async_client.cookies
            assert all(r["target"] == "http://knowledge.example/mcp" for r in proxy.requests)
            assert all(
                r["headers"]["proxy-authorization"].startswith("Basic ") for r in proxy.requests
            )
            count = len(proxy.requests)
            forbidden = MCPSessionOwner(
                configuration("http://control.example/private/v1"), ["search"], proxy=borrowed
            )
            with pytest.raises(MCPError):
                await forbidden.prepare(deadline=deadline())
            await forbidden.close(deadline=deadline())
            assert len(proxy.requests) == count
            await borrowed.request("GET", "http://public.example/ordinary")
            assert "authorization" not in proxy.requests[-1]["headers"]
            await adapter.close()

    asyncio.run(scenario())


def test_configured_timeout_values_reach_httpx_and_earlier_prepare_deadline_wins() -> None:
    from contractor_runtime.toolsets.mcp.transport import create_mcp_http_client

    async def scenario() -> None:
        observed = []

        async def send(request: httpx.Request) -> httpx.Response:
            observed.append(request.extensions["timeout"])
            return httpx.Response(200)

        config = configuration("http://mcp.example/mcp").model_copy(
            update={"connect_timeout_seconds": 3, "request_timeout_seconds": 7}
        )
        client, _ = create_mcp_http_client(config, transport=httpx.MockTransport(send))
        async with client:
            await client.post(config.endpoint)
        assert observed == [{"connect": 3, "read": 7, "write": 7, "pool": 3}]
        async with MCPFixture() as server:
            server.block_method = "initialize"
            owner = MCPSessionOwner(configuration(server.url), ["search"])
            started = asyncio.get_running_loop().time()
            with pytest.raises(MCPError, match="prepare_timeout"):
                await owner.prepare(deadline=deadline(0.08))
            await owner.close(deadline=deadline())
            assert asyncio.get_running_loop().time() - started < 0.8
            assert owner.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    ["auth", "redirect", "cursor", "missing", "duplicate", "schema", "large", "large_sse"],
)
def test_bounded_discovery_failure_closes_all_sdk_contexts(failure) -> None:
    async def scenario() -> None:
        async with MCPFixture(sse=failure == "large_sse") as server:
            if failure == "auth":
                server.required_header = "unavailable"
            elif failure == "redirect":
                server.redirect = True
            elif failure == "cursor":
                server.repeated_cursor = True
            elif failure == "missing":
                server.pages = [[remote_tool("other")]]
            elif failure == "duplicate":
                server.pages = [[remote_tool("search")], [remote_tool("search")]]
            elif failure == "schema":
                server.pages = [
                    [
                        remote_tool(
                            "search",
                            inputSchema={
                                "type": "object",
                                "$ref": "https://external.example/schema",
                            },
                        )
                    ]
                ]
            elif failure in {"large", "large_sse"}:
                server.raw_response = b"x" * (2 * 1024 * 1024 + 1)
            owner = MCPSessionOwner(configuration(server.url), ["search"])
            with pytest.raises(MCPError):
                await owner.prepare(deadline=deadline())
            await owner.close(deadline=deadline())
            assert owner.closed
            assert not server.calls
            assert not any("redirect" in request["target"] for request in server.requests)

    asyncio.run(scenario())
