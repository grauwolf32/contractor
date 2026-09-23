"""Target policy enforced by the production direct transport and the proxy route."""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from test_http_toolset import FakeArtifactClient, close_tools, make_tools, proxy_runtime_settings

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts import HTTPOriginTargetSettings, RuntimeSettings
from contractor_runtime.toolsets.common.target_policy import (
    IPAddress,
    TargetPolicyConfig,
    TargetUnresolved,
    parse_private_networks,
)
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory


@dataclass
class LoopbackServer:
    port: int
    paths: list[str] = field(default_factory=list)
    redirects: dict[str, str] = field(default_factory=dict)


@asynccontextmanager
async def loopback_server() -> AsyncIterator[LoopbackServer]:
    state = LoopbackServer(port=0)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            path = head.split(b" ", 2)[1].decode("ascii")
            state.paths.append(path)
            location = state.redirects.get(path)
            if location is not None:
                writer.write(
                    b"HTTP/1.1 302 Found\r\nLocation: "
                    + location.encode("ascii")
                    + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            else:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                    b"Content-Length: 2\r\nConnection: close\r\n\r\nok"
                )
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    state.port = server.sockets[0].getsockname()[1]
    async with server:
        yield state


def table_resolver(table: dict[str, tuple[str, ...]], calls: list[str] | None = None):
    async def resolve(host: str, port: int) -> tuple[IPAddress, ...]:
        del port
        if calls is not None:
            calls.append(host)
        if host not in table:
            raise TargetUnresolved
        return tuple(ipaddress.ip_address(value) for value in table[host])

    return resolve


def direct_settings(**overrides) -> RuntimeSettings:
    values = {
        "llmGatewayUrl": "https://gateway.example/v1",
        "artifactApiUrl": "https://control.example/private/v1",
        "requestTimeoutSeconds": 10,
    }
    values.update(overrides)
    return RuntimeSettings(**values)


async def denied(tool, url: str) -> None:
    with pytest.raises(HTTPToolError) as failure:
        await tool(url)
    assert failure.value.code == "http_target_denied"
    assert failure.value.retryable is False


def test_names_resolving_to_loopback_are_denied_at_connect_time(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with loopback_server() as server:
            config = TargetPolicyConfig(
                resolver=table_resolver(
                    {"rebind.example": ("127.0.0.1",), "ula.example": ("fd00::1",)}
                )
            )
            factory = HTTPToolsetFactory(lambda *_: FakeArtifactClient(), target_policy=config)
            tools = await make_tools(factory, tmp_path, settings=direct_settings())
            try:
                for url in (
                    f"http://rebind.example:{server.port}/",
                    f"http://ula.example:{server.port}/",
                    f"http://127.1:{server.port}/",
                    f"http://2130706433:{server.port}/",
                    f"http://0x7f000001:{server.port}/",
                    f"http://localhost:{server.port}/",
                ):
                    await denied(tools["http_request"], url)
                assert server.paths == []
            finally:
                await close_tools(tools)

    asyncio.run(scenario())


def test_loopback_project_target_is_reachable_in_direct_mode(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with loopback_server() as server, loopback_server() as other:
            server.redirects["/elsewhere"] = f"http://127.0.0.1:{other.port}/private"
            factory = HTTPToolsetFactory(
                lambda *_: FakeArtifactClient(), target_policy=TargetPolicyConfig()
            )
            settings = direct_settings(
                httpOriginTarget=HTTPOriginTargetSettings(url=f"http://127.0.0.1:{server.port}/")
            )
            tools = await make_tools(factory, tmp_path, settings=settings)
            request = tools["http_request"]
            try:
                first = await request(f"http://127.0.0.1:{server.port}/one")
                assert first["status"] == 200
                assert first["body_preview"] == "ok"
                alias = await request(f"http://127.1:{server.port}/two")
                assert alias["status"] == 200
                # Another loopback port stays private, including via a redirect.
                await denied(request, f"http://127.0.0.1:{other.port}/direct")
                await denied(request, f"http://127.0.0.1:{server.port}/elsewhere")
                assert server.paths == ["/one", "/two", "/elsewhere"]
                assert other.paths == []
            finally:
                await close_tools(tools)

    asyncio.run(scenario())


def test_operator_networks_allow_loopback_but_never_runtime_endpoints(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with loopback_server() as target, loopback_server() as artifacts:
            calls: list[str] = []
            config = TargetPolicyConfig(
                private_networks=parse_private_networks(["127.0.0.0/8"]),
                resolver=table_resolver(
                    {
                        "app.example": ("10.255.0.1", "127.0.0.1"),
                        "artifacts.example": ("127.0.0.1",),
                    },
                    calls,
                ),
            )
            factory = HTTPToolsetFactory(lambda *_: FakeArtifactClient(), target_policy=config)
            settings = direct_settings(
                artifactApiUrl=f"https://artifacts.example:{artifacts.port}/private/v1"
            )
            tools = await make_tools(factory, tmp_path, settings=settings)
            request = tools["http_request"]
            try:
                calls.clear()
                # The denied private candidate is skipped; the socket is pinned
                # to the permitted answer from the same single lookup.
                allowed = await request(f"http://app.example:{target.port}/allowed")
                assert allowed["status"] == 200
                assert calls == ["app.example"]
                for url in (
                    f"http://127.0.0.1:{artifacts.port}/",
                    f"http://127.1:{artifacts.port}/",
                    f"http://[::1]:{artifacts.port}/",
                    f"http://artifacts.example:{artifacts.port}/",
                    f"http://169.254.169.254:{target.port}/",
                ):
                    await denied(request, url)
                assert target.paths == ["/allowed"]
                assert artifacts.paths == []
            finally:
                await close_tools(tools)

    asyncio.run(scenario())


def test_proxy_route_denial_is_target_denied_and_never_retried(tmp_path: Path) -> None:
    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, request=request)

    async def scenario() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        config = TargetPolicyConfig(
            private_networks=parse_private_networks(["127.0.0.0/8"]),
            resolver=table_resolver({}),
        )
        factory = HTTPToolsetFactory(lambda *_: FakeArtifactClient(), target_policy=config)
        tools = await make_tools(
            factory,
            tmp_path,
            settings=proxy_runtime_settings(),
            adapter_handles=AdapterHandles(
                tool_http=ProxyHTTPClient(client, forbidden_hosts=("127.0.0.1", "localhost"))
            ),
        )
        try:
            # The operator network passes the Runtime check; the route still
            # refuses loopback and that refusal is a policy denial.
            await denied(tools["http_request"], "http://127.0.0.1:8080/")
            await denied(tools["http_request"], "http://169.254.169.254/latest")
            await denied(tools["http_request"], "https://proxy.example/")
            assert sent == []
            history = await tools["http_history"]()
            assert history == []
        finally:
            await close_tools(tools)
            await client.aclose()

    asyncio.run(scenario())
