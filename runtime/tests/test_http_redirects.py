"""Redirect cookie isolation and response ownership on both HTTP transports."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_http_toolset import FakeArtifactClient, close_tools, make_tools, proxy_runtime_settings

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory


async def redirect_tools(
    tmp_path: Path, handler: Any, proxied: bool
) -> tuple[dict[str, Any], httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    factory = HTTPToolsetFactory(lambda *_: FakeArtifactClient(), lambda: client)
    kwargs: dict[str, Any] = {}
    if proxied:
        kwargs = {
            "settings": proxy_runtime_settings(),
            "adapter_handles": AdapterHandles(tool_http=ProxyHTTPClient(client)),
        }
    return await make_tools(factory, tmp_path, **kwargs), client


@pytest.mark.parametrize("proxied", [False, True])
@pytest.mark.parametrize(
    "destination", ["https://other.example.com/end", "https://app.example.com:8443/end"]
)
def test_redirect_does_not_forward_transport_cookies(
    tmp_path: Path, proxied: bool, destination: str
) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={
                    "Location": destination,
                    "Set-Cookie": "sid=secret; Domain=.example.com; Path=/",
                },
                request=request,
            )
        return httpx.Response(200, content=b"", request=request)

    async def scenario() -> None:
        tools, client = await redirect_tools(tmp_path, handler, proxied)
        try:
            await tools["http_session_set"](auth={"kind": "bearer", "token": "secret"})
            await tools["http_request"]("https://app.example.com/start")
            assert len(observed) == 2
            assert observed[0].headers["authorization"] == "Bearer secret"
            assert "cookie" not in observed[1].headers
            assert "authorization" not in observed[1].headers
        finally:
            await close_tools(tools)
            await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("proxied", [False, True])
def test_same_origin_redirect_cookies_survive_and_can_be_deleted(
    tmp_path: Path, proxied: bool
) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"Location": "/end", "Set-Cookie": "sid=secret; Path=/"},
                request=request,
            )
        headers = {"Set-Cookie": "sid=; Max-Age=0; Path=/"} if request.url.path == "/delete" else {}
        return httpx.Response(200, headers=headers, content=b"", request=request)

    async def scenario() -> None:
        tools, client = await redirect_tools(tmp_path, handler, proxied)
        try:
            await tools["http_request"]("https://app.example.com/start")
            await tools["http_request"]("https://app.example.com/after")
            assert observed[1].headers["cookie"] == "sid=secret"
            assert observed[2].headers["cookie"] == "sid=secret"
            await tools["http_request"]("https://app.example.com/delete")
            await tools["http_request"]("https://app.example.com/after-delete")
            assert "cookie" not in observed[-1].headers
        finally:
            await close_tools(tools)
            await client.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("proxied", [False, True])
@pytest.mark.parametrize(
    "destination", ["https://control.example/private", "file:///etc/passwd", "https://[invalid"]
)
def test_rejected_redirect_closes_stream_without_committing_cookies(
    tmp_path: Path, proxied: bool, destination: str
) -> None:
    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"unconsumed response"

        async def aclose(self) -> None:
            self.closed = True

    stream = Stream()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"Location": destination, "Set-Cookie": "sid=secret; Path=/"},
            stream=stream,
            request=request,
        )

    async def scenario() -> None:
        tools, client = await redirect_tools(tmp_path, handler, proxied)
        try:
            with pytest.raises(HTTPToolError):
                await tools["http_request"]("https://app.example.com/start")
            assert stream.closed
            assert (await tools["http_session_get"]())["cookie_count"] == 0
        finally:
            await close_tools(tools)
            await client.aclose()

    asyncio.run(scenario())
