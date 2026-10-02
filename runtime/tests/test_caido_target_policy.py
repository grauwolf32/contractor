"""Caido request-bound actions obey the same allocation destination policy as HTTP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from test_caido_read_tools import FakeArtifactClient, close_tools, create_tools, request_detail
from test_http_toolset import close_tools as close_http_tools
from test_http_toolset import make_tools

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts import (
    CaidoSettings,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.factories import built_in_factories
from contractor_runtime.toolsets.caido.tools import CaidoToolError, CaidoToolsetFactory
from contractor_runtime.toolsets.common.target_policy import (
    TargetPolicyConfig,
    TargetUnresolved,
    parse_allowed_networks,
)
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory


async def _unresolved(_host: str, _port: int):
    raise TargetUnresolved


def _settings(*, origin: str | None = None, proxy: bool = False) -> RuntimeSettings:
    values: dict[str, object] = {
        "llmGatewayUrl": "https://gateway.example/v1",
        "llmGatewayToken": "gateway-token",
        "artifactApiUrl": "https://control.example/private/v1",
        "caido": CaidoSettings(
            adapter="caido-graphql@1",
            endpoint="https://caido.example",
            bearerToken="caido-token",
            requestTimeoutSeconds=5,
        ),
        "requestTimeoutSeconds": 10,
    }
    if origin is not None:
        values["httpOriginTarget"] = HTTPOriginTargetSettings(url=origin)
    if proxy:
        values["httpProxy"] = HTTPProxySettings(
            adapter="http-proxy@1", proxyUrl="https://proxy.example", targets=["tool-http"]
        )
    return RuntimeSettings(**values)


def _policy(*, allow_loopback: bool = False) -> TargetPolicyConfig:
    return TargetPolicyConfig(
        protected_urls=("http://127.0.0.1:9443",),
        allowed_networks=parse_allowed_networks(["127.0.0.0/8"]) if allow_loopback else (),
        resolver=_unresolved,
    )


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("169.254.169.254", 80),
        ("metadata.google.internal", 80),
        ("127.1", 9443),
        ("gateway.example", 443),
        ("control.example", 443),
        ("caido.example", 443),
    ],
)
def test_raw_replay_denies_protected_targets_before_graphql(
    tmp_path: Path, host: str, port: int
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected Caido GraphQL request: {request.url}")

    async def scenario() -> None:
        factory = CaidoToolsetFactory(
            lambda *_: FakeArtifactClient(), target_policy=_policy(allow_loopback=True)
        )
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
            factory=factory,
        )
        try:
            with pytest.raises(CaidoToolError) as denied:
                await tools["caido_replay"](
                    raw_request="GET / HTTP/1.1\r\nHost: ignored.example\r\n\r\n",
                    host=host,
                    port=port,
                    wait=False,
                )
            assert denied.value.code == "caido_target_denied"
            assert denied.value.retryable is False
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["caido_replay", "caido_automate_run", "caido_workflow_run"])
@pytest.mark.parametrize(
    ("host", "port"),
    [("169.254.169.254", 80), ("127.1", 9443), ("gateway.example", 443)],
)
def test_captured_request_actions_deny_before_mutation(
    tmp_path: Path, action: str, host: str, port: int
) -> None:
    operations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operations.append(payload["operationName"])
        assert payload["operationName"] == "RequestDetail"
        detail = request_detail(b"POST / HTTP/1.1\r\nHost: ignored.example\r\n\r\nTARGET", b"")
        detail.update({"host": host, "port": port, "isTls": port == 443})
        return httpx.Response(200, json={"data": {"request": detail}}, request=request)

    async def scenario() -> None:
        factory = CaidoToolsetFactory(
            lambda *_: FakeArtifactClient(), target_policy=_policy(allow_loopback=True)
        )
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={action},
            factory=factory,
        )
        try:
            with pytest.raises(CaidoToolError) as denied:
                if action == "caido_replay":
                    await tools[action](request_id="request-1", wait=False)
                elif action == "caido_automate_run":
                    await tools[action]("request-1", targets=["TARGET"], payloads=["probe"])
                else:
                    await tools[action]("workflow-1", request_id="request-1")
            assert denied.value.code == "caido_target_denied"
            assert denied.value.retryable is False
            assert operations == ["RequestDetail"]
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("host", "port", "origin", "allow_loopback"),
    [
        ("127.1", 8080, "http://127.0.0.1:8080/", False),
        ("127.2", 9090, None, True),
        ("10.0.0.5", 8080, None, False),
        ("public.example", 443, None, False),
    ],
)
def test_raw_replay_keeps_permitted_targets(
    tmp_path: Path, host: str, port: int, origin: str | None, allow_loopback: bool
) -> None:
    operations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        operation = payload["operationName"]
        operations.append(operation)
        if operation == "CreateReplaySession":
            data = {
                "createReplaySession": {
                    "session": {"id": "replay-1", "name": "replay", "activeEntry": None}
                }
            }
        elif operation == "StartReplayTask":
            data = {
                "startReplayTask": {
                    "error": None,
                    "task": {"id": "task-1", "replayEntry": {"id": "entry-1"}},
                }
            }
        else:
            raise AssertionError(operation)
        return httpx.Response(200, json={"data": data}, request=request)

    async def scenario() -> None:
        factory = CaidoToolsetFactory(
            lambda *_: FakeArtifactClient(), target_policy=_policy(allow_loopback=allow_loopback)
        )
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
            factory=factory,
            runtime_settings=_settings(origin=origin),
        )
        try:
            result = await tools["caido_replay"](
                raw_request="GET / HTTP/1.1\r\nHost: ignored.example\r\n\r\n",
                host=host,
                port=port,
                wait=False,
            )
            assert result["status"] == "started"
            assert operations == ["CreateReplaySession", "StartReplayTask"]
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())


def test_caido_and_http_proxy_reject_the_same_destinations(tmp_path: Path) -> None:
    caido_calls: list[httpx.Request] = []
    proxy_calls: list[httpx.Request] = []

    async def caido_handler(request: httpx.Request) -> httpx.Response:
        caido_calls.append(request)
        raise AssertionError("denied Caido target reached GraphQL")

    async def proxy_handler(request: httpx.Request) -> httpx.Response:
        proxy_calls.append(request)
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        settings = _settings(proxy=True)
        policy = _policy(allow_loopback=True)
        caido_tools, _state, caido_handle = await create_tools(
            tmp_path,
            caido_handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
            factory=CaidoToolsetFactory(lambda *_: FakeArtifactClient(), target_policy=policy),
            runtime_settings=settings,
        )
        client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_handler), trust_env=False)
        http_tools = await make_tools(
            HTTPToolsetFactory(lambda *_: FakeArtifactClient(), target_policy=policy),
            tmp_path,
            settings=settings,
            adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(client)),
        )
        try:
            for host, port in (
                ("169.254.169.254", 80),
                ("metadata.google.internal", 80),
                ("127.1", 9443),
                ("gateway.example", 443),
                ("control.example", 443),
                ("caido.example", 443),
            ):
                with pytest.raises(HTTPToolError) as http_denied:
                    await http_tools["http_request"](f"http://{host}:{port}/")
                with pytest.raises(CaidoToolError) as caido_denied:
                    await caido_tools["caido_replay"](
                        raw_request="GET / HTTP/1.1\r\nHost: ignored.example\r\n\r\n",
                        host=host,
                        port=port,
                        wait=False,
                    )
                assert http_denied.value.code == "http_target_denied"
                assert caido_denied.value.code == "caido_target_denied"
            assert caido_calls == []
            assert proxy_calls == []
        finally:
            await close_http_tools(http_tools)
            await close_tools(caido_tools)
            await caido_handle.close()
            await client.aclose()

    asyncio.run(scenario())


def test_builtin_registry_passes_process_protected_targets_to_caido(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected Caido GraphQL request: {request.url}")

    async def scenario() -> None:
        policy = TargetPolicyConfig(
            protected_urls=("https://private-runtime.example:8443",), resolver=_unresolved
        )
        registry = built_in_factories(
            tmp_path,
            artifact_client_factory=lambda *_: FakeArtifactClient(),
            target_policy=policy,
        )
        factory = registry.toolsets["caido@1"]
        assert isinstance(factory, CaidoToolsetFactory)
        tools, _state, handle = await create_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
            factory=factory,
        )
        try:
            with pytest.raises(CaidoToolError) as denied:
                await tools["caido_replay"](
                    raw_request="GET / HTTP/1.1\r\nHost: ignored.example\r\n\r\n",
                    host="private-runtime.example",
                    port=8443,
                    wait=False,
                )
            assert denied.value.code == "caido_target_denied"
        finally:
            await close_tools(tools)
            await handle.close()

    asyncio.run(scenario())
