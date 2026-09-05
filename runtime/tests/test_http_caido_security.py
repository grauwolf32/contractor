"""Adversarial input and transport bounds for HTTP/Caido tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_http_toolset import FakeArtifactClient, close_tools, create_tools, make_tools

import contractor_runtime.adapters.caido_graphql as caido_graphql
import contractor_runtime.toolsets.http_tools as http_tools
from contractor_runtime.adapters.caido_graphql import CaidoClientError, CaidoGraphQLClient
from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.http_tools import HTTPToolError


def test_session_header_and_cookie_limits_apply_to_atomic_merged_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_HEADER_BYTES", 24)
    monkeypatch.setattr(http_tools, "MAX_COOKIES", 2)
    transport_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal transport_calls
        transport_calls += 1
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        await tools["http_session_set"](
            headers={"X-Default": "d" * 8},
            cookies={"one": "1", "two": "2"},
            replace_headers=True,
            replace_cookies=True,
        )

        with pytest.raises(HTTPToolError) as headers:
            await tools["http_session_set"](headers={"X-Request": "r" * 8})
        assert headers.value.code == "http_request_invalid"
        with pytest.raises(HTTPToolError) as cookies:
            await tools["http_session_set"](cookies={"three": "3"})
        assert cookies.value.code == "http_request_invalid"
        with pytest.raises(HTTPToolError) as combined:
            await tools["http_session_set"](
                headers={"X-Replaced": "ok"},
                cookies={"three": "3"},
                replace_headers=True,
            )
        assert combined.value.code == "http_request_invalid"

        # Both rejected sparse updates are preflighted before mutation.
        view = await tools["http_session_get"]()
        assert view["default_headers"] == {"X-Default": "d" * 8}
        assert view["cookie_names"] == ["one", "two"]
        assert transport_calls == 0
        await close_tools(tools)

    asyncio.run(scenario())


def test_response_cookie_overflow_erases_cookie_state_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_COOKIES", 2)
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/overflow":
            return httpx.Response(
                200,
                content=b"must-not-be-selected",
                headers=[
                    ("content-type", "text/plain"),
                    ("set-cookie", "one=1; Path=/"),
                    ("set-cookie", "two=2; Path=/"),
                    ("set-cookie", "three=3; Path=/"),
                ],
                request=request,
            )
        return httpx.Response(200, content=b"", request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        await tools["http_session_set"](cookies={"existing": "secret"})
        with pytest.raises(HTTPToolError) as overflow:
            await tools["http_request"]("https://target.example/overflow")
        assert overflow.value.code == "http_request_failed"
        assert artifacts.writes == 0
        with pytest.raises(HTTPToolError) as missing:
            await tools["http_read_body"](1)
        assert missing.value.code == "http_body_not_found"
        assert (await tools["http_session_get"]())["cookie_count"] == 0

        result = await tools["http_request"]("https://target.example/after")
        assert result["request_id"] == 2
        assert "cookie" not in observed[-1].headers
        await close_tools(tools)

    asyncio.run(scenario())


def test_url_header_query_and_body_injection_fail_before_transport(tmp_path: Path) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"unexpected", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        request = tools["http_request"]
        cases: list[tuple[dict[str, Any], str]] = [
            ({"url": "https://user:password@target.example/"}, "http_request_invalid"),
            ({"url": "https://target.example/#fragment"}, "http_request_invalid"),
            ({"url": "http://localhost/"}, "http_target_denied"),
            ({"url": "http://service.localhost/"}, "http_target_denied"),
            ({"url": "http://127.0.0.1/"}, "http_target_denied"),
            ({"url": "http://[::1]/"}, "http_target_denied"),
            ({"url": "https://gateway.example/private"}, "http_target_denied"),
            ({"url": "https://control.example/private/v1/run"}, "http_target_denied"),
            (
                {"url": "https://target.example/", "headers": {"Host": "other.example"}},
                "http_request_invalid",
            ),
            (
                {
                    "url": "https://target.example/",
                    "headers": {"Proxy-Authorization": "Bearer injected"},
                },
                "http_request_invalid",
            ),
            (
                {"url": "https://target.example/", "headers": {"X-Test": "x\r\ny"}},
                "http_request_invalid",
            ),
            (
                {"url": "https://target.example/", "query": {"line\nfeed": "x"}},
                "http_request_invalid",
            ),
            (
                {"url": "https://target.example/", "query": {"value": float("nan")}},
                "http_request_invalid",
            ),
            (
                {
                    "url": "https://target.example/",
                    "method": "POST",
                    "body_type": "none",
                    "body": "smuggled",
                },
                "http_request_invalid",
            ),
        ]
        for arguments, code in cases:
            with pytest.raises(HTTPToolError) as failure:
                await request(**arguments)
            assert failure.value.code == code
        assert calls == 0
        await close_tools(tools)

    asyncio.run(scenario())


def test_omitted_http_timeout_is_bounded_when_allocation_timeout_exceeds_tool_limit(
    tmp_path: Path,
) -> None:
    observed_timeout: dict[str, float] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_timeout.update(request.extensions["timeout"])
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()

        def direct() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        factory = http_tools.HTTPToolsetFactory(
            lambda _allocation, _settings: artifacts,
            direct,
        )
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            llmGatewayToken="gateway-token",
            artifactApiUrl="https://control.example/private/v1",
            requestTimeoutSeconds=180,
        )
        tools = await make_tools(factory, tmp_path, settings=settings)
        await tools["http_request"]("https://target.example/")
        assert observed_timeout == {
            "connect": 120.0,
            "read": 120.0,
            "write": 120.0,
            "pool": 120.0,
        }
        await close_tools(tools)

    asyncio.run(scenario())


def test_redirect_limit_private_hop_and_cross_origin_auth_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_REDIRECTS", 2)
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        path = request.url.path
        if path == "/private":
            return httpx.Response(
                302,
                headers={"location": "https://control.example/private/v1"},
                request=request,
            )
        if path == "/cross":
            return httpx.Response(
                302,
                headers={"location": "https://other.example/final"},
                request=request,
            )
        if path == "/ok-0":
            return httpx.Response(302, headers={"location": "/ok-1"}, request=request)
        if path == "/ok-1":
            return httpx.Response(302, headers={"location": "/ok-final"}, request=request)
        if path.startswith("/loop-"):
            index = int(path.rsplit("-", 1)[1])
            return httpx.Response(302, headers={"location": f"/loop-{index + 1}"}, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        await tools["http_session_set"](
            cookies={"sid": "private"},
            auth={"kind": "bearer", "token": "private"},
        )

        with pytest.raises(HTTPToolError) as denied:
            await tools["http_request"]("https://target.example/private", follow_redirects=True)
        assert denied.value.code == "http_target_denied"
        assert [str(item.url) for item in observed] == ["https://target.example/private"]

        observed.clear()
        cross = await tools["http_request"]("https://target.example/cross", follow_redirects=True)
        assert cross["status"] == 200
        assert observed[0].headers["authorization"] == "Bearer private"
        assert "authorization" not in observed[1].headers
        assert "cookie" not in observed[1].headers

        observed.clear()
        exact = await tools["http_request"]("https://target.example/ok-0", follow_redirects=True)
        assert exact["redirects"] == 2
        assert len(observed) == 3

        observed.clear()
        with pytest.raises(HTTPToolError) as excessive:
            await tools["http_request"]("https://target.example/loop-0", follow_redirects=True)
        assert excessive.value.code == "http_request_failed"
        assert len(observed) == 3
        await close_tools(tools)

    asyncio.run(scenario())


def test_http_response_stream_limit_is_exact_and_never_partially_selects_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_RESPONSE_BODY_BYTES", 64)
    artifacts = FakeArtifactClient()

    async def handler(request: httpx.Request) -> httpx.Response:
        size = 64 if request.url.path == "/exact" else 65
        return httpx.Response(
            200,
            content=b"x" * size,
            headers={"content-type": "text/plain"},
            request=request,
        )

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        exact = await tools["http_request"]("https://target.example/exact")
        assert exact["content_length"] == 64
        assert artifacts.writes == 1
        with pytest.raises(HTTPToolError) as oversized:
            await tools["http_request"]("https://target.example/oversized")
        assert oversized.value.code == "http_response_too_large"
        assert artifacts.writes == 1
        assert [item["request_id"] for item in await tools["http_history"]()] == [1]
        await close_tools(tools)

    asyncio.run(scenario())


def test_caido_static_operation_variable_and_response_shape_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        calls: list[httpx.Request] = []

        async def valid(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"data": {"scopes": []}}, request=request)

        client = caido_client(valid)
        with pytest.raises(CaidoClientError) as arbitrary:
            await client.execute("query { arbitrarySecretField }")
        assert arbitrary.value.code == "caido_request_invalid"
        assert calls == []

        monkeypatch.setattr(caido_graphql, "MAX_CAIDO_VARIABLE_DEPTH", 2)
        with pytest.raises(CaidoClientError) as deep_variable:
            await client.execute("scopes", {"a": {"b": {"c": 1}}})
        assert deep_variable.value.code == "caido_request_invalid"
        assert calls == []

        monkeypatch.setattr(caido_graphql, "MAX_CAIDO_VARIABLE_ITEMS", 3)
        with pytest.raises(CaidoClientError) as many_variables:
            await client.execute("scopes", {"items": [1, 2, 3]})
        assert many_variables.value.code == "caido_request_invalid"
        assert calls == []
        await client.close()

        fixtures: list[tuple[bytes, str]] = [
            (b'{"data":{"value":1,"value":2}}', "caido_response_invalid"),
            (b'{"data":{"a":{"b":{"c":1}}}}', "caido_response_invalid"),
            (b'{"data":{"items":[1,2,3]}}', "caido_response_invalid"),
            (b"{" + b"x" * 64 + b"}", "caido_response_too_large"),
        ]
        monkeypatch.setattr(caido_graphql, "MAX_CAIDO_RESPONSE_DEPTH", 2)
        monkeypatch.setattr(caido_graphql, "MAX_CAIDO_RESPONSE_ITEMS", 5)
        for index, (payload, code) in enumerate(fixtures):
            monkeypatch.setattr(
                caido_graphql,
                "MAX_CAIDO_RESPONSE_BYTES",
                32 if index == len(fixtures) - 1 else 1024,
            )
            bounded = caido_client(
                lambda request, body=payload: httpx.Response(200, content=body, request=request)
            )
            with pytest.raises(CaidoClientError) as failure:
                await bounded.execute("scopes")
            assert failure.value.code == code
            await bounded.close()

    asyncio.run(scenario())


def caido_client(handler: Any) -> CaidoGraphQLClient:
    return CaidoGraphQLClient(
        endpoint="https://caido.example",
        bearer_token="caido-secret",
        ca_bundle_pem=None,
        timeout_seconds=5,
        metrics=RuntimeAdapterMetricsState(),
        transport=httpx.MockTransport(handler),
    )
