from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import contractor_runtime.toolsets.http.tools as http_tools
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import (
    ArtifactRef,
    HTTPOriginTargetSettings,
    HTTPProxySettings,
    RuntimeSettings,
)
from contractor_runtime.toolsets.http.tools import (
    HTTP_BODY_MEDIA_TYPE,
    HTTPToolError,
    HTTPToolsetFactory,
)
from contractor_runtime.workspace import AllocationWorkspace

SECRET = "recognizable-http-session-secret"
TARGET_SECRET = "recognizable-project-origin-secret"


def test_direct_text_binary_status_redirect_and_exact_body_reads(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"Location": "/text"}, request=request)
        if request.url.path == "/text":
            return httpx.Response(
                404,
                content="hello world",
                headers={"Content-Type": "text/plain; charset=utf-8", "Set-Cookie": "sid=x"},
                request=request,
            )
        return httpx.Response(
            500,
            content=b"\x00\xffpayload",
            headers={"Content-Type": "application/octet-stream"},
            request=request,
        )

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, state = await create_tools(tmp_path, handler, artifacts=artifacts)
        request = tools["http_request"]
        read = tools["http_read_body"]

        text = await request("https://target.example/redirect", follow_redirects=True)
        assert text["status"] == 404
        assert text["body_kind"] == "text"
        assert text["body_preview"] == "hello world"
        assert text["redirects"] == 1
        exact = ArtifactRef.model_validate(text["body_artifact"]).require_exact()
        assert artifacts.media_types[(exact.namespace, exact.name, exact.revision)] == (
            HTTP_BODY_MEDIA_TYPE
        )
        text_slice = await read(text["request_id"], offset=6, length=5)
        assert text_slice == {
            "request_id": 1,
            "kind": "text",
            "unit": "characters",
            "offset": 6,
            "length": 5,
            "total": 11,
            "eof": True,
            "data": "world",
        }

        binary = await request("https://target.example/binary")
        assert binary["status"] == 500
        assert binary["retries"] == 2
        assert binary["body_kind"] == "binary"
        assert binary["body_preview"] is None
        binary_slice = await read(binary["request_id"], offset=1, length=3)
        assert base64.b64decode(binary_slice["data_b64"]) == b"\xffpa"
        assert [item.url.path for item in calls] == [
            "/redirect",
            "/text",
            "/binary",
            "/binary",
            "/binary",
        ]
        assert state.metrics.counters["tool_calls"] == 4
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_request_validation_private_origin_and_retry_policy(tmp_path: Path) -> None:
    counts: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counts[path] = counts.get(path, 0) + 1
        if path == "/retry" and counts[path] < 3:
            return httpx.Response(503, content=b"retry", request=request)
        return httpx.Response(503 if path == "/post" else 200, content=b"done", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        request = tools["http_request"]

        retried = await request("https://target.example/retry")
        assert retried["status"] == 200
        assert retried["retries"] == 2
        posted = await request(
            "https://target.example/post",
            method="POST",
            body_type="json",
            body={"safe": True},
        )
        assert posted["status"] == 503
        assert posted["retries"] == 0
        assert counts == {"/retry": 3, "/post": 1}

        for kwargs, code in (
            ({"url": "https://control.example/private"}, "http_target_denied"),
            (
                {"url": "https://target.example", "headers": {"X-Test": "x\r\ny"}},
                "http_request_invalid",
            ),
            (
                {"url": "https://target.example", "headers": {"Host": "elsewhere"}},
                "http_request_invalid",
            ),
            ({"url": "file:///tmp/private"}, "http_request_invalid"),
        ):
            with pytest.raises(HTTPToolError) as failure:
                await request(**kwargs)
            assert failure.value.code == code

    asyncio.run(scenario())


def test_session_is_redacted_serialized_and_history_is_bounded(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        observed.append(request)
        active -= 1
        if request.url.path == "/redirect-other":
            return httpx.Response(
                302,
                headers={"Location": "https://other.example/final"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, state = await create_tools(tmp_path, handler)
        await tools["http_session_set"](
            cookies={"sid": SECRET},
            headers={"X-API-Key": SECRET, "X-Safe": "visible"},
            auth={"kind": "bearer", "token": SECRET},
        )
        view = await tools["http_session_get"]()
        assert view == {
            "auth_kind": "bearer",
            "default_headers": {"X-API-Key": "[REDACTED]", "X-Safe": "visible"},
            "cookie_names": ["sid"],
            "cookie_count": 1,
            "history_count": 0,
        }
        assert SECRET not in repr(view)

        await asyncio.gather(
            tools["http_request"]("https://target.example/one"),
            tools["http_request"]("https://target.example/two"),
        )
        assert maximum_active == 1
        assert len(observed) == 2
        assert all(item.headers["authorization"] == f"Bearer {SECRET}" for item in observed)
        assert all("sid=" in item.headers["cookie"] for item in observed)
        await tools["http_request"]("https://target.example/redirect-other")
        assert observed[-1].url.host == "other.example"
        assert "authorization" not in observed[-1].headers
        assert "cookie" not in observed[-1].headers
        history = await tools["http_history"]()
        assert [item["request_id"] for item in history] == [1, 2, 3]
        assert all("body_preview" not in item for item in history)

        cleared = await tools["http_session_clear"]()
        assert cleared["auth_kind"] == "none"
        assert cleared["cookie_count"] == 0
        assert cleared["history_count"] == 0
        assert SECRET not in repr(state.metrics)

    asyncio.run(scenario())


def test_project_authorization_is_exact_origin_hidden_and_erased(tmp_path: Path) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/redirect":
            return httpx.Response(
                302,
                headers={"Location": "https://other.example/final"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        state = WorkerState()

        def direct() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        factory = HTTPToolsetFactory(lambda _allocation, _settings: FakeArtifactClient(), direct)
        settings = RuntimeSettings(
            llmGatewayUrl="https://gateway.example/v1",
            artifactApiUrl="https://control.example/private/v1",
            httpOriginTarget=HTTPOriginTargetSettings(
                url="https://target.example/application",
                bearerToken=TARGET_SECRET,
            ),
            requestTimeoutSeconds=30,
        )
        tools = await make_tools(factory, tmp_path, state=state, settings=settings)

        await tools["http_request"](
            "https://target.example/one",
            headers={"Authorization": "Bearer model-supplied"},
        )
        await tools["http_request"](
            "https://target.example:444/two",
            headers={"Authorization": "Bearer model-other-origin"},
        )
        await tools["http_request"]("https://target.example/redirect", follow_redirects=True)

        assert observed[0].headers["authorization"] == f"Bearer {TARGET_SECRET}"
        assert observed[1].headers["authorization"] == "Bearer model-other-origin"
        assert observed[2].headers["authorization"] == f"Bearer {TARGET_SECRET}"
        assert observed[3].url.host == "other.example"
        assert "authorization" not in observed[3].headers
        session = await tools["http_session_get"]()
        assert TARGET_SECRET not in repr(session)
        assert TARGET_SECRET not in repr(state)

        private_session = tools["http_request"]._session
        await close_tools(tools)
        assert private_session._target_origin is None
        assert private_session._target_authorization is None

    asyncio.run(scenario())


def test_proxy_route_is_required_and_target_errors_are_returned(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=b"target failure", request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        settings = proxy_runtime_settings()
        factory = HTTPToolsetFactory(lambda _allocation, _settings: artifacts)
        with pytest.raises(RuntimeError, match="resolved tool-http"):
            await make_tools(factory, tmp_path, settings=settings)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
        handle = ProxyHTTPClient(client)
        tools = await make_tools(
            factory,
            tmp_path,
            settings=settings,
            adapter_handles=AdapterHandles(tool_http=handle),
        )
        result = await tools["http_request"]("https://target.example/failure")
        assert result["status"] == 502
        assert result["retries"] == 2
        await close_tools(tools)
        await client.aclose()

    asyncio.run(scenario())


def test_oversized_response_and_proxy_auth_failure_leave_no_body_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(http_tools, "MAX_RESPONSE_BODY_BYTES", 32)

    async def oversized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 33, request=request)

    async def proxy_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(407, content=b"must not be surfaced", request=request)

    async def scenario() -> None:
        artifacts = FakeArtifactClient()
        tools, _state = await create_tools(tmp_path, oversized, artifacts=artifacts)
        with pytest.raises(HTTPToolError) as too_large:
            await tools["http_request"]("https://target.example/large")
        assert too_large.value.code == "http_response_too_large"
        assert artifacts.writes == 0

        client = httpx.AsyncClient(transport=httpx.MockTransport(proxy_auth), trust_env=False)
        proxy = ProxyHTTPClient(client)
        factory = HTTPToolsetFactory(
            lambda _allocation, _settings: artifacts,
            lambda: (_ for _ in ()).throw(AssertionError("direct fallback")),
        )
        proxied = await make_tools(
            factory,
            tmp_path,
            settings=proxy_runtime_settings(),
            adapter_handles=AdapterHandles(tool_http=proxy),
        )
        with pytest.raises(HTTPToolError) as failed:
            await proxied["http_request"]("https://target.example/private")
        assert failed.value.code == "http_request_failed"
        assert artifacts.writes == 0
        await client.aclose()

    asyncio.run(scenario())


async def create_tools(
    tmp_path: Path,
    handler: Any,
    *,
    artifacts: FakeArtifactClient | None = None,
) -> tuple[dict[str, Any], WorkerState]:
    artifacts = artifacts or FakeArtifactClient()
    state = WorkerState()

    def direct() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

    factory = HTTPToolsetFactory(lambda _allocation, _settings: artifacts, direct)
    tools = await make_tools(factory, tmp_path, state=state)
    return tools, state


async def make_tools(
    factory: HTTPToolsetFactory,
    tmp_path: Path,
    *,
    state: WorkerState | None = None,
    settings: RuntimeSettings | None = None,
    adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
) -> dict[str, Any]:
    state = state or WorkerState()
    settings = settings or RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=30,
    )
    workspace_path = tmp_path / "allocation"
    result = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation-http",
        run_id="run-http",
        namespace="worker",
        runtime_settings=settings,
        workspace=AllocationWorkspace(root=tmp_path, path=workspace_path),
        state=state,
        adapter_handles=adapter_handles,
    )
    return dict(result)


async def close_tools(tools: dict[str, Any]) -> None:
    for tool in tools.values():
        await tool.close()


def proxy_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings(
        llmGatewayUrl="https://gateway.example/v1",
        llmGatewayToken="gateway-token",
        artifactApiUrl="https://control.example/private/v1",
        httpProxy=HTTPProxySettings(
            adapter="http-proxy@1",
            proxyUrl="https://proxy.example",
            targets=["tool-http"],
        ),
        requestTimeoutSeconds=30,
    )


class FakeArtifactClient:
    def __init__(self) -> None:
        self.payloads: dict[tuple[str, str, str], bytes] = {}
        self.media_types: dict[tuple[str, str, str], str] = {}
        self.writes = 0

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        assert expected_revision is None
        self.writes += 1
        exact = target.model_copy(update={"revision": f"revision-{self.writes}"})
        key = (exact.namespace, exact.name, exact.revision)
        self.payloads[key] = data
        self.media_types[key] = media_type
        return SimpleNamespace(artifact=exact)

    async def read_artifact(self, ref: ArtifactRef) -> Any:
        exact = ref.require_exact()
        key = (exact.namespace, exact.name, exact.revision)
        if key not in self.payloads:
            raise RuntimeError("not found")
        return SimpleNamespace(data=self.payloads[key], media_type=self.media_types[key])
