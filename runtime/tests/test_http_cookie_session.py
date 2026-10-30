from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from http_cookie_store import CookieArtifactStore

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts import HTTPOriginTargetSettings, RuntimeSettings
from contractor_runtime.core.http_session import matching_cookies
from contractor_runtime.toolsets.common.http_limits import MAX_COOKIE_SESSION_BYTES, MAX_COOKIES
from contractor_runtime.toolsets.http.cookies import (
    COOKIE_BINDING,
    CookieCheckpoint,
    decode_cookies,
    encode_cookies,
)
from contractor_runtime.toolsets.http.tools import HTTPToolError, HTTPToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

ORIGIN = ("https", "target.example", 443)


async def tools_for(
    tmp_path: Path,
    store: CookieArtifactStore,
    handler: Any,
    *,
    proxy: bool = False,
    namespace: str = "worker",
    origin: str = "https://target.example",
    origin_token: str | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    factory = HTTPToolsetFactory(lambda: client, artifact_client_factory=lambda *_: store, now=now)
    settings = RuntimeSettings(
        llmGatewayUrl="https://gateway.example",
        llmGatewayToken="infrastructure-canary",
        artifactApiUrl="https://control.example/private/v1",
        requestTimeoutSeconds=5,
        httpOriginTarget=HTTPOriginTargetSettings(url=origin, bearerToken=origin_token),
    )
    state = WorkerState()
    tools = dict(
        await factory.create_selected(
            selected=sorted(factory.exported_tools),
            allocation_id="replacement",
            run_id="run",
            namespace=namespace,
            runtime_settings=settings,
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path / "slot"),
            state=state,
            adapter_handles=AdapterHandles(tool_http=ProxyHTTPClient(client))
            if proxy
            else AdapterHandles(),
        )
    )
    # The mock adapter owns its client in production; tests must close it too.
    tools["_client"] = client
    tools["_state"] = state
    return tools


async def close(tools: dict[str, Any]) -> None:
    await tools["http_request"].close()
    await tools["_client"].aclose()


@pytest.mark.parametrize("proxy", [False, True])
def test_replacement_restores_login_and_clear_without_auth_or_history(
    tmp_path: Path,
    proxy: bool,
) -> None:
    async def scenario() -> None:
        store = CookieArtifactStore()
        seen = []

        def target(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path == "/login":
                return httpx.Response(
                    200, headers={"set-cookie": "sid=target-cookie; Path=/; HttpOnly"}
                )
            return httpx.Response(
                200 if request.headers.get("cookie") == "sid=target-cookie" else 401
            )

        first = await tools_for(tmp_path, store, target, proxy=proxy)
        await first["http_session_set"](auth={"kind": "bearer", "token": "ephemeral-auth"})
        await first["http_request"]("https://target.example/login", method="POST")
        assert store.writes == 1
        stored = store.current[("worker", COOKIE_BINDING)]
        assert b"target-cookie" in stored.data
        assert b"ephemeral-auth" not in stored.data and b"infrastructure-canary" not in stored.data
        await close(first)
        second = await tools_for(tmp_path, store, target, proxy=proxy)
        assert await second["http_session_get"]() == {
            "auth_kind": "none",
            "default_headers": {},
            "cookie_names": ["sid"],
            "cookie_count": 1,
            "history_count": 0,
        }
        assert (await second["http_request"]("https://target.example/private"))["status"] == 200
        assert "authorization" not in seen[-1].headers
        assert store.writes == 1  # unchanged jar does not append revisions
        other = await tools_for(tmp_path, store, target, proxy=proxy, namespace="other")
        assert (await other["http_request"]("https://target.example/private"))["status"] == 401
        await close(other)
        await second["http_session_clear"]()
        await close(second)
        third = await tools_for(tmp_path, store, target, proxy=proxy)
        assert (await third["http_request"]("https://target.example/private"))["status"] == 401
        assert store.writes == 2
        await close(third)
        with pytest.raises(HTTPToolError, match="http_cookie_state_invalid"):
            await tools_for(tmp_path, store, target, origin="https://other.example")

    asyncio.run(scenario())


@pytest.mark.parametrize("proxy", [False, True])
@pytest.mark.parametrize("operation", ["get", "merge"])
def test_expired_cookies_leave_live_session_before_view_or_merge(tmp_path, proxy, operation):
    async def scenario():
        store = CookieArtifactStore()
        now = [2524608000.0]  # 2050; cookies below expire in 2080.

        def target(request):
            return httpx.Response(
                200,
                headers=[("set-cookie", "session=ready; Path=/")]
                + [
                    ("set-cookie", f"c{i}=temporary; Expires=Mon, 01 Jan 2080 00:00:00 GMT; Path=/")
                    for i in range(127)
                ],
            )

        tools = await tools_for(tmp_path, store, target, proxy=proxy, now=lambda: now[0])
        try:
            assert (await tools["http_request"]("https://target.example/prime"))["status"] == 200
            assert (await tools["http_session_get"]())["cookie_count"] == 128
            now[0] = 4102444800.0  # 2100, without sleeping or changing system time.
            if operation == "get":
                view = await tools["http_session_get"]()
                assert view["cookie_names"] == ["session"] and view["cookie_count"] == 1
                assert store.writes == 1
            view = await tools["http_session_set"](cookies={"fresh": "value"})
            assert view["cookie_names"] == ["fresh", "session"] and view["cookie_count"] == 2
            assert store.writes == 2
            restored = decode_cookies(
                store.current[("worker", COOKIE_BINDING)].data, ORIGIN, now[0]
            )
            assert {cookie.name for cookie in restored.jar} == {"session", "fresh"}
        finally:
            await close(tools)

    asyncio.run(scenario())


def test_encoding_removes_expired_entries_from_source_jar():
    cookies = httpx.Cookies({"session": "ready"})
    for i in range(127):
        cookies.set(f"c{i}", "temporary")
    for cookie in cookies.jar:
        if cookie.name != "session":
            cookie.expires = 1001
    encoded = encode_cookies(cookies, None, 1002)
    assert {cookie.name for cookie in cookies.jar} == {"session"}
    assert {cookie.name for cookie in decode_cookies(encoded, None, 1002).jar} == {"session"}


def test_matching_expiry_and_all_cookie_attributes_roundtrip() -> None:
    cookies = httpx.Cookies()
    response = httpx.Response(
        200,
        headers=[
            ("set-cookie", "host=one; Path=/private; Secure; HttpOnly; SameSite=Strict"),
            ("set-cookie", "domain=two; Domain=target.example; Path=/"),
            ("set-cookie", "expired=gone; Expires=Mon, 01 Jan 2080 00:00:00 GMT; Path=/"),
        ],
        request=httpx.Request("GET", "https://target.example/login"),
    )
    cookies.extract_cookies(response)
    before = {c.name: vars(c).copy() for c in cookies.jar}
    restored = decode_cookies(encode_cookies(cookies, ORIGIN, 0), ORIGIN, 0)
    assert {c.name: vars(c).copy() for c in restored.jar} == before
    for url, names in [
        ("https://target.example/private", {"host", "domain", "expired"}),
        ("http://target.example/private", {"domain", "expired"}),
        ("https://target.example/public", {"domain", "expired"}),
        ("https://sub.target.example/private", {"domain"}),
        ("https://other.example/private", set()),
    ]:
        request = httpx.Request("GET", url, cookies=matching_cookies(restored, url))
        actual = {
            item.split("=", 1)[0] for item in request.headers.get("cookie", "").split("; ") if item
        }
        assert actual == names
    expired_at = before["expired"]["expires"]
    later = decode_cookies(encode_cookies(cookies, ORIGIN, 0), ORIGIN, expired_at + 1)
    assert {c.name for c in later.jar} == {"host", "domain"}


def test_shared_cookie_document_strictness_and_exact_limits() -> None:
    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "api/testdata/http-cookie-session.json").read_bytes()
    )
    data = json.dumps(fixture["document"], separators=(",", ":"), sort_keys=True).encode()
    cookies = decode_cookies(data, ORIGIN, 0)
    assert encode_cookies(cookies, ORIGIN, 0) == data
    assert fixture["limits"]["count"] == MAX_COOKIES
    assert fixture["limits"]["bytes"] == MAX_COOKIE_SESSION_BYTES
    count = httpx.Cookies()
    for i in range(MAX_COOKIES):
        count.set(f"cookie-{i}", "x")
    assert len(decode_cookies(encode_cookies(count, None, 0), None, 0)) == MAX_COOKIES
    count.set("one-over", "x")
    with pytest.raises(HTTPToolError):
        encode_cookies(count, None, 0)
    # One cookie exactly fills the encoded JSON budget, including all metadata.
    size = httpx.Cookies({"sid": ""})
    overhead = len(encode_cookies(size, None, 0))
    size.set("sid", "x" * (MAX_COOKIE_SESSION_BYTES - overhead))
    exact = encode_cookies(size, None, 0)
    assert len(exact) == MAX_COOKIE_SESSION_BYTES
    assert len(decode_cookies(exact, None, 0)) == 1
    size.set("sid", "x" * (MAX_COOKIE_SESSION_BYTES - overhead + 1))
    with pytest.raises(HTTPToolError):
        encode_cookies(size, None, 0)
    # The budget counts encoded UTF-8 bytes. A cookie value must be ASCII to be
    # sent back, but a path (never sent in a Cookie header) may be multibyte.
    remaining = MAX_COOKIE_SESSION_BYTES - overhead
    multibyte = httpx.Cookies()
    multibyte.set("sid", "", path="/" + "é" * (remaining // 2) + "x" * (remaining % 2))
    assert len(encode_cookies(multibyte, None, 0)) == MAX_COOKIE_SESSION_BYTES
    multibyte = httpx.Cookies()
    multibyte.set("sid", "", path="/" + "é" * (remaining // 2 + 1))
    with pytest.raises(HTTPToolError):
        encode_cookies(multibyte, None, 0)
    # A cookie that could not be sent back is invalid on write and on read.
    for unsendable in ("caf\u00e9", "trailing ", "a\x0bb"):
        with pytest.raises(HTTPToolError, match="http_cookie_state_invalid"):
            encode_cookies(httpx.Cookies({"sid": unsendable}), None, 0)
    for invalid in [
        data + b" ",  # valid whitespace but one-over bound covered separately below
    ]:
        assert len(decode_cookies(invalid, ORIGIN, 0)) == 1
    for invalid in [
        data.replace(b'"schema_version":1', b'"schema_version":2'),
        data.replace(b'"schema_version":1', b'"schema_version":true'),
        data.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
        data[:-1] + b',"unknown":0}',
        b'{"cookies":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
        exact + b" ",
        b"[]",
        data.replace(b'"value":"target-cookie"', b'"value":"caf\\u00e9"'),
        data.replace(b'"value":"target-cookie"', b'"value":"target-cookie "'),
        data.replace(b'"name":"sid"', b'"name":" sid"'),
    ]:
        with pytest.raises(HTTPToolError, match="http_cookie_state_invalid"):
            decode_cookies(invalid, ORIGIN, 0)


def test_response_loss_reconciles_exact_bytes_and_conflict_never_overwrites() -> None:
    class Lost(CookieArtifactStore):
        async def write_artifact(self, *args: Any, **kwargs: Any) -> Any:
            value = await super().write_artifact(*args, **kwargs)
            if self.writes == 1:
                raise ArtifactTransportError("response lost")
            return value

    async def scenario() -> None:
        store = Lost()
        first = CookieCheckpoint(store, "worker", None, lambda: 0, 2)
        stale = CookieCheckpoint(store, "worker", None, lambda: 0, 2)
        await first.restore()
        await stale.restore()
        await first.save(httpx.Cookies({"sid": "one"}))
        assert store.writes == 1
        with pytest.raises(HTTPToolError, match="http_cookie_state_conflict"):
            await stale.save(httpx.Cookies({"sid": "stale"}))
        with pytest.raises(HTTPToolError, match="http_cookie_state_conflict"):
            await stale.recover()
        assert store.writes == 1
        # A second inspection may adopt the newer state, never replay stale bytes.
        assert (await stale.recover()).get("sid") == "one"
        assert store.writes == 1
        assert b'"one"' in store.current[("worker", COOKIE_BINDING)].data

    asyncio.run(scenario())


def test_completed_post_reports_checkpoint_failure_and_get_repairs_only_storage(
    tmp_path: Path,
) -> None:
    class Down(CookieArtifactStore):
        down = True

        async def write_artifact(self, *args: Any, **kwargs: Any) -> Any:
            if self.down:
                raise ArtifactAPIError(503, "storage_transaction_conflict", True)
            return await super().write_artifact(*args, **kwargs)

    async def scenario() -> None:
        store = Down()
        posts = 0

        def target(request: httpx.Request) -> httpx.Response:
            nonlocal posts
            posts += 1
            return httpx.Response(
                201,
                headers={"set-cookie": "sid=one; Path=/", "content-type": "text/plain"},
                content=b"created",
            )

        tools = await tools_for(tmp_path, store, target)
        result = await tools["http_request"]("https://target.example/create", method="POST")
        assert result["status"] == 201 and result["body_preview"] == "created"
        assert result["cookie_checkpoint"]["state"] == "unconfirmed"
        assert result["cookie_checkpoint"]["code"] == "http_cookie_storage_unavailable"
        with pytest.raises(HTTPToolError):
            await tools["http_request"]("https://target.example/create", method="POST")
        store.down = False
        assert (await tools["http_session_get"]())["cookie_names"] == ["sid"]
        assert posts == 1 and store.writes == 1
        await close(tools)

    asyncio.run(scenario())


def test_cancelled_checkpoint_is_joined_before_owner_can_close() -> None:
    class Held(CookieArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def write_artifact(self, *args: Any, **kwargs: Any) -> Any:
            self.entered.set()
            await self.release.wait()
            return await super().write_artifact(*args, **kwargs)

    async def scenario() -> None:
        store = Held()
        checkpoint = CookieCheckpoint(store, "worker", None, lambda: 0, 2)
        task = asyncio.create_task(checkpoint.save(httpx.Cookies({"sid": "one"})))
        await store.entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        restored = CookieCheckpoint(store, "worker", None, lambda: 0, 2)
        assert len(await restored.restore()) == 1 and store.writes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("proxy", [False, True])
def test_cancelled_incomplete_response_never_replaces_durable_cookies(
    tmp_path: Path,
    proxy: bool,
) -> None:
    class Incomplete(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def __aiter__(self):
            self.entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"

    async def scenario() -> None:
        store = CookieArtifactStore()
        stream = Incomplete()

        def target(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/incomplete":
                return httpx.Response(
                    200, headers={"set-cookie": "sid=uncommitted; Path=/"}, stream=stream
                )
            return httpx.Response(200)

        first = await tools_for(tmp_path, store, target, proxy=proxy)
        await first["http_session_set"](cookies={"sid": "committed"})
        task = asyncio.create_task(first["http_request"]("https://target.example/incomplete"))
        await stream.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await close(first)
        second = await tools_for(tmp_path, store, target, proxy=proxy)
        assert (await second["http_session_get"]())["cookie_names"] == ["sid"]
        assert b"uncommitted" not in store.current[("worker", COOKIE_BINDING)].data
        assert store.writes == 1
        await close(second)

    asyncio.run(scenario())


def test_cancelled_clear_cannot_resurrect_its_committed_cookie_jar(tmp_path: Path) -> None:
    class HeldClear(CookieArtifactStore):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def write_artifact(self, *args: Any, **kwargs: Any) -> Any:
            if self.writes == 1:
                self.entered.set()
                await self.release.wait()
            return await super().write_artifact(*args, **kwargs)

    async def scenario() -> None:
        store = HeldClear()
        seen = []

        def target(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200)

        tools = await tools_for(tmp_path, store, target)
        await tools["http_session_set"](
            cookies={"sid": "old"},
            auth={"kind": "bearer", "token": "ephemeral"},
        )
        clearing = asyncio.create_task(tools["http_session_clear"]())
        await store.entered.wait()
        clearing.cancel()
        store.release.set()
        with pytest.raises(asyncio.CancelledError):
            await clearing
        assert (await tools["http_session_get"]())["cookie_count"] == 0
        await tools["http_request"]("https://target.example/private")
        assert "cookie" not in seen[-1].headers and "authorization" not in seen[-1].headers
        assert store.writes == 2
        await close(tools)

    asyncio.run(scenario())


@pytest.mark.parametrize("proxy", [False, True])
def test_target_cookie_that_cannot_be_sent_back_is_ignored(tmp_path: Path, proxy: bool) -> None:
    async def scenario() -> None:
        store = CookieArtifactStore()
        sent: list[str | None] = []

        def target(request: httpx.Request) -> httpx.Response:
            sent.append(request.headers.get("cookie"))
            if request.url.path == "/login":
                return httpx.Response(200, headers={"set-cookie": "sid=good; Path=/"})
            if request.url.path == "/set":
                return httpx.Response(
                    200,
                    headers=[
                        # UTF-8 that httpx cannot encode, replacing a good cookie.
                        (b"set-cookie", "sid=caf\u00e9; Path=/".encode()),
                        (b"set-cookie", b"ff=a\x0cb; Path=/"),
                        (b"set-cookie", b"other=fine; Path=/"),
                    ],
                )
            return httpx.Response(200)

        tools = await tools_for(tmp_path, store, target, proxy=proxy)
        await tools["http_request"]("https://target.example/login")
        result = await tools["http_request"]("https://target.example/set")
        assert result["status"] == 200
        assert "cookie_checkpoint" not in result
        assert (await tools["http_session_get"]())["cookie_names"] == ["other", "sid"]
        # The next request to the origin is sent, with the cookie the bad one
        # would have replaced.
        assert (await tools["http_request"]("https://target.example/next"))["status"] == 200
        assert sorted((sent[-1] or "").split("; ")) == ["other=fine", "sid=good"]
        payload = store.current[("worker", COOKIE_BINDING)].data
        assert b"good" in payload and b"fine" in payload
        assert b"caf" not in payload and b'"ff"' not in payload
        await close(tools)
        restored = await tools_for(tmp_path, store, target, proxy=proxy)
        assert (await restored["http_session_get"]())["cookie_names"] == ["other", "sid"]
        await close(restored)

    asyncio.run(scenario())


def test_only_target_cookies_persist_and_metrics_remain_content_free(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = CookieArtifactStore()
        project_auth = "project-authorization-canary"

        def target(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"] == "Bearer " + project_auth
            return httpx.Response(200, headers={"set-cookie": "sid=target-only-canary; Path=/"})

        tools = await tools_for(tmp_path, store, target, origin_token=project_auth)
        await tools["http_session_set"](
            headers={"X-Api-Key": "ephemeral-header-canary"},
            auth={"kind": "bearer", "token": "ephemeral-auth-canary"},
        )
        result = await tools["http_request"]("https://target.example/private")
        projection = await tools["http_session_get"]()
        payload = store.current[("worker", COOKIE_BINDING)].data
        assert b"target-only-canary" in payload
        for excluded in (
            project_auth,
            "infrastructure-canary",
            "ephemeral-header-canary",
            "ephemeral-auth-canary",
        ):
            assert excluded.encode() not in payload
        metrics = repr(tools["_state"].metrics.snapshot())
        for excluded in (
            project_auth,
            "target-only-canary",
            "ephemeral-header-canary",
            "ephemeral-auth-canary",
        ):
            assert excluded not in metrics
            assert excluded not in repr(result)
            assert excluded not in repr(projection)
        await close(tools)

    asyncio.run(scenario())
