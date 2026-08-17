"""Unit tests for HTTPClient connection lifecycle.

Regression for a connection-pool leak: ``http_tools`` builds an ``HTTPClient``
and returns only tool closures — there is no teardown seam reachable from the
agent factories, so the persistent ``httpx.AsyncClient`` opened in ``__init__``
was never closed. The client is now created per request and closed via
``async with``; these tests pin that contract (no persistent client, each
per-request client is closed, cookies persist across requests).
"""
from __future__ import annotations

import asyncio
import json
import warnings

import httpx
import pytest

from contractor.tools.http import HTTPClient, http_tools

_EXPECTED_TOOLS = {
    "http_request",
    "http_read_body",
    "http_history",
    "http_session_set",
    "http_session_get",
    "http_session_clear",
}


def test_http_tools_public_surface_unchanged():
    tools = http_tools(name="t")
    assert {t.__name__ for t in tools} == _EXPECTED_TOOLS


def test_no_persistent_async_client_leaks_on_build():
    # Building tools / a client must not open a long-lived httpx client.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        cli = HTTPClient(name="t")
        assert not hasattr(cli, "_client")


def test_cookie_jar_lives_on_the_client():
    cli = HTTPClient(name="t")
    cli.set_cookies({"a": "b"})
    assert cli.get_cookies() == {"a": "b"}
    cli.clear_session_state()
    assert cli.get_cookies() == {}


def _mock_client_factory(created: list[httpx.AsyncClient]):
    def fake_new_client(self: HTTPClient, timeout: float | None = None) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"set-cookie": "sid=42; Path=/"}, text="ok"
                )
            ),
            cookies=self._cookies,
        )
        created.append(client)
        return client

    return fake_new_client


@pytest.mark.asyncio
async def test_request_closes_its_client(monkeypatch):
    created: list[httpx.AsyncClient] = []
    monkeypatch.setattr(HTTPClient, "_new_client", _mock_client_factory(created))

    cli = HTTPClient(name="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        record = await cli.request(url="http://example.test/", method="GET")

    assert record["status"] == 200
    assert created, "expected a per-request client to be created"
    assert all(c.is_closed for c in created), "per-request clients must be closed"


@pytest.mark.asyncio
async def test_cookies_persist_across_requests(monkeypatch):
    created: list[httpx.AsyncClient] = []
    monkeypatch.setattr(HTTPClient, "_new_client", _mock_client_factory(created))

    cli = HTTPClient(name="t")
    await cli.request(url="http://example.test/", method="GET")
    # The Set-Cookie from the first response is retained via the shared jar,
    # even though that request's client has since been closed.
    assert cli.get_cookies().get("sid") == "42"
    assert len(created) == 1

    await cli.request(url="http://example.test/again", method="GET")
    assert len(created) == 2
    assert all(c.is_closed for c in created)


@pytest.mark.asyncio
async def test_aclose_is_a_safe_noop():
    # Kept for backward compatibility with ``async with HTTPClient(...)`` and
    # explicit aclose() call sites; must not raise even without a live client.
    cli = HTTPClient(name="t")
    await cli.aclose()
    async with HTTPClient(name="t") as ctx_cli:
        assert ctx_cli is not None


class _MemoryArtifactContext:
    def __init__(self) -> None:
        self.artifacts = {}

    async def load_artifact(self, filename: str):
        return self.artifacts.get(filename)

    async def save_artifact(self, filename: str, artifact):
        # Force a scheduling point so the test exercises the request lock rather
        # than accidentally passing because persistence completed synchronously.
        await asyncio.sleep(0)
        self.artifacts[filename] = artifact
        return 1


class _FailingFinalSessionContext(_MemoryArtifactContext):
    def __init__(self, *, block: bool = False) -> None:
        super().__init__()
        self.block = block
        self.session_saves = 0
        self.final_save_started = asyncio.Event()

    async def save_artifact(self, filename: str, artifact):
        if filename.endswith("/session.json"):
            self.session_saves += 1
            if self.session_saves == 2:
                self.final_save_started.set()
                if self.block:
                    await asyncio.Event().wait()
                raise OSError("session persistence failed")
        return await super().save_artifact(filename, artifact)


@pytest.mark.asyncio
async def test_overlapping_requests_reserve_distinct_persisted_ids(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=request.url.path.encode(),
            request=request,
        )

    def fake_new_client(
        self: HTTPClient, timeout: float | None = None
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(HTTPClient, "_new_client", fake_new_client)
    ctx = _MemoryArtifactContext()
    cli = HTTPClient(name="race")

    first, second = await asyncio.gather(
        cli.request(url="http://example.test/first", ctx=ctx),
        cli.request(url="http://example.test/second", ctx=ctx),
    )

    assert (first["request_id"], second["request_id"]) == (1, 2)
    first_body = json.loads(ctx.artifacts["http/race/responses/00000001.json"].text)
    second_body = json.loads(ctx.artifacts["http/race/responses/00000002.json"].text)
    assert first_body["text"] == "/first"
    assert second_body["text"] == "/second"

    session = json.loads(ctx.artifacts["http/race/session.json"].text)
    assert session["next_request_id"] == 3
    assert [entry["request_id"] for entry in session["history"]] == [1, 2]


@pytest.mark.asyncio
async def test_separate_clients_share_artifact_namespace_lock(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=request.url.path.encode(),
            request=request,
        )

    def fake_new_client(
        self: HTTPClient, timeout: float | None = None
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(HTTPClient, "_new_client", fake_new_client)
    ctx = _MemoryArtifactContext()
    first_client = HTTPClient(name="shared")
    second_client = HTTPClient(name="shared")

    first, second = await asyncio.gather(
        first_client.request(url="http://example.test/first", ctx=ctx),
        second_client.request(url="http://example.test/second", ctx=ctx),
    )

    assert {first["request_id"], second["request_id"]} == {1, 2}
    for record, expected in ((first, "/first"), (second, "/second")):
        body = json.loads(ctx.artifacts[record["body_artifact"]].text)
        assert body["text"] == expected

    session = json.loads(ctx.artifacts["http/shared/session.json"].text)
    assert session["next_request_id"] == 3
    assert [entry["request_id"] for entry in session["history"]] == [1, 2]


@pytest.mark.asyncio
async def test_final_session_save_failure_does_not_reuse_body_id(monkeypatch):
    def fake_new_client(
        self: HTTPClient, timeout: float | None = None
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/plain"},
                    content=request.url.path.encode(),
                    request=request,
                )
            )
        )

    monkeypatch.setattr(HTTPClient, "_new_client", fake_new_client)
    ctx = _FailingFinalSessionContext()

    with pytest.raises(OSError, match="session persistence failed"):
        await HTTPClient(name="save-failure").request(
            url="http://example.test/first", ctx=ctx
        )

    first_body_name = "http/save-failure/responses/00000001.json"
    assert json.loads(ctx.artifacts[first_body_name].text)["text"] == "/first"

    second = await HTTPClient(name="save-failure").request(
        url="http://example.test/second", ctx=ctx
    )

    assert second["request_id"] == 2
    assert json.loads(ctx.artifacts[first_body_name].text)["text"] == "/first"
    assert json.loads(ctx.artifacts[second["body_artifact"]].text)["text"] == "/second"


@pytest.mark.asyncio
async def test_cancellation_after_body_save_does_not_reuse_body_id(monkeypatch):
    def fake_new_client(
        self: HTTPClient, timeout: float | None = None
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/plain"},
                    content=request.url.path.encode(),
                    request=request,
                )
            )
        )

    monkeypatch.setattr(HTTPClient, "_new_client", fake_new_client)
    ctx = _FailingFinalSessionContext(block=True)
    request = asyncio.create_task(
        HTTPClient(name="cancelled").request(
            url="http://example.test/first", ctx=ctx
        )
    )

    await asyncio.wait_for(ctx.final_save_started.wait(), timeout=1)
    first_body_name = "http/cancelled/responses/00000001.json"
    assert json.loads(ctx.artifacts[first_body_name].text)["text"] == "/first"
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    second = await HTTPClient(name="cancelled").request(
        url="http://example.test/second", ctx=ctx
    )

    assert second["request_id"] == 2
    assert json.loads(ctx.artifacts[first_body_name].text)["text"] == "/first"
    assert json.loads(ctx.artifacts[second["body_artifact"]].text)["text"] == "/second"
