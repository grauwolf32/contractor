"""Unit tests for HTTPClient connection lifecycle.

Regression for a connection-pool leak: ``http_tools`` builds an ``HTTPClient``
and returns only tool closures — there is no teardown seam reachable from the
agent factories, so the persistent ``httpx.AsyncClient`` opened in ``__init__``
was never closed. The client is now created per request and closed via
``async with``; these tests pin that contract (no persistent client, each
per-request client is closed, cookies persist across requests).
"""
from __future__ import annotations

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
