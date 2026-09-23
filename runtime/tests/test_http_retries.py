"""Retry waits for http_request: bounded backoff, Retry-After and the call deadline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from test_http_toolset import close_tools, create_tools

import contractor_runtime.toolsets.http.tools as http_tools
from contractor_runtime.toolsets.http.tools import HTTPToolError


def record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(http_tools, "_retry_sleep", sleep)
    return sleeps


def test_retryable_statuses_back_off_and_honour_bounded_retry_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = record_sleeps(monkeypatch)
    counts: dict[str, int] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counts[path] = counts.get(path, 0) + 1
        if path == "/backoff" and counts[path] < 3:
            return httpx.Response(429, request=request)
        if path == "/after" and counts[path] == 1:
            return httpx.Response(503, headers={"retry-after": "2"}, request=request)
        if path == "/long":
            return httpx.Response(503, headers={"retry-after": "30"}, request=request)
        if path == "/tight":
            return httpx.Response(503, headers={"retry-after": "3"}, request=request)
        if path == "/junk" and counts[path] == 1:
            return httpx.Response(503, headers={"retry-after": "soon"}, request=request)
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        request = tools["http_request"]

        backoff = await request("https://target.example/backoff")
        assert (backoff["status"], backoff["retries"]) == (200, 2)
        assert sleeps == [0.25, 0.5]

        sleeps.clear()
        after = await request("https://target.example/after")
        assert (after["status"], after["retries"]) == (200, 1)
        assert sleeps == [2.0]

        # A Retry-After beyond the cap returns the target's answer at once.
        sleeps.clear()
        long = await request("https://target.example/long")
        assert (long["status"], long["retries"]) == (503, 0)
        assert sleeps == []

        # So does a wait that would leave the retry too little of the deadline.
        tight = await request("https://target.example/tight", timeout=4)
        assert (tight["status"], tight["retries"]) == (503, 0)
        assert sleeps == []

        # An unparsable Retry-After falls back to the ordinary backoff.
        junk = await request("https://target.example/junk")
        assert (junk["status"], junk["retries"]) == (200, 1)
        assert sleeps == [0.25]
        assert counts == {"/backoff": 3, "/after": 2, "/long": 1, "/tight": 1, "/junk": 2}
        await close_tools(tools)

    asyncio.run(scenario())


def test_transport_failures_back_off_within_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps = record_sleeps(monkeypatch)
    attempts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.path)
        raise httpx.ConnectError("unreachable", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        with pytest.raises(HTTPToolError) as failure:
            await tools["http_request"]("https://target.example/down")
        assert failure.value.code == "http_request_failed"
        assert len(attempts) == 3
        assert sleeps == [0.25, 0.5]

        # With one second left, the second wait no longer fits the deadline.
        sleeps.clear()
        attempts.clear()
        monkeypatch.setattr(http_tools, "RETRY_BACKOFF_SECONDS", 0.4)
        with pytest.raises(HTTPToolError):
            await tools["http_request"]("https://target.example/down", timeout=1)
        assert len(attempts) == 2
        assert sleeps == [0.4]
        await close_tools(tools)

    asyncio.run(scenario())


def test_retry_waits_really_pause_between_attempts(tmp_path: Path) -> None:
    seen: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(asyncio.get_running_loop().time())
        if len(seen) == 1:
            return httpx.Response(503, headers={"retry-after": "1"}, request=request)
        return httpx.Response(200, request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        result = await tools["http_request"]("https://target.example/")
        assert (result["status"], result["retries"]) == (200, 1)
        assert seen[1] - seen[0] >= 0.95
        await close_tools(tools)

    asyncio.run(scenario())
