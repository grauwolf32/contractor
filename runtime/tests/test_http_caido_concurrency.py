"""Cancellation and concurrency invariants for allocation-owned HTTP/Caido state."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_caido_read_tools import close_tools as close_caido_tools
from test_caido_read_tools import create_tools as create_caido_tools
from test_http_toolset import FakeArtifactClient, close_tools, create_tools

from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.http.tools import HTTPToolError


class BlockingArtifactClient(FakeArtifactClient):
    """Commit the first body and then expose a cancellation point to the test."""

    def __init__(self) -> None:
        super().__init__()
        self.committed = asyncio.Event()
        self.block_after_commit = True

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        result = await super().write_artifact(
            target,
            data=data,
            media_type=media_type,
            expected_revision=expected_revision,
        )
        if self.block_after_commit:
            self.committed.set()
            await asyncio.Event().wait()
        return result


def test_http_cancellation_after_body_commit_never_reuses_id_or_selects_orphan(
    tmp_path: Path,
) -> None:
    observed: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        headers = {"content-type": "text/plain"}
        if request.url.path == "/first":
            headers["set-cookie"] = "cancelled=must-not-survive; Path=/"
        return httpx.Response(
            200,
            content=request.url.path.encode(),
            headers=headers,
            request=request,
        )

    async def scenario() -> None:
        artifacts = BlockingArtifactClient()
        tools, _state = await create_tools(tmp_path, handler, artifacts=artifacts)
        first = asyncio.create_task(tools["http_request"]("https://target.example/first"))
        await asyncio.wait_for(artifacts.committed.wait(), timeout=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert artifacts.writes == 1
        first_key = next(iter(artifacts.payloads))
        assert first_key[1].endswith(".000001")
        with pytest.raises(HTTPToolError) as missing:
            await tools["http_read_body"](1)
        assert missing.value.code == "http_body_not_found"
        assert await tools["http_history"]() == []

        artifacts.block_after_commit = False
        second = await tools["http_request"]("https://target.example/second")
        assert second["request_id"] == 2
        assert second["body_artifact"]["name"].endswith(".000002")
        assert "cookie" not in observed[-1].headers
        assert (await tools["http_read_body"](2))["data"] == "/second"
        assert [item["request_id"] for item in await tools["http_history"]()] == [2]
        assert (
            artifacts.payloads[first_key]
            != artifacts.payloads[
                (
                    second["body_artifact"]["namespace"],
                    second["body_artifact"]["name"],
                    second["body_artifact"]["revision"],
                )
            ]
        )
        await close_tools(tools)

    asyncio.run(scenario())


def test_http_requests_and_body_reads_are_serialized_and_exact(tmp_path: Path) -> None:
    active = 0
    maximum_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return httpx.Response(
            200,
            content=request.url.path.encode(),
            headers={"content-type": "text/plain"},
            request=request,
        )

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        results = await asyncio.gather(
            *(tools["http_request"](f"https://target.example/item-{index}") for index in range(8))
        )
        assert maximum_active == 1
        assert {item["request_id"] for item in results} == set(range(1, 9))

        bodies = await asyncio.gather(
            *(tools["http_read_body"](item["request_id"]) for item in results)
        )
        by_id = {body["request_id"]: body["data"] for body in bodies}
        for result in results:
            assert by_id[result["request_id"]] == httpx.URL(result["final_url"]).path
        await close_tools(tools)

    asyncio.run(scenario())


def test_http_close_waits_for_active_request_then_erases_the_session(tmp_path: Path) -> None:
    entered = asyncio.Event()
    proceed = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await proceed.wait()
        return httpx.Response(200, content=b"ok", request=request)

    async def scenario() -> None:
        tools, _state = await create_tools(tmp_path, handler)
        request = asyncio.create_task(tools["http_request"]("https://target.example/active"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        closing = asyncio.create_task(close_tools(tools))
        await asyncio.sleep(0)
        assert not closing.done()

        proceed.set()
        assert (await request)["request_id"] == 1
        await closing
        with pytest.raises(HTTPToolError) as closed:
            await tools["http_request"]("https://target.example/after-close")
        assert closed.value.code == "http_request_failed"

    asyncio.run(scenario())


def test_cancelled_caido_mutation_consumes_opaque_tag_and_slot_remains_usable(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        first_mutation_started = asyncio.Event()
        payloads: list[dict[str, Any]] = []
        block_first = True

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal block_first
            payload = json.loads(request.content)
            payloads.append(payload)
            operation = payload["operationName"]
            if operation == "CreateReplaySession" and block_first:
                block_first = False
                first_mutation_started.set()
                await asyncio.Event().wait()
            if operation == "CreateReplaySession":
                data = {
                    "createReplaySession": {
                        "session": {
                            "id": "session-2",
                            "name": "replay",
                            "activeEntry": {"id": "entry-2"},
                        }
                    }
                }
            elif operation == "StartReplayTask":
                data = {
                    "startReplayTask": {
                        "error": None,
                        "task": {"id": "task-2", "replayEntry": {"id": "entry-2"}},
                    }
                }
            else:
                raise AssertionError(f"unexpected operation {operation}")
            return httpx.Response(200, json={"data": data}, request=request)

        tools, _state, handle = await create_caido_tools(
            tmp_path,
            handler,
            FakeArtifactClient(),
            selected={"caido_replay"},
        )
        first = asyncio.create_task(
            tools["caido_replay"](
                raw_request="GET /first HTTP/1.1\r\nHost: target.example\r\n\r\n",
                host="target.example",
                wait=False,
            )
        )
        await asyncio.wait_for(first_mutation_started.wait(), timeout=1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        second = await tools["caido_replay"](
            raw_request="GET /second HTTP/1.1\r\nHost: target.example\r\n\r\n",
            host="target.example",
            wait=False,
        )
        assert second["status"] == "started"
        assert second["request_tag"].endswith("-c000002")

        create_payloads = [
            item for item in payloads if item["operationName"] == "CreateReplaySession"
        ]
        assert len(create_payloads) == 2
        submitted = [
            base64.b64decode(item["variables"]["input"]["requestSource"]["raw"]["raw"])
            for item in create_payloads
        ]
        assert b"-c000001\r\n" in submitted[0]
        assert b"-c000002\r\n" in submitted[1]
        await close_caido_tools(tools)
        await handle.close()
        assert handle.closed

    asyncio.run(scenario())
