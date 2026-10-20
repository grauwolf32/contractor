from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from test_caido_adapter import adapter_context
from test_caido_read_tools import CAIDO_TOKEN, FakeArtifactClient, close_tools

from contractor_runtime.adapters import caido_graphql
from contractor_runtime.adapters.caido_graphql import CaidoGraphQLAdapter
from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import CaidoSettings, RuntimeSettings
from contractor_runtime.toolsets.caido import tools as caido_tools
from contractor_runtime.toolsets.caido.tools import CaidoToolError, CaidoToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

FIXTURE = json.loads((Path(__file__).parent / "testdata" / "caido-sitemap.json").read_text())


@asynccontextmanager
async def sitemap_server(tmp_path: Path, nodes: list[dict[str, Any]], *, descendant=False):
    requests = []
    tasks: set[asyncio.Task[Any]] = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            fields = {}
            for line in headers.splitlines()[1:]:
                name, separator, value = line.partition(":")
                if separator:
                    fields[name.lower()] = value.strip()
            assert fields["authorization"] == f"Bearer {CAIDO_TOKEN}"
            length = int(fields["content-length"])
            payload = json.loads(await reader.readexactly(length))
            requests.append(payload)
            field = "sitemapDescendantEntries" if descendant else "sitemapRootEntries"
            body = json.dumps({"data": {field: {"nodes": nodes}}}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    endpoint = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    settings = CaidoSettings(
        adapter="caido-graphql@1",
        endpoint=endpoint,
        bearerToken=CAIDO_TOKEN,
        requestTimeoutSeconds=5,
    )
    adapter = CaidoGraphQLAdapter(adapter_context(), settings)
    state = WorkerState()
    tools = dict(
        await CaidoToolsetFactory(lambda *_: FakeArtifactClient()).create_selected(
            selected=["caido_sitemap"],
            allocation_id="sitemap-allocation",
            run_id="sitemap-run",
            namespace="analyst",
            runtime_settings=RuntimeSettings(
                artifactApiUrl="https://control.example/private/v1",
                requestTimeoutSeconds=10,
                caido=settings,
            ),
            workspace=AllocationWorkspace(root=tmp_path, path=tmp_path / "allocation"),
            state=state,
            adapter_handles=adapter.handles,
        )
    )
    try:
        yield tools["caido_sitemap"], requests, state, adapter
    finally:
        await close_tools(tools)
        await adapter.close()
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


@pytest.mark.parametrize("count", FIXTURE["cardinalities"])
@pytest.mark.parametrize("kind", ["root", "descendant"])
def test_sitemap_pages_real_graphql_collection(tmp_path: Path, count: int, kind: str) -> None:
    nodes = [{**FIXTURE[kind], "id": str(100 + index)} for index in range(count)]

    async def scenario():
        async with sitemap_server(tmp_path, nodes, descendant=kind == "descendant") as (
            tool,
            requests,
            state,
            _adapter,
        ):
            first_args = (
                {"parent_id": "1", "depth": "ALL"}
                if kind == "descendant"
                else {"scope_id": "scope-1"}
            )
            page = await tool(**first_args)
            seen = []
            while True:
                assert len(page["entries"]) <= 100
                assert (
                    len(json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode())
                    <= 1024 * 1024
                )
                assert page["total_entries"] == count
                assert page["offset"] == len(seen)
                seen.extend(entry["id"] for entry in page["entries"])
                assert page["truncated"] == (len(seen) < count)
                if not page["truncated"]:
                    assert "next_cursor" not in page
                    break
                assert page["entries"]
                page = await tool(cursor=page["next_cursor"])
            assert seen == [node["id"] for node in nodes]
            assert len(requests) == 1
            assert requests[0]["variables"] == (
                {"parentId": "1", "depth": "ALL"}
                if kind == "descendant"
                else {"scopeId": "scope-1"}
            )
            assert CAIDO_TOKEN not in repr(state.metrics)

    asyncio.run(scenario())


def test_sitemap_fits_utf8_bytes_and_preserves_snapshot(tmp_path: Path) -> None:
    nodes = [
        {**FIXTURE["root"], "id": str(index), "label": "🧭" * 2048, "kind": "界" * 2730}
        for index in range(107)
    ]

    async def scenario():
        async with sitemap_server(tmp_path, nodes) as (tool, requests, _state, _adapter):
            first = await tool()
            assert 0 < len(first["entries"]) < 100
            cursor = first["next_cursor"]
            nodes[:] = [{**FIXTURE["root"], "id": "new"}]
            page = first
            seen = []
            while True:
                assert len(json.dumps(page, ensure_ascii=False).encode()) <= 1024 * 1024
                seen.extend(entry["id"] for entry in page["entries"])
                if not page["truncated"]:
                    break
                page = await tool(cursor=page["next_cursor"])
            assert seen == [str(index) for index in range(107)]
            assert await tool(cursor=cursor) == await tool(cursor=cursor)
            assert len(requests) == 1
            assert (await tool())["entries"][0]["id"] == "new"
            with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                await tool(cursor=cursor)
            assert len(requests) == 2

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments",
    [{"limit": 0}, {"limit": 101}, {"limit": True}, {"cursor": None}, {"cursor": "unknown.1"}],
)
def test_sitemap_rejects_invalid_inputs_without_fetch(tmp_path: Path, arguments) -> None:
    async def scenario():
        async with sitemap_server(tmp_path, []) as (tool, requests, _state, _adapter):
            with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                await tool(**arguments)
            assert requests == []

    asyncio.run(scenario())


def test_sitemap_cursor_binding_and_release(tmp_path: Path) -> None:
    nodes = [{**FIXTURE["root"], "id": str(index)} for index in range(107)]

    async def scenario():
        async with sitemap_server(tmp_path, nodes) as (tool, requests, _state, _adapter):
            first = await tool(limit=7)
            cursor = first["next_cursor"]
            for invalid in (
                cursor.rsplit(".", 1)[0] + ".107",
                cursor.replace(".7", ".07"),
                cursor + ".1",
            ):
                with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                    await tool(cursor=invalid)
            for arguments in ({"parent_id": "1"}, {"scope_id": "1"}, {"depth": "ALL"}):
                with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                    await tool(cursor=cursor, **arguments)
            async with sitemap_server(tmp_path, nodes) as (other, foreign, _state, _adapter):
                await other()
                with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                    await other(cursor=cursor)
                assert len(foreign) == 1
            assert len(requests) == 1
            await tool.close()
            assert tool._session._sitemap_snapshot is None
            with pytest.raises(CaidoToolError, match="caido_request_failed"):
                await tool(cursor=cursor)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["malformed", "oversized"])
def test_sitemap_rejects_invalid_source_beyond_first_page(tmp_path: Path, monkeypatch, failure):
    nodes = [{**FIXTURE["root"], "id": str(index)} for index in range(107)]

    async def scenario():
        async with sitemap_server(tmp_path, nodes) as (tool, requests, state, _adapter):
            cursor = (await tool())["next_cursor"]
            if failure == "malformed":
                nodes[-1]["hasDescendants"] = "invalid"
            else:
                monkeypatch.setattr(caido_graphql, "MAX_CAIDO_RESPONSE_BYTES", 1024)
            with pytest.raises(CaidoToolError) as error:
                await tool()
            assert error.value.code == (
                "caido_response_invalid" if failure == "malformed" else "caido_response_too_large"
            )
            assert tool._session._sitemap_snapshot is None
            with pytest.raises(CaidoToolError, match="caido_request_invalid"):
                await tool(cursor=cursor)
            assert len(requests) == 2
            assert CAIDO_TOKEN not in f"{error.value!r} {state.metrics!r}"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["decode", "snapshot", "page"])
@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_sitemap_joins_processing_before_release(tmp_path: Path, monkeypatch, phase, stop):
    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        if phase == "decode":
            target, name = caido_graphql, "_decode_response"
        elif phase == "snapshot":
            target, name = caido_tools, "_sitemap_snapshot"
        else:
            target, name = caido_tools._SitemapSnapshot, "page"
        original = getattr(target, name)

        def blocked(*args, **kwargs):
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(10), "test did not release processing"
                return original(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(target, name, blocked)
        async with sitemap_server(tmp_path, [FIXTURE["root"]]) as (
            tool,
            _requests,
            _state,
            _adapter,
        ):
            budget = asyncio.timeout(None)

            async def call():
                async with budget:
                    return await tool()

            task = asyncio.create_task(call())
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                if stop == "deadline":
                    budget.reschedule(loop.time())
                else:
                    task.cancel()
                close = asyncio.create_task(tool.close())
                await asyncio.sleep(0)
                assert not task.done()
                assert not close.done()
            finally:
                release.set()
            with pytest.raises(TimeoutError if stop == "deadline" else asyncio.CancelledError):
                await task
            await close
            assert finished.is_set()
            assert tool._session._sitemap_snapshot is None

    asyncio.run(scenario())
