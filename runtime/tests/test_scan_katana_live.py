"""Optional real Katana checks; all crawl targets are controlled loopback servers."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import pytest
from target_policy_fixtures import SCAN_TEST_POLICY

from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.scan.tools import KatanaTool, ScanToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

KATANA_BINARY = os.environ.get("KATANA_TEST_BINARY") or shutil.which("katana")
pytestmark = pytest.mark.skipif(not KATANA_BINARY, reason="optional Katana binary is absent")


class CapturedArtifacts:
    def __init__(self):
        self.writes = []

    async def read_artifact(self, ref, **kwargs):
        raise ArtifactAPIError(404, "artifact_not_found", False)

    async def write_artifact(self, ref, *, data, media_type, expected_revision):
        assert expected_revision is None
        assert ref.revision is None
        exact = ArtifactRef(namespace=ref.namespace, name=ref.name, revision="targets-1")
        self.writes.append((exact, data, media_type))
        return SimpleNamespace(artifact=exact, media_type=media_type, size=len(data))


async def make_live_katana(tmp_path, monkeypatch):
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    (executable_dir / "katana").symlink_to(Path(KATANA_BINARY).resolve())
    monkeypatch.setenv("PATH", str(executable_dir))
    artifacts = CapturedArtifacts()
    factory = ScanToolsetFactory(
        lambda *_: artifacts, scanners=[KatanaTool], target_policy=SCAN_TEST_POLICY
    )
    assert await factory.probe() == {"scan_katana"}
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkerState()
    tools = await factory.create_selected(
        selected=["scan_katana"],
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=30
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
    )
    return tools["scan_katana"], artifacts, state, workspace


@asynccontextmanager
async def fixture_server(respond):
    requests = []
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            method, target, _ = headers.split(b"\r\n", 1)[0].decode().split(" ")
            requests.append((method, target))
            status, extra_headers, body = await respond(target)
            writer.write(
                f"HTTP/1.1 {status}\r\nConnection: close\r\n"
                "Content-Type: text/html\r\nSet-Cookie: session=private-canary\r\n"
                f"{extra_headers}Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
        origin = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            yield origin, requests
        finally:
            for task in tuple(handlers):
                task.cancel()
            if handlers:
                await asyncio.gather(*tuple(handlers), return_exceptions=True)


async def plain_page(_target):
    return "200 OK", "", b"fixture page"


def exported_targets(result, artifacts):
    assert len(artifacts.writes) == 1
    ref, data, media_type = artifacts.writes[0]
    assert result["targetsArtifact"] == ref.model_dump(by_alias=True)
    assert ref.namespace == "scanner" and ref.name == "targets"
    assert media_type == "text/vnd.contractor.target-list"
    assert data.endswith(b"\n") and len(data) <= 128 * 1024
    targets = data.decode().splitlines()
    assert targets == sorted(set(targets))
    return targets


def test_installed_katana_same_origin_depth_and_no_redirect_or_form_submission(
    tmp_path, monkeypatch
):
    async def scenario():
        async with fixture_server(plain_page) as (foreign_origin, foreign_requests):

            async def respond(target):
                if target == "/":
                    body = (
                        '<a href="/one">one</a><a href="/redirect">redirect</a>'
                        f'<a href="{foreign_origin}/external">external</a>'
                        f'<a href="{origin.replace("127.0.0.1", "localhost")}/alias">alias</a>'
                        '<form method="post" action="/submit"><input name="token" '
                        'value="private-canary"></form><script>fetch("/js-hidden")</script>'
                    ).encode()
                    return "200 OK", "", body
                if target == "/one":
                    return "200 OK", "", b'<a href="/two">two</a><a href="/two">duplicate</a>'
                if target == "/two":
                    return "200 OK", "", b'<a href="/three">over depth</a>'
                if target == "/redirect":
                    return "302 Found", f"Location: {foreign_origin}/redirected\r\n", b"redirect"
                return await plain_page(target)

            async with fixture_server(respond) as (origin, requests):
                tool, artifacts, state, workspace = await make_live_katana(tmp_path, monkeypatch)
                try:
                    result = await tool(
                        origin + "/",
                        max_depth=2,
                        max_pages=20,
                        rate_limit=100,
                        timeout_seconds=15,
                    )
                finally:
                    await tool.close()
            assert result["status"] == "completed", result
            assert result["discoveryComplete"] is False
            assert result["source"]["seed"] == origin + "/"
            assert result["source"]["origin"] == origin
            assert result["coverage"]["depthLimited"] == 1
            assert "depth_limit" in result["coverage"]["limitations"]
            assert result["stdout"] == result["stderr"] == ""
            assert "private-canary" not in repr(result)
            assert exported_targets(result, artifacts) == sorted(
                origin + path for path in ["/", "/one", "/two", "/redirect"]
            )
            assert set(requests) == {
                ("GET", "/"),
                ("GET", "/one"),
                ("GET", "/two"),
                ("GET", "/redirect"),
            }
            assert len(requests) == 4
            assert not foreign_requests
            assert state.metrics.counters["tool_calls"] == 1
            assert not list(workspace.iterdir())

    asyncio.run(scenario())


def test_installed_katana_page_budget_bounds_actual_requests(tmp_path, monkeypatch):
    async def scenario():
        async def respond(target):
            if target == "/":
                body = "".join(f'<a href="/item/{index}">{index}</a>' for index in range(30))
                return "200 OK", "", body.encode()
            return await plain_page(target)

        async with fixture_server(respond) as (origin, requests):
            tool, artifacts, _, workspace = await make_live_katana(tmp_path, monkeypatch)
            try:
                result = await tool(origin + "/", max_pages=3, rate_limit=100, timeout_seconds=15)
            finally:
                await tool.close()
        assert result["status"] == "completed", result
        assert result["discoveryComplete"] is False
        assert "page_limit" in result["coverage"]["limitations"]
        assert len(requests) == 3
        assert set(exported_targets(result, artifacts)) == {origin + path for _, path in requests}
        assert not list(workspace.iterdir())

    asyncio.run(scenario())


def test_installed_katana_seed_query_comma_cannot_add_another_origin(tmp_path, monkeypatch):
    async def scenario():
        async with fixture_server(plain_page) as (foreign_origin, foreign_requests):
            async with fixture_server(plain_page) as (origin, requests):
                target = f"/comma?one=1,{foreign_origin}/injected"
                tool, artifacts, _, workspace = await make_live_katana(tmp_path, monkeypatch)
                try:
                    result = await tool(origin + target, rate_limit=100, timeout_seconds=10)
                finally:
                    await tool.close()
            assert result["status"] == "completed", result
            assert requests == [("GET", target)]
            assert not foreign_requests
            assert exported_targets(result, artifacts) == [origin + target]
            assert not list(workspace.iterdir())

    asyncio.run(scenario())


def test_installed_katana_timeout_does_not_publish_unvisited_target(tmp_path, monkeypatch):
    async def scenario():
        async def respond(target):
            await asyncio.sleep(10)
            return await plain_page(target)

        async with fixture_server(respond) as (origin, requests):
            tool, artifacts, state, workspace = await make_live_katana(tmp_path, monkeypatch)
            try:
                result = await tool(origin + "/", timeout_seconds=1)
            finally:
                await tool.close()
        assert result["status"] == "failed", result
        assert result["errorCode"]
        assert result["discoveryComplete"] is False
        assert not artifacts.writes
        assert len(requests) <= 1
        assert state.metrics.counters["tool_errors"] == 1
        assert not list(workspace.iterdir())

    asyncio.run(scenario())


def test_installed_katana_transport_failure_does_not_export_target(tmp_path, monkeypatch):
    async def scenario():
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
            tool, artifacts, state, workspace = await make_live_katana(tmp_path, monkeypatch)
            try:
                result = await tool(f"http://127.0.0.1:{port}/", timeout_seconds=10)
            finally:
                await tool.close()
        assert result["status"] == "failed", result
        assert result["errorCode"]
        assert result["discoveryComplete"] is False
        assert not artifacts.writes
        assert state.metrics.counters["tool_errors"] == 1
        assert not list(workspace.iterdir())

    asyncio.run(scenario())
