"""Optional checks of the real ffuf binary against loopback fixtures."""

from __future__ import annotations

import asyncio
import shutil
from contextlib import suppress
from types import SimpleNamespace

import pytest
from target_policy_fixtures import SCAN_TEST_POLICY

from contractor_runtime.allocation import WorkerState
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.scan.tools import FFUFTool, ScanToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace

pytestmark = pytest.mark.skipif(
    shutil.which("ffuf") is None, reason="optional ffuf binary is absent"
)
REF = {"namespace": "inputs", "name": "wordlist", "revision": "list-1"}


async def make_live_ffuf(tmp_path, data):
    async def read_artifact(ref, *, max_bytes):
        assert ref.model_dump(by_alias=True) == REF
        assert len(data) <= max_bytes
        return SimpleNamespace(data=data, media_type="text/vnd.contractor.wordlist")

    factory = ScanToolsetFactory(
        lambda *_: SimpleNamespace(read_artifact=read_artifact),
        scanners=[FFUFTool],
        target_policy=SCAN_TEST_POLICY,
    )
    assert await factory.probe() == {"scan_ffuf"}
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = WorkerState()
    tools = await factory.create_selected(
        selected=["scan_ffuf"],
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=30
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
    )
    return tools["scan_ffuf"], state


@pytest.mark.parametrize(
    "filter_name,filter_value", [("filter_status", "404"), ("filter_size", "7")]
)
def test_installed_ffuf_preserves_payloads_and_filters_local_responses(
    tmp_path, filter_name, filter_value
):
    async def scenario():
        paths, handlers = [], set()

        async def handle(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                head = await reader.readuntil(b"\r\n\r\n")
                paths.append(head.split(b"\r\n", 1)[0])
                missing = head.startswith(b"GET /miss ")
                body = b"missing" if missing else b"found"
                status = b"404 Not Found" if missing else b"200 OK"
                writer.write(
                    b"HTTP/1.1 "
                    + status
                    + b"\r\nConnection: close\r\nContent-Type: text/plain\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                with suppress(ConnectionError):
                    await writer.wait_closed()
                handlers.discard(asyncio.current_task())

        tool, state = await make_live_ffuf(
            tmp_path, b"hit\r\nmiss\r\n keep \r\n#tag\r\nhit\r\n\r\n"
        )
        try:
            async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                result = await tool(
                    f"http://127.0.0.1:{port}/FUZZ",
                    REF,
                    rate=100,
                    timeout_seconds=10,
                    **{filter_name: filter_value},
                )
                if handlers:
                    await asyncio.gather(*handlers)
            assert result["status"] == "completed", result
            assert result["scanComplete"] is True
            assert result["wordlistArtifact"] == REF
            assert result["wordlistEntries"] == result["payloadsAttempted"] == 6
            assert result["requestErrors"] == 0
            assert result["invalidResultLines"] == 0
            assert result["resultsTruncated"] is False
            assert [item["input"]["FUZZ"] for item in result["results"]] == [
                "hit",
                " keep ",
                "#tag",
                "hit",
                "",
            ]
            assert all(item["status"] == 200 for item in result["results"])
            assert len(paths) == 6
            assert b"GET /%20keep%20 HTTP/1.1" in paths
            assert state.metrics.counters["tool_calls"] == 1
            assert result["stdout"] == result["stderr"] == ""
            assert not list((tmp_path / "workspace").iterdir())
        finally:
            await tool.close()

    asyncio.run(scenario())


def test_installed_ffuf_transport_errors_are_not_success(tmp_path):
    async def scenario():
        # Reserve a loopback port without listening: requests deterministically fail.
        import socket

        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            tool, state = await make_live_ffuf(tmp_path, b"probe\n")
            try:
                result = await tool(f"http://127.0.0.1:{port}/FUZZ", REF, timeout_seconds=10)
                assert result["exitCode"] == 0
                assert result["status"] == "failed"
                assert result["errorCode"] == "scan_request_failed"
                assert result["scanComplete"] is False
                assert result["requestErrors"] > 0
                assert state.metrics.counters["tool_errors"] == 1
                assert not list((tmp_path / "workspace").iterdir())
            finally:
                await tool.close()

    asyncio.run(scenario())
