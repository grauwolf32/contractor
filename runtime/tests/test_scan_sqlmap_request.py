from __future__ import annotations

import asyncio
import json
from base64 import b64encode
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from target_policy_fixtures import SCAN_TEST_POLICY
from test_scan_http_request import request_document
from test_scan_toolset import executable

import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactClient, ArtifactHTTPResponse
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.http_request import MAX_BODY_BYTES, MAX_REQUEST_ARTIFACT_BYTES
from contractor_runtime.workspace import AllocationWorkspace

REQUEST_REF = {"namespace": "inputs", "name": "request", "revision": "request-1"}


class RequestTransport:
    def __init__(
        self, data=None, *, media_type="application/json", revision="request-1", block=False
    ):
        self.data = json.dumps(request_document()).encode() if data is None else data
        self.media_type = media_type
        self.revision = revision
        self.block = block
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.calls += 1
        assert method == "GET"
        assert urlsplit(path).path == "/allocations/allocation/artifacts/inputs/request"
        assert parse_qs(urlsplit(path).query) == {"revision": ["request-1"]}
        assert max_response_bytes == MAX_REQUEST_ARTIFACT_BYTES
        self.started.set()
        if self.block:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        return ArtifactHTTPResponse(
            200,
            {
                "content-type": self.media_type,
                "content-length": str(len(self.data)),
                "etag": json.dumps(self.revision),
                "x-contractor-binding-created-at": "2026-09-19T00:00:00Z",
                "x-contractor-revision-created-at": "2026-09-19T00:00:00Z",
            },
            self.data,
        )


async def make_sqlmap(tmp_path, transport):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    factory = scan.ScanToolsetFactory(
        artifact_client_factory=lambda allocation, settings: ArtifactClient(allocation, transport),
        target_policy=SCAN_TEST_POLICY,
    )
    state = WorkerState()
    tools = await factory.create_selected(
        selected=["scan_sqlmap"],
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=60
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
    )
    return tools["scan_sqlmap"], state


def forbid_launch(tmp_path, monkeypatch):
    executable(tmp_path, "sqlmap", "raise AssertionError('unexpected scanner launch')")
    monkeypatch.setenv("PATH", str(tmp_path))

    async def forbidden(*args, **kwargs):
        pytest.fail("invalid input launched a scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)


def assert_failed_call_is_private(state, tmp_path):
    assert state.metrics.counters["tool_calls"] == 1
    assert state.metrics.counters["tool_errors"] == 1
    assert state.metrics.tool_calls[0].arguments == {}
    assert "canary" not in repr(state.metrics)
    assert list((tmp_path / "workspace").iterdir()) == []


@pytest.mark.parametrize(
    "request_ref",
    [
        {},
        {"namespace": "inputs", "name": "request"},
        {**REQUEST_REF, "revision": None},
        {**REQUEST_REF, "revision": ""},
        {**REQUEST_REF, "extra": "secret-canary"},
        {**REQUEST_REF, "namespace": "skills"},
        {**REQUEST_REF, "namespace": "scanner", "name": "memory.secret-canary"},
        {**REQUEST_REF, "namespace": "scanner", "name": "http.body.secret-canary"},
        "secret-canary",
        ["secret-canary"],
    ],
)
def test_invalid_or_reserved_ref_fails_before_artifact_read(tmp_path, monkeypatch, request_ref):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = RequestTransport()
        tool, state = await make_sqlmap(tmp_path, transport)
        try:
            with pytest.raises(scan.ScanInputError) as caught:
                await tool(request_ref=request_ref)
            assert "canary" not in str(caught.value)
            assert transport.calls == 0
            assert_failed_call_is_private(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("argument", ["url", "parameter", "data", "cookie"])
def test_request_ref_rejects_every_url_mode_argument_before_read(tmp_path, monkeypatch, argument):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = RequestTransport()
        tool, state = await make_sqlmap(tmp_path, transport)
        try:
            with pytest.raises(scan.ScanInputError) as caught:
                await tool(request_ref=REQUEST_REF, **{argument: "secret-canary"})
            assert "canary" not in str(caught.value)
            assert transport.calls == 0
            assert_failed_call_is_private(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"media_type": "text/plain"}, id="wrong-media-type"),
        pytest.param({"revision": "wrong-revision-canary"}, id="wrong-revision"),
        pytest.param({"data": b'{"secret-canary":'}, id="invalid-json"),
        pytest.param({"data": b"\xffsecret-canary"}, id="invalid-utf8"),
        pytest.param(
            {"data": b"secret-canary" + b" " * MAX_REQUEST_ARTIFACT_BYTES},
            id="oversized-artifact",
        ),
        pytest.param(
            {"data": json.dumps(request_document(body="a" * (MAX_BODY_BYTES + 1))).encode()},
            id="oversized-body",
        ),
        pytest.param(
            {"data": json.dumps(request_document(schemaVersion=2)).encode()},
            id="unsupported-schema",
        ),
        pytest.param(
            {
                "data": json.dumps(
                    request_document(
                        headers=[{"name": "X-Header", "value": "secret-canary\r\nInjected: yes"}]
                    )
                ).encode()
            },
            id="invalid-header",
        ),
        pytest.param(
            {
                "data": json.dumps(
                    request_document(headers=[{"name": "Transfer-Encoding", "value": "chunked"}])
                ).encode()
            },
            id="unsupported-framing",
        ),
        pytest.param(
            {
                "data": json.dumps(
                    request_document(
                        body=(
                            '<PORT>80</PORT><ReQuEsT BaSe64="TrUe"><![CDATA['
                            + b64encode(
                                b"GET /?id=7 HTTP/1.1\r\nHost: replacement-canary.invalid\r\n\r\n"
                            ).decode("ascii")
                            + "]]></ReQuEsT>"
                        )
                    )
                ).encode()
            },
            id="mixed-case-xml-nested-request",
        ),
    ],
)
def test_invalid_artifact_is_rejected_before_launch(tmp_path, monkeypatch, options):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = RequestTransport(**options)
        tool, state = await make_sqlmap(tmp_path, transport)
        try:
            with pytest.raises(ToolInputError) as caught:
                await tool(request_ref=REQUEST_REF)
            assert caught.value.code == "tool_input_invalid"
            assert not caught.value.retryable
            assert "canary" not in str(caught.value)
            assert transport.calls == 1
            assert_failed_call_is_private(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest?id=7",
        "http://[fd00:ec2::254]/latest?id=7",
        "https://artifacts.invalid/private?id=7",
    ],
)
def test_request_artifact_destination_is_checked_before_launch(tmp_path, monkeypatch, url):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = RequestTransport(json.dumps(request_document(url=url)).encode())
        tool, state = await make_sqlmap(tmp_path, transport)
        try:
            result = await tool(request_ref=REQUEST_REF)
            assert result["status"] == "failed"
            assert result["errorCode"] == "scan_target_denied"
            assert result["requestArtifact"] == REQUEST_REF
            assert transport.calls == 1
            assert "canary" not in repr(result)
            assert_failed_call_is_private(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["artifact-read", "scanner"])
@pytest.mark.parametrize("stop", ["cancel", "close", "timeout"])
def test_stopping_prepared_request_cleans_files_processes_and_metrics(
    tmp_path, monkeypatch, phase, stop
):
    marker = tmp_path / "process.json"
    executable(
        tmp_path,
        "sqlmap",
        "import json, os, pathlib, sys, time\n"
        "request = pathlib.Path(sys.argv[sys.argv.index('-r') + 1])\n"
        "assert b'body-canary' in request.read_bytes()\n"
        "print(request.read_text(), flush=True)\n"
        "print(request.read_text(), file=sys.stderr, flush=True)\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        f"    marker = pathlib.Path({str(marker)!r})\n"
        "    marker.with_suffix('.tmp').write_text(json.dumps(\n"
        "        {'parent': os.getpid(), 'child': child, 'request': str(request)}))\n"
        "    marker.with_suffix('.tmp').replace(marker)\n"
        "    time.sleep(60)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        transport = RequestTransport(block=phase == "artifact-read")
        tool, state = await make_sqlmap(tmp_path, transport)
        task = asyncio.create_task(
            tool(request_ref=REQUEST_REF, timeout_seconds=1 if stop == "timeout" else 30)
        )
        try:
            async with asyncio.timeout(3):
                await transport.started.wait()
                if phase == "scanner":
                    while not marker.exists():
                        await asyncio.sleep(0.01)
                    process = json.loads(marker.read_text())
                    assert Path(process["request"]).exists()
                else:
                    assert not list((tmp_path / "workspace").rglob("request.http"))

            if stop == "timeout":
                result = await asyncio.wait_for(task, 3)
                assert result["status"] == "failed"
                assert result["errorCode"] == "scan_timeout"
                assert result["requestArtifact"] == REQUEST_REF
                assert result["diagnosticsRedacted"] is True
                assert result["stdout"] == result["stderr"] == ""
                assert "canary" not in repr(result)
            else:
                if stop == "cancel":
                    task.cancel()
                else:
                    await asyncio.wait_for(tool.close(), 3)
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 3)

            if phase == "artifact-read":
                assert transport.cancelled.is_set()
                assert not marker.exists()
            else:
                assert not Path(process["request"]).exists()
                assert not Path(f"/proc/{process['parent']}").exists()
                child = Path(f"/proc/{process['child']}/stat")
                # A killed orphan may remain a zombie until the host's init reaps it.
                async with asyncio.timeout(3):
                    while child.exists() and child.read_text().split()[2] != "Z":
                        await asyncio.sleep(0.01)
            assert_failed_call_is_private(state, tmp_path)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await tool.close()

    asyncio.run(scenario())
