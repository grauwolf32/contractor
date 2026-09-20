from __future__ import annotations

import asyncio
import json
from base64 import b64encode
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from test_scan_toolset import executable

import contractor_runtime.toolsets.scan.ffuf as ffuf_module
import contractor_runtime.toolsets.scan.process as process_module
import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactClient, ArtifactHTTPResponse
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.wordlist import (
    MAX_WORDLIST_ARTIFACT_BYTES,
    MAX_WORDLIST_LINES,
    MAX_WORDLIST_PAYLOAD_BYTES,
)
from contractor_runtime.workspace import AllocationWorkspace

WORDLIST_REF = {"namespace": "inputs", "name": "wordlist", "revision": "wordlist-1"}
TARGET = "https://target.invalid/FUZZ"


def ffuf_result(payload="foo", *, position=1, **changes):
    result = {
        "input": {"FUZZ": b64encode(payload.encode()).decode()},
        "position": position,
        "status": 200,
        "length": 2,
        "words": 1,
        "lines": 1,
        "url": "https://target.invalid/foo",
        "duration": 123,
        "content-type": "text/plain",
        "redirectlocation": "",
    }
    result.update(changes)
    return result


def ffuf_progress(attempted=1, total=1, errors=0):
    return (
        f":: Progress: [{attempted}/{total}] :: Job [1/1] :: 10 req/sec :: "
        f"Duration: [0:00:00] :: Errors: {errors} ::"
    )


def install_ffuf(tmp_path, monkeypatch, *, results=None, progress=None, exit_code=0):
    records = [ffuf_result()] if results is None else results
    output = "\n".join(json.dumps(record) for record in records)
    diagnostics = ffuf_progress() if progress is None else progress
    executable(
        tmp_path,
        "ffuf",
        "import sys\n"
        "if sys.argv[1:] == ['-V']:\n"
        "    print('fixture-ffuf')\n"
        "    sys.exit(0)\n"
        f"print({output!r})\n"
        f"print({diagnostics!r}, file=sys.stderr)\n"
        f"sys.exit({exit_code})",
    )
    monkeypatch.setenv("PATH", str(tmp_path))


class WordlistTransport:
    def __init__(
        self,
        data=b"foo\n",
        *,
        media_type="text/plain",
        revision="wordlist-1",
        block=False,
        status=200,
    ):
        self.data = data
        self.media_type = media_type
        self.revision = revision
        self.block = block
        self.status = status
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def request(self, method, path, *, headers, body, max_response_bytes):
        self.calls += 1
        assert method == "GET"
        assert urlsplit(path).path == "/allocations/allocation/artifacts/inputs/wordlist"
        assert parse_qs(urlsplit(path).query) == {"revision": ["wordlist-1"]}
        assert max_response_bytes == MAX_WORDLIST_ARTIFACT_BYTES
        self.started.set()
        if self.block:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()
        return ArtifactHTTPResponse(
            self.status,
            {
                "content-type": self.media_type,
                "content-length": str(len(self.data)),
                "etag": json.dumps(self.revision),
                "x-contractor-binding-created-at": "2026-09-19T00:00:00Z",
                "x-contractor-revision-created-at": "2026-09-19T00:00:00Z",
            },
            self.data,
        )


async def make_ffuf(tmp_path, transport):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    factory = scan.ScanToolsetFactory(
        artifact_client_factory=lambda allocation, settings: ArtifactClient(allocation, transport)
    )
    state = WorkerState()
    tools = await factory.create_selected(
        selected=["scan_ffuf"],
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=60
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
    )
    return tools["scan_ffuf"], state


def forbid_launch(tmp_path, monkeypatch):
    executable(tmp_path, "ffuf", "raise AssertionError('unexpected scanner launch')")
    monkeypatch.setenv("PATH", str(tmp_path))

    async def forbidden(*args, **kwargs):
        pytest.fail("invalid input launched a scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)


def assert_private_metrics_and_cleanup(state, tmp_path, *, failed=True):
    assert state.metrics.counters["tool_calls"] == 1
    if failed:
        assert state.metrics.counters["tool_errors"] == 1
    assert state.metrics.tool_calls[0].arguments == {}
    assert "canary" not in repr(state.metrics)
    assert list((tmp_path / "workspace").iterdir()) == []


@pytest.mark.parametrize("media_type", ["text/plain", "text/vnd.contractor.wordlist"])
@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_exact_wordlist_private_file_arguments_results_and_cleanup(
    tmp_path, monkeypatch, media_type, line_ending
):
    raw = " first payload \n#comment-canary\nsame\nsame\n\nключ\nlast".encode()
    data = raw.replace(b"\n", line_ending)
    record = ffuf_result("ключ", position=6, url="https://target.invalid/%D0%BA%D0%BB%D1%8E%D1%87")
    executable(
        tmp_path,
        "ffuf",
        "import json, os, pathlib, stat, sys\n"
        "args = sys.argv[1:]\n"
        "wordlist_arg = args[args.index('-w') + 1]\n"
        "assert wordlist_arg.endswith(':FUZZ')\n"
        "wordlist = pathlib.Path(wordlist_arg[:-5])\n"
        "assert wordlist.name == 'wordlist.txt'\n"
        "assert stat.S_IMODE(wordlist.stat().st_mode) == 0o600\n"
        "assert stat.S_IMODE(wordlist.parent.stat().st_mode) == 0o700\n"
        f"assert wordlist.read_bytes() == {raw!r}\n"
        f"assert args[args.index('-u') + 1] == {TARGET!r}\n"
        "for flag, value in {'-t': '1', '-rate': '7', '-timeout': '10', '-mc': '200-299,401',\n"
        "                    '-fc': '404', '-fs': '0,12-20', '-fw': '0', '-fl': '0'}.items():\n"
        "    assert args[args.index(flag) + 1] == value\n"
        "assert '-json' in args and '-noninteractive' in args\n"
        "assert '-s' not in args and '-r' not in args and '-recursion' not in args\n"
        "assert os.environ['HOME'] == str(wordlist.parent)\n"
        "assert 'PRIVATE_GATEWAY_KEY' not in os.environ\n"
        "assert 'HTTPS_PROXY' not in os.environ\n"
        f"print({json.dumps(record)!r})\n"
        "print('diagnostic-canary', file=sys.stderr)\n"
        f"print({ffuf_progress(7, 7)!r}, file=sys.stderr)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PRIVATE_GATEWAY_KEY", "private-canary")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy-canary.invalid")

    async def scenario():
        transport = WordlistTransport(data, media_type=media_type)
        tool, state = await make_ffuf(tmp_path, transport)
        try:
            result = await tool(
                TARGET,
                WORDLIST_REF,
                rate=7,
                match_status="200-299,401",
                filter_status="404",
                filter_size="0,12-20",
                filter_words="0",
                filter_lines="0",
            )
            assert result["status"] == "completed", result
            assert result["errorCode"] is None
            assert result["scanComplete"] is True
            assert result["wordlistArtifact"] == WORDLIST_REF
            assert result["wordlistEntries"] == 7
            assert result["payloadsAttempted"] == 7
            assert result["requestErrors"] == 0
            assert result["diagnosticsRedacted"] is True
            assert result["stdout"] == result["stderr"] == ""
            assert result["results"] == [
                {
                    "input": {"FUZZ": "ключ"},
                    "position": 6,
                    "status": 200,
                    "length": 2,
                    "words": 1,
                    "lines": 1,
                    "url": record["url"],
                    "durationNs": 123,
                    "contentType": "text/plain",
                    "redirectLocation": "",
                }
            ]
            assert not result["resultsTruncated"]
            assert "canary" not in repr(result)
            assert transport.calls == 1
            assert_private_metrics_and_cleanup(state, tmp_path, failed=False)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "wordlist_ref",
    [
        {},
        {"namespace": "inputs", "name": "wordlist"},
        {**WORDLIST_REF, "revision": None},
        {**WORDLIST_REF, "revision": ""},
        {**WORDLIST_REF, "extra": "secret-canary"},
        {**WORDLIST_REF, "namespace": "skills"},
        {**WORDLIST_REF, "namespace": "scanner", "name": "memory.secret-canary"},
        {**WORDLIST_REF, "namespace": "scanner", "name": "http.body.secret-canary"},
        "secret-canary",
        ["secret-canary"],
        None,
    ],
)
def test_invalid_refs_fail_before_read(tmp_path, monkeypatch, wordlist_ref):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = WordlistTransport()
        tool, state = await make_ffuf(tmp_path, transport)
        try:
            with pytest.raises(ToolInputError) as caught:
                await tool(TARGET, wordlist_ref)
            assert "canary" not in str(caught.value)
            assert transport.calls == 0
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "https://target.invalid/no-marker"},
        {"url": "https://FUZZ.invalid/path"},
        {"url": "https://target.invalid:FUZZ/path"},
        {"url": "https://user:secret-canary@target.invalid/FUZZ"},
        {"url": "https://target.invalid/FUZZ#secret-canary"},
        {"url": "https://target.invalid/FUZZ?token=FFUFHASH"},
        {"url": "file:///FUZZ"},
        {"url": "https://target.invalid/FUZZ\nsecret-canary"},
        {"rate": 0},
        {"rate": 1001},
        {"rate": True},
        {"rate": "10"},
        {"timeout_seconds": 0},
        {"timeout_seconds": True},
        {"match_status": ""},
        {"match_status": "--help"},
        {"match_status": "999"},
        {"match_status": True},
        {"match_status": "300-200"},
        {"filter_status": "all"},
        {"filter_status": "-1"},
        {"filter_status": 404},
        {"filter_size": "1,,2"},
        {"filter_size": "20-10"},
        {"filter_size": "00000000000"},
        {"filter_size": "0-00000000000"},
        {"filter_size": True},
        {"filter_words": "secret-canary"},
        {"filter_words": []},
        {"filter_lines": "1\nsecret-canary"},
        {"filter_lines": "1;2"},
    ],
)
def test_invalid_arguments_fail_before_read(tmp_path, monkeypatch, changes):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = WordlistTransport()
        tool, state = await make_ffuf(tmp_path, transport)
        try:
            with pytest.raises(ToolInputError) as caught:
                await tool(**{"url": TARGET, "wordlist_ref": WORDLIST_REF, **changes})
            assert "canary" not in str(caught.value)
            assert transport.calls == 0
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"media_type": "application/json"}, id="wrong-media-type"),
        pytest.param({"revision": "other-revision-canary"}, id="wrong-revision"),
        pytest.param({"data": b""}, id="empty"),
        pytest.param({"data": b"\xffsecret-canary"}, id="invalid-utf8"),
        pytest.param({"data": b"foo\rsecret-canary"}, id="bare-cr"),
        pytest.param({"data": b"foo\x00secret-canary"}, id="nul"),
        pytest.param({"data": b"foo\tsecret-canary"}, id="control-character"),
        pytest.param({"data": b"secret-canary-FFUFHASH\n"}, id="reserved-marker"),
        pytest.param({"data": b"\xef\xbb\xbfsecret-canary"}, id="byte-order-mark"),
        pytest.param({"data": b"a" * (MAX_WORDLIST_ARTIFACT_BYTES + 1)}, id="oversized-artifact"),
        pytest.param({"data": b"a" * (MAX_WORDLIST_PAYLOAD_BYTES + 1)}, id="oversized-payload"),
        pytest.param({"data": b"a\n" * (MAX_WORDLIST_LINES + 1)}, id="too-many-payloads"),
        pytest.param({"status": 403, "data": b'{"message":"secret-canary"}'}, id="forbidden"),
        pytest.param({"status": 404, "data": b'{"message":"secret-canary"}'}, id="not-found"),
    ],
)
def test_invalid_artifact_never_launches(tmp_path, monkeypatch, options):
    forbid_launch(tmp_path, monkeypatch)

    async def scenario():
        transport = WordlistTransport(**options)
        tool, state = await make_ffuf(tmp_path, transport)
        try:
            with pytest.raises(ToolInputError) as caught:
                await tool(TARGET, WORDLIST_REF)
            assert caught.value.code == "tool_input_invalid"
            assert not caught.value.retryable
            assert "canary" not in str(caught.value)
            assert transport.calls == 1
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options,error_code,attempted,errors",
    [
        ({"progress": "diagnostic-canary"}, "scan_incomplete", None, None),
        ({"progress": ffuf_progress(0, 1)}, "scan_incomplete", 0, 0),
        ({"progress": ffuf_progress(1, 1, 1)}, "scan_request_failed", 1, 1),
        ({"exit_code": 2}, "scanner_failed", 1, 0),
        (
            {"results": [ffuf_result(), {"invalid": "secret-canary"}]},
            "invalid_scanner_output",
            1,
            0,
        ),
    ],
)
def test_partial_failed_or_malformed_scans_are_never_complete(
    tmp_path, monkeypatch, options, error_code, attempted, errors
):
    install_ffuf(tmp_path, monkeypatch, **options)

    async def scenario():
        tool, state = await make_ffuf(tmp_path, WordlistTransport())
        try:
            result = await tool(TARGET, WORDLIST_REF)
            assert result["status"] == "failed", result
            assert result["errorCode"] == error_code
            assert result["scanComplete"] is False
            assert result["payloadsAttempted"] == attempted
            assert result["requestErrors"] == errors
            assert result["results"][0]["input"] == {"FUZZ": "foo"}
            assert result["stdout"] == result["stderr"] == ""
            assert "canary" not in repr(result)
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "record",
    [
        ffuf_result(input={"FUZZ": "%%%secret-canary"}),
        ffuf_result(input={"FUZZ": b64encode(b"\xff").decode()}),
        ffuf_result(status=True),
        ffuf_result(status=0),
        ffuf_result(length=-1),
        ffuf_result(position=0),
        ffuf_result(duration="secret-canary"),
        ffuf_result(url=["secret-canary"]),
        ffuf_result(url="file:///secret-canary"),
        ffuf_result(url="https://user:secret-canary@target.invalid/foo"),
        {"input": {"FUZZ": "Zm9v"}},
    ],
)
def test_invalid_result_records_are_dropped(tmp_path, monkeypatch, record):
    install_ffuf(tmp_path, monkeypatch, results=[record])

    async def scenario():
        tool, state = await make_ffuf(tmp_path, WordlistTransport())
        try:
            result = await tool(TARGET, WORDLIST_REF)
            assert result["errorCode"] == "invalid_scanner_output"
            assert result["scanComplete"] is False
            assert result["results"] == []
            assert result["invalidResultLines"] == 1
            assert "canary" not in repr(result)
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("bound", ["records", "bytes"])
def test_results_are_bounded_with_explicit_truncation(tmp_path, monkeypatch, bound):
    count = 110 if bound == "records" else 50
    payload = "x" if bound == "records" else "x" * MAX_WORDLIST_PAYLOAD_BYTES
    records = [ffuf_result(payload, position=index + 1) for index in range(count)]
    install_ffuf(tmp_path, monkeypatch, results=records, progress=ffuf_progress(count, count))

    async def scenario():
        tool, state = await make_ffuf(
            tmp_path, WordlistTransport((payload + "\n").encode() * count)
        )
        try:
            result = await tool(TARGET, WORDLIST_REF)
            assert result["resultsTruncated"] is True
            assert 0 < len(result["results"]) <= 100
            assert len(result["results"]) < count
            assert len(json.dumps(result["results"], separators=(",", ":")).encode()) <= 128 * 1024
            assert result["wordlistEntries"] == count
            assert result["payloadsAttempted"] == count
            assert result["requestErrors"] == 0
            assert_private_metrics_and_cleanup(state, tmp_path, failed=False)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("offset", [-1, 0])
def test_result_byte_limit_includes_json_array_punctuation(monkeypatch, offset):
    output = process_module.ProcessResult(
        0,
        stdout="\n".join(json.dumps(ffuf_result("тест", position=i)) for i in (1, 2)).encode(),
        stderr=ffuf_progress(2, 2).encode(),
    )
    complete = ffuf_module.ffuf_observation(output, 2)
    encoded = json.dumps(complete["results"], ensure_ascii=False, separators=(",", ":")).encode()
    limit = len(encoded) + offset
    monkeypatch.setattr(ffuf_module, "MAX_FFUF_RESULTS_BYTES", limit)

    result = ffuf_module.ffuf_observation(output, 2)

    assert len(result["results"]) == (1 if offset < 0 else 2)
    assert result["resultsTruncated"] is (offset < 0)
    assert result["scanComplete"] is True
    assert (
        len(json.dumps(result["results"], ensure_ascii=False, separators=(",", ":")).encode())
        <= limit
    )


def test_output_overflow_preserves_bounded_partial_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(process_module, "MAX_OUTPUT_BYTES", 4096)
    record = ffuf_result()
    executable(
        tmp_path,
        "ffuf",
        f"print({json.dumps(record)!r}, flush=True)\n"
        "while True:\n"
        "    print('output-canary' * 1000, flush=True)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        tool, state = await make_ffuf(tmp_path, WordlistTransport())
        try:
            result = await asyncio.wait_for(tool(TARGET, WORDLIST_REF), 3)
            assert result["errorCode"] == "output_limit_exceeded"
            assert result["outputLimitExceeded"] is True
            assert result["scanComplete"] is False
            assert result["results"][0]["input"] == {"FUZZ": "foo"}
            assert result["stdout"] == result["stderr"] == ""
            assert "canary" not in repr(result)
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            await tool.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["artifact-read", "scanner"])
@pytest.mark.parametrize("stop", ["cancel", "close", "timeout"])
def test_stopping_ffuf_cleans_files_processes_and_metrics(tmp_path, monkeypatch, phase, stop):
    marker = tmp_path / "process.json"
    executable(
        tmp_path,
        "ffuf",
        "import json, os, pathlib, sys, time\n"
        "wordlist = pathlib.Path(sys.argv[sys.argv.index('-w') + 1][:-5])\n"
        "assert b'payload-canary' in wordlist.read_bytes()\n"
        "print(wordlist.read_text(), flush=True)\n"
        "print(wordlist.read_text(), file=sys.stderr, flush=True)\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        f"    marker = pathlib.Path({str(marker)!r})\n"
        "    marker.with_suffix('.tmp').write_text(json.dumps(\n"
        "        {'parent': os.getpid(), 'child': child, 'wordlist': str(wordlist)}))\n"
        "    marker.with_suffix('.tmp').replace(marker)\n"
        "    time.sleep(60)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        transport = WordlistTransport(b"payload-canary\n", block=phase == "artifact-read")
        tool, state = await make_ffuf(tmp_path, transport)
        task = asyncio.create_task(
            tool(TARGET, WORDLIST_REF, timeout_seconds=1 if stop == "timeout" else 30)
        )
        try:
            async with asyncio.timeout(3):
                await transport.started.wait()
                if phase == "scanner":
                    while not marker.exists():
                        await asyncio.sleep(0.01)
                    process = json.loads(marker.read_text())
                    assert Path(process["wordlist"]).exists()
                else:
                    assert not list((tmp_path / "workspace").rglob("wordlist.txt"))

            if stop == "timeout":
                result = await asyncio.wait_for(task, 3)
                assert result["status"] == "failed"
                assert result["errorCode"] == "scan_timeout"
                assert result["scanComplete"] is False
                assert result["wordlistArtifact"] == WORDLIST_REF
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
                assert not Path(process["wordlist"]).exists()
                assert not Path(f"/proc/{process['parent']}").exists()
                child = Path(f"/proc/{process['child']}/stat")
                async with asyncio.timeout(3):
                    while child.exists() and child.read_text().split()[2] != "Z":
                        await asyncio.sleep(0.01)
            assert_private_metrics_and_cleanup(state, tmp_path)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await tool.close()

    asyncio.run(scenario())
