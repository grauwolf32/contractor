from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

import pytest
from target_policy_fixtures import SCAN_TEST_POLICY
from test_scan_toolset import executable

import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.allocation import WorkerState
from contractor_runtime.artifacts import ArtifactAPIError
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.common.input_errors import ToolInputError
from contractor_runtime.toolsets.scan.katana import (
    MAX_TARGET_BYTES,
    canonical_url,
    katana_observation,
    scope_regex,
)
from contractor_runtime.toolsets.scan.process import ProcessResult
from contractor_runtime.workspace import AllocationWorkspace

SEED = "http://target.invalid/"


def record(path="/", **changes):
    value = {
        "request": {"method": "GET", "endpoint": SEED.rstrip("/") + path, "source": SEED},
        "response": {
            "status_code": 200,
            "headers": {"Set-Cookie": "secret-canary"},
            "body": "private-body-canary",
        },
    }
    value.update(changes)
    return value


def parse(records, **changes):
    result = ProcessResult(0, stdout=b"\n".join(json.dumps(x).encode() for x in records), **changes)
    return katana_observation(
        result, seed=SEED, max_depth=2, max_pages=100, rate_limit=10, timeout_seconds=60
    )


def test_deterministic_export_deduplicates_and_keeps_only_observed_same_origin_get():
    observations = [
        record("/b"),
        record("/a"),
        record("/a"),
        record(request={"method": "GET", "endpoint": "http://foreign.invalid/"}),
        record(request={"method": "POST", "endpoint": SEED}),
        record("/deep", response=None, error="max depth reached"),
        record("/unvisited", response=None),
    ]
    result, data = parse(observations)
    reverse, other = parse(list(reversed(observations)))
    assert result == reverse and data == other
    assert data == b"http://target.invalid/a\nhttp://target.invalid/b\n"
    assert result["targetsDigest"] == "sha256:" + hashlib.sha256(data).hexdigest()
    assert result["status"] == "completed" and result["discoveryComplete"] is False
    assert result["coverage"]["exportedTargets"] == 2
    assert result["coverage"]["duplicates"] == 1
    assert result["coverage"]["depthLimited"] == 1
    assert result["coverage"]["unsupportedMethods"] == 1
    assert result["coverage"]["outOfScope"] == 1
    assert result["coverage"]["unvisited"] == 1
    assert "canary" not in json.dumps(result)
    assert result["stdout"] == result["stderr"] == ""


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"request": {}},
        record(response={"status_code": True}),
        record(request={"method": "GET", "endpoint": "http://user:pass@target.invalid/"}),
        record(request={"method": "GET", "endpoint": SEED, "source": "file:///etc/passwd"}),
    ],
)
def test_invalid_observations_fail_without_erasing_valid_partial_targets(bad):
    result, data = parse([record(), bad])
    assert result["status"] == "failed" and result["errorCode"] == "invalid_scanner_output"
    assert result["invalidResultLines"] == 1
    assert data == SEED.encode() + b"\n"


def test_output_and_transport_failures_remain_incomplete_and_bounded():
    result, data = parse([record(), record("/error", response=None, error="private-error-canary")])
    assert result["status"] == "failed" and result["requestErrors"] == 1
    assert "canary" not in json.dumps(result)
    failed, partial = parse([record()], error_code="scan_timeout")
    assert failed["errorCode"] == "scan_timeout" and partial == data
    empty, empty_data = parse([])
    assert empty["errorCode"] == "no_discovered_targets" and empty_data == b""
    truncated, data = parse([record(f"/{i:04}") for i in range(101)])
    assert truncated["resultsTruncated"] is True
    assert len(data.splitlines()) == 100
    assert truncated["coverage"]["omittedTargets"] == 1
    long, data = parse([record("/" + str(i) + "x" * 8000) for i in range(30)])
    assert long["resultsTruncated"] is True
    assert 0 < len(data) <= MAX_TARGET_BYTES


@pytest.mark.parametrize("line", [b'{"request":{},"request":{}}', b'{"x":NaN}', b"\xff", b"[]"])
def test_malformed_json_lines_are_explicit(line):
    result, data = katana_observation(
        ProcessResult(0, stdout=line),
        seed=SEED,
        max_depth=2,
        max_pages=100,
        rate_limit=10,
        timeout_seconds=60,
    )
    assert result["errorCode"] == "invalid_scanner_output" and not data


@pytest.mark.parametrize(
    "seed,allowed,forbidden",
    [
        (
            "HTTP://Example.TEST:80",
            ["http://example.test/", "http://EXAMPLE.test:80/?x"],
            [
                "https://example.test/",
                "http://example.test:81/",
                "http://example.test.evil/",
                "http://example.test@evil/",
            ],
        ),
        ("http://[::1]:8080/", ["http://[::1]:8080/a"], ["http://[::1]/", "http://[::1]:8081/"]),
    ],
)
def test_exact_origin_scope(seed, allowed, forbidden):
    pattern = re.compile(scope_regex(canonical_url(seed)))
    assert all(pattern.search(url) for url in allowed)
    assert not any(pattern.search(url) for url in forbidden)


class Artifacts:
    def __init__(self, *, existing=False, fail_write=False):
        self.existing, self.fail_write = existing, fail_write
        self.reads, self.writes = [], []

    async def read_artifact(self, ref, *, max_bytes):
        self.reads.append(ref)
        if self.existing:
            return SimpleNamespace(data=b"already exists")
        raise ArtifactAPIError(404, "artifact_not_found", False)

    async def write_artifact(self, ref, *, data, media_type, expected_revision):
        self.writes.append((ref, data, media_type, expected_revision))
        if self.fail_write:
            raise RuntimeError("private-publication-canary")
        return SimpleNamespace(
            artifact=ArtifactRef(namespace=ref.namespace, name=ref.name, revision="r1")
        )


async def make_tool(tmp_path, artifacts):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    factory = scan.ScanToolsetFactory(
        lambda *_: artifacts, scanners=[scan.KatanaTool], target_policy=SCAN_TEST_POLICY
    )
    state = WorkerState()
    tools = await factory.create_selected(
        selected=["scan_katana"],
        allocation_id="a",
        run_id="r",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            artifactApiUrl="https://artifacts.invalid", requestTimeoutSeconds=10
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
    )
    return tools["scan_katana"], state


def test_private_single_seed_flags_exact_publication_and_cleanup(tmp_path, monkeypatch):
    seed = SEED + '?next=a,http://other.invalid/"quote"'
    executable(
        tmp_path,
        "katana",
        (
            "import json, os, pathlib, stat, sys\n"
            "args=sys.argv[1:]\n"
            "p=pathlib.Path(args[args.index('-list')+1])\n"
            f"assert p.read_text() == {seed + chr(10)!r}\n"
            "assert stat.S_IMODE(p.stat().st_mode)==0o600\n"
            "assert pathlib.Path(args[args.index('-config')+1]).read_text()=='{}\\n'\n"
            "assert os.environ['HOME']==str(p.parent)\n"
            "assert 'PRIVATE_KEY' not in os.environ and 'HTTPS_PROXY' not in os.environ\n"
            "for k,v in {'-d':'3','-mdp':'7','-rl':'2','-retry':'0',\n"
            "            '-c':'1','-p':'1','-ct':'8s','-timeout':'3'}.items():\n"
            "    assert args[args.index(k)+1]==v\n"
            "assert all(x in args for x in ['-dr','-duc','-or','-ob','-duf'])\n"
            "assert all(x not in args for x in ['-u','-headless','-aff','-jc','-ns','-kb'])\n"
            f"print({json.dumps(record())!r})\n"
            "print('private-diagnostics-canary',file=sys.stderr)"
        ),
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("PRIVATE_KEY", "canary")
    monkeypatch.setenv("HTTPS_PROXY", "http://canary.invalid")

    async def scenario():
        artifacts = Artifacts()
        tool, state = await make_tool(tmp_path, artifacts)
        result = await tool(seed, max_depth=3, max_pages=7, rate_limit=2, timeout_seconds=10)
        await tool.close()
        assert result["status"] == "completed", result
        assert result["artifacts"] == {"targets": result["targetsArtifact"]}
        assert result["targetsArtifact"] == {
            "namespace": "scanner",
            "name": "targets",
            "revision": "r1",
        }
        assert artifacts.writes[0][1:] == (
            SEED.encode() + b"\n",
            "text/vnd.contractor.target-list",
            None,
        )
        assert state.metrics.counters["tool_calls"] == 1
        assert "canary" not in repr(result) + repr(state.metrics)
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "file:///etc/passwd"},
        {"url": SEED + "\nhttp://foreign.invalid"},
        {"url": "http://user:secret@target.invalid/"},
        {"url": SEED + "%zz"},
        {"url": SEED + "\\evil"},
        {"url": "http://target.invalid:0/"},
        {"url": SEED, "max_depth": 0},
        {"url": SEED, "max_depth": 6},
        {"url": SEED, "max_pages": 1001},
        {"url": SEED, "max_pages": True},
        {"url": SEED, "rate_limit": 0},
        {"url": SEED, "timeout_seconds": 0},
    ],
)
def test_invalid_arguments_never_launch_or_access_artifacts(tmp_path, monkeypatch, kwargs):
    async def forbidden(*_args, **_kwargs):
        pytest.fail("invalid input launched scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)

    async def scenario():
        artifacts = Artifacts()
        tool, state = await make_tool(tmp_path, artifacts)
        with pytest.raises(ToolInputError):
            await tool(**kwargs)
        assert not artifacts.reads and not artifacts.writes
        assert state.metrics.counters["tool_errors"] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "existing,write_failure,code",
    [
        (True, False, "scan_output_exists"),
        (False, True, "scan_artifact_failed"),
    ],
)
def test_artifact_failures_are_explicit_without_overwrite(
    tmp_path, monkeypatch, existing, write_failure, code
):
    source = (
        "raise AssertionError('must not launch')"
        if existing
        else f"print({json.dumps(record())!r})"
    )
    executable(tmp_path, "katana", source)
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        artifacts = Artifacts(existing=existing, fail_write=write_failure)
        tool, state = await make_tool(tmp_path, artifacts)
        result = await tool(SEED)
        assert result["status"] == "failed" and result["errorCode"] == code
        assert result["artifacts"] == {} and result["targetsArtifact"] is None
        assert state.metrics.counters["tool_errors"] == 1
        assert "canary" not in repr(result) + repr(state.metrics)
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


def test_publication_failure_preserves_primary_process_error(tmp_path, monkeypatch):
    executable(tmp_path, "katana", "raise AssertionError('mock process required')")
    monkeypatch.setenv("PATH", str(tmp_path))

    async def failed_process(*_args, **_kwargs):
        return ProcessResult(None, stdout=json.dumps(record()).encode(), error_code="scan_timeout")

    monkeypatch.setattr(scan, "run_process", failed_process)

    async def scenario():
        tool, _ = await make_tool(tmp_path, Artifacts(fail_write=True))
        result = await tool(SEED)
        assert result["errorCode"] == "scan_timeout"
        assert result["artifactErrorCode"] == "scan_artifact_failed"
        assert result["targetsArtifact"] is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "version,accepted", [("1.5.0", False), ("1.6.1", False), ("1.7.0", True), ("1.8.0", False)]
)
def test_unsupported_binary_version_only_disables_katana(tmp_path, monkeypatch, version, accepted):
    executable(
        tmp_path, "katana", f"import sys; print('Current version: v{version}', file=sys.stderr)"
    )
    executable(tmp_path, "naabu", "print('fixture')")
    monkeypatch.setenv("PATH", str(tmp_path))
    available = asyncio.run(scan.ScanToolsetFactory().probe())
    assert available == ({"scan_naabu", "scan_katana"} if accepted else {"scan_naabu"})
