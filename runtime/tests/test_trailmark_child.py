from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

import contractor_runtime.toolsets.code_analysis.trailmark_host as host_module
from contractor_runtime.projectfs import LocalWorkspaceProvider, hydrate_workspace
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.code_analysis.trailmark_host import TrailmarkChildHost


@pytest.mark.parametrize("mode", ["direct", "overlay"])
def test_child_graph_uses_exact_effective_snapshot_not_local_underlay(
    tmp_path: Path, mode: str
) -> None:
    async def scenario() -> None:
        initial = (
            "def EffectiveOnly():\n    return 1\n"
            if mode == "direct"
            else "def UnderlayOnly():\n    return 1\n"
        )
        spec, reader = workspace_inputs([("source", "", archive({"src/app.py": initial.encode()}))])
        spec.mode = mode  # type: ignore[assignment]
        provider = LocalWorkspaceProvider(settings("local", tmp_path / "provider"))
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id=f"graph-{mode}",
            timeout_seconds=5,
        )
        if mode == "overlay":
            await session.write_text("src/app.py", "def EffectiveOnly():\n    return 2\n")
            snapshot = await session.snapshot()
        else:
            snapshot = await session.snapshot()
            physical = Path(session.storage.root) / "run_workdir" / "src" / "app.py"
            physical.write_text("def UnderlayOnly():\n    return 9\n", encoding="utf-8")

        scratch = tmp_path / "allocation-scratch"
        host = TrailmarkChildHost(scratch)
        try:
            result = await host.build(snapshot)
            symbols = await host.symbols()
            names = {item.name for item in symbols.items}
            assert result.snapshot_digest == snapshot.digest
            assert result.coverage.analyzed_files == 1
            assert "EffectiveOnly" in names
            assert "UnderlayOnly" not in names
            assert {item.path for item in symbols.items} == {"src/app.py"}
        finally:
            await host.close()
            await session.close()
            await provider.cleanup(session.storage)
        assert list(scratch.iterdir()) == []

    asyncio.run(scenario())


def test_mirror_admission_is_lexicographic_bounded_and_reports_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_module, "MAX_GRAPH_FILES", 1)
    files = (
        _file("b.py", "def OmittedByFileLimit():\n    pass\n"),
        _file("a.py", "def Admitted():\n    pass\n"),
        _file("unsupported.scala", "def ScalaOnly = 1\n"),
        _file("oversized.py", "x" * 80),
    )
    monkeypatch.setattr(host_module, "MAX_GRAPH_FILE_BYTES", 64)
    snapshot = WorkspaceSnapshot(
        directories=(),
        files=files,
        binary_paths=("image.bin",),
        digest="sha256:" + "2" * 64,
    )

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path / "scratch")
        try:
            result = await host.build(snapshot)
            page = await host.symbols()
            assert {item.name for item in page.items} >= {"Admitted"}
            assert "OmittedByFileLimit" not in {item.name for item in page.items}
            assert result.coverage.wire() == {
                "analyzedFiles": 1,
                "analyzedBytes": files[1].size,
                "binaryFiles": 1,
                "unsupportedSourceFiles": 1,
                "oversizedFiles": 1,
                "parseErrors": 0,
                "incomplete": True,
                "reasons": ["file_limit"],
            }
        finally:
            await host.close()

    asyncio.run(scenario())


def test_child_boundary_contains_no_runtime_secret_or_physical_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "runtime-token-canary-never-forward"
    physical_canary = str(tmp_path)
    captured: dict[str, Any] = {}
    original = asyncio.create_subprocess_exec

    async def capture(*arguments: str, **keywords: Any) -> asyncio.subprocess.Process:
        captured["arguments"] = arguments
        captured["environment"] = keywords.get("env")
        return await original(*arguments, **keywords)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    source = "def VisibleSymbol():\n    return 1\n"
    snapshot = WorkspaceSnapshot(
        directories=(),
        files=(_file("safe.py", source),),
        binary_paths=(),
        digest="sha256:" + "3" * 64,
    )

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path / "physical-host-canary" / "scratch")
        try:
            await host.build(snapshot)
            page = await host.symbols()
            assert page.items
            assert all(not item.path.startswith("/") for item in page.items)
            assert host.stderr_observed is False
        finally:
            await host.close()

    asyncio.run(scenario())
    boundary = json.dumps(
        {"arguments": captured["arguments"], "environment": captured["environment"]},
        sort_keys=True,
    )
    assert secret not in boundary
    assert physical_canary not in boundary
    assert "safe.py" not in boundary
    assert source not in boundary
    assert set(captured["environment"]) == set(host_module._SAFE_CHILD_ENVIRONMENT)

    request = host_module._encode_request(
        {
            "schemaVersion": "1.0",
            "requestId": "r1",
            "operation": "build",
            "arguments": {"snapshotDigest": snapshot.digest, "coverage": {}},
        }
    )
    assert secret.encode() not in request
    assert physical_canary.encode() not in request
    assert source.encode() not in request
    assert b"safe.py" not in request


@pytest.mark.parametrize(
    "payload",
    [
        b"{not-json",
        b'{"schemaVersion":"1.0","schemaVersion":"1.0","requestId":"r1",'
        b'"operation":"summary","arguments":{}}',
        json.dumps(
            {
                "schemaVersion": "9.0",
                "requestId": "r1",
                "operation": "summary",
                "arguments": {},
            }
        ).encode(),
    ],
)
def test_child_rejects_malformed_or_wrong_schema_frames(tmp_path: Path, payload: bytes) -> None:
    async def scenario() -> None:
        process = await _raw_child(tmp_path)
        stdout, stderr = await process.communicate(struct.pack(">I", len(payload)) + payload)
        assert process.returncode not in {None, 0}
        assert stdout == b""
        assert stderr == b""

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "wire",
    [
        struct.pack(">I", host_module.MAX_REQUEST_BYTES + 1),
        b"\x00\x00",
        struct.pack(">I", 20) + b"{}",
    ],
)
def test_child_rejects_oversized_and_partial_frames(tmp_path: Path, wire: bytes) -> None:
    async def scenario() -> None:
        process = await _raw_child(tmp_path)
        stdout, stderr = await process.communicate(wire)
        assert process.returncode not in {None, 0}
        assert stdout == b""
        assert stderr == b""

    asyncio.run(scenario())


def test_child_returns_bounded_error_with_exact_request_id(tmp_path: Path) -> None:
    request = {
        "schemaVersion": "1.0",
        "requestId": "request-exact-1",
        "operation": "not-supported",
        "arguments": {},
    }
    payload = json.dumps(request, separators=(",", ":"), sort_keys=True).encode()

    async def scenario() -> None:
        process = await _raw_child(tmp_path)
        stdout, stderr = await process.communicate(struct.pack(">I", len(payload)) + payload)
        assert process.returncode == 0
        assert stderr == b""
        length = struct.unpack(">I", stdout[:4])[0]
        assert length == len(stdout) - 4
        response = json.loads(stdout[4:])
        assert response == {
            "schemaVersion": "1.0",
            "requestId": "request-exact-1",
            "ok": False,
            "code": "unsupported_operation",
            "retryable": False,
        }

    asyncio.run(scenario())


async def _raw_child(root: Path) -> asyncio.subprocess.Process:
    root.mkdir(parents=True, exist_ok=True)
    return await asyncio.create_subprocess_exec(
        *host_module._child_command(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=root,
        env=dict(host_module._SAFE_CHILD_ENVIRONMENT),
        start_new_session=True,
    )


def _file(path: str, text: str) -> WorkspaceTextFile:
    return WorkspaceTextFile(path=path, text=text, size=len(text.encode("utf-8")))
