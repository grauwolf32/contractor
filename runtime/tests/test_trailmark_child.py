from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

import contractor_runtime.toolsets.code_analysis.trailmark_child as child_module
import contractor_runtime.toolsets.code_analysis.trailmark_host as host_module
from contractor_runtime.projectfs import LocalWorkspaceProvider, hydrate_workspace
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    TrailmarkChildHost,
    TrailmarkHostError,
)


@pytest.mark.parametrize(
    "platform,hard_limit,expected",
    [
        ("linux", -1, 1024**3),
        ("linux", 512 * 1024**2, 512 * 1024**2),
        ("darwin", -1, 5 * 1024**3),
        ("darwin", 6 * 1024**3, 5 * 1024**3),
        ("darwin", 3 * 1024**3, 3 * 1024**3),
    ],
)
def test_address_space_budget_accounts_for_darwin_mappings_and_hard_limit(
    monkeypatch, platform, hard_limit, expected
):
    monkeypatch.setattr(child_module.sys, "platform", platform)
    monkeypatch.setattr(child_module.resource, "getrlimit", lambda _: (hard_limit, hard_limit))

    def virtual_size():
        assert platform == "darwin"
        return 4 * 1024**3

    monkeypatch.setattr(child_module, "_darwin_virtual_size_bytes", virtual_size)
    assert child_module._maximum_address_space_bytes() == expected


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


def test_root_init_symbols_survive_same_digest_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = WorkspaceSnapshot(
        directories=(),
        files=(
            _file(
                "__init__.py", "def helper():\n    return 1\n\ndef main():\n    return helper()\n"
            ),
            _file("pkg/mod.py", "def leaf():\n    return 2\n"),
        ),
        binary_paths=(),
        digest="sha256:" + "a" * 64,
    )

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path / "scratch")
        try:
            await host.build(snapshot)
            first_mirror = host._mirror
            assert first_mirror is not None
            original = await host.symbols()
            by_name = {item.name: item for item in original.items}
            assert {"__init__", "helper", "main", "leaf"} <= by_name.keys()
            assert all("code-analysis-mirror-" not in item.name for item in original.items)
            assert [
                item.name
                for item in (await host.find_symbols("__init__.helper", offset=0, limit=10)).items
            ] == ["helper"]
            assert (
                await host.find_symbols(first_mirror.path.name + ".helper", offset=0, limit=10)
            ).items == ()

            helper_id = by_name["helper"].symbol_id
            main_id = by_name["main"].symbol_id
            leaf_id = by_name["leaf"].symbol_id

            async def assert_old_ids_work() -> None:
                callers = await host.relationships("find_callers", helper_id, offset=0, limit=10)
                assert [item.symbol.name for item in callers.items] == ["main"]
                callees = await host.relationships("find_callees", main_id, offset=0, limit=10)
                assert [item.symbol.name for item in callees.items] == ["helper"]
                paths = await host.paths_between(main_id, helper_id, max_depth=3, limit=10)
                assert [[item.name for item in path] for path in paths.items] == [
                    ["main", "helper"]
                ]
                entrypoint_paths = await host.entrypoint_paths_to(helper_id, max_depth=3, limit=10)
                assert [[item.name for item in path] for path in entrypoint_paths.items] == [
                    ["main", "helper"]
                ]
                surface = await host.attack_surface(offset=0, limit=10)
                assert [item.symbol.name for item in surface.items] == ["main"]
                assert all(
                    "code-analysis-mirror-" not in (item.description or "")
                    for item in surface.items
                )
                assert [
                    item.name
                    for item in (await host.find_symbols("pkg.mod.leaf", offset=0, limit=10)).items
                ] == ["leaf"]
                assert (await host.find_symbols("leaf", offset=0, limit=10)).items[
                    0
                ].symbol_id == leaf_id

            await assert_old_ids_work()
            await host.invalidate()
            assert not first_mirror.path.exists()
            await host.build(snapshot)
            assert host._mirror is not None and host._mirror.path != first_mirror.path
            assert (await host.symbols()).items == original.items
            await assert_old_ids_work()

            original_request = host._request_once_locked

            async def time_out_query(
                operation: str, arguments: object, *, timeout: float
            ) -> object:
                if operation == "find_symbol":
                    raise TimeoutError
                return await original_request(operation, arguments, timeout=timeout)

            monkeypatch.setattr(host, "_request_once_locked", time_out_query)
            with pytest.raises(TrailmarkHostError) as timed_out:
                await host.find_symbols("helper", offset=0, limit=10)
            assert timed_out.value.code == "code_analysis_query_timeout"
            assert host.mirror_exists is False
            monkeypatch.setattr(host, "_request_once_locked", original_request)
            await host.build(snapshot)
            assert (await host.symbols()).items == original.items
            await assert_old_ids_work()
        finally:
            await host.close()
        assert list((tmp_path / "scratch").iterdir()) == []

    asyncio.run(scenario())


def test_root_init_proxy_symbols_survive_rebuilds_without_mirror_names(tmp_path: Path) -> None:
    snapshot = WorkspaceSnapshot(
        directories=("pkg",),
        files=(
            _file(
                "__init__.py",
                "class Service:\n"
                "    def run(self):\n"
                "        print('ready')\n"
                "        return self.missing()\n",
            ),
            _file("pkg/mod.py", "def leaf():\n    print('leaf')\n"),
        ),
        binary_paths=(),
        digest="sha256:" + "b" * 64,
    )

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path / "scratch")
        try:
            await host.build(snapshot)
            first_mirror = host._mirror
            assert first_mirror is not None
            original = await host.symbols()
            proxies = {item.name: item for item in original.items if item.kind == "proxy"}
            assert set(proxies) == {"__init__:print", "__init__:Service.missing", "pkg.mod:print"}
            assert [
                item.symbol_id
                for item in (await host.find_symbols("__init__:print", offset=0, limit=10)).items
            ] == [proxies["__init__:print"].symbol_id]
            assert (
                await host.find_symbols(first_mirror.path.name + ":print", offset=0, limit=10)
            ).items == ()

            async def assert_old_proxy_ids_work() -> None:
                for name, caller in (
                    ("__init__:print", "run"),
                    ("__init__:Service.missing", "run"),
                    ("pkg.mod:print", "leaf"),
                ):
                    callers = await host.relationships(
                        "find_callers", proxies[name].symbol_id, offset=0, limit=10
                    )
                    assert [item.symbol.name for item in callers.items] == [caller]

            await assert_old_proxy_ids_work()
            await host.invalidate()
            await host.build(snapshot)
            assert host._mirror is not None and host._mirror.path != first_mirror.path
            assert (await host.symbols()).items == original.items
            await assert_old_proxy_ids_work()
        finally:
            await host.close()
        assert list((tmp_path / "scratch").iterdir()) == []

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
                "excludedFiles": 0,
                "parseErrors": 0,
                "incomplete": True,
                "reasons": ["file_limit"],
            }
        finally:
            await host.close()

    asyncio.run(scenario())


def test_mirror_admission_skips_directories_the_graph_engine_never_walks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Sorted admission reaches build/, node_modules/ and vendor/ before src/.
    # Their files must not exhaust the budget that Trailmark's walk would
    # then spend on nothing.
    monkeypatch.setattr(host_module, "MAX_GRAPH_FILES", 1)
    files = (
        _file(".github/scripts/release.py", "def HiddenDirectory():\n    pass\n"),
        _file("build/generated.py", "def BuildOutput():\n    pass\n"),
        _file("node_modules/pkg/index.js", "function Dependency() {}\n"),
        _file("node_modules/pkg/types.scala", "def ScalaDependency = 1\n"),
        _file("pkg/vendor/lib.go", "package lib\nfunc Vendored() {}\n"),
        _file("src/app.py", "def Application():\n    pass\n"),
        _file("src/notes.md", "not source"),
    )
    snapshot = WorkspaceSnapshot(
        directories=(),
        files=files,
        binary_paths=(),
        digest="sha256:" + "3" * 64,
    )

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path / "scratch")
        try:
            result = await host.build(snapshot)
            page = await host.symbols()
            names = {item.name for item in page.items}
            assert "Application" in names
            assert not names & {"HiddenDirectory", "BuildOutput", "Dependency", "Vendored"}
            assert result.coverage.wire() == {
                "analyzedFiles": 1,
                "analyzedBytes": files[5].size,
                "binaryFiles": 0,
                "unsupportedSourceFiles": 0,
                "oversizedFiles": 0,
                "excludedFiles": 5,
                "parseErrors": 0,
                "incomplete": False,
                "reasons": [],
            }
            assert host._mirror is not None
            mirrored = sorted(
                path.relative_to(host._mirror.path).as_posix()
                for path in host._mirror.path.rglob("*")
                if path.is_file()
            )
            assert mirrored == [".trailmark/entrypoints.toml", "src/app.py"]
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
