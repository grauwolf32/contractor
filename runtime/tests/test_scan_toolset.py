from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest
from google.adk.tools import FunctionTool

import contractor_runtime.toolsets.scan.process as process_module
import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.allocation import WorkerState
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.factories import FactoryRegistry, StubADKWorkerRuntimeFactory
from contractor_runtime.toolsets.scan.process import run_process
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory


def executable(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}\n")
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("available_scanner", scan.SCANNERS, ids=lambda scanner: scanner.name)
def test_capabilities_check_each_binary_independently(tmp_path, monkeypatch, available_scanner):
    failures = iter(["import time; time.sleep(60)", "import sys; sys.exit(2)", "exit(3)"])
    for scanner in scan.SCANNERS:
        source = "import sys; sys.exit(0)" if scanner is available_scanner else next(failures)
        executable(tmp_path, scanner.binary, source)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(scan, "PROBE_TIMEOUT_SECONDS", 0.2)

    async def scenario():
        factory = scan.ScanToolsetFactory()
        registry = FactoryRegistry(
            worker_runtimes={"adk@1": StubADKWorkerRuntimeFactory()},
            toolsets={factory.ref: factory},
            sandbox_profiles={"local-workdir@1": LocalWorkdirFactory(tmp_path / "work")},
        )
        snapshot = await discover_capabilities(registry)
        for scanner in scan.SCANNERS:
            assert snapshot.supports_tools("scan@1", [scanner.name]) == (
                scanner is available_scanner
            )
        assert [item.model_dump(by_alias=True) for item in snapshot.wire_toolsets()] == [
            {"ref": "scan@1", "tools": [available_scanner.name]}
        ]

        # Inaccessible, absent, and unloadable executables must all be negative.
        (tmp_path / "nuclei").chmod(0o600)
        (tmp_path / "sqlmap").unlink()
        (tmp_path / "naabu").write_text("#!/missing/interpreter\n")
        (tmp_path / "ffuf").unlink()
        missing_factory = scan.ScanToolsetFactory()
        empty = await discover_capabilities(
            FactoryRegistry(
                worker_runtimes=registry.worker_runtimes,
                toolsets={missing_factory.ref: missing_factory},
                sandbox_profiles=registry.sandbox_profiles,
            )
        )
        assert not empty.has_toolset("scan@1")
        assert empty.wire_toolsets() == []
        assert empty.runtimes == ("adk@1",)

    asyncio.run(scenario())


async def make_tools(tmp_path, *, selected=None, proxy=False, scanners=scan.SCANNERS):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    factory = scan.ScanToolsetFactory(templates_directory=templates, scanners=scanners)
    state = WorkerState()
    tools = await factory.create_selected(
        selected=sorted(factory.exported_tools) if selected is None else selected,
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://gateway.invalid",
            artifactApiUrl="https://artifacts.invalid",
            requestTimeoutSeconds=60,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        state=state,
        adapter_handles=AdapterHandles(tool_subprocess=object() if proxy else None),
    )
    return tools, state


def install_echo_scanners(tmp_path, monkeypatch):
    for scanner in scan.SCANNERS:
        executable(
            tmp_path,
            scanner.binary,
            "import json, os, sys\n"
            "print(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ), "
            "'cwd': os.getcwd()}))",
        )
    monkeypatch.setenv("PATH", str(tmp_path))


def test_selection_adk_arguments_isolation_metrics_and_cleanup(tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)
    monkeypatch.setenv("PRIVATE_GATEWAY_KEY", "secret-canary")
    monkeypatch.setenv("HTTPS_PROXY", "https://secret-canary.invalid")

    async def scenario():
        tools, state = await make_tools(tmp_path)
        for name, tool in tools.items():
            declaration = FunctionTool(tool)._get_declaration()
            assert declaration.name == name
            assert tool.description == tool.__doc__
            assert declaration.description == inspect.cleandoc(tool.description)
            required = {
                "scan_naabu": ["host"],
                "scan_nuclei": ["url"],
                "scan_sqlmap": [],
                "scan_ffuf": ["url", "wordlist_ref"],
            }
            assert declaration.parameters_json_schema.get("required", []) == required[name]

        nuclei = await tools["scan_nuclei"](
            "https://target.invalid/?q=$(touch%20owned)",
            template_ids="CVE-2024-*",
            tags="cve",
            severity="high,critical",
            rate_limit=3,
        )
        assert nuclei["status"] == "completed"
        captured = nuclei["results"][0]
        args = captured["args"]
        assert args[args.index("-u") + 1] == "https://target.invalid/?q=$(touch%20owned)"
        assert args[args.index("-type") + 1] == "http"
        assert args[args.index("-rate-limit") + 1] == "3"
        assert args[args.index("-t") + 1] == str(tmp_path / "templates")
        assert "-disable-update-check" in args and "-no-interactsh" in args
        assert "secret-canary" not in repr(captured)
        assert captured["env"]["HOME"] == captured["cwd"]
        assert not Path(captured["cwd"]).exists()

        sqlmap = await tools["scan_sqlmap"](
            "https://target.invalid/?id=1",
            parameter="id",
            data="id=1&name=test",
            cookie="session=secret-canary",
            level=2,
            risk=1,
        )
        sql_output = json.loads(sqlmap["stdout"])
        assert "--batch" in sql_output["args"]
        assert "--cookie=session=secret-canary" in sql_output["args"]
        assert "--data=id=1&name=test" in sql_output["args"]
        assert any(arg.startswith("--output-dir=") for arg in sql_output["args"])
        assert not Path(sql_output["cwd"]).exists()

        naabu = await tools["scan_naabu"]("::1", ports="80,443,8000-8010", rate=5)
        args = naabu["results"][0]["args"]
        assert args[args.index("-scan-type") + 1] == "c"
        assert args[args.index("-host") + 1] == "::1"
        assert state.metrics.counters["tool_calls"] == 3
        assert "secret-canary" not in repr(state.metrics)
        assert list((tmp_path / "workspace").iterdir()) == []

        await tools["scan_naabu"].close()
        assert (await tools["scan_sqlmap"]("http://target.invalid"))["errorCode"] == "scan_closed"
        selected, _ = await make_tools(tmp_path, selected=["scan_naabu"])
        assert set(selected) == {"scan_naabu"}
        with pytest.raises(ValueError, match="unknown selected"):
            await make_tools(tmp_path, selected=["exec_command"])

    asyncio.run(scenario())


def test_all_scanners_available_from_relative_path(tmp_path, monkeypatch):
    for scanner in scan.SCANNERS:
        arguments = list(scanner.version_arguments)
        executable(
            tmp_path,
            scanner.binary,
            f"import sys; sys.exit(0 if sys.argv[1:] == {arguments!r} else 1)",
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", ".")
    factory = scan.ScanToolsetFactory()
    assert asyncio.run(factory.probe()) == factory.exported_tools


def test_registered_adapter_owns_binary_preparation_and_decoding(tmp_path, monkeypatch):
    class FixtureTool(scan.ScanTool):
        name = "scan_fixture"
        binary = "custom-executable"
        version_arguments = ("version", "--local")
        description = "Run the local adapter fixture."

        def prepare(self, directory, templates):
            payload = directory / "input.txt"
            payload.write_text("fixture payload")
            return [str(payload)]

        def observation(self, result):
            return {**super().observation(result), "match": result.stdout.decode().strip()}

        async def __call__(self):
            return await self._call(5, lambda: ["scan"])

    executable(
        tmp_path,
        FixtureTool.binary,
        "import pathlib, sys\n"
        "if sys.argv[1:] == ['version', '--local']:\n"
        "    sys.exit(0)\n"
        "assert sys.argv[1] == 'scan'\n"
        "print(pathlib.Path(sys.argv[2]).read_text())",
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    scanners = (*scan.SCANNERS, FixtureTool)
    factory = scan.ScanToolsetFactory(scanners=scanners)
    assert asyncio.run(factory.probe()) == {"scan_fixture"}
    assert factory.exported_tools == {
        "scan_nuclei",
        "scan_sqlmap",
        "scan_naabu",
        "scan_ffuf",
        "scan_fixture",
    }
    assert factory.infrastructure_channels["scan_fixture"] == {"runtime-subprocess-launcher"}

    async def scenario():
        tools, state = await make_tools(tmp_path, selected=["scan_fixture"], scanners=scanners)
        result = await tools["scan_fixture"]()
        assert result["status"] == "completed"
        assert result["scanner"] == "custom-executable"
        assert result["match"] == "fixture payload"
        assert state.metrics.counters["tool_calls"] == 1
        assert list((tmp_path / "workspace").iterdir()) == []
        await tools["scan_fixture"].close()

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="duplicate scan tool names"):
        scan.ScanToolsetFactory(scanners=(FixtureTool, FixtureTool))


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("scan_nuclei", {"url": "file:///etc/passwd"}),
        ("scan_nuclei", {"url": "http://user:pass@target.invalid"}),
        ("scan_nuclei", {"url": "http://target.invalid", "template_ids": "../../file"}),
        ("scan_nuclei", {"url": "http://target.invalid", "tags": "--help"}),
        ("scan_nuclei", {"url": "http://target.invalid", "severity": "bogus"}),
        ("scan_sqlmap", {"url": "http://target.invalid", "cookie": "a=1\nInjected: yes"}),
        ("scan_sqlmap", {"url": "http://target.invalid", "level": True}),
        ("scan_sqlmap", {"url": "http://target.invalid", "risk": 4}),
        ("scan_naabu", {"host": "--help"}),
        ("scan_naabu", {"host": "127.0.0.1/24"}),
        ("scan_naabu", {"host": "127.0.0.1", "ports": "1-65535"}),
        ("scan_naabu", {"host": "127.0.0.1", "ports": "443-80"}),
        ("scan_naabu", {"host": "127.0.0.1", "timeout_seconds": 0}),
    ],
)
def test_invalid_arguments_do_not_launch(name, arguments, tmp_path, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("invalid arguments launched a scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)

    async def scenario():
        tools, state = await make_tools(tmp_path)
        with pytest.raises(scan.ScanInputError):
            await tools[name](**arguments)
        assert state.metrics.counters["tool_calls"] == 1

    asyncio.run(scenario())


def test_proxy_missing_binary_and_missing_templates_are_explicit(tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_tools(tmp_path, proxy=True)
        for name in tools:
            args = (
                {"host": "target.invalid"}
                if name == "scan_naabu"
                else {"url": "http://target.invalid"}
            )
            if name == "scan_ffuf":
                args = {
                    "url": "http://target.invalid/FUZZ",
                    "wordlist_ref": {"namespace": "inputs", "name": "wordlist", "revision": "r1"},
                }
            result = await tools[name](**args)
            assert result["status"] == "failed"
            assert result["errorCode"] == "scan_proxy_unsupported"
        tools, _ = await make_tools(tmp_path)
        (tmp_path / "templates").rmdir()
        assert (await tools["scan_nuclei"]("http://target.invalid"))[
            "errorCode"
        ] == "nuclei_templates_unavailable"
        (tmp_path / "naabu").unlink()
        assert (await tools["scan_naabu"]("target.invalid"))["errorCode"] == "scanner_unavailable"
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())


def test_process_failure_timeout_overflow_and_preview(tmp_path, monkeypatch):
    async def run(source, timeout=2):
        return await run_process([sys.executable, "-c", source], tmp_path, timeout)

    async def scenario():
        failed = await run("import sys; print('diagnostic', file=sys.stderr); sys.exit(7)")
        assert failed.error_code == "scanner_failed" and failed.exit_code == 7
        assert failed.stderr == b"diagnostic\n"
        timed_out = await run("import time; print('partial', flush=True); time.sleep(60)", 0.2)
        assert timed_out.error_code == "scan_timeout"
        assert timed_out.stdout == b"partial\n"
        preview = await run("print('x' * 40000)")
        assert preview.error_code is None
        assert preview.observation()["stdoutTruncated"]
        monkeypatch.setattr(process_module, "MAX_OUTPUT_BYTES", 1000)
        overflow = await asyncio.wait_for(run("while True: print('x' * 10000)"), 3)
        assert overflow.error_code == "output_limit_exceeded"
        assert len(overflow.stdout) + len(overflow.stderr) == 1000

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_cancellation_and_close_reap_process_group(tmp_path, monkeypatch, stop):
    marker = tmp_path / "child-pid"
    executable(
        tmp_path,
        "naabu",
        "import os, signal, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    time.sleep(60)\n"
        "else:\n"
        f"    open({str(marker)!r}, 'w').write(str(child))\n"
        "    time.sleep(60)\n",
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        tools, _ = await make_tools(tmp_path)
        tool = tools["scan_naabu"]
        task = asyncio.create_task(tool("127.0.0.1"))
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        if stop == "cancel":
            task.cancel()
        else:
            await asyncio.wait_for(tool.close(), 3)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        child = int(marker.read_text())
        # A dead child may remain a zombie until the host's init reaps it.
        stat = Path(f"/proc/{child}/stat")
        async with asyncio.timeout(3):
            while stat.exists() and stat.read_text().split()[2] != "Z":
                await asyncio.sleep(0.01)
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())


def test_json_results_are_bounded_and_invalid_output_is_failure(tmp_path, monkeypatch):
    executable(tmp_path, "naabu", "print('not-json')")
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        tools, _ = await make_tools(tmp_path)
        result = await tools["scan_naabu"]("127.0.0.1")
        assert result["errorCode"] == "invalid_scanner_output"
        assert result["invalidResultLines"] == 1

    asyncio.run(scenario())
    results, truncated, invalid = scan._json_lines(b'{"port":80}\n' * 120 + b"[]\n")
    assert len(results) == 100 and truncated and invalid == 1


def test_sqlmap_techniques_never_expose_arbitrary_diagnostics():
    assert scan._sqlmap_techniques(
        b"Authorization: Bearer secret-canary\n"
        b"    Type: boolean-based blind\n"
        b"    Payload: id=1 AND secret-canary\n"
        b"    Type: UNION query\n"
        b"    Type: secret-canary\n"
        b"    Type: error-based secret-canary\n"
        b"    Type: boolean-based blind\n"
    ) == ["boolean-based blind", "union query"]


@pytest.mark.skipif(shutil.which("nuclei") is None, reason="optional nuclei binary is absent")
def test_installed_nuclei_with_local_http_fixture(tmp_path):
    async def handle(reader, writer):
        try:
            await reader.readuntil(b"\r\n\r\n")
            body = b"contractor-scan-fixture"
            writer.write(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: text/plain\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        (tmp_path / "templates" / "fixture.yaml").write_text(
            "id: contractor-test-http\n"
            "info:\n  name: Local test fixture\n  author: contractor\n  severity: info\n"
            "http:\n  - method: GET\n    path:\n      - '{{BaseURL}}/'\n"
            "    matchers:\n      - type: word\n        words:\n"
            "          - contractor-scan-fixture\n"
        )
        async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await tools["scan_nuclei"](
                f"http://127.0.0.1:{port}",
                template_ids="contractor-test-http",
                timeout_seconds=15,
            )
        assert result["status"] == "completed", result
        assert [item["template-id"] for item in result["results"]] == ["contractor-test-http"]
        await tools["scan_nuclei"].close()
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())
