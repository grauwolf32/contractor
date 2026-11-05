from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import os
import shlex
import shutil
import sys
import tempfile
import threading
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from fakes.process_markers import atomic_marker_source
from fakes.worker_runtime import StubADKWorkerRuntimeFactory
from fakes.worker_state import WorkerState
from google.adk.tools import FunctionTool
from target_policy_fixtures import SCAN_TEST_POLICY

import contractor_runtime.threads as threads_module
import contractor_runtime.toolsets.scan.process as process_module
import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import HTTPProxySettings, RuntimeSettings
from contractor_runtime.core.target_policy import (
    TargetPolicyConfig,
    TargetUnresolved,
    parse_allowed_networks,
)
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.toolsets.scan.ffuf import ffuf_observation
from contractor_runtime.toolsets.scan.katana import TARGET_LIST_MEDIA_TYPE, katana_observation
from contractor_runtime.toolsets.scan.process import ProcessResult, run_process
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory


def executable(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}\n")
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("available_scanner", scan.SCANNERS, ids=lambda scanner: scanner.name)
def test_capabilities_check_each_binary_independently(
    tmp_path, monkeypatch, scanner_probe_budget, available_scanner
):
    failures = iter(["exit 5", "exit 2", "exit 3", "exit 4"])
    for scanner in scan.SCANNERS:
        # These probes test scanner independence, not Python interpreter startup time.
        source = (
            "printf '%s\\n' 'Current version: v1.7.0'"
            if scanner is available_scanner
            else next(failures)
        )
        path = tmp_path / scanner.binary
        path.write_text(f"#!/bin/sh\n{source}\n")
        path.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))

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
        (tmp_path / "katana").unlink()
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


def test_hung_scanner_probe_does_not_hide_a_healthy_binary(
    tmp_path, monkeypatch, scanner_probe_budget
):
    # Both children must start before the test expires the hung child's timer.
    marker = tmp_path / "hung.pid"
    hanging = tmp_path / "nuclei"
    hanging.write_text(
        f"#!/bin/sh\nprintf '%s' $$ > {shlex.quote(str(marker))}\n"
        "printf 'hung-ready\\n'\nexec /bin/sleep 120\n"
    )
    hanging.chmod(0o700)
    healthy = tmp_path / "katana"
    healthy.write_text("#!/bin/sh\nprintf '%s\\n' 'Current version: v1.7.0'\n")
    healthy.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))

    async def scenario():
        ready, healthy_finished = asyncio.Event(), asyncio.Event()
        original_read = asyncio.StreamReader.read
        original_run = scan.run_process

        async def observed_read(self, n=-1):
            data = await original_read(self, n)
            if b"hung-ready" in data:
                ready.set()
            return data

        async def observed_run(command, *args):
            result = await original_run(command, *args)
            if Path(command[0]).name == "katana":
                healthy_finished.set()
            return result

        monkeypatch.setattr(asyncio.StreamReader, "read", observed_read)
        monkeypatch.setattr(scan, "run_process", observed_run)
        factory = scan.ScanToolsetFactory()
        running = asyncio.create_task(factory.probe())
        try:
            await asyncio.wait_for(asyncio.gather(ready.wait(), healthy_finished.wait()), 30)
            assert not running.done()
            advance_clock(scanner_probe_budget)
            assert await asyncio.wait_for(running, 30) == frozenset({"scan_katana"})
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)

    run_with_advanceable_clock(scenario())


async def make_tools(tmp_path, *, selected=None, proxy=False, scanners=scan.SCANNERS):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    factory = scan.ScanToolsetFactory(
        templates_directory=templates, scanners=scanners, target_policy=SCAN_TEST_POLICY
    )
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
        resources=state.resources,
        adapter_handles=AdapterHandles(tool_subprocess=object() if proxy else None),
    )
    return tools, state


def install_echo_scanners(tmp_path, monkeypatch):
    for scanner in scan.SCANNERS:
        executable(
            tmp_path,
            scanner.binary,
            "import json, os, pathlib, stat, sys\n"
            "args = sys.argv[1:]\n"
            "target = pathlib.Path(args[args.index('-l') + 1]) if '-l' in args else None\n"
            "print(json.dumps({'args': args, 'env': dict(os.environ), "
            "'cwd': os.getcwd(), "
            "'targetFile': target.read_text() if target else None, "
            "'targetMode': stat.S_IMODE(target.stat().st_mode) if target else None}))",
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
                "scan_katana": ["url"],
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
        assert "-u" not in args
        assert args[args.index("-l") + 1].endswith("/target.txt")
        assert captured["targetFile"] == "https://target.invalid/?q=$(touch%20owned)\n"
        assert captured["targetMode"] == 0o600
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


def test_nuclei_comma_url_remains_one_private_target(tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)
    url = "http://target.invalid/?q=1,http://169.254.169.254/latest/meta-data/"

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        result = await tools["scan_nuclei"](url)
        assert result["status"] == "completed", result
        captured = result["results"][0]
        assert "-u" not in captured["args"]
        assert captured["targetFile"] == url + "\n"
        assert captured["targetMode"] == 0o600
        assert not Path(captured["cwd"]).exists()
        await tools["scan_nuclei"].close()

    asyncio.run(scenario())


def test_all_scanners_available_from_relative_path(tmp_path, monkeypatch):
    for scanner in scan.SCANNERS:
        arguments = list(scanner.version_arguments)
        executable(
            tmp_path,
            scanner.binary,
            "print('Current version: v1.7.0')\n"
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
        "scan_katana",
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


async def make_policy_tools(tmp_path, *, networks=(), http_proxy=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    templates = tmp_path / "templates"
    templates.mkdir(exist_ok=True)
    table = {
        "rebind.example": "127.0.0.1",
        "gateway.internal": "10.0.0.7",
        "artifacts.internal": "10.0.0.8",
        "app.example": "93.184.215.14",
    }

    async def resolver(host, port):
        if host not in table:
            raise TargetUnresolved
        return (ipaddress.ip_address(table[host]),)

    factory = scan.ScanToolsetFactory(
        templates_directory=templates,
        target_policy=TargetPolicyConfig(
            allowed_networks=parse_allowed_networks(networks), resolver=resolver
        ),
    )
    state = WorkerState()
    tools = await factory.create_selected(
        selected=sorted(factory.exported_tools),
        allocation_id="allocation",
        run_id="run",
        namespace="scanner",
        runtime_settings=RuntimeSettings(
            llmGatewayUrl="https://gateway.internal",
            artifactApiUrl="https://artifacts.internal:8443/private",
            httpProxy=http_proxy,
            requestTimeoutSeconds=60,
        ),
        workspace=AllocationWorkspace(root=tmp_path, path=workspace),
        resources=state.resources,
    )
    return tools, state


@pytest.mark.parametrize(
    "name,arguments,code",
    [
        ("scan_nuclei", {"url": "http://127.1:8080/"}, "scan_target_denied"),
        ("scan_nuclei", {"url": "http://2130706433/"}, "scan_target_denied"),
        ("scan_nuclei", {"url": "http://metadata.google.internal/"}, "scan_target_denied"),
        ("scan_nuclei", {"url": "http://rebind.example/"}, "scan_target_denied"),
        ("scan_nuclei", {"url": "https://gateway.internal/"}, "scan_target_denied"),
        ("scan_nuclei", {"url": "http://missing.example/"}, "scan_target_unresolved"),
        ("scan_sqlmap", {"url": "http://[::ffff:169.254.169.254]/?id=1"}, "scan_target_denied"),
        ("scan_katana", {"url": "http://169.254.10.10/"}, "scan_target_denied"),
        ("scan_katana", {"url": "http://[fe80::1]/"}, "scan_target_denied"),
        ("scan_naabu", {"host": "0x7f000001"}, "scan_target_denied"),
        ("scan_naabu", {"host": "169.254.169.254"}, "scan_target_denied"),
        ("scan_naabu", {"host": "rebind.example", "ports": "80"}, "scan_target_denied"),
        (
            "scan_ffuf",
            {
                "url": "http://127.0.0.1/FUZZ",
                "wordlist_ref": {"namespace": "inputs", "name": "wordlist", "revision": "r1"},
            },
            "scan_target_denied",
        ),
    ],
)
def test_scan_destinations_are_checked_before_launch(name, arguments, code, tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)

    async def forbidden(*args, **kwargs):
        pytest.fail("a denied destination launched a scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)

    async def scenario():
        tools, state = await make_policy_tools(tmp_path)
        result = await tools[name](**arguments)
        assert result["status"] == "failed"
        assert result["errorCode"] == code
        assert state.metrics.tool_calls[0].arguments is None
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())


def test_private_scan_targets_need_no_operator_network(tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_policy_tools(tmp_path)
        for url in ("http://10.0.0.5/", "http://192.168.1.10:8080/", "http://[fd00::5]/"):
            result = await tools["scan_nuclei"](url)
            assert result["status"] == "completed", url
        # A Runtime endpoint inside a private network stays denied.
        protected = await tools["scan_nuclei"]("https://10.0.0.8:8443/")
        assert protected["errorCode"] == "scan_target_denied"

    asyncio.run(scenario())


def test_operator_networks_allow_scans_except_runtime_endpoints(tmp_path, monkeypatch):
    install_echo_scanners(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_policy_tools(tmp_path, networks=("127.0.0.0/8",))
        allowed = await tools["scan_nuclei"]("http://127.0.0.1:3000/")
        assert allowed["status"] == "completed"
        public = await tools["scan_naabu"]("app.example", ports="80,443")
        assert public["status"] == "completed"
        # Any selected port that reaches a Runtime endpoint denies the scan.
        protected = await tools["scan_naabu"]("10.0.0.8", ports="8000-9000")
        assert protected["errorCode"] == "scan_target_denied"
        other = await tools["scan_naabu"]("10.0.0.8", ports="80")
        assert other["status"] == "completed"

    asyncio.run(scenario())


@pytest.mark.parametrize("targets", [["tool-http"], ["llm-gateway", "tool-http"]])
def test_tool_http_proxy_route_fails_scanners_closed(tmp_path, monkeypatch, targets):
    install_echo_scanners(tmp_path, monkeypatch)

    async def forbidden(*args, **kwargs):
        pytest.fail("a proxied allocation launched a direct scanner")

    monkeypatch.setattr(scan, "run_process", forbidden)

    async def scenario():
        proxy = HTTPProxySettings(
            adapter="http-proxy@1", proxyUrl="http://proxy.internal:3128", targets=targets
        )
        tools, _ = await make_policy_tools(tmp_path, http_proxy=proxy)
        result = await tools["scan_nuclei"]("http://app.example/")
        assert result["errorCode"] == "scan_proxy_unsupported"
        result = await tools["scan_naabu"]("app.example")
        assert result["errorCode"] == "scan_proxy_unsupported"

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
        + atomic_marker_source(marker, "str(child)", indentation="    ")
        + "    time.sleep(60)\n",
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
        # A dead child may remain a zombie until the host's init reaps it,
        # and the host may reap it between any two reads of its /proc entry.
        stat = Path(f"/proc/{child}/stat")

        def running() -> bool:
            try:
                return stat.read_text().split()[2] != "Z"
            except (FileNotFoundError, ProcessLookupError):
                return False

        async with asyncio.timeout(3):
            while running():
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


@pytest.mark.skipif(shutil.which("nuclei") is None, reason="optional nuclei binary is absent")
def test_installed_nuclei_comma_url_does_not_scan_second_origin(tmp_path):
    requests = {"allowed": 0, "other": 0}

    def handler(origin):
        async def handle(reader, writer):
            try:
                await reader.readuntil(b"\r\n\r\n")
                requests[origin] += 1
                body = b"contractor-scan-fixture"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        return handle

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        (tmp_path / "templates" / "fixture.yaml").write_text(
            "id: contractor-test-http\n"
            "info:\n  name: Local test fixture\n  author: contractor\n  severity: info\n"
            "http:\n  - method: GET\n    path:\n      - '{{BaseURL}}/'\n"
        )
        async with (
            await asyncio.start_server(handler("allowed"), "127.0.0.1", 0) as allowed,
            await asyncio.start_server(handler("other"), "127.0.0.1", 0) as other,
        ):
            allowed_port = allowed.sockets[0].getsockname()[1]
            other_port = other.sockets[0].getsockname()[1]
            url = f"http://127.0.0.1:{allowed_port}/?q=1,http://127.0.0.1:{other_port}/"
            result = await tools["scan_nuclei"](
                url, template_ids="contractor-test-http", timeout_seconds=20
            )
        assert result["status"] == "completed", result
        assert requests["allowed"] > 0
        assert requests["other"] == 0
        await tools["scan_nuclei"].close()

    asyncio.run(scenario())


# The installed package path is shared by discovery, standalone tools and the
# model-free Worker. Test private bytes and provenance on the actual adapter.
def test_shared_nuclei_inputs():
    from contractor_runtime.core.input_errors import ToolInputError
    from contractor_runtime.toolsets.scan.nuclei_inputs import validate_nuclei_arguments

    root = Path(__file__).parents[2]
    fixture = json.loads((root / "api/testdata/nuclei-input-cases.json").read_text())
    from contractor_runtime.toolsets.scan.nuclei_rulepack import (
        MAX_FILE_BYTES,
        MAX_ZIP_BYTES,
        RULEPACK_MEDIA_TYPE,
        valid_rulepack_metadata,
    )

    assert fixture["rulepackLimits"] == {
        "zipBytes": MAX_ZIP_BYTES,
        "fileBytes": MAX_FILE_BYTES,
        "mediaType": RULEPACK_MEDIA_TYPE,
    }
    for case in fixture["rulepackMetadataCases"]:
        assert valid_rulepack_metadata(case["mediaType"], case["size"]) == case["valid"], case[
            "name"
        ]
    for case in fixture["cases"]:
        try:
            validate_nuclei_arguments(case["arguments"])
            valid = True
        except ToolInputError:
            valid = False
        assert valid == case["valid"], case["name"]


def test_shared_nuclei_package_canonical_manifest():
    from contractor_runtime.digests import jcs_digest

    root = Path(__file__).parents[2]
    fixture = json.loads((root / "api/testdata/nuclei-package-manifest.json").read_text())
    assert jcs_digest(fixture["manifest"]) == fixture["digest"]


NUCLEI_PIN_FIXTURE = (
    "id: contractor-pin-fixture\ninfo:\n  name: Pinned local fixture\n"
    "  author: contractor\n  severity: info\n  tags: contractor,fixture\n"
    'http:\n  - method: GET\n    path: ["{{BaseURL}}/"]\n'
    "    matchers:\n      - type: word\n        words: [contractor-pin-fixture]\n"
)


def package_pin(source, destination):
    import time

    from contractor_runtime.toolsets.scan.nuclei_package import snapshot_package

    return snapshot_package(source, destination, time.monotonic() + 30).digest


def install_pin_scanner(tmp_path, monkeypatch, *, mode="completed", mutate=False):
    marker = tmp_path / "scans"
    pid = tmp_path / "scan.pid"
    source = (
        "import json, os, pathlib, sys, time\n"
        "if '-version' in sys.argv:\n"
        "    print('Nuclei Engine Version: v3.11.1')\n"
        "elif '-validate' in sys.argv:\n"
        f"    sys.exit({2 if mode == 'invalid_rules' else 0})\n"
        "elif '-tl' in sys.argv:\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('-t')+1])\n"
        "    print('\\n'.join(str(p.relative_to(root)) for p in sorted(root.rglob('*.yaml'))))\n"
        "else:\n"
        f"    pathlib.Path({str(marker)!r}).open('a').write('scan\\n')\n"
        + atomic_marker_source(pid, "str(os.getpid())", indentation="    ")
        + "    root = pathlib.Path(sys.argv[sys.argv.index('-t')+1])\n"
    )
    if mutate:
        source += (
            f"    pathlib.Path({str(tmp_path / 'templates/fixture.yaml')!r})"
            ".write_text('mutated installed bytes')\n"
        )
    if mode == "timeout":
        source += "    time.sleep(60)\n"
    elif mode == "failed":
        source += "    sys.exit(2)\n"
    else:
        source += "    print(json.dumps({'privateBytes': next(root.glob('*.yaml')).read_text()}))\n"
    executable(tmp_path, "nuclei", source)
    monkeypatch.setenv("PATH", str(tmp_path))
    return marker, pid


@pytest.mark.parametrize("change", ["template", "resource", "replacement", "symlink"])
def test_nuclei_package_mismatch_never_launches(tmp_path, monkeypatch, change):
    from test_tool_worker import ArtifactStore

    from contractor_runtime.artifacts import ArtifactClient

    marker, _ = install_pin_scanner(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        tool._session.artifact_client = ArtifactClient("allocation", ArtifactStore())
        source = tmp_path / "templates"
        (source / "fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
        (source / "words.txt").write_text("first")
        digest = package_pin(source, tmp_path / "reference")
        if change == "template":
            (source / "fixture.yaml").write_text(NUCLEI_PIN_FIXTURE + "# changed\n")
        elif change == "resource":
            (source / "words.txt").write_text("second")
        elif change == "replacement":
            digest = "sha256:" + "f" * 64
        else:
            (source / "words.txt").unlink()
            (source / "words.txt").symlink_to(tmp_path / "reference/words.txt")
        value = await tool("http://target.invalid/", templates_digest=digest)
        assert value["status"] == "failed"
        assert value["errorCode"] == (
            "nuclei_package_invalid" if change == "symlink" else "nuclei_package_mismatch"
        )
        assert not marker.exists()
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout", "unknown"])
def test_shared_nuclei_pinned_worker_manifest_receipt_and_private_bytes(
    tmp_path, monkeypatch, outcome
):
    from test_tool_worker import ArtifactStore, request, tool_spec, worker

    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.contracts.tool_execution import ToolArgumentBinding
    from contractor_runtime.digests import _agent_template_digest

    marker, pid = install_pin_scanner(tmp_path, monkeypatch, mode=outcome, mutate=True)

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        store, spec = ArtifactStore(), tool_spec()
        tool._session.artifact_client = ArtifactClient(spec.allocation_id, store)
        tool._session.namespace = spec.namespace
        source = tmp_path / "templates"
        (source / "fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
        digest = package_pin(source, tmp_path / "reference")
        (source / ".git").mkdir()
        (source / ".git/metadata.yaml").write_text("Git metadata must not reach the scanner")
        for name, value in {
            "templates_digest": digest,
            "timeout_seconds": 1 if outcome == "timeout" else 15,
        }.items():
            spec.agent_template.execution.arguments[name] = ToolArgumentBinding(
                source="literal", value=value
            )
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        value = request(spec, target="http://target.invalid/")
        runtime = await worker(tmp_path, store, tool, spec=spec)
        store.fail_terminal = outcome == "unknown"
        completion = await runtime.invoke(value)
        expected = {
            "failed": "tool_execution_failed",
            "timeout": "tool_execution_failed",
            "unknown": "tool_outcome_unknown",
        }
        if outcome in expected:
            assert completion.failure.code == expected[outcome], completion
        else:
            assert completion.failure is None, completion
        data, _, revision = store.values[(spec.namespace, "template-manifest")]
        manifest = json.loads(data)
        assert manifest == {
            "schemaVersion": 2,
            "packageDigest": digest,
            "engineVersion": "v3.11.1",
            "selectedTemplateCount": 1,
            "source": {"kind": "installed"},
        }
        if outcome == "completed":
            report = json.loads(store.values[(spec.namespace, "report")][0])
            observation = report["observation"]
            assert observation["results"][0]["privateBytes"] == NUCLEI_PIN_FIXTURE
            assert observation["templateManifest"]["revision"] == revision
            assert "artifacts" not in observation
            assert set(completion.result.artifacts) == {"report"}
            assert (
                observation["expectedPackageDigest"] == observation["actualPackageDigest"] == digest
            )
        # Re-created allocation uses durable receipts, even though installed bytes changed.
        recreated = await worker(tmp_path, store, tool, spec=spec)
        replay = await recreated.invoke(value)
        assert replay.failure == completion.failure
        assert marker.read_text() == "scan\n"
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())
        with pytest.raises(ProcessLookupError):
            import os

            os.kill(int(pid.read_text()), 0)

    asyncio.run(scenario())


def test_shared_nuclei_large_package_manifest_survives_worker_and_replay(tmp_path, monkeypatch):
    from test_tool_worker import ArtifactStore, request, tool_spec, worker

    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.contracts.tool_execution import ToolArgumentBinding
    from contractor_runtime.digests import _agent_template_digest

    marker, _ = install_pin_scanner(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        source = tmp_path / "templates"
        (source / "fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
        for index in range(12_000):
            (source / f"rule-{index:05}.yaml").write_text(
                NUCLEI_PIN_FIXTURE.replace("contractor-pin-fixture", f"rule-{index:05}")
            )
        digest = package_pin(source, tmp_path / "reference")
        store, spec = ArtifactStore(), tool_spec()
        tool._session.artifact_client = ArtifactClient(spec.allocation_id, store)
        tool._session.namespace = spec.namespace
        spec.agent_template.execution.arguments["templates_digest"] = ToolArgumentBinding(
            source="literal", value=digest
        )
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        value = request(spec)
        runtime = await worker(tmp_path, store, tool, spec=spec)
        completion = await runtime.invoke(value)
        assert completion.failure is None, completion
        manifest = store.values[(spec.namespace, "template-manifest")][0]
        assert len(manifest) < 512
        assert json.loads(manifest) == {
            "schemaVersion": 2,
            "packageDigest": digest,
            "engineVersion": "v3.11.1",
            "selectedTemplateCount": 12_001,
            "source": {"kind": "installed"},
        }
        assert len(store.values[(spec.namespace, "report")][0]) < 512 * 1024
        recreated = await worker(tmp_path, store, tool, spec=spec)
        replay = await recreated.invoke(value)
        assert replay.result.artifacts == completion.result.artifacts
        assert marker.read_text() == "scan\n"
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bound", ["files", "file_bytes", "total_bytes", "manifest_bytes", "deadline"]
)
def test_nuclei_package_bounds(tmp_path, monkeypatch, bound):
    import time

    from contractor_runtime.toolsets.scan import nuclei_package as package

    source = tmp_path / "source"
    source.mkdir()
    (source / "a.yaml").write_text(NUCLEI_PIN_FIXTURE)
    (source / "words.txt").write_text("resource")
    if bound != "deadline":
        constant = {
            "files": "MAX_FILES",
            "file_bytes": "MAX_FILE_BYTES",
            "total_bytes": "MAX_PACKAGE_BYTES",
            "manifest_bytes": "MAX_MANIFEST_BYTES",
        }[bound]
        monkeypatch.setattr(package, constant, 1)
    with pytest.raises(package.PackageError) as error:
        package.snapshot_package(
            source, tmp_path / "private", 0 if bound == "deadline" else time.monotonic() + 5
        )
    assert str(error.value) == ("scan_timeout" if bound == "deadline" else "nuclei_package_invalid")


def test_nuclei_snapshot_thread_joined_before_cancelled_cleanup(tmp_path, monkeypatch):
    import threading

    from contractor_runtime import threads

    entered, finish, exited = threading.Event(), threading.Event(), threading.Event()

    def copy():
        entered.set()
        finish.wait()
        (tmp_path / "complete").write_text("joined")
        exited.set()

    async def scenario():
        task = asyncio.create_task(
            threads.to_thread_until_done(copy, name="nuclei-package-snapshot")
        )
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("nuclei") is None, reason="optional nuclei binary is absent")
def test_installed_nuclei_pinned_package_and_manifest(tmp_path):
    from test_tool_worker import ArtifactStore

    from contractor_runtime.artifacts import ArtifactClient

    requests = []

    async def handle(reader, writer):
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        body = b"contractor-pin-fixture"
        writer.write(
            b"HTTP/1.1 200 OK\r\nConnection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        store = ArtifactStore()
        tool._session.artifact_client = ArtifactClient("allocation", store)
        (tmp_path / "templates/fixture.yaml").write_text(
            NUCLEI_PIN_FIXTURE.replace("{{BaseURL}}/", "{{BaseURL}}/{{word}}").replace(
                "    matchers:", "    payloads:\n      word: helpers/words.txt\n    matchers:"
            )
        )
        (tmp_path / "templates/helpers").mkdir()
        (tmp_path / "templates/helpers/words.txt").write_text("private-resource\n")
        digest = package_pin(tmp_path / "templates", tmp_path / "reference")
        async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await tool(
                f"http://127.0.0.1:{port}/", templates_digest=digest, timeout_seconds=15
            )
        assert result["status"] == "completed", result
        assert result["results"][0]["template-id"] == "contractor-pin-fixture"
        assert result["templateManifest"]["revision"]
        assert len(requests) == 1 and requests[0].startswith(b"GET /private-resource ")
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


def test_shared_nuclei_cancel_joins_process_and_never_rescans(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from test_tool_worker import ArtifactStore, request, tool_spec, worker

    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.contracts.tool_execution import ToolArgumentBinding
    from contractor_runtime.digests import _agent_template_digest

    marker, pid = install_pin_scanner(tmp_path, monkeypatch, mode="timeout")

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        store, spec = ArtifactStore(), tool_spec()
        tool._session.artifact_client = ArtifactClient(spec.allocation_id, store)
        tool._session.namespace = spec.namespace
        (tmp_path / "templates/fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
        digest = package_pin(tmp_path / "templates", tmp_path / "reference")
        spec.agent_template.execution.arguments["templates_digest"] = ToolArgumentBinding(
            source="literal", value=digest
        )
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        value = request(spec)
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(value))
        async with asyncio.timeout(5):
            while not pid.exists():
                await asyncio.sleep(0.01)
        await runtime.abort(datetime.now(UTC) + timedelta(seconds=5))
        with pytest.raises(asyncio.CancelledError):
            await task
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())
        with pytest.raises(ProcessLookupError):
            import os

            os.kill(int(pid.read_text()), 0)
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(value)).failure.code == "tool_outcome_unknown"
        assert marker.read_text() == "scan\n"
        assert (spec.namespace, "template-manifest") in store.values

    asyncio.run(scenario())


@pytest.mark.parametrize("ambiguous", [False, True])
def test_shared_nuclei_manifest_publication_failure_never_scans(tmp_path, monkeypatch, ambiguous):
    from test_tool_worker import ArtifactStore, request, tool_spec, worker

    from contractor_runtime.artifacts import ArtifactClient, ArtifactTransportError
    from contractor_runtime.contracts.tool_execution import ToolArgumentBinding
    from contractor_runtime.digests import _agent_template_digest

    class ManifestFailureStore(ArtifactStore):
        async def request(self, method, path, **kwargs):
            if method == "PUT" and path.endswith("/template-manifest"):
                if ambiguous:
                    await super().request(method, path, **kwargs)
                raise ArtifactTransportError("manifest response unavailable")
            return await super().request(method, path, **kwargs)

    marker, _ = install_pin_scanner(tmp_path, monkeypatch)

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        tool = tools["scan_nuclei"]
        store, spec = ManifestFailureStore(), tool_spec()
        tool._session.artifact_client = ArtifactClient(spec.allocation_id, store)
        tool._session.namespace = spec.namespace
        (tmp_path / "templates/fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
        digest = package_pin(tmp_path / "templates", tmp_path / "reference")
        spec.agent_template.execution.arguments["templates_digest"] = ToolArgumentBinding(
            source="literal", value=digest
        )
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        value = request(spec)
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(value)).failure.code == "tool_execution_failed"
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(value)).failure.code == "tool_execution_failed"
        assert not marker.exists()
        assert ((spec.namespace, "template-manifest") in store.values) == ambiguous
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


def rulepack_zip(entries, *, stored=False):
    import io
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream, "w", zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    ) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return stream.getvalue()


def test_rulepack_complete_pack_and_real_cardinality_bounds(tmp_path):
    import time

    from contractor_runtime.toolsets.scan.nuclei_rulepack import (
        MAX_FILES,
        RULEPACK_MEDIA_TYPE,
        materialize_rulepack,
    )

    entries = [
        (f"rule-{i}.yaml", NUCLEI_PIN_FIXTURE.replace("contractor-pin-fixture", f"fixture-{i}"))
        for i in range(MAX_FILES)
    ]
    snapshot, identifiers, digest = materialize_rulepack(
        rulepack_zip(entries),
        RULEPACK_MEDIA_TYPE,
        tmp_path / "accepted",
        time.monotonic() + 30,
    )
    assert len(snapshot.files) == len(identifiers) == MAX_FILES
    assert snapshot.digest.startswith("sha256:") and digest != snapshot.digest
    entries.append(("overflow.yaml", NUCLEI_PIN_FIXTURE))
    from contractor_runtime.toolsets.scan.nuclei_package import PackageError

    with pytest.raises(PackageError, match="nuclei_rulepack_invalid"):
        materialize_rulepack(
            rulepack_zip(entries), RULEPACK_MEDIA_TYPE, tmp_path / "overflow", time.monotonic() + 30
        )


@pytest.mark.parametrize(
    "case",
    [
        "traversal",
        "absolute",
        "directory",
        "backslash",
        "symlink",
        "duplicate",
        "case",
        "duplicate_id",
        "non_template",
        "non_http",
        "mixed",
        "alias",
        "duplicate_key",
        "multi_document",
        "invalid_utf8",
        "oversize",
        "expanded",
        "zip_bytes",
        "deadline",
        "empty",
        "unknown_media",
        "too_deep",
        "tokens",
        "lying_count",
        "truncated",
        "appended",
    ],
)
def test_rulepack_rejects_entire_invalid_pack(tmp_path, case):
    import stat
    import struct
    import time
    import zipfile

    from contractor_runtime.toolsets.scan.nuclei_package import PackageError
    from contractor_runtime.toolsets.scan.nuclei_rulepack import (
        MAX_FILE_BYTES,
        RULEPACK_MEDIA_TYPE,
        materialize_rulepack,
    )

    content = NUCLEI_PIN_FIXTURE
    entries = [("valid.yaml", content)]
    media = RULEPACK_MEDIA_TYPE
    deadline = time.monotonic() + 30
    if case in {"traversal", "absolute", "directory", "backslash"}:
        entries.append(
            (
                {
                    "traversal": "../evil.yaml",
                    "absolute": "/evil.yaml",
                    "directory": "nested/evil.yaml",
                    "backslash": "dir\\evil.yaml",
                }[case],
                content,
            )
        )
    elif case == "symlink":
        link = zipfile.ZipInfo("link.yaml")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        entries.append((link, "valid.yaml"))
    elif case in {"duplicate", "case"}:
        entries.append(("valid.yaml" if case == "duplicate" else "VALID.yaml", content))
    elif case == "duplicate_id":
        entries.append(("other.yaml", content))
    elif case in {"non_template", "non_http", "mixed"}:
        entries.append(
            (
                "bad.yaml",
                "words: [test]"
                if case == "non_template"
                else content.replace("http:", "dns:")
                if case == "non_http"
                else content + "\ncode: [{engine: [sh], source: echo}]\n",
            )
        )
    elif case == "alias":
        entries.append(("bad.yaml", content + "\nvariables: {a: &a value, b: *a}\n"))
    elif case == "duplicate_key":
        entries.append(("bad.yaml", content + "\nid: different\n"))
    elif case == "multi_document":
        entries.append(("bad.yaml", content + "\n---\n" + content))
    elif case == "invalid_utf8":
        entries.append(("bad.yaml", b"\xff"))
    elif case == "oversize":
        entries.append(("bad.yaml", "x" * (MAX_FILE_BYTES + 1)))
    elif case == "zip_bytes":
        entries = [
            (
                f"{i}.yaml",
                content.replace("contractor-pin-fixture", f"fixture-{i}")
                + "#"
                + "x" * (MAX_FILE_BYTES - len(content) - 2),
            )
            for i in range(17)
        ]
    elif case == "expanded":
        # Each file is valid and individually bounded; aggregate expansion fails.
        entries = [
            (
                f"{i}.yaml",
                content.replace("contractor-pin-fixture", f"fixture-{i}")
                + "#"
                + "x" * (MAX_FILE_BYTES - len(content) - 2),
            )
            for i in range(66)
        ]
    elif case == "deadline":
        deadline = 0
    elif case == "empty":
        entries = []
    elif case == "unknown_media":
        media = "application/zip"
    elif case == "too_deep":
        entries.append(("bad.yaml", content + "\nvariables: " + "[" * 65 + "0" + "]" * 65))
    elif case == "tokens":
        entries.append(("bad.yaml", content + "\nvariables: [" + "a," * 60000 + "a]"))
    if case == "duplicate":
        # Building this malformed fixture intentionally emits zipfile's warning.
        with pytest.warns(UserWarning, match="Duplicate name: 'valid.yaml'"):
            payload = rulepack_zip(entries)
    else:
        payload = rulepack_zip(entries, stored=case == "zip_bytes")
    if case == "lying_count":
        payload = bytearray(payload)
        position = payload.rfind(b"PK\x05\x06")
        struct.pack_into("<HH", payload, position + 8, 0, 0)
        payload = bytes(payload)
    elif case == "truncated":
        payload = payload[:-15]
    elif case == "appended":
        payload += b"extra"
    with pytest.raises(
        PackageError, match="scan_timeout" if case == "deadline" else "nuclei_rulepack_invalid"
    ):
        materialize_rulepack(payload, media, tmp_path / "pack", deadline)


@pytest.mark.parametrize(
    "archive, case",
    [
        (False, "valid"),
        (True, "valid"),
        (True, "invalid_rules"),
        (False, "wrong_media"),
        (False, "large_yaml"),
        (False, "missing"),
        (False, "timeout"),
    ],
)
def test_rulepack_adapter_uses_only_exact_pack_without_installed_rules(
    tmp_path, monkeypatch, archive, case
):
    from urllib.parse import parse_qs

    from test_tool_worker import ArtifactStore

    from contractor_runtime.artifacts import ArtifactClient
    from contractor_runtime.toolsets.scan.nuclei_rulepack import RULEPACK_MEDIA_TYPE

    class RulepackStore(ArtifactStore):
        """Records the revision every read of the supplied pack asks for."""

        def __init__(self):
            super().__init__()
            self.pack_reads = []

        async def request(self, method, path, **kwargs):
            parsed = urlsplit(path)
            for kind, suffix in (
                ("metadata", "/inputs/rulepack/metadata"),
                ("payload", "/inputs/rulepack"),
            ):
                if method == "GET" and parsed.path.endswith(suffix):
                    self.pack_reads.append((kind, parse_qs(parsed.query).get("revision")))
            return await super().request(method, path, **kwargs)

    marker, _ = install_pin_scanner(tmp_path, monkeypatch, mode=case)

    async def scenario():
        tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
        (tmp_path / "templates").rmdir()  # Custom mode requires no installation.
        store = RulepackStore()
        data = (
            rulepack_zip([("fixture.yaml", NUCLEI_PIN_FIXTURE)])
            if archive
            else NUCLEI_PIN_FIXTURE.encode()
        )
        media = RULEPACK_MEDIA_TYPE if archive else "application/yaml"
        if case == "wrong_media":
            media = "application/zip"
        if case == "large_yaml":
            data = data + b"#" + b"x" * (1024 * 1024)
        store.values[("inputs", "rulepack")] = (data, media, "revision-pack")
        if case == "missing":
            store.values.clear()
        tool = tools["scan_nuclei"]
        tool._session.artifact_client = ArtifactClient("allocation", store)
        result = await tool(
            "http://target.invalid/",
            templates_digest="sha256:" + "f" * 64,
            rulepack_ref={"namespace": "inputs", "name": "rulepack", "revision": "revision-pack"},
            timeout_seconds=1 if case == "timeout" else 30,
        )
        # The pack is resolved and read only at its exact revision, never the latest.
        assert store.pack_reads[0] == ("metadata", ["revision-pack"])
        assert all(revisions == ["revision-pack"] for _, revisions in store.pack_reads)
        if case not in {"valid", "timeout"}:
            assert result["errorCode"] == (
                "scan_artifact_unavailable" if case == "missing" else "nuclei_rulepack_invalid"
            ), result
            assert not marker.exists()
            await tool.close()
            assert not list((tmp_path / "workspace").iterdir())
            return
        if case == "timeout":
            assert result["errorCode"] == "scan_timeout", result
            await tool.close()
            assert not list((tmp_path / "workspace").iterdir())
            return
        assert result["status"] == "completed", result
        assert result["source"] == "rulepack" and result["selectedTemplateCount"] == 1
        manifest = json.loads(store.values[("scanner", "template-manifest")][0])
        assert manifest["schemaVersion"] == 2
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["digest"].startswith("sha256:")
        assert manifest["selectedTemplates"][0]["id"] == "contractor-pin-fixture"
        assert manifest["selectedTemplates"][0]["digest"] == manifest["files"][0]["digest"]
        assert manifest["source"]["artifact"]["revision"] == "revision-pack"
        assert manifest["source"]["contentDigest"] == result["sourceContentDigest"]
        assert marker.read_text() == "scan\n"
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())


# Off-loop scanner work. These are structural checks, not event-loop gap
# bounds: a worker thread can hold the GIL, so wall-clock gaps are not
# deterministic. Thread identity proves where each step ran, and a gate that
# holds the worker shows the loop still running callbacks meanwhile.

_SCRATCH_EVENTS = frozenset({"tempfile.mkdtemp", "os.mkdir", "open", "shutil.rmtree"})
_scratch_recorders: list[ScratchRecorder] = []


def _audit_scratch(event: str, args: tuple[object, ...]) -> None:
    if not _scratch_recorders or event not in _SCRATCH_EVENTS or not args:
        return
    path = args[0]
    if isinstance(path, (str, bytes, os.PathLike)):
        for recorder in tuple(_scratch_recorders):
            recorder.audit(event, os.fsdecode(os.fspath(path)))


# Audit hooks cannot be removed; this one is inert while no recorder is active.
sys.addaudithook(_audit_scratch)


class Gate:
    """Holds one worker-thread step until the test releases it.

    A step reached on the event-loop thread is recorded instead of held:
    holding it would stop the loop that has to release it.
    """

    def __init__(self, loop_thread: int) -> None:
        self.loop_thread = loop_thread
        self.entered = threading.Event()
        self.release = threading.Event()
        self.on_loop = False

    def hold(self) -> None:
        if threading.get_ident() == self.loop_thread:
            self.on_loop = True
            return
        self.entered.set()
        if not self.release.wait(30):
            raise RuntimeError("the test never released the gate")


class ScratchRecorder:
    """Audited filesystem operations below one directory, with their threads."""

    def __init__(self, root: Path, loop_thread: int) -> None:
        self.root = f"{root}{os.sep}"
        self.loop_thread = loop_thread
        self.operations: list[tuple[str, str]] = []
        self.on_loop: list[tuple[str, str]] = []
        # (event, base name or None for any name) -> gate held before the operation.
        self.gates: dict[tuple[str, str | None], Gate] = {}

    def __enter__(self) -> ScratchRecorder:
        _scratch_recorders.append(self)
        return self

    def __exit__(self, *_exception: object) -> None:
        _scratch_recorders.remove(self)

    def audit(self, event: str, path: str) -> None:
        if not path.startswith(self.root):
            return
        name = os.path.basename(path)
        self.operations.append((event, name))
        if threading.get_ident() == self.loop_thread:
            self.on_loop.append((event, name))
        gate = self.gates.get((event, name)) or self.gates.get((event, None))
        if gate is not None:
            gate.hold()

    def mark(self, label: str) -> int:
        self.operations.append((label, ""))
        return len(self.operations) - 1

    def indices(self, event: str) -> list[int]:
        return [index for index, (name, _) in enumerate(self.operations) if name == event]


OFF_LOOP_WORDLIST = {"namespace": "inputs", "name": "wordlist", "revision": "wordlist-1"}
OFF_LOOP_REQUEST = {"namespace": "inputs", "name": "request", "revision": "request-1"}
OFF_LOOP_FFUF_URL = "https://target.invalid/FUZZ"
OFF_LOOP_KATANA_SEED = "http://target.invalid/"
OFF_LOOP_CASES = (
    "scan_nuclei",
    "scan_sqlmap",
    "scan_sqlmap_request",
    "scan_naabu",
    "scan_ffuf",
    "scan_katana",
)
# Parsers whose thread identity is recorded; every observation projects the
# process result through ProcessResult.observation as well.
OFF_LOOP_PARSERS = {
    "scan_nuclei": ((scan, "_json_lines"),),
    "scan_sqlmap": (),
    "scan_sqlmap_request": ((scan, "parse_http_request"), (scan, "_sqlmap_techniques")),
    "scan_naabu": ((scan, "_json_lines"),),
    "scan_ffuf": ((scan, "parse_wordlist"), (scan, "ffuf_observation")),
    "scan_katana": ((scan, "katana_observation"),),
}
OFF_LOOP_PRIVATE_FILES = {
    "scan_nuclei": {"target.txt"},
    "scan_sqlmap": set(),
    "scan_sqlmap_request": {"request.http"},
    "scan_naabu": set(),
    "scan_ffuf": {"wordlist.txt"},
    "scan_katana": {"seed.txt", "config.yaml"},
}
# Scratch creation, adapter preparation, observation and scratch removal,
# plus each adapter's materialization: a fixed count, whatever the output size.
OFF_LOOP_SUBMISSIONS = {
    "scan_nuclei": 6,
    "scan_sqlmap": 4,
    "scan_sqlmap_request": 6,
    "scan_naabu": 4,
    "scan_ffuf": 5,
    "scan_katana": 5,
}
FFUF_OFF_LOOP_RECORD = {
    "input": {"FUZZ": b64encode(b"foo").decode()},
    "position": 1,
    "status": 200,
    "length": 2,
    "words": 1,
    "lines": 1,
    "url": "https://target.invalid/foo",
    "duration": 123,
    "content-type": "text/plain",
    "redirectlocation": "",
}
FFUF_OFF_LOOP_PROGRESS = (
    b":: Progress: [1/1] :: Job [1/1] :: 10 req/sec :: Duration: [0:00:00] :: Errors: 0 ::\n"
)


def scanner_output(case: str, *, large: bool) -> tuple[bytes, bytes]:
    """Valid scanner stdout and stderr; large output nearly fills the 1 MiB bound."""

    def line(index: int) -> bytes:
        if case in {"scan_nuclei", "scan_naabu"}:
            value = {"host": "target.invalid", "port": index % 65535 + 1, "info": "x" * 160}
            return json.dumps(value).encode()
        if case == "scan_ffuf":
            return json.dumps(FFUF_OFF_LOOP_RECORD).encode()
        if case == "scan_katana":
            return json.dumps(
                {
                    "request": {
                        "method": "GET",
                        "endpoint": f"{OFF_LOOP_KATANA_SEED}p{index:06}",
                        "source": OFF_LOOP_KATANA_SEED,
                    },
                    "response": {"status_code": 200},
                }
            ).encode()
        if case == "scan_sqlmap_request":
            return b"    Type: boolean-based blind" if index % 2 else b"    Payload: id=7 AND 1=1"
        return b"[INFO] testing connection to the target URL"

    stderr = FFUF_OFF_LOOP_PROGRESS if case == "scan_ffuf" else b""
    limit = process_module.MAX_OUTPUT_BYTES - len(stderr)
    lines, size = [], 0
    while True:
        item = line(len(lines)) + b"\n"
        if lines and (not large or size + len(item) > limit):
            return b"".join(lines), stderr
        lines.append(item)
        size += len(item)


def off_loop_store(*, output: str | None = None, failure: str | None = None):
    """An Artifact API fake holding the scanner inputs.

    ``failure`` fails metadata reads of ``scanner/<output>`` as an unavailable
    store ("unavailable") or a lost response ("transport").
    """

    from test_scan_http_request import request_document
    from test_tool_worker import ArtifactStore

    from contractor_runtime.artifacts import ArtifactTransportError

    class Store(ArtifactStore):
        async def request(self, method, path, **kwargs):
            self.threads.append(threading.get_ident())
            metadata = urlsplit(path).path.endswith(f"/scanner/{output}/metadata")
            if failure is not None and method == "GET" and metadata:
                self.reads.append(("metadata", "scanner", output))
                if failure == "transport":
                    raise ArtifactTransportError("metadata response lost")
                return self.response(503, {"code": "storage_unavailable", "retryable": True})
            return await super().request(method, path, **kwargs)

    store = Store()
    store.threads = []
    store.values[("inputs", "wordlist")] = (b"foo\n", "text/plain", "wordlist-1")
    store.values[("inputs", "request")] = (
        json.dumps(request_document()).encode(),
        "application/json",
        "request-1",
    )
    return store


async def off_loop_scanner(tmp_path, monkeypatch, case, stdout, stderr=b"", *, store=None):
    """One scanner whose process returns ``stdout``/``stderr`` without launching."""

    from contractor_runtime.artifacts import ArtifactClient

    install_echo_scanners(tmp_path, monkeypatch)
    launches = []

    async def process(command, directory, timeout):
        launches.append(command)
        return ProcessResult(0, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(scan, "run_process", process)
    name = "scan_sqlmap" if case == "scan_sqlmap_request" else case
    tools, state = await make_tools(tmp_path, selected=[name])
    tool = tools[name]
    store = off_loop_store() if store is None else store
    tool._session.artifact_client = ArtifactClient("allocation", store)
    arguments = {
        "scan_nuclei": ("http://target.invalid/",),
        "scan_sqlmap": ("http://target.invalid/?id=1",),
        "scan_sqlmap_request": (),
        "scan_naabu": ("target.invalid",),
        "scan_ffuf": (OFF_LOOP_FFUF_URL, OFF_LOOP_WORDLIST),
        "scan_katana": (OFF_LOOP_KATANA_SEED,),
    }[case]
    options = {"request_ref": OFF_LOOP_REQUEST} if case == "scan_sqlmap_request" else {}
    return SimpleNamespace(
        tool=tool,
        state=state,
        store=store,
        launches=launches,
        call=lambda **extra: tool(*arguments, **options, **extra),
    )


def record_threads(patch, targets) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    for owner, name in targets:
        original = getattr(owner, name)

        def recorded(*args, _original=original, _name=name, **kwargs):
            calls.append((_name, threading.get_ident()))
            return _original(*args, **kwargs)

        patch.setattr(owner, name, recorded)
    return calls


def count_thread_submissions(patch) -> list[str]:
    names: list[str] = []
    original = threads_module.to_thread_until_done

    async def counted(function, *args, name):
        names.append(name)
        return await original(function, *args, name=name)

    patch.setattr(threads_module, "to_thread_until_done", counted)
    return names


def direct_observation(case: str, tool, result: ProcessResult) -> dict:
    """The observation each adapter's parser computes, called directly."""

    if case == "scan_ffuf":
        return {**ffuf_observation(result, 1), "wordlistArtifact": OFF_LOOP_WORDLIST}
    if case == "scan_katana":
        value, _ = katana_observation(
            result,
            seed=OFF_LOOP_KATANA_SEED,
            max_depth=2,
            max_pages=100,
            rate_limit=10,
            timeout_seconds=60,
        )
        return value
    if case == "scan_sqlmap_request":
        techniques = scan._sqlmap_techniques(result.stdout)
        return {
            **result.observation(),
            "scanner": "sqlmap",
            "stdout": "",
            "stderr": "",
            "diagnosticsRedacted": True,
            "requestArtifact": OFF_LOOP_REQUEST,
            "injectionOutcome": "reported" if techniques else "unknown",
            "injectionTechniques": techniques,
        }
    return tool.observation(result)


@pytest.mark.parametrize("case", OFF_LOOP_CASES)
def test_scanner_parsing_and_scratch_files_stay_off_the_event_loop(tmp_path, monkeypatch, case):
    stdout, stderr = scanner_output(case, large=True)
    assert len(stdout) + len(stderr) > process_module.MAX_OUTPUT_BYTES - 1024

    async def scenario():
        scanner = await off_loop_scanner(tmp_path, monkeypatch, case, stdout, stderr)
        loop_thread = threading.get_ident()
        workspace = tmp_path / "workspace"
        try:
            with (
                monkeypatch.context() as patch,
                ScratchRecorder(workspace, loop_thread) as recorder,
            ):
                parsers = record_threads(
                    patch, ((ProcessResult, "observation"), *OFF_LOOP_PARSERS[case])
                )
                submissions = count_thread_submissions(patch)
                result = await scanner.call()
        finally:
            await scanner.tool.close()

        assert len(scanner.launches) == 1
        # Parsing ran, and never on the loop thread.
        expected_parsers = {"observation", *(name for _, name in OFF_LOOP_PARSERS[case])}
        assert {name for name, _ in parsers} == expected_parsers
        assert loop_thread not in {thread for _, thread in parsers}
        # Scratch creation, private inputs and removal ran, and never on the loop.
        events = {event for event, _ in recorder.operations}
        assert {"tempfile.mkdtemp", "os.mkdir", "shutil.rmtree"} <= events
        opened = {name for event, name in recorder.operations if event == "open"}
        assert OFF_LOOP_PRIVATE_FILES[case] <= opened
        assert recorder.on_loop == []
        assert len(submissions) == OFF_LOOP_SUBMISSIONS[case], submissions
        # Artifact reads and publication stay awaited on the loop.
        assert set(scanner.store.threads) <= {loop_thread}
        assert list(workspace.iterdir()) == []

        # The threaded observation equals a direct parse of the same output.
        expected = direct_observation(
            case, scanner.tool, ProcessResult(0, stdout=stdout, stderr=stderr)
        )
        observed = dict(result)
        if case == "scan_katana":
            published = scanner.store.values[("scanner", "targets")]
            _, data = katana_observation(
                ProcessResult(0, stdout=stdout, stderr=stderr),
                seed=OFF_LOOP_KATANA_SEED,
                max_depth=2,
                max_pages=100,
                rate_limit=10,
                timeout_seconds=60,
            )
            assert published[:2] == (data, TARGET_LIST_MEDIA_TYPE)
            exact = {"namespace": "scanner", "name": "targets", "revision": published[2]}
            assert observed.pop("targetsArtifact") == exact
            assert observed.pop("artifacts") == {"targets": exact}
            del expected["targetsArtifact"], expected["artifacts"]
        assert observed == expected
        assert result["stdoutTruncated"] is True
        if case not in {"scan_sqlmap", "scan_sqlmap_request"}:
            assert result["resultsTruncated"] is True and len(result["results"]) == 100

    asyncio.run(scenario())


def test_probe_scratch_stays_off_the_event_loop(tmp_path, monkeypatch):
    executable(tmp_path, "naabu", "print('fixture')")
    monkeypatch.setenv("PATH", str(tmp_path))
    probe_root = tmp_path / "probe-tmp"
    probe_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(probe_root))

    async def scenario():
        factory = scan.ScanToolsetFactory(scanners=[scan.NaabuTool])
        with ScratchRecorder(probe_root, threading.get_ident()) as recorder:
            assert await factory.probe() == {"scan_naabu"}
        events = {event for event, _ in recorder.operations}
        assert {"tempfile.mkdtemp", "shutil.rmtree"} <= events
        assert recorder.on_loop == []
        assert list(probe_root.iterdir()) == []

    asyncio.run(scenario())


# (case, stage): the held worker-thread step of one invocation.
OFF_LOOP_STAGES = (
    ("scan_ffuf", "create"),
    ("scan_nuclei", "prepare"),
    ("scan_ffuf", "materialize"),
    ("scan_sqlmap_request", "materialize"),
    ("scan_katana", "materialize"),
    ("scan_ffuf", "parse"),
    ("scan_katana", "parse"),
    ("scan_naabu", "parse"),
    ("scan_ffuf", "remove"),
)


def hold_stage(patch, recorder: ScratchRecorder, case: str, stage: str, gate: Gate) -> None:
    if stage == "create":
        recorder.gates[("tempfile.mkdtemp", None)] = gate
    elif stage == "prepare":
        recorder.gates[("os.mkdir", "nuclei-templates")] = gate
    elif stage == "materialize":
        name = {
            "scan_ffuf": "wordlist.txt",
            "scan_sqlmap_request": "request.http",
            "scan_katana": "seed.txt",
        }[case]
        recorder.gates[("open", name)] = gate
    elif stage == "remove":
        recorder.gates[("shutil.rmtree", None)] = gate
    else:
        owner, name = {
            "scan_ffuf": (scan, "ffuf_observation"),
            "scan_katana": (scan, "katana_observation"),
            "scan_naabu": (scan, "_json_lines"),
        }[case]
        original = getattr(owner, name)

        def held(*args, **kwargs):
            gate.hold()
            return original(*args, **kwargs)

        patch.setattr(owner, name, held)


async def until_held(gate: Gate, task: asyncio.Task) -> None:
    """Wait until a worker thread holds the gate; a step on the loop is not held."""

    async with asyncio.timeout(30):
        while not gate.entered.is_set() and not task.done():
            await asyncio.sleep(0.001)
    assert not gate.on_loop, "the held step ran on the event-loop thread"
    assert gate.entered.is_set()


@pytest.mark.parametrize("case,stage", OFF_LOOP_STAGES)
def test_held_scanner_thread_leaves_the_event_loop_running(tmp_path, monkeypatch, case, stage):
    stdout, stderr = scanner_output(case, large=False)

    async def scenario():
        scanner = await off_loop_scanner(tmp_path, monkeypatch, case, stdout, stderr)
        loop = asyncio.get_running_loop()
        gate = Gate(threading.get_ident())
        task = None
        try:
            with (
                monkeypatch.context() as patch,
                ScratchRecorder(tmp_path / "workspace", gate.loop_thread) as recorder,
            ):
                hold_stage(patch, recorder, case, stage, gate)
                task = asyncio.create_task(scanner.call())
                await until_held(gate, task)
                # An independent callback runs while the worker is still held.
                progressed = asyncio.Event()
                loop.call_soon(progressed.set)
                await progressed.wait()
                assert not task.done() and not gate.release.is_set()
                gate.release.set()
                result = await task
        finally:
            gate.release.set()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await scanner.tool.close()
        assert result["status"] == "completed", result
        assert recorder.on_loop == []
        assert len(scanner.launches) == 1
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["cancel", "close"])
@pytest.mark.parametrize("case,stage", OFF_LOOP_STAGES)
def test_stopped_scanner_joins_held_thread_before_cleanup_and_unlock(
    tmp_path, monkeypatch, case, stage, stop
):
    stdout, stderr = scanner_output(case, large=False)

    async def scenario():
        scanner = await off_loop_scanner(tmp_path, monkeypatch, case, stdout, stderr)
        session = scanner.tool._session
        gate = Gate(threading.get_ident())
        task = closing = None
        try:
            with (
                monkeypatch.context() as patch,
                ScratchRecorder(tmp_path / "workspace", gate.loop_thread) as recorder,
            ):
                hold_stage(patch, recorder, case, stage, gate)
                task = asyncio.create_task(scanner.call())
                await until_held(gate, task)
                held = len(recorder.operations)
                if stop == "cancel":
                    task.cancel()
                else:
                    closing = asyncio.create_task(session.close())
                for _ in range(20):
                    await asyncio.sleep(0)
                # The stopped invocation waits for the held thread: no cleanup
                # and no lock release happen while it still runs.
                assert not task.done() and session._lock.locked()
                assert closing is None or not closing.done()
                assert "shutil.rmtree" not in {event for event, _ in recorder.operations[held:]}
                released = recorder.mark("released")
                gate.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                if closing is not None:
                    await closing
        finally:
            gate.release.set()
            for pending in (task, closing):
                if pending is not None:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
            await scanner.tool.close()

        assert not session._lock.locked()
        assert list((tmp_path / "workspace").iterdir()) == []
        removals = recorder.indices("shutil.rmtree")
        assert len(removals) == 1
        if stage in {"create", "prepare", "materialize"}:
            # The scratch is removed only after the held thread finished.
            assert removals[0] > released
        if stage == "create":
            # The directory the thread created after the stop is removed too.
            assert recorder.indices("os.mkdir")[0] > released
        assert ("scanner", "targets") not in scanner.store.values
        assert scanner.state.metrics.counters["tool_errors"] == 1
        if stop == "close":
            assert (await scanner.call())["errorCode"] == "scan_closed"

    asyncio.run(scenario())


def test_deadline_during_held_materialization_joins_thread_and_times_out(tmp_path, monkeypatch):
    stdout, stderr = scanner_output("scan_ffuf", large=False)

    async def scenario():
        scanner = await off_loop_scanner(tmp_path, monkeypatch, "scan_ffuf", stdout, stderr)
        session = scanner.tool._session
        gate = Gate(threading.get_ident())
        task = None
        try:
            with (
                monkeypatch.context() as patch,
                ScratchRecorder(tmp_path / "workspace", gate.loop_thread) as recorder,
            ):
                hold_stage(patch, recorder, "scan_ffuf", "materialize", gate)
                # The test owns the scan deadline: the hour-long budget never
                # expires by itself, and expires only once the thread is held.
                deadlines = []
                real_timeout_at = asyncio.timeout_at

                def recorded_timeout_at(when):
                    deadlines.append(real_timeout_at(when))
                    return deadlines[-1]

                patch.setattr(asyncio, "timeout_at", recorded_timeout_at)
                task = asyncio.create_task(scanner.call(timeout_seconds=3600))
                await until_held(gate, task)
                assert len(deadlines) == 1 and not session._task.cancelling()
                deadlines[0].reschedule(asyncio.get_running_loop().time())
                # The expired deadline cancels the invocation while the thread is held.
                async with asyncio.timeout(30):
                    while not session._task.cancelling():
                        await asyncio.sleep(0)
                for _ in range(20):
                    await asyncio.sleep(0)
                assert deadlines[0].expired()
                assert not task.done() and session._lock.locked()
                released = recorder.mark("released")
                gate.release.set()
                result = await task
        finally:
            gate.release.set()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await scanner.tool.close()
        assert result["status"] == "failed" and result["errorCode"] == "scan_timeout"
        assert result["scanComplete"] is False
        assert not scanner.launches
        assert recorder.indices("shutil.rmtree")[0] > released
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("output", ["targets", "template-manifest"])
@pytest.mark.parametrize("binding", ["missing", "existing", "unavailable", "transport"])
def test_create_only_scanner_output_is_checked_by_metadata_without_payload(
    tmp_path, monkeypatch, output, binding
):
    failure = binding if binding in {"unavailable", "transport"} else None
    store = off_loop_store(output=output, failure=failure)
    existing = (b"existing", "application/octet-stream", "existing-1")
    if binding == "existing":
        store.values[("scanner", output)] = existing

    async def scenario():
        if output == "targets":
            stdout, stderr = scanner_output("scan_katana", large=False)
            scanner = await off_loop_scanner(
                tmp_path, monkeypatch, "scan_katana", stdout, stderr, store=store
            )
            tool = scanner.tool
            call = scanner.call
            launched = lambda: bool(scanner.launches)  # noqa: E731
        else:
            from contractor_runtime.artifacts import ArtifactClient

            _, pid = install_pin_scanner(tmp_path, monkeypatch)
            tools, _ = await make_tools(tmp_path, selected=["scan_nuclei"])
            tool = tools["scan_nuclei"]
            tool._session.artifact_client = ArtifactClient("allocation", store)
            (tmp_path / "templates/fixture.yaml").write_text(NUCLEI_PIN_FIXTURE)
            digest = package_pin(tmp_path / "templates", tmp_path / "reference")

            def call():
                return tool("http://target.invalid/", templates_digest=digest, timeout_seconds=30)

            launched = pid.exists
        try:
            result = await call()
        finally:
            await tool.close()

        assert ("metadata", "scanner", output) in store.reads
        # An existing payload is never downloaded to learn that it exists.
        assert ("payload", "scanner", output) not in store.reads
        if binding == "missing":
            assert result["status"] == "completed", result
            assert launched()
            assert ("scanner", output) in store.values
        else:
            expected = {"existing": "scan_output_exists"}.get(binding, "scan_artifact_unavailable")
            assert result["status"] == "failed" and result["errorCode"] == expected, result
            assert not launched()
            assert store.values.get(("scanner", output)) == (
                existing if binding == "existing" else None
            )
        assert list((tmp_path / "workspace").iterdir()) == []

    asyncio.run(scenario())
