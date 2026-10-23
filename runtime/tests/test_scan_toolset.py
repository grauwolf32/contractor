from __future__ import annotations

import asyncio
import inspect
import ipaddress
import json
import shutil
import sys
from pathlib import Path

import pytest
from fakes.worker_runtime import StubADKWorkerRuntimeFactory
from google.adk.tools import FunctionTool
from target_policy_fixtures import SCAN_TEST_POLICY

import contractor_runtime.toolsets.scan.process as process_module
import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.allocation import WorkerState
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.contracts import HTTPProxySettings, RuntimeSettings
from contractor_runtime.core.target_policy import (
    TargetPolicyConfig,
    TargetUnresolved,
    parse_allowed_networks,
)
from contractor_runtime.factories import FactoryRegistry
from contractor_runtime.toolsets.scan.process import run_process
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory


def executable(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n{source}\n")
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("available_scanner", scan.SCANNERS, ids=lambda scanner: scanner.name)
def test_capabilities_check_each_binary_independently(tmp_path, monkeypatch, available_scanner):
    failures = iter(["exec /bin/sleep 60", "exit 2", "exit 3", "exit 4"])
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
        state=state,
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
        state=state,
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
        assert state.metrics.tool_calls[0].arguments == {}
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
        "elif '-tl' in sys.argv:\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('-t')+1])\n"
        "    print('\\n'.join(str(p.relative_to(root)) for p in sorted(root.rglob('*.yaml'))))\n"
        "else:\n"
        f"    pathlib.Path({str(marker)!r}).open('a').write('scan\\n')\n"
        f"    pathlib.Path({str(pid)!r}).write_text(str(os.getpid()))\n"
        "    root = pathlib.Path(sys.argv[sys.argv.index('-t')+1])\n"
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
        source += "    print(json.dumps({'privateBytes': (root / 'fixture.yaml').read_text()}))\n"
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
    from contractor_runtime.contracts import ArtifactRef
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
        for name, value in {
            "templates_digest": digest,
            "timeout_seconds": 1 if outcome == "timeout" else 15,
        }.items():
            spec.agent_template.execution.arguments[name] = ToolArgumentBinding(
                source="literal", value=value
            )
        spec.agent_template.ref.digest = _agent_template_digest(spec.agent_template)
        value = request(spec, target="http://target.invalid/")
        value.result_artifacts["template_manifest"] = ArtifactRef(
            namespace=spec.namespace, name="template-manifest"
        )
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
        assert manifest["packageDigest"] == digest
        assert manifest["selectedTemplates"][0]["id"] == "contractor-pin-fixture"
        assert manifest["engineVersion"] == "v3.11.1"
        if outcome == "completed":
            report = json.loads(store.values[(spec.namespace, "report")][0])
            observation = report["observation"]
            assert observation["results"][0]["privateBytes"] == NUCLEI_PIN_FIXTURE
            assert observation["templateManifest"]["revision"] == revision
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
    from contractor_runtime.contracts import ArtifactRef
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
        value.result_artifacts["template_manifest"] = ArtifactRef(
            namespace=spec.namespace, name="template-manifest"
        )
        runtime = await worker(tmp_path, store, tool, spec=spec)
        completion = await runtime.invoke(value)
        assert completion.failure is None, completion
        manifest = store.values[(spec.namespace, "template-manifest")][0]
        assert 512 * 1024 < len(manifest) < 16 * 1024 * 1024
        assert len(json.loads(manifest)["selectedTemplates"]) == 12_001
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
    from contractor_runtime.contracts import ArtifactRef
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
        value.result_artifacts["template_manifest"] = ArtifactRef(
            namespace=spec.namespace, name="template-manifest"
        )
        runtime = await worker(tmp_path, store, tool, spec=spec)
        task = asyncio.create_task(runtime.invoke(value))
        async with asyncio.timeout(5):
            while not pid.exists() or not pid.read_text():
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
    from contractor_runtime.contracts import ArtifactRef
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
        value.result_artifacts["template_manifest"] = ArtifactRef(
            namespace=spec.namespace, name="template-manifest"
        )
        runtime = await worker(tmp_path, store, tool, spec=spec)
        assert (await runtime.invoke(value)).failure.code == "tool_execution_failed"
        recreated = await worker(tmp_path, store, tool, spec=spec)
        assert (await recreated.invoke(value)).failure.code == "tool_execution_failed"
        assert not marker.exists()
        assert ((spec.namespace, "template-manifest") in store.values) == ambiguous
        await tool.close()
        assert not list((tmp_path / "workspace").iterdir())

    asyncio.run(scenario())
