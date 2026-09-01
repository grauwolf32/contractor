from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import ssl
import subprocess
import sys
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import contractor_runtime.cli as runtime_cli
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState


def test_shutdown_cancels_inflight_heartbeat_and_stops_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        settings = make_settings(tmp_path)
        state = RuntimeState(instance_id="runtime-cli")
        stop = asyncio.Event()
        transport = InflightTransport()
        fake_server = FakeServer()
        monkeypatch.setattr(
            runtime_cli,
            "runtime_agent_client_context",
            lambda **_: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )
        monkeypatch.setattr(
            runtime_cli,
            "runtime_agent_server_context",
            lambda **_: ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
        )

        task = asyncio.create_task(
            runtime_cli.serve(
                settings,
                state=state,
                transport=transport,
                stop_requested=stop,
                server_factory=lambda _: fake_server,  # type: ignore[arg-type,return-value]
                install_signal_handlers=False,
            )
        )
        await asyncio.wait_for(transport.heartbeat_started.wait(), timeout=2)
        stop.set()
        await asyncio.wait_for(task, timeout=2)
        assert transport.heartbeat_cancelled
        assert fake_server.stopped
        assert (await state.snapshot()).process_state is ProcessState.STOPPING
        await asyncio.sleep(0)
        current = asyncio.current_task()
        leaked = [
            pending
            for pending in asyncio.all_tasks()
            if pending is not current and not pending.done()
        ]
        assert leaked == []

    asyncio.run(scenario())


def test_process_sigterm_during_real_mtls_heartbeat_exits_cleanly(tmp_path: Path) -> None:
    pki = generate_pki(tmp_path / "pki")
    heartbeat_started = threading.Event()
    release_heartbeat = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            if self.path == "/private/v1/agents/register":
                response = {
                    "apiVersion": "contractor/v1alpha1",
                    "privateProtocolVersion": 2,
                    "runtimeAgentId": "a" * 64,
                    "labels": [],
                    "labelRevision": 1,
                    "heartbeatIntervalSeconds": 10,
                    "confirmedLeaseSeconds": 60,
                }
            else:
                heartbeat_started.set()
                release_heartbeat.wait(timeout=10)
                response = {
                    "apiVersion": "contractor/v1alpha1",
                    "ackSeq": payload["heartbeatSeq"],
                    "action": "continue",
                }
            encoded = json.dumps(response, separators=(",", ":")).encode()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

        def log_message(self, _: str, *__: object) -> None:
            return

    control_plane = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    control_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    control_tls.minimum_version = ssl.TLSVersion.TLSv1_3
    control_tls.verify_mode = ssl.CERT_REQUIRED
    control_tls.load_verify_locations(cafile=pki["ca"])
    control_tls.load_cert_chain(
        certfile=pki["control_plane_certificate"], keyfile=pki["control_plane_key"]
    )
    control_plane.socket = control_tls.wrap_socket(control_plane.socket, server_side=True)
    control_thread = threading.Thread(target=control_plane.serve_forever)
    control_thread.start()
    runtime_port = unused_tcp_port()
    control_port = int(control_plane.server_address[1])
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "contractor_runtime",
            "--control-plane-url",
            f"https://localhost:{control_port}",
            "--advertised-control-url",
            f"https://localhost:{runtime_port}",
            "--advertised-a2a-url",
            f"https://localhost:{runtime_port}",
            "--ca-file",
            str(pki["ca"]),
            "--certificate-file",
            str(pki["agent_certificate"]),
            "--private-key-file",
            str(pki["agent_key"]),
            "--listen",
            f"127.0.0.1:{runtime_port}",
            "--work-root",
            str(tmp_path / "work"),
            "--request-timeout-seconds",
            "30",
            "--shutdown-grace-seconds",
            "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert heartbeat_started.wait(timeout=8), "Runtime Agent did not begin its heartbeat"
        process.terminate()
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=2)
            pytest.fail(f"Runtime Agent ignored SIGTERM\nstdout:\n{stdout}\nstderr:\n{stderr}")
        assert return_code == 0
        stdout, stderr = process.communicate()
        assert "Traceback" not in stderr
        assert pki["agent_key"].read_text(encoding="utf-8") not in stdout + stderr
    finally:
        release_heartbeat.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        control_plane.shutdown()
        control_plane.server_close()
        control_thread.join(timeout=3)
    assert not control_thread.is_alive()


class FakeServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False
        self.force_exit = False
        self.stopped = False

    async def serve(self) -> None:
        self.started = True
        while not self.should_exit and not self.force_exit:
            await asyncio.sleep(0)
        self.stopped = True


class InflightTransport:
    def __init__(self) -> None:
        self.heartbeat_started = asyncio.Event()
        self.heartbeat_cancelled = False
        self.calls = 0

    async def post_json(self, _: str, __: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "apiVersion": "contractor/v1alpha1",
                "heartbeatIntervalSeconds": 10,
                "confirmedLeaseSeconds": 60,
            }
        self.heartbeat_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.heartbeat_cancelled = True
            raise
        raise AssertionError("unreachable")


def make_settings(tmp_path: Path) -> Settings:
    placeholder = tmp_path / "placeholder"
    return Settings(
        control_plane_url="https://localhost:8443",
        advertised_control_url="https://localhost:9443",
        advertised_a2a_url="https://localhost:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
        work_root=tmp_path / "work",
        request_timeout_seconds=1,
        shutdown_grace_seconds=1,
    )


def generate_pki(root: Path) -> dict[str, Path]:
    repository = Path(__file__).resolve().parents[2]
    for arguments in (
        ("init-ca", "--root", str(root)),
        ("issue-control-plane", "--root", str(root)),
        ("issue-agent", "--root", str(root), "--name", "agent-1"),
    ):
        subprocess.run(
            ["go", "run", "./cmd/contractor-pki", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    return {
        "ca": root / "ca.crt",
        "control_plane_certificate": root / "control-plane.crt",
        "control_plane_key": root / "control-plane.key",
        "agent_certificate": root / "agents" / "agent-1.crt",
        "agent_key": root / "agents" / "agent-1.key",
    }


def unused_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
