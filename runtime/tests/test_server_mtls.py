from __future__ import annotations

import asyncio
import http.client
import socket
import ssl
import subprocess
import threading
import time
from pathlib import Path

import pytest

from contractor_runtime.mtls import runtime_agent_server_context
from contractor_runtime.server import RuntimeServer, create_app, create_server_config
from contractor_runtime.settings import Settings
from contractor_runtime.state import RuntimeState


def test_ca_valid_non_control_plane_peer_is_rejected_before_dispatch(tmp_path: Path) -> None:
    pki = generate_pki(tmp_path / "pki")
    port = unused_tcp_port()
    settings = Settings(
        control_plane_url="https://localhost:8443",
        advertised_control_url=f"https://localhost:{port}",
        advertised_a2a_url=f"https://localhost:{port}",
        ca_file=pki["ca"],
        certificate_file=pki["agent_certificate"],
        private_key_file=pki["agent_key"],
        host="127.0.0.1",
        port=port,
        work_root=tmp_path / "work",
    )
    state = RuntimeState(instance_id="runtime-server-test")
    server_context = runtime_agent_server_context(
        ca_file=settings.ca_file,
        certificate_file=settings.certificate_file,
        private_key_file=settings.private_key_file,
    )
    server = RuntimeServer(create_server_config(settings, create_app(state), server_context))
    failures: list[BaseException] = []

    def run_server() -> None:
        try:
            asyncio.run(server.serve())
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run_server)
    thread.start()
    wait_started(server, thread)
    try:
        agent_context = client_context(pki, "agent_certificate", "agent_key")
        with pytest.raises((ssl.SSLError, OSError, http.client.HTTPException)):
            request(port, agent_context, "POST", "/private/v1/allocations/a/prepare")
        assert asyncio.run(state.snapshot()).route_dispatches == 0

        control_plane_context = client_context(
            pki, "control_plane_certificate", "control_plane_key"
        )
        status, body = request(
            port, control_plane_context, "POST", "/private/v1/allocations/a/prepare"
        )
        assert status == 503
        assert b"allocation_service_unavailable" in body
        assert asyncio.run(state.snapshot()).route_dispatches == 1
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not failures


def request(port: int, context: ssl.SSLContext, method: str, path: str) -> tuple[int, bytes]:
    connection = http.client.HTTPSConnection("localhost", port, context=context, timeout=2)
    try:
        connection.request(method, path, body=b"{}", headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def client_context(pki: dict[str, Path], certificate: str, key: str) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=pki["ca"])
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certfile=pki[certificate], keyfile=pki[key])
    return context


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


def wait_started(server: RuntimeServer, thread: threading.Thread) -> None:
    deadline = time.monotonic() + 5
    while not server.started:
        if not thread.is_alive():
            raise AssertionError("Runtime Agent test server exited during startup")
        if time.monotonic() >= deadline:
            raise AssertionError("Runtime Agent test server did not start")
        time.sleep(0.01)
