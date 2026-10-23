"""Compatibility with the pinned upstream server, not a protocol-shaped fake."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import smtplib
import socket
import struct
import subprocess
import time
from contextlib import ExitStack

import httpx
import pytest
from test_interactsh import Artifacts

from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.adapters.interactsh import InteractshClient
from contractor_runtime.contracts import InteractshSettings
from contractor_runtime.toolsets.interactsh.tools import _Session


def test_real_interactsh_dns_http_smtp_and_owned_cleanup(tmp_path):
    binary = os.environ.get("CONTRACTOR_TEST_INTERACTSH_SERVER")
    if not binary:
        pytest.skip("run make test-interactsh-protocol for the pinned upstream server")
    # Hold all three chosen ports until process startup, preventing duplicates.
    with ExitStack() as stack:
        ports = []
        for _ in range(3):
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            ports.append(listener.getsockname()[1])
    with (tmp_path / "server.log").open("w+") as log:
        process = subprocess.Popen([binary, *map(str, ports)], stdout=log, stderr=log)
        try:
            # A finite readiness budget, not a wall-clock performance assertion.
            for _ in range(100):
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail(f"upstream server exited: {log.read()}")
                try:
                    with socket.create_connection(("127.0.0.1", ports[0]), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                pytest.fail("upstream HTTP listener did not become ready")
            asyncio.run(_scenario(ports))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


async def _scenario(ports):
    now = [0]
    settings = InteractshSettings(
        adapter="interactsh-http@1",
        endpoint=f"http://localhost:{ports[0]}",
        token="local-protocol-test-token",
        requestTimeoutSeconds=10,
    )
    handle = InteractshClient(
        settings, timeout_seconds=10, metrics=RuntimeAdapterMetricsState(), monotonic=lambda: now[0]
    )
    artifacts = Artifacts()
    session = _Session(handle, artifacts, "worker", ())
    try:
        registration = await session.register(
            "local target HTTP request, DNS lookup and SMTP action"
        )
        assert registration["state"] == "registered"
        payload = registration["payload"]
        async with httpx.AsyncClient(trust_env=False) as target:
            response = await target.get(
                f"http://127.0.0.1:{ports[0]}/callback-proof", headers={"Host": payload}
            )
            assert response.status_code == 200
        await asyncio.to_thread(_dns_callback, payload, ports[1])
        await asyncio.to_thread(_smtp_callback, payload, ports[2])
        result = await session.poll(registration["handle"])
        assert {entry["protocol"] for entry in result["events"]} == {"http", "dns", "smtp"}
        assert not result["truncated"] and not result["coverage_gap"]
        evidence = json.loads(next(iter(artifacts.rows.values())).data)
        events = [json.loads(base64.b64decode(raw)) for raw in evidence["events_base64"]]
        assert all(row["unique-id"] == payload.split(".")[0] for row in events)
        assert any("callback-proof" in row.get("raw-request", "") for row in events)
        assert evidence["trigger"]["attribution"] == "unverified"
        now[0] += 2
        empty = await session.poll(registration["handle"])
        assert empty["state"] == "no_interaction"
        item = handle.registration(registration["handle"])
        correlation, secret = item.correlation, item.secret
        await session.close()
        await handle.close()
        assert item.key is None and not item.secret
        async with httpx.AsyncClient(trust_env=False) as probe:
            after = await probe.get(
                f"http://localhost:{ports[0]}/poll",
                headers={"Authorization": "local-protocol-test-token"},
                params={"id": correlation, "secret": secret},
            )
            assert after.status_code == 400  # actual upstream session was removed
    finally:
        await session.close()
        await handle.close()


def _dns_callback(payload, port):
    labels = b"".join(bytes([len(part)]) + part.encode() for part in payload.split(".")) + b"\0"
    query = struct.pack("!HHHHHH", 17, 0x100, 1, 0, 0, 0) + labels + struct.pack("!HH", 1, 1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as target:
        target.settimeout(3)
        target.sendto(query, ("127.0.0.1", port))
        response = target.recv(4096)
        assert response[:2] == query[:2]


def _smtp_callback(payload, port):
    with smtplib.SMTP("127.0.0.1", port, timeout=3) as target:
        target.sendmail(
            "target@example.test",
            [f"evidence@{payload}"],
            "Subject: callback proof\r\n\r\nlocal target message\r\n",
        )
