from __future__ import annotations

import asyncio
import json
import logging
import ssl
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from contractor_runtime.control_client import ControlClient, ControlClientError, MTLSJSONTransport
from contractor_runtime.mtls import runtime_agent_client_context
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState


def test_registration_commits_idle_and_uses_server_timing() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-register")
        transport = FakeTransport(
            [
                {
                    "apiVersion": "contractor/v1alpha1",
                    "heartbeatIntervalSeconds": 12,
                    "confirmedLeaseSeconds": 72,
                }
            ]
        )
        client = ControlClient(make_settings(), state, transport)
        response = await client.register()
        assert response.heartbeat_interval_seconds == 12
        assert client.timing.heartbeat_interval_seconds == 12
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        assert transport.requests[0][0] == "/private/v1/agents/register"
        assert transport.requests[0][1]["instanceId"] == "runtime-register"

    asyncio.run(scenario())


def test_response_loss_never_advances_echoed_ack() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-heartbeat")
        transport = FakeTransport(
            [
                registration_response(),
                heartbeat_response(1),
                TimeoutError("response containing private-token was lost"),
                heartbeat_response(3),
            ]
        )
        client = ControlClient(make_settings(), state, transport)
        await client.register()
        await client.heartbeat_once()
        assert client.sequence == 1
        assert client.echoed_ack == 1
        with pytest.raises(TimeoutError):
            await client.heartbeat_once()
        assert client.sequence == 2
        assert client.echoed_ack == 1
        await client.heartbeat_once()
        assert client.sequence == 3
        assert client.echoed_ack == 3
        heartbeat_requests = [
            payload for path, payload in transport.requests if "heartbeat" in path
        ]
        assert [request["heartbeatSeq"] for request in heartbeat_requests] == [1, 2, 3]
        assert [request["echoedAckSeq"] for request in heartbeat_requests] == [0, 1, 1]

    asyncio.run(scenario())


def test_wrong_ack_is_rejected_without_changing_echo() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-wrong-ack")
        transport = FakeTransport([registration_response(), heartbeat_response(9)])
        client = ControlClient(make_settings(), state, transport)
        await client.register()
        with pytest.raises(ControlClientError):
            await client.heartbeat_once()
        assert client.sequence == 1
        assert client.echoed_ack == 0

    asyncio.run(scenario())


def test_retry_backoff_is_jittered_but_bounded() -> None:
    low = ControlClient(make_settings(), RuntimeState(), FakeTransport([]), jitter=lambda: 0.0)
    high = ControlClient(make_settings(), RuntimeState(), FakeTransport([]), jitter=lambda: 1.0)
    assert low._backoff(1) == 0.375
    assert high._backoff(100) == 5.0


def test_retry_logging_does_not_include_exception_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-redaction")
        stop = asyncio.Event()

        async def stop_after_retry(_: float) -> None:
            stop.set()

        transport = FakeTransport([RuntimeError("recognizable-private-token")])
        client = ControlClient(make_settings(), state, transport, sleep=stop_after_retry)
        assert await client.register_until_stopped(stop) is False

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())
    assert "recognizable-private-token" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_inflight_heartbeat_cancels_cleanly() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-cancel")
        waiting = WaitingTransport()
        client = ControlClient(make_settings(), state, waiting)
        await state.mark_registered()
        task = asyncio.create_task(client.run_heartbeats(asyncio.Event()))
        await waiting.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.done()

    asyncio.run(scenario())


def test_real_mtls_control_transport_registers_and_heartbeats(
    tmp_path: Path,
) -> None:
    pki = generate_pki(tmp_path / "pki")

    async def scenario() -> None:
        received: list[dict[str, Any]] = []
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.minimum_version = ssl.TLSVersion.TLSv1_3
        server_context.verify_mode = ssl.CERT_REQUIRED
        server_context.load_verify_locations(cafile=pki["ca"])
        server_context.load_cert_chain(
            certfile=pki["control_plane_certificate"], keyfile=pki["control_plane_key"]
        )

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                request_line = await reader.readline()
                headers: dict[str, str] = {}
                while True:
                    line = await reader.readline()
                    if line == b"\r\n":
                        break
                    name, _, value = line.partition(b":")
                    headers[name.decode().lower()] = value.decode().strip()
                payload = json.loads(await reader.readexactly(int(headers["content-length"])))
                received.append(payload)
                if request_line.startswith(b"POST /private/v1/agents/register "):
                    response = registration_response()
                else:
                    response = heartbeat_response(payload["heartbeatSeq"])
                encoded = json.dumps(response, separators=(",", ":")).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(encoded)).encode()
                    + b"\r\nConnection: close\r\n\r\n"
                    + encoded
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=server_context)
        port = server.sockets[0].getsockname()[1]
        settings = Settings(
            control_plane_url=f"https://localhost:{port}",
            advertised_control_url="https://localhost:9443",
            advertised_a2a_url="https://localhost:9444",
            ca_file=pki["ca"],
            certificate_file=pki["agent_certificate"],
            private_key_file=pki["agent_key"],
        )
        client_context = runtime_agent_client_context(
            ca_file=pki["ca"],
            certificate_file=pki["agent_certificate"],
            private_key_file=pki["agent_key"],
        )
        transport = MTLSJSONTransport(settings.control_plane_url, client_context, 2)
        state = RuntimeState(instance_id="runtime-real-mtls")
        client = ControlClient(settings, state, transport)
        try:
            await client.register()
            await client.heartbeat_once()
            await client.heartbeat_once()
        finally:
            server.close()
            await server.wait_closed()
        assert [item.get("heartbeatSeq") for item in received[1:]] == [1, 2]
        assert [item.get("echoedAckSeq") for item in received[1:]] == [0, 1]

    asyncio.run(scenario())


class FakeTransport:
    def __init__(self, responses: list[Mapping[str, Any] | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, Mapping[str, Any]]] = []

    async def post_json(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append((path, payload))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class WaitingTransport:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def post_json(self, _: str, __: Mapping[str, Any]) -> Mapping[str, Any]:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def registration_response() -> Mapping[str, Any]:
    return {
        "apiVersion": "contractor/v1alpha1",
        "heartbeatIntervalSeconds": 10,
        "confirmedLeaseSeconds": 60,
    }


def heartbeat_response(sequence: int) -> Mapping[str, Any]:
    return {"apiVersion": "contractor/v1alpha1", "ackSeq": sequence, "action": "continue"}


def make_settings() -> Settings:
    placeholder = Path("/tmp/contractor-test-placeholder")
    return Settings(
        control_plane_url="https://localhost:8443",
        advertised_control_url="https://localhost:9443",
        advertised_a2a_url="https://localhost:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
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
