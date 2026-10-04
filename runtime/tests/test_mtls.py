from __future__ import annotations

import logging
import socket
import ssl
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from cryptography import x509

from contractor_runtime.mtls import (
    CONTROL_PLANE_URI_PREFIX,
    log_runtime_certificate_expiry,
    runtime_agent_client_context,
    runtime_agent_server_context,
    verify_control_plane_peer,
)


def test_runtime_certificate_expiry_logs_and_warns_with_injected_clock(
    deployment_pki: PKI, caplog: pytest.LogCaptureFixture
) -> None:
    certificate = x509.load_pem_x509_certificate(deployment_pki.agent_certificate.read_bytes())
    expires = certificate.not_valid_after_utc
    with caplog.at_level(logging.INFO, logger="contractor_runtime.mtls"):
        logged = log_runtime_certificate_expiry(
            deployment_pki.agent_certificate,
            warning_days=30,
            now=expires - timedelta(days=31),
        )
    assert logged == expires
    assert any(
        record.levelno == logging.INFO and expires.isoformat() in record.message
        for record in caplog.records
    )
    assert not any(record.levelno == logging.WARNING for record in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="contractor_runtime.mtls"):
        log_runtime_certificate_expiry(
            deployment_pki.agent_certificate,
            warning_days=30,
            now=expires - timedelta(days=29),
        )
    assert any(record.levelno == logging.WARNING for record in caplog.records)


@dataclass(frozen=True)
class PKI:
    root: Path
    ca: Path
    control_plane_certificate: Path
    control_plane_key: Path
    agent_certificate: Path
    agent_key: Path


@pytest.fixture(scope="module")
def deployment_pki(tmp_path_factory: pytest.TempPathFactory) -> PKI:
    return generate_pki(tmp_path_factory.mktemp("deployment-pki"))


@pytest.fixture(scope="module")
def foreign_pki(tmp_path_factory: pytest.TempPathFactory) -> PKI:
    return generate_pki(tmp_path_factory.mktemp("foreign-pki"))


def generate_pki(root: Path) -> PKI:
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
    return PKI(
        root=root,
        ca=root / "ca.crt",
        control_plane_certificate=root / "control-plane.crt",
        control_plane_key=root / "control-plane.key",
        agent_certificate=root / "agents" / "agent-1.crt",
        agent_key=root / "agents" / "agent-1.key",
    )


def test_runtime_agent_outgoing_chain_hostname_and_role_verification(
    deployment_pki: PKI,
) -> None:
    server_context = private_server_context(
        deployment_pki,
        deployment_pki.control_plane_certificate,
        deployment_pki.control_plane_key,
    )
    client_context = runtime_agent_client_context(
        ca_file=deployment_pki.ca,
        certificate_file=deployment_pki.agent_certificate,
        private_key_file=deployment_pki.agent_key,
    )

    def connect(address: tuple[str, int]) -> bytes:
        raw = socket.create_connection(address, timeout=2)
        with client_context.wrap_socket(raw, server_hostname="localhost") as connection:
            verify_control_plane_peer(connection)
            return connection.recv(2)

    assert exchange_once(server_context, connect) == b"ok"
    assert client_context.check_hostname is True
    assert client_context.verify_mode == ssl.CERT_REQUIRED
    assert client_context.minimum_version == ssl.TLSVersion.TLSv1_3


def test_runtime_agent_incoming_rejects_non_control_plane_role(
    deployment_pki: PKI,
) -> None:
    server_context = runtime_agent_server_context(
        ca_file=deployment_pki.ca,
        certificate_file=deployment_pki.agent_certificate,
        private_key_file=deployment_pki.agent_key,
    )
    valid_client = private_client_context(
        deployment_pki,
        deployment_pki.control_plane_certificate,
        deployment_pki.control_plane_key,
    )

    def connect(address: tuple[str, int]) -> bytes:
        raw = socket.create_connection(address, timeout=2)
        with valid_client.wrap_socket(raw, server_hostname="localhost") as connection:
            return connection.recv(2)

    assert exchange_once(server_context, connect, verify_server_peer=True) == b"ok"

    agent_client = private_client_context(
        deployment_pki,
        deployment_pki.agent_certificate,
        deployment_pki.agent_key,
    )

    def connect_as_agent(address: tuple[str, int]) -> bytes:
        raw = socket.create_connection(address, timeout=2)
        with agent_client.wrap_socket(raw, server_hostname="localhost") as connection:
            return connection.recv(2)

    with pytest.raises((ssl.SSLError, ConnectionError)):
        exchange_once(server_context, connect_as_agent, verify_server_peer=True)


def test_runtime_agent_rejects_ca_valid_server_without_control_plane_uri(
    deployment_pki: PKI,
) -> None:
    server_context = private_server_context(
        deployment_pki,
        deployment_pki.agent_certificate,
        deployment_pki.agent_key,
    )
    client_context = runtime_agent_client_context(
        ca_file=deployment_pki.ca,
        certificate_file=deployment_pki.agent_certificate,
        private_key_file=deployment_pki.agent_key,
    )

    def connect(address: tuple[str, int]) -> bytes:
        raw = socket.create_connection(address, timeout=2)
        with (
            pytest.raises(ssl.SSLCertVerificationError),
            client_context.wrap_socket(raw, server_hostname="localhost") as connection,
        ):
            verify_control_plane_peer(connection)
        return b"rejected"

    assert exchange_once(server_context, connect, allow_server_error=True) == b"rejected"


def test_runtime_agent_rejects_foreign_ca_server(
    deployment_pki: PKI,
    foreign_pki: PKI,
) -> None:
    server_context = private_server_context(
        foreign_pki,
        foreign_pki.control_plane_certificate,
        foreign_pki.control_plane_key,
    )
    client_context = runtime_agent_client_context(
        ca_file=deployment_pki.ca,
        certificate_file=deployment_pki.agent_certificate,
        private_key_file=deployment_pki.agent_key,
    )

    def connect(address: tuple[str, int]) -> bytes:
        raw = socket.create_connection(address, timeout=2)
        with (
            pytest.raises(ssl.SSLCertVerificationError),
            client_context.wrap_socket(raw, server_hostname="localhost") as connection,
        ):
            verify_control_plane_peer(connection)
        return b"rejected"

    assert exchange_once(server_context, connect, allow_server_error=True) == b"rejected"


def test_control_plane_uri_requires_nonempty_suffix() -> None:
    class Peer:
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        def getpeercert(self) -> dict:
            return {"subjectAltName": (("URI", CONTROL_PLANE_URI_PREFIX + self.suffix),)}

    verify_control_plane_peer(Peer("cp-1"))  # type: ignore[arg-type]
    with pytest.raises(ssl.SSLCertVerificationError):
        verify_control_plane_peer(Peer(""))  # type: ignore[arg-type]


def private_server_context(pki: PKI, certificate: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=pki.ca)
    context.load_cert_chain(certfile=certificate, keyfile=key)
    return context


def private_client_context(pki: PKI, certificate: Path, key: Path) -> ssl.SSLContext:
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=pki.ca)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(certfile=certificate, keyfile=key)
    return context


def exchange_once(
    server_context: ssl.SSLContext,
    client: Callable[[tuple[str, int]], bytes],
    *,
    verify_server_peer: bool = False,
    allow_server_error: bool = False,
) -> bytes:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(3)
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            raw, _ = listener.accept()
            if verify_server_peer:
                connection = server_context.wrap_socket(raw, server_side=True)
                try:
                    verify_control_plane_peer(connection)
                except BaseException:
                    connection.close()
                    raise
            else:
                connection = server_context.wrap_socket(raw, server_side=True)
            with connection:
                connection.sendall(b"ok")
        except BaseException as error:  # the assertion below reports thread failures
            errors.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        address = listener.getsockname()
        result = client((address[0], address[1]))
    finally:
        listener.close()
        thread.join(timeout=3)
    assert not thread.is_alive()
    if not allow_server_error and errors:
        raise errors[0]
    return result
