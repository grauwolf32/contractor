"""Role-specific mTLS helpers for the private Runtime Agent boundary."""

from __future__ import annotations

import socket
import ssl
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

CONTROL_PLANE_URI_PREFIX = "urn:contractor:control-plane:"


def runtime_agent_client_context(
    *,
    ca_file: str | PathLike[str],
    certificate_file: str | PathLike[str],
    private_key_file: str | PathLike[str],
) -> ssl.SSLContext:
    """Create the Agent's normal hostname-verifying outgoing TLS context.

    Call :func:`wrap_control_plane_client_socket` (or call
    :func:`verify_control_plane_peer` immediately after a framework completes
    its TLS handshake) to add the required Control Plane URI SAN check.
    """

    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_cert_chain(certfile=certificate_file, keyfile=private_key_file)
    return context


def runtime_agent_server_context(
    *,
    ca_file: str | PathLike[str],
    certificate_file: str | PathLike[str],
    private_key_file: str | PathLike[str],
) -> ssl.SSLContext:
    """Create the Agent's incoming context requiring a CA-valid client cert.

    The stdlib has no portable certificate-verification callback. Acceptors
    must therefore use :func:`wrap_control_plane_server_socket` or invoke
    :func:`verify_control_plane_peer` immediately after ``accept``/handshake
    and before reading an HTTP byte.
    """

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=ca_file)
    context.load_cert_chain(certfile=certificate_file, keyfile=private_key_file)
    return context


def verify_control_plane_peer(connection: ssl.SSLSocket | ssl.SSLObject) -> None:
    """Require the reserved Control Plane URI SAN on an already verified peer."""

    certificate = connection.getpeercert()
    if not certificate:
        raise ssl.SSLCertVerificationError("peer has no verified certificate")
    subject_alt_names = certificate.get("subjectAltName", ())
    if not isinstance(subject_alt_names, Sequence):
        raise ssl.SSLCertVerificationError("peer certificate has invalid subjectAltName data")
    for entry in subject_alt_names:
        if (
            isinstance(entry, Sequence)
            and len(entry) == 2
            and entry[0] == "URI"
            and isinstance(entry[1], str)
            and entry[1].startswith(CONTROL_PLANE_URI_PREFIX)
            and len(entry[1]) > len(CONTROL_PLANE_URI_PREFIX)
        ):
            return
    raise ssl.SSLCertVerificationError("peer certificate lacks the Control Plane URI SAN")


def verify_control_plane_certificate(certificate: Mapping[str, Any]) -> None:
    """Testable certificate-dictionary variant of the post-handshake check."""

    class _Peer:
        def getpeercert(self) -> Mapping[str, Any]:
            return certificate

    verify_control_plane_peer(_Peer())  # type: ignore[arg-type]


def wrap_control_plane_client_socket(
    context: ssl.SSLContext,
    raw_socket: socket.socket,
    *,
    server_hostname: str,
) -> ssl.SSLSocket:
    """Perform chain, hostname, and role verification before returning a socket."""

    connection = context.wrap_socket(raw_socket, server_hostname=server_hostname)
    try:
        verify_control_plane_peer(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def wrap_control_plane_server_socket(
    context: ssl.SSLContext,
    raw_socket: socket.socket,
) -> ssl.SSLSocket:
    """Perform client-chain and Control Plane role verification on accept."""

    connection = context.wrap_socket(raw_socket, server_side=True)
    try:
        verify_control_plane_peer(connection)
    except BaseException:
        connection.close()
        raise
    return connection
