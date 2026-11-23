"""Role-specific mTLS helpers for the private Runtime Agent boundary."""

from __future__ import annotations

import logging
import math
import ssl
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from os import PathLike
from pathlib import Path

from cryptography import x509

CONTROL_PLANE_URI_PREFIX = "urn:contractor:control-plane:"
logger = logging.getLogger(__name__)


def log_runtime_certificate_expiry(
    certificate_file: str | PathLike[str],
    *,
    ca_file: str | PathLike[str],
    warning_days: float,
    now: datetime | None = None,
) -> datetime:
    """Log the Runtime leaf's and deployment CA's expiry and warn within the window.

    A leaf never outlives its CA, so an approaching CA expiry breaks every
    private mTLS link and warrants its own warning that leaf renewal cannot fix.
    """

    if not math.isfinite(warning_days) or not 0 < warning_days <= 3650:
        raise ValueError("certificate expiry warning days must be between 0 and 3650")
    current = now or datetime.now(UTC)
    certificate = x509.load_pem_x509_certificate(Path(certificate_file).read_bytes())
    expires = certificate.not_valid_after_utc
    logger.info("runtime agent mTLS certificate expires at %s", expires.isoformat())
    if expires <= current + timedelta(days=warning_days):
        logger.warning(
            "runtime agent mTLS certificate expires within %.1f days: %s",
            warning_days,
            expires.isoformat(),
        )
    ca_certificate = x509.load_pem_x509_certificate(Path(ca_file).read_bytes())
    ca_expires = ca_certificate.not_valid_after_utc
    logger.info("deployment CA certificate expires at %s", ca_expires.isoformat())
    if ca_expires <= current + timedelta(days=warning_days):
        logger.warning(
            "deployment CA certificate expires within %.1f days; "
            "rotate the CA and reissue leaves: %s",
            warning_days,
            ca_expires.isoformat(),
        )
    return expires


def runtime_agent_client_context(
    *,
    ca_file: str | PathLike[str],
    certificate_file: str | PathLike[str],
    private_key_file: str | PathLike[str],
) -> ssl.SSLContext:
    """Create the Agent's normal hostname-verifying outgoing TLS context.

    Call :func:`verify_control_plane_peer` immediately after the TLS handshake
    to add the required Control Plane URI SAN check.
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
    must invoke :func:`verify_control_plane_peer` immediately after the TLS
    handshake and before reading an HTTP byte.
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
