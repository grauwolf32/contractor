"""Reproduce stock grpcio's peer-role gap using disposable loopback certificates.

This is evidence for a transport choice, not a passing production security gate.
No application, existing PKI, external service or database is used.
"""

from __future__ import annotations

import json
import socket
import ssl
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

BASELINE = "9fd3ac96d32c696fa39988df85e32f6bb6d1d83b"
CONTROL_ROLE = "urn:contractor:control-plane:probe"


def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def name(value):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, value)])


def pem_key(value):
    return value.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def certificate(public_key, subject, issuer, signer, *, ca=False, role=None, host="localhost"):
    now = datetime.now(UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()), critical=False
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=not ca,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=ca,
                crl_sign=ca,
                encipher_only=None,
                decipher_only=None,
            ),
            critical=True,
        )
    )
    if not ca:
        sans = [x509.DNSName(host)]
        if role:
            sans.append(x509.UniformResourceIdentifier(role))
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
        builder = builder.add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
    return builder.sign(signer, hashes.SHA256()).public_bytes(serialization.Encoding.PEM)


def inspect_standard_tls(port, root, cert, private, directory):
    files = [directory / file for file in ("root.pem", "client.pem", "client.key")]
    for file, data in zip(files, (root, cert, private), strict=True):
        file.write_bytes(data)
    context = ssl.create_default_context(cafile=str(files[0]))
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.load_cert_chain(str(files[1]), str(files[2]))
    context.set_alpn_protocols(["h2"])
    with (
        socket.create_connection(("localhost", port), timeout=3) as connection,
        context.wrap_socket(connection, server_hostname="localhost") as secured,
    ):
        return any(
            kind == "URI"
            and value.startswith("urn:contractor:control-plane:")
            and len(value) > len("urn:contractor:control-plane:")
            for kind, value in secured.getpeercert().get("subjectAltName", ())
        )


def main():
    ca_key = key()
    ca_name = name("disposable grpc probe CA")
    root = certificate(ca_key.public_key(), ca_name, ca_name, ca_key, ca=True)
    client_key = key()
    client_cert = certificate(client_key.public_key(), name("runtime"), ca_name, ca_key)
    observations = []
    with tempfile.TemporaryDirectory(prefix="contractor-grpc-probe-") as temporary:
        directory = Path(temporary)
        for case, role, host, foreign in [
            ("control_plane", CONTROL_ROLE, "localhost", False),
            ("runtime_as_server", None, "localhost", False),
            ("wrong_hostname", CONTROL_ROLE, "wrong.invalid", False),
            ("foreign_ca", CONTROL_ROLE, "localhost", True),
        ]:
            server_key = key()
            issuer_key = key() if foreign else ca_key
            server_cert = certificate(
                server_key.public_key(),
                name(case),
                ca_name,
                issuer_key,
                role=role,
                host=host,
            )
            received = []

            def receive(request, _context, captured=received):
                captured.append(request)
                return b"probe-reply"

            executor = ThreadPoolExecutor(max_workers=1)
            server = grpc.server(executor)
            server.add_generic_rpc_handlers(
                (
                    grpc.method_handlers_generic_handler(
                        "probe.Peer",
                        {"Check": grpc.unary_unary_rpc_method_handler(receive)},
                    ),
                )
            )
            port = server.add_secure_port(
                "127.0.0.1:0",
                grpc.ssl_server_credentials(
                    [(pem_key(server_key), server_cert)],
                    root_certificates=root,
                    require_client_auth=True,
                ),
            )
            if port == 0:
                raise RuntimeError("probe listener did not bind")
            server.start()
            channel = grpc.secure_channel(
                f"localhost:{port}",
                grpc.ssl_channel_credentials(
                    root,
                    pem_key(client_key),
                    client_cert,
                ),
                options=(("grpc.enable_retries", 0),),
            )
            result = "accepted"
            try:
                try:
                    channel.unary_unary("/probe.Peer/Check")(b"probe-only", timeout=3)
                except grpc.RpcError as error:
                    result = error.code().name
                try:
                    role_verified = inspect_standard_tls(
                        port,
                        root,
                        client_cert,
                        pem_key(client_key),
                        directory,
                    )
                except (OSError, ssl.SSLError):
                    role_verified = False
                observations.append(
                    {
                        "case": case,
                        "stock_grpc_result": result,
                        "requests_received": len(received),
                        "existing_role_check_accepts": role_verified,
                    }
                )
            finally:
                channel.close()
                server.stop(0).wait(timeout=3)
                executor.shutdown(wait=True)
    expected = [
        ("control_plane", "accepted", 1, True),
        ("runtime_as_server", "accepted", 1, False),
        ("wrong_hostname", "UNAVAILABLE", 0, False),
        ("foreign_ca", "UNAVAILABLE", 0, False),
    ]
    actual = [tuple(row.values()) for row in observations]
    if actual != expected:
        raise RuntimeError(f"probe observations changed: {observations}")
    report = {
        "baseline": BASELINE,
        "grpcio": grpc.__version__,
        "experiment": "stock Python gRPC validates chain/name, not Contractor URI role",
        "cases": observations,
        "production_acceptance": (
            "blocked: select a role-preserving transport or explicit PKI change"
        ),
        "scope": "loopback Python grpcio only; no Go/A2A/production interoperability claimed",
        "cleanup": "all owned listeners/channels/executors closed; temporary PKI deleted",
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
