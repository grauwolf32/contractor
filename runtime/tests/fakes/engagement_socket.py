"""Real local socket observers for controlled preflight conformance.

Only DNS and protocol replies are fixtures. Request bytes are read from actual
TCP/TLS sockets; the production policy and native HTTP engine stay live.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import ssl
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta

import psutil
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from contractor_runtime.contracts.engagement_allocation import EngagementAllocationProjection
from contractor_runtime.core.target_policy import TargetPolicyConfig
from fakes.engagement import preflight_spec
from fakes.http_transport import HTTPTransport


def private_interface():
    requested = os.environ.get("CONTRACTOR_HTTP_TEST_BIND_ADDRESS")
    ranges = tuple(
        ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
    )
    active = psutil.net_if_stats()
    for name, entries in psutil.net_if_addrs().items():
        if name not in active or not active[name].isup:
            continue
        for entry in entries:
            if entry.family == socket.AF_INET:
                value = ipaddress.ip_address(entry.address)
                if any(value in network for network in ranges) and (
                    requested is None or str(value) == requested
                ):
                    return str(value)
    if requested is not None:
        raise RuntimeError("test bind address must be an assigned active RFC1918 address")
    raise RuntimeError("controlled HTTP conformance requires a real RFC1918 interface")


class SocketTarget:
    """Capture every application byte, including incomplete request headers."""

    def __init__(self, responder=None, *, ssl_context=None):
        self.address = private_interface()
        self.responder = responder
        self.ssl_context = ssl_context
        self.connections = []
        self.requests = []
        self.tasks = set()
        self.writers = set()
        self.first_byte = asyncio.Event()
        self.request_received = asyncio.Event()
        self.all_closed = asyncio.Event()
        self.server = None

    async def __aenter__(self):
        self.server = await asyncio.start_server(
            self._connected, self.address, 0, ssl=self.ssl_context
        )
        self.port = self.server.sockets[0].getsockname()[1]
        scheme = "https" if self.ssl_context else "http"
        self.origin = f"{scheme}://target.example:{self.port}"
        return self

    def _connected(self, reader, writer):
        task = asyncio.create_task(self._observe(reader, writer))
        self.tasks.add(task)

    async def _observe(self, reader, writer):
        observed = bytearray()
        self.connections.append(observed)
        self.writers.add(writer)
        self.all_closed.clear()
        try:
            async with asyncio.timeout(30):
                pending = bytearray()
                while True:
                    while b"\r\n\r\n" not in pending:
                        data = await reader.read(8192)
                        if not data:
                            return
                        observed.extend(data)
                        pending.extend(data)
                        self.first_byte.set()
                        if len(pending) > 64 << 10:
                            raise AssertionError("fixture received an unbounded request")
                    end = pending.index(b"\r\n\r\n") + 4
                    head = bytes(pending[:end])
                    del pending[:end]
                    lengths = [
                        line.split(b":", 1)[1].strip()
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length:")
                    ]
                    length = int(lengths[0]) if lengths else 0
                    if len(pending) < length:
                        data = await reader.readexactly(length - len(pending))
                        observed.extend(data)
                        pending.extend(data)
                    message = head + bytes(pending[:length])
                    del pending[:length]
                    self.requests.append(message)
                    self.request_received.set()
                    if self.responder is None:
                        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                        await writer.drain()
                    else:
                        await self.responder(self, reader, writer, message)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionResetError, BrokenPipeError):
                await writer.wait_closed()
            self.writers.discard(writer)
            if not self.writers:
                self.all_closed.set()

    @property
    def written(self):
        return b"".join(self.connections)

    async def __aexit__(self, *_):
        self.server.close()
        await self.server.wait_closed()
        for writer in tuple(self.writers):
            writer.close()
        if self.tasks:
            async with asyncio.timeout(5):
                await asyncio.gather(*self.tasks)


def socket_spec(target, *, path="/allowed", body_limit=262144, change=None):
    spec = preflight_spec(body_limit=body_limit)
    projection = spec.engagement.model_dump(mode="json", by_alias=True, exclude_none=True)
    assignment = projection["http"]
    assignment["routes"][0].update(origin=target.origin, addressRanges=[target.address + "/32"])
    assignment["identities"][0]["origins"] = [target.origin]
    if change is not None:
        change(projection)
    spec.engagement = EngagementAllocationProjection.model_validate_json(json.dumps(projection))
    return spec


def resolver_for(target, answers=None, *, calls=None, protected=None):

    async def resolve(host, port):
        if calls is not None:
            calls.append((host, port))
        if host == "target.example":
            values = answers() if callable(answers) else answers or [target.address]
        elif protected is not None and host in protected:
            values = protected[host]
        else:
            values = ["10.255.255.254"]
        return tuple(ipaddress.ip_address(value) for value in values)

    return TargetPolicyConfig(resolver=resolve)


@asynccontextmanager
async def preflight(target, *, spec=None, config=None, limits=None, **kwargs):
    spec = spec or socket_spec(target)
    spec.engagement.http.limits = limits
    config = config or resolver_for(target)
    resource = HTTPTransport(
        spec,
        protected_urls=config.protected_urls,
        resolver=config.resolver,
        **kwargs,
    )
    resource.test_url = target.origin + getattr(spec, "test_path", "/allowed")
    try:
        yield (resource, None)
    finally:
        await resource.close()


def launch(resource, *, clock=time.monotonic, lease=None, url=None):
    resource.configure(authority_deadline=lease or (lambda: clock() + 60))

    async def execute():
        outcome, _ = await resource.request(url or resource.test_url)
        return outcome

    return asyncio.create_task(execute())


def tls_contexts(directory):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "target.example")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("target.example")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    certificate, private_key = (directory / "target.pem", directory / "target.key")
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private_key.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    private_key.chmod(384)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(certificate, private_key)
    names = []
    server.set_servername_callback(lambda _, name, __: names.append(name))
    return (server, ssl.create_default_context(cafile=certificate), names)
