# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["mitmproxy==12.3.1", "httpx[http2]==0.28.1"]
# ///
"""Assess an imported proxy engine against bounded, private local fixtures.

This is an opt-in executable experiment, not Contractor transport implementation.
No target beyond the ephemeral loopback fixtures is contacted by protocol probes.
Dependency resolution may consult package indexes, using a disposable project copy.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import inspect
import ipaddress
import json
import platform
import shutil
import ssl
import sys
import tempfile
import time
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import h2.config
import h2.connection
import h2.events
import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from mitmproxy import addons, ctx, http, master, options
from mitmproxy.addons import next_layer, proxyserver, tlsconfig
from wsproto import WSConnection
from wsproto.connection import ConnectionType
from wsproto.events import AcceptConnection, CloseConnection, Request, TextMessage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/src"))
from contractor_runtime.core.async_tasks import join_owned_task  # noqa: E402

VERSION = "12.3.1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def certificates(directory):
    """Private fixture CA and a localhost server/client certificate."""
    now = datetime.now(UTC)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Assessment fixture CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(ca.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    ca_path, cert_path, key_path = (directory / name for name in ("ca.pem", "leaf.pem", "key.pem"))
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    client_path = directory / "client.pem"
    client_path.write_bytes(key_path.read_bytes() + cert_path.read_bytes())
    client_path.chmod(0o600)
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_path, key_path)
    server.set_alpn_protocols(["h2", "http/1.1"])
    mutual = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    mutual.load_cert_chain(cert_path, key_path)
    mutual.load_verify_locations(cafile=ca_path)
    mutual.verify_mode = ssl.CERT_REQUIRED
    mutual.set_alpn_protocols(["http/1.1"])
    return ca_path, client_path, server, mutual


class Origin:
    """Bounded target fixture with independently observed protocol/request facts."""

    def __init__(self):
        self.servers = []
        self.tasks = set()
        self.trace = []
        self.connections = 0
        self.stream_release = asyncio.Event()
        self.stall_started = asyncio.Event()
        self.stall_closed = asyncio.Event()

    async def listen(self, tls=None):
        server = await asyncio.start_server(self.accept, "127.0.0.1", 0, ssl=tls)
        self.servers.append(server)
        return server.sockets[0].getsockname()[1]

    def accept(self, reader, writer):
        task = asyncio.create_task(self.serve(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def serve(self, reader, writer):
        self.connections += 1
        tls = writer.get_extra_info("ssl_object")
        try:
            async with asyncio.timeout(15):
                if tls and tls.selected_alpn_protocol() == "h2":
                    await self.serve_h2(reader, writer)
                else:
                    await self.serve_h1(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError, TimeoutError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, ssl.SSLError):
                await writer.wait_closed()

    async def serve_h1(self, reader, writer):
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            method, path, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
            headers = dict(
                line.split(b":", 1) for line in head.lower().split(b"\r\n")[1:] if b":" in line
            )
            body = await reader.readexactly(int(headers.get(b"content-length", b"0")))
            require(len(self.trace) < 64, "fixture request bound exceeded")
            self.trace.append({"method": method, "path": path, "protocol": "HTTP/1.1"})
            if headers.get(b"upgrade", b"").strip() == b"websocket":
                await self.websocket(reader, writer, head)
                return
            if path in {"/events", "/stall"}:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
                    b"b\r\ndata: one\n\n\r\n"
                )
                await writer.drain()
                if path == "/stall":
                    self.stall_started.set()
                    await reader.read(1)
                    self.stall_closed.set()
                else:
                    await self.stream_release.wait()
                    writer.write(b"0\r\n\r\n")
                    await writer.drain()
                return
            response = json.dumps(
                {"method": method, "path": path, "body_bytes": len(body)}
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(response)).encode()
                + b"\r\n\r\n"
                + response
            )
            await writer.drain()
            if headers.get(b"connection", b"").strip() == b"close":
                return

    async def serve_h2(self, reader, writer):
        connection = h2.connection.H2Connection(
            h2.config.H2Configuration(client_side=False, header_encoding="utf-8")
        )
        connection.initiate_connection()
        writer.write(connection.data_to_send())
        streams = {}
        while data := await reader.read(65536):
            for event in connection.receive_data(data):
                if isinstance(event, h2.events.RequestReceived):
                    streams[event.stream_id] = dict(event.headers)
                elif isinstance(event, h2.events.DataReceived):
                    connection.acknowledge_received_data(
                        event.flow_controlled_length, event.stream_id
                    )
                elif isinstance(event, h2.events.StreamEnded):
                    headers = streams.pop(event.stream_id)
                    self.trace.append(
                        {
                            "method": headers[":method"],
                            "path": headers[":path"],
                            "protocol": "HTTP/2",
                        }
                    )
                    connection.send_headers(
                        event.stream_id, [(":status", "200"), ("content-type", "text/plain")]
                    )
                    connection.send_data(event.stream_id, b"h2 fixture", end_stream=True)
            writer.write(connection.data_to_send())
            await writer.drain()

    @staticmethod
    async def websocket(reader, writer, head):
        connection = WSConnection(ConnectionType.SERVER)
        connection.receive_data(head)
        require(
            any(isinstance(event, Request) for event in connection.events()), "missing WS request"
        )
        writer.write(connection.send(AcceptConnection()))
        await writer.drain()
        while data := await reader.read(4096):
            connection.receive_data(data)
            for event in connection.events():
                if isinstance(event, TextMessage):
                    writer.write(connection.send(TextMessage(data=event.data)))
                elif isinstance(event, CloseConnection):
                    writer.write(connection.send(event.response()))
                    await writer.drain()
                    return
            await writer.drain()

    async def close(self):
        for server in self.servers:
            server.close()
            await server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*tuple(self.tasks), return_exceptions=True)


class ProbeHooks:
    """Demonstrate narrow callbacks, not complete engagement enforcement."""

    def __init__(self):
        self.ready = asyncio.Event()
        self.observations = []
        self.connections = 0
        self.websocket_messages = 0
        self.prefer_http2 = False
        self.streamed_bodies_retained = []

    def running(self):
        self.ready.set()

    def requestheaders(self, flow):
        if urlsplit(flow.request.url).path == "/denied":
            flow.metadata["assessment_local_reply"] = True
            flow.response = http.Response.make(403, b"fixture policy denied")
        elif urlsplit(flow.request.url).path == "/hook-error":
            raise RuntimeError("deliberate fixture policy callback failure")

    def responseheaders(self, flow):
        self.observations.append(
            {
                "method": flow.request.method,
                "path": flow.request.path,
                "status": flow.response.status_code,
                "local_reply": bool(flow.metadata.get("assessment_local_reply")),
            }
        )
        if flow.request.path in {"/events", "/stall"}:
            flow.response.stream = True

    def server_connect(self, data):
        self.connections += 1

    def websocket_message(self, flow):
        self.websocket_messages += 1

    def tls_clienthello(self, data):
        if self.prefer_http2 and b"h2" in data.client_hello.alpn_protocols:
            data.context.client.alpn = b"h2"

    def response(self, flow):
        if flow.request.path == "/events":
            self.streamed_bodies_retained.append(flow.response.raw_content is not None)


class EmbeddedProxy:
    """Small assembly plus explicitly assessed embedding cleanup."""

    def __init__(self, directory, origin_ca):
        self.hooks = ProbeHooks()
        self.options = options.Options(
            listen_host="127.0.0.1",
            listen_port=0,
            confdir=str(directory),
            upstream_cert=False,
            http2=True,
            rawtcp=False,
            ssl_verify_upstream_trusted_ca=str(origin_ca),
        )
        self.master = master.Master(self.options, with_termlog=False)
        self.server = proxyserver.Proxyserver()
        self.master.addons.add(
            self.server, next_layer.NextLayer(), tlsconfig.TlsConfig(), self.hooks
        )
        self.options.update(connection_strategy="lazy", body_size_limit="64k")
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.master.run())
        await asyncio.wait_for(self.hooks.ready.wait(), 10)
        require(not self.task.done(), "proxy exited during startup")
        self.address = self.server.listen_addrs()[0]
        self.url = f"http://{self.address[0]}:{self.address[1]}"
        self.client_tls = ssl.create_default_context(
            cafile=str(Path(self.options.confdir) / "mitmproxy-ca-cert.pem")
        )

    async def stop_master(self):
        self.master.shutdown()
        await asyncio.wait_for(self.task, 5)

    async def cleanup(self):
        """Reuse Contractor's owned join; only engine shutdown is specific here."""
        task = asyncio.create_task(self._cleanup_engine())
        interrupted = await join_owned_task(task)
        task.result()
        if interrupted:
            raise asyncio.CancelledError

    async def _cleanup_engine(self):
        if self.task and not self.task.done():
            await self.stop_master()
        await self.server.servers.update([])
        tasks = {
            io.handler
            for handler in tuple(self.server.connections.values())
            for io in handler.transports.values()
            if io.handler is not None and not io.handler.done()
        }
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def can_connect(address):
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(*address), 1)
    except OSError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


async def ws_probe(proxy, port):
    reader, writer = await asyncio.open_connection(*proxy.address)
    connection = WSConnection(ConnectionType.CLIENT)
    writer.write(
        connection.send(Request(host=f"127.0.0.1:{port}", target=f"http://127.0.0.1:{port}/ws"))
    )
    await writer.drain()
    while True:
        connection.receive_data(await reader.read(4096))
        if any(isinstance(event, AcceptConnection) for event in connection.events()):
            break
    writer.write(connection.send(TextMessage(data="fixture ping")))
    await writer.drain()
    echoed = False
    while not echoed:
        connection.receive_data(await reader.read(4096))
        echoed = any(
            isinstance(event, TextMessage) and event.data == "fixture ping"
            for event in connection.events()
        )
    writer.write(connection.send(CloseConnection(code=1000)))
    await writer.drain()
    writer.close()
    await writer.wait_closed()
    return {"echoed": echoed, "message_events": proxy.hooks.websocket_messages}


async def dependency_probe(directory):
    """Resolve the actual Runtime project with the candidate in a disposable copy."""
    target = directory / "resolution"
    target.mkdir()
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copyfile(ROOT / "runtime" / name, target / name)
    project = target / "pyproject.toml"
    project.write_text(
        project.read_text().replace(
            "dependencies = [", f'dependencies = [\n  "mitmproxy=={VERSION}",', 1
        )
    )
    before = tomllib.loads((ROOT / "runtime/uv.lock").read_text())
    exit_code, stdout, stderr = await run_command(
        "uv",
        "lock",
        "--project",
        str(target),
        "--python",
        "3.13",
    )
    result = {"resolved": exit_code == 0, "exit_code": exit_code}
    if exit_code:
        # Package solver diagnostics only; never print process environment.
        result["diagnostic"] = (
            (stdout + stderr).decode()[-6000:].replace(str(directory), "[temporary]")
        )
        return result
    after = tomllib.loads((target / "uv.lock").read_text())
    original = {p["name"]: p.get("version") for p in before["package"]}
    resolved = {p["name"]: p.get("version") for p in after["package"]}
    result["added"] = {name: version for name, version in resolved.items() if name not in original}
    result["changed"] = {
        name: {"before": original[name], "after": version}
        for name, version in resolved.items()
        if name in original and version != original[name]
    }
    return result


async def run_command(*arguments):
    process = await asyncio.create_subprocess_exec(
        *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 120)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.communicate()
        raise
    return process.returncode, stdout, stderr


async def runtime_import_probe():
    code = (
        "import json, importlib.metadata as m; "
        "import contractor_runtime.allocation.preparation; "
        "import contractor_runtime.adapters.http_transport; "
        "import contractor_runtime.adapters.http_proxy_bridge; "
        "from mitmproxy import master; "
        "print(json.dumps({name:m.version(name) for name in "
        "('mitmproxy','google-adk','cryptography','mitmproxy-rs')}))"
    )
    exit_code, stdout, stderr = await run_command(
        "uv",
        "run",
        "--project",
        str(ROOT / "runtime"),
        "--locked",
        "--with",
        f"mitmproxy=={VERSION}",
        "python",
        "-c",
        code,
    )
    require(exit_code == 0, "Runtime/mitmproxy overlay import failed: " + stderr.decode()[-1000:])
    return {"imported_together": True, "versions": json.loads(stdout)}


async def assess(directory):
    ca, client_certificate, tls, mutual_tls = certificates(directory)
    origin = Origin()
    old_context = {name: getattr(ctx, name, None) for name in ("master", "options", "log")}
    proxy = EmbeddedProxy(directory / "proxy-ca", ca)
    results = {}
    try:
        untrusted = directory / "untrusted"
        untrusted.mkdir()
        _, _, untrusted_tls, _ = certificates(untrusted)
        plain_port = await origin.listen()
        tls_port = await origin.listen(tls)
        mutual_port = await origin.listen(mutual_tls)
        untrusted_port = await origin.listen(untrusted_tls)
        await proxy.start()
        plain = f"http://127.0.0.1:{plain_port}"
        secure = f"https://localhost:{tls_port}"
        async with httpx.AsyncClient(
            proxy=proxy.url, verify=proxy.client_tls, trust_env=False
        ) as client:
            connections = origin.connections
            first = await client.get(plain + "/one")
            second = await client.post(plain + "/two", content=b"fixture body")
            require(first.status_code == second.status_code == 200, "HTTP/1 request failed")
            require(second.json()["body_bytes"] == 12, "POST body changed")
            results["http1"] = {
                "requests": 2,
                "origin_connections": origin.connections - connections,
            }
            connects = proxy.hooks.connections
            denied = await client.get(plain + "/denied")
            require(denied.status_code == 403, "request admission hook failed")
            require(
                not any(row["path"] == "/denied" for row in origin.trace),
                "denied target received request",
            )
            results["path_admission"] = {
                "status": 403,
                "upstream_connects": proxy.hooks.connections - connects,
            }
            connects = proxy.hooks.connections
            denied = await client.get(secure + "/denied")
            require(denied.status_code == 403, "HTTPS request admission failed")
            require(proxy.hooks.connections == connects, "HTTPS denial preconnected to origin")
            results["path_admission"]["https_upstream_connects"] = (
                proxy.hooks.connections - connects
            )
            secured = await client.get(secure + "/tls")
            require(secured.status_code == 200, "verified HTTPS failed")
            results["https"] = {
                "client_verifies_generated_certificate": True,
                "upstream_verification_enabled": not proxy.options.ssl_insecure,
            }
            untrusted_response = await client.get(f"https://localhost:{untrusted_port}/untrusted")
            require(
                untrusted_response.status_code == 502, "untrusted upstream certificate accepted"
            )
            results["https"]["untrusted_upstream_status"] = untrusted_response.status_code
            hook_error = await client.get(plain + "/hook-error")
            results["policy_hook_exception"] = {
                "status": hook_error.status_code,
                "target_received_request": any(
                    row["path"] == "/hook-error" for row in origin.trace
                ),
            }
            async with client.stream("GET", plain + "/events") as response:
                chunks = response.aiter_raw()
                chunk = await asyncio.wait_for(anext(chunks), 3)
                require(chunk == b"data: one\n\n", "streamed response changed")
                require(
                    not origin.stream_release.is_set(), "fixture released before streamed delivery"
                )
                results["streaming"] = {"first_chunk_before_origin_completion": True}
                origin.stream_release.set()
                async for _ in chunks:
                    pass
            results["streaming"]["stored_body"] = any(proxy.hooks.streamed_bodies_retained)

        async with httpx.AsyncClient(
            proxy=proxy.url, verify=proxy.client_tls, http2=True, trust_env=False
        ) as client:
            response = await client.get(secure + "/alpn-default")
            results["http2_lazy_default"] = {"client_protocol": response.http_version}
        proxy.hooks.prefer_http2 = True
        before_h2 = origin.connections
        async with httpx.AsyncClient(
            proxy=proxy.url, verify=proxy.client_tls, http2=True, trust_env=False
        ) as client:
            response = await client.get(secure + "/h2")
            responses = await asyncio.gather(*(client.get(secure + f"/h2-{i}") for i in range(4)))
            require(
                all(r.http_version == "HTTP/2" for r in [response, *responses]),
                "client HTTP/2 missing",
            )
            require(
                all(
                    row["protocol"] == "HTTP/2"
                    for row in origin.trace
                    if row["path"].startswith("/h2")
                ),
                "origin HTTP/2 missing",
            )
            results["http2"] = {
                "client_and_origin_http2": True,
                "requests": 5,
                "origin_connections": origin.connections - before_h2,
            }
        proxy.hooks.prefer_http2 = False

        results["websocket"] = await ws_probe(proxy, plain_port)
        async with httpx.AsyncClient(
            proxy=proxy.url, verify=proxy.client_tls, trust_env=False
        ) as client:
            no_certificate = await client.get(f"https://localhost:{mutual_port}/mutual")
            require(no_certificate.status_code == 502, "mTLS origin accepted missing client cert")
        proxy.options.update(client_certs=str(client_certificate))
        async with httpx.AsyncClient(
            proxy=proxy.url, verify=proxy.client_tls, trust_env=False
        ) as client:
            response = await client.get(f"https://localhost:{mutual_port}/mutual")
            require(response.status_code == 200, "upstream mutual TLS failed")
            results["mtls"] = {
                "upstream_required_client_certificate": True,
                "missing_client_certificate_status": no_certificate.status_code,
                "with_client_certificate_status": response.status_code,
            }

        # Leave a real stream open while stopping Master, then assess what remains.
        client = httpx.AsyncClient(proxy=proxy.url, trust_env=False)
        stream = client.stream("GET", plain + "/stall")
        response = await stream.__aenter__()
        await asyncio.wait_for(origin.stall_started.wait(), 3)
        chunks = response.aiter_raw()
        await asyncio.wait_for(anext(chunks), 3)
        await proxy.stop_master()
        results["master_shutdown"] = {
            "run_returned": True,
            "listener_still_accepts": await can_connect(proxy.address),
            "active_connections": len(proxy.server.connections),
            "origin_stream_closed": origin.stall_closed.is_set(),
            "global_context_retains_master": ctx.master is proxy.master,
        }
        listener_stopped = asyncio.Event()

        async def servers_changed():
            if not proxy.server.servers.is_updating:
                listener_stopped.set()

        proxy.server.servers.changed.connect(servers_changed)
        proxy.options.update(server=False)
        await asyncio.wait_for(listener_stopped.wait(), 3)
        results["server_false_option"] = {
            "listener_still_accepts": await can_connect(proxy.address),
            "active_connections": len(proxy.server.connections),
            "origin_stream_closed": origin.stall_closed.is_set(),
        }
        await proxy.cleanup()
        await stream.__aexit__(None, None, None)
        await client.aclose()
        await asyncio.wait_for(origin.stall_closed.wait(), 3)
        results["explicit_cleanup"] = {
            "listener_still_accepts": await can_connect(proxy.address),
            "active_connections": len(proxy.server.connections),
            "origin_stream_closed": origin.stall_closed.is_set(),
        }
        require(
            not results["explicit_cleanup"]["listener_still_accepts"], "listener survived cleanup"
        )
        require(
            results["explicit_cleanup"]["active_connections"] == 0,
            "proxy handlers survived cleanup",
        )
        results["observations"] = {
            "responseheaders_events": len(proxy.hooks.observations),
            "websocket_message_events": proxy.hooks.websocket_messages,
            "fields": ["method", "path", "status", "local_reply"],
            "synthetic_policy_response_events": sum(
                row["local_reply"] for row in proxy.hooks.observations
            ),
            "archive_addon_installed": proxy.master.addons.get("save") is not None,
        }
        results["target_requests"] = len(origin.trace)
        return results
    finally:
        await proxy.cleanup()
        await origin.close()
        for name, value in old_context.items():
            if value is None:
                with contextlib.suppress(AttributeError):
                    delattr(ctx, name)
            else:
                setattr(ctx, name, value)


async def main(output):
    started = time.monotonic()
    source_paths = [ROOT / "runtime" / name for name in ("pyproject.toml", "uv.lock")]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths]
    result = {
        "schema": "mitmproxy-embedding-assessment@1",
        "python": platform.python_version(),
        "platform": platform.system() + " " + platform.machine(),
        "mitmproxy": importlib.metadata.version("mitmproxy"),
        "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "default_addons": [type(addon).__name__ for addon in addons.default_addons()],
        "minimal_engine_addons": ["Proxyserver", "NextLayer", "TlsConfig"],
        "experiment_hook_addon": "ProbeHooks",
        "reused_cleanup_primitive": "contractor_runtime.core.async_tasks.join_owned_task",
        "assembly_class_lines": len(inspect.getsource(EmbeddedProxy).splitlines()),
        "direct_dependencies": importlib.metadata.requires("mitmproxy"),
    }
    require(result["mitmproxy"] == VERSION, "unexpected mitmproxy version")
    with tempfile.TemporaryDirectory(prefix="mitmproxy-assessment-") as temporary:
        directory = Path(temporary)
        result["dependency_resolution"] = await dependency_probe(directory)
        result["runtime_import"] = await runtime_import_probe()
        async with asyncio.timeout(60):
            result["repetitions"] = []
            for index in range(3):
                sample_directory = directory / f"sample-{index}"
                sample_directory.mkdir()
                result["repetitions"].append(await assess(sample_directory))
            result["probes"] = result["repetitions"][0]
    after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths]
    require(before == after, "production dependency files changed")
    result["production_dependency_files_unchanged"] = True
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "probes": result["probes"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
