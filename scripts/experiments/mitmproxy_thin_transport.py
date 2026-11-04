"""Try one native outgoing engine behind Contractor policy and a thin facade.

Run with the locked Runtime environment plus mitmproxy==12.3.1. This experiment
uses owned non-loopback local fixtures, never a public target. It is not a
production HTTPTransport replacement or a full allocation conformance gate.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import hmac
import importlib.metadata
import inspect
import ipaddress
import json
import secrets
import shutil
import ssl
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil
from mitmproxy import ctx, http, master, options
from mitmproxy.addons import next_layer, proxyserver, tlsconfig
from mitmproxy.proxy import commands, events

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/src"))
from mitmproxy_embedding import certificates  # noqa: E402

from contractor_runtime.adapters.http_policy import HTTPRoutePolicy  # noqa: E402
from contractor_runtime.adapters.http_session import IdentitySession, PrivateResponse  # noqa: E402
from contractor_runtime.contracts.engagement import canonical_url  # noqa: E402
from contractor_runtime.contracts.engagement_allocation import HTTPAssignment  # noqa: E402
from contractor_runtime.contracts.settings import (  # noqa: E402
    HTTPIdentityMaterial,
    RuntimeSettings,
)
from contractor_runtime.core.async_tasks import join_owned_task  # noqa: E402

VERSION = "12.3.1"
NUCLEI_VERSION = "3.11.1"
BODY_LIMIT = 65536


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fixture_address():
    for name, rows in sorted(psutil.net_if_addrs().items()):
        if name.startswith(("utun", "tun", "tap")):
            continue
        for row in rows:
            if row.family.name == "AF_INET":
                address = ipaddress.ip_address(row.address)
                if address.is_private and not (
                    address.is_loopback or address.is_link_local or address.is_reserved
                ):
                    return str(address)
    raise RuntimeError("a non-loopback private local fixture address is required")


@dataclass
class RequestState:
    identity: str
    url: str = ""
    policy: object = None
    addresses: tuple = ()
    peer_verified: bool = False
    acquired: bool = False
    epoch: object = None
    denied: bool = False


class Authority:
    """Existing policy/session primitives plus experimental allocation admission."""

    def __init__(self, assignment, settings, resolver):
        self.policy = HTTPRoutePolicy(assignment, resolver=resolver)
        self.settings = settings
        self.sessions = {
            identity.id: IdentitySession(
                identity, (settings.http_identities or {}).get(identity.id)
            )
            for identity in assignment.identities
        }
        self.token = secrets.token_hex(24)
        self.clock = 10.0
        self.deadline = 100.0
        self.closed = False
        self.clients = {}
        self.semaphore = asyncio.Semaphore(3)
        self.allowed_requests = 100
        self.requests = 0
        self.target_write_checks = 0
        self.blocked_writes = 0
        self.peer_checks = 0
        self.observations = []
        self.gates = {}
        self.peer_fault = False
        self.callback_fault = False
        self.ready = asyncio.Event()
        self.server = None

    def require_active(self):
        if self.closed or self.clock >= self.deadline:
            raise RuntimeError("authority expired")

    def authenticate(self, flow):
        self.require_active()
        header = flow.request.headers.pop("proxy-authorization", "")
        state = self.clients.get(flow.client_conn.id)
        if header:
            scheme, encoded = header.split(" ", 1)
            require(scheme.lower() == "basic", "proxy authentication required")
            identity, token = base64.b64decode(encoded, validate=True).decode().split(":", 1)
            require(hmac.compare_digest(token, self.token), "invalid proxy capability")
            require(identity in self.sessions, "unassigned identity")
            require(state is None or state.identity == identity, "connection identity changed")
            state = RequestState(identity)
            self.clients[flow.client_conn.id] = state
        require(state is not None, "missing proxy capability")
        return state

    def finish(self, flow):
        state = self.clients.get(flow.client_conn.id)
        if state is not None and state.acquired:
            state.acquired = False
            self.semaphore.release()

    def deny(self, flow, status=403):
        flow.response = http.Response.make(status, b"fixture proxy denied")
        flow.metadata["synthetic"] = True
        state = self.clients.get(flow.client_conn.id)
        if state is not None:
            state.denied = True
        self.finish(flow)

    def running(self):
        self.ready.set()

    def http_connect(self, flow):
        try:
            state = self.authenticate(flow)
            origin = canonical_url(f"https://{flow.request.host}:{flow.request.port}/")[:-1]
            require(
                origin in self.sessions[state.identity].identity.origins, "CONNECT origin denied"
            )
        except Exception:
            self.deny(flow, 407)

    async def requestheaders(self, flow):
        try:
            state = self.authenticate(flow)
            session = self.sessions[state.identity]
            url = flow.request.url
            require(canonical_url(url) == url, "noncanonical URL")
            state.url = url
            state.policy = self.policy.for_url(url, session.identity, flow.request.method)
            parsed = urlsplit(url)
            state.addresses = await state.policy.resolve(
                parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
            )
            if self.callback_fault:
                raise RuntimeError("injected admission callback failure")
            await self.semaphore.acquire()
            state.acquired = True
            self.require_active()
            require(self.requests < self.allowed_requests, "request allowance exhausted")
            self.requests += 1
            state.epoch = session.epoch
            for name in ("authorization", "cookie", "proxy-connection"):
                flow.request.headers.pop(name, None)
            private = session.request(url)
            for name, value in private.headers:
                flow.request.headers[name.decode()] = value.decode()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.deny(flow)

    async def server_connect(self, data):
        try:
            self.require_active()
            state = self.clients[data.client.id]
            require(state.policy is not None and not state.denied, "unadmitted connection")
            parsed = urlsplit(state.url)
            expected = (parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
            require(data.server.address == expected, "connection differs from admitted origin")
            state.addresses = await state.policy.resolve(*expected)
            self.require_active()
            data.server.sni = expected[0]
            data.server.address = (str(state.addresses[0]), expected[1])
        except asyncio.CancelledError:
            raise
        except Exception:
            data.server.error = "Contractor connection denied"

    async def server_connected(self, data):
        state = self.clients[data.client.id]
        peer = data.server.peername
        expected = (str(state.addresses[0]), data.server.address[1])
        self.peer_checks += 1
        state.peer_verified = peer[:2] == expected and not self.peer_fault
        gate = self.gates.get("tcp")
        if gate is not None:
            gate[0].set()
            await gate[1].wait()

    async def tls_established_server(self, data):
        gate = self.gates.get("tls")
        if gate is not None:
            gate[0].set()
            await gate[1].wait()

    def responseheaders(self, flow):
        if flow.metadata.get("synthetic"):
            return
        state = self.clients.get(flow.client_conn.id)
        if state is not None and not state.denied:
            self.sessions[state.identity].receive(
                PrivateResponse(
                    state.url, flow.response.status_code, list(flow.response.headers.fields)
                ),
                state.epoch,
            )

    def response(self, flow):
        self.observations.append(
            {
                "path": flow.request.path,
                "status": flow.response.status_code,
                "synthetic": bool(flow.metadata.get("synthetic")),
            }
        )
        self.finish(flow)

    def error(self, flow):
        self.finish(flow)

    def client_disconnected(self, client):
        state = self.clients.pop(client.id, None)
        if state is not None and state.acquired:
            self.semaphore.release()


class GuardedLayer:
    """Intercept native commands; the engine's network server remains unchanged."""

    def __init__(self, inner, owner):
        self.inner, self.owner = inner, owner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def handle_event(self, event):
        try:
            if isinstance(event, events.DataReceived):
                self.owner.require_active()
            for command in self.inner.handle_event(event):
                if isinstance(command, (commands.SendData, commands.OpenConnection)):
                    self.owner.require_active()
                    if command.connection is not self.context.client:
                        state = self.owner.clients[self.context.client.id]
                        require(not state.denied, "denied exchange")
                        if isinstance(command, commands.SendData):
                            self.owner.target_write_checks += 1
                            require(state.peer_verified, "target peer was not verified")
                yield command
        except Exception:
            self.owner.blocked_writes += 1
            yield commands.CloseConnection(self.context.client)


class GuardedProxyserver(proxyserver.Proxyserver):
    name = "proxyserver"

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    @contextlib.contextmanager
    def register_connection(self, connection_id, handler):
        handler.layer = GuardedLayer(handler.layer, self.owner)
        with super().register_connection(connection_id, handler):
            yield


class ThinHTTPTransport:
    """Programmatic ingress only: mitmproxy owns every outgoing target exchange."""

    def __init__(self, proxy_url, client_tls):
        self.proxy_url, self.client_tls = proxy_url, client_tls
        self.clients = {}
        self.tasks = set()

    async def request(self, url, *, identity="anonymous", method="GET", body=b""):
        client = self.clients.get(identity)
        if client is None:
            client = httpx.AsyncClient(
                proxy=self.proxy_url(identity),
                verify=self.client_tls,
                trust_env=False,
                timeout=10,
                follow_redirects=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            )
            self.clients[identity] = client
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            async with client.stream(method, url, content=body) as response:
                data = bytearray()
                async for chunk in response.aiter_raw():
                    require(len(data) + len(chunk) <= BODY_LIMIT, "facade body bound exceeded")
                    data.extend(chunk)
                return response.status_code, bytes(data)
        finally:
            self.tasks.discard(task)

    async def close(self):
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()


class Engine:
    def __init__(self, owner, directory, origin_ca):
        self.owner = owner
        self.previous = {name: getattr(ctx, name, None) for name in ("master", "log", "options")}
        opts = options.Options(
            confdir=str(directory),
            listen_host="127.0.0.1",
            listen_port=0,
            upstream_cert=False,
            rawtcp=False,
            http2=False,
            ssl_verify_upstream_trusted_ca=str(origin_ca),
        )
        self.master = master.Master(opts, with_termlog=False)
        self.server = GuardedProxyserver(owner)
        owner.server = self.server
        self.master.addons.add(self.server, next_layer.NextLayer(), tlsconfig.TlsConfig(), owner)
        opts.update(connection_strategy="lazy", body_size_limit="64k", store_streamed_bodies=False)
        self.task = None
        self.facade = None
        self.closed = False
        self.scanner_tasks = set()
        self.scanner_processes_joined = 0

    async def start(self):
        await self.owner.policy.initialize(self.owner.settings)
        self.task = asyncio.create_task(self.master.run())
        await asyncio.wait_for(self.owner.ready.wait(), 10)
        self.address = self.server.listen_addrs()[0]
        client_tls = ssl.create_default_context(
            cafile=str(Path(self.master.options.confdir) / "mitmproxy-ca-cert.pem")
        )
        self.facade = ThinHTTPTransport(self.proxy_url, client_tls)
        return self.facade

    def proxy_url(self, identity="anonymous"):
        host, port = self.address
        return f"http://{identity}:{self.owner.token}@{host}:{port}"

    async def close(self):
        if self.closed:
            return

        async def cleanup():
            self.owner.closed = True
            for _, release in self.owner.gates.values():
                release.set()
            scanners = tuple(self.scanner_tasks)
            for scanner in scanners:
                scanner.cancel()
            await asyncio.gather(*scanners, return_exceptions=True)
            if self.facade is not None:
                await self.facade.close()
            if self.task is not None:
                self.master.shutdown()
                await self.task
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
            for session in self.owner.sessions.values():
                session.close()
            self.owner.token = ""
            for name, value in self.previous.items():
                if value is None:
                    ctx.__dict__.pop(name, None)
                else:
                    setattr(ctx, name, value)
            self.closed = True

        task = asyncio.create_task(cleanup())
        cancelled = await join_owned_task(task)
        task.result()
        if cancelled:
            raise asyncio.CancelledError


class Origin:
    def __init__(self, address):
        self.address = address
        self.servers, self.tasks, self.writers, self.trace = [], set(), set(), []
        self.connections = 0
        self.stall_started, self.stall_closed = asyncio.Event(), asyncio.Event()
        self.both_stalls_started = asyncio.Event()
        self.stalled = self.closed_stalls = 0
        self.sni = []

    async def listen(self, tls=None):
        if tls is not None:
            tls.set_servername_callback(lambda _, name, __: self.sni.append(name))
        server = await asyncio.start_server(self.accept, self.address, 0, ssl=tls)
        self.servers.append(server)
        return server.sockets[0].getsockname()[1]

    def accept(self, reader, writer):
        self.connections += 1
        self.writers.add(writer)
        task = asyncio.create_task(self.serve(reader, writer))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def serve(self, reader, writer):
        try:
            async with asyncio.timeout(20):
                head = await reader.readuntil(b"\r\n\r\n")
                method, path, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
                headers = {
                    key.lower(): value.strip()
                    for key, value in (
                        line.split(b":", 1) for line in head.split(b"\r\n")[1:] if b":" in line
                    )
                }
                length = int(headers.get(b"content-length", b"0"))
                require(length <= BODY_LIMIT, "fixture request too large")
                body = await reader.readexactly(length)
                self.trace.append(
                    {
                        "method": method,
                        "path": path,
                        "body_bytes": len(body),
                        "host": headers.get(b"host", b"").decode(),
                        "bearer": headers.get(b"authorization") == b"Bearer thin-fixture-canary",
                        "cookie": b"learned=fixture" in headers.get(b"cookie", b""),
                    }
                )
                if path.startswith("/allowed/stall"):
                    self.stalled += 1
                    self.stall_started.set()
                    if self.stalled == 2:
                        self.both_stalls_started.set()
                    await reader.read(1)
                    self.closed_stalls += 1
                    if self.closed_stalls == 2:
                        self.stall_closed.set()
                    return
                cookie = (
                    b"Set-Cookie: learned=fixture; Path=/\r\n" if path.endswith("/cookie") else b""
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\nConnection: close\r\n"
                    + cookie
                    + b"\r\nthin-fixture"
                )
                await writer.drain()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            self.writers.discard(writer)

    async def close(self):
        for server in self.servers:
            server.close()
            await server.wait_closed()
        for writer in tuple(self.writers):
            writer.close()
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def owner_for(address, http_port, https_port):
    origins = [f"http://target.example:{http_port}", f"https://localhost:{https_port}"]
    assignment = HTTPAssignment.model_validate_json(
        json.dumps(
            {
                "adapter": "http-target@1",
                "routes": [
                    {
                        "origin": origin,
                        "pathPrefixes": ["/allowed"],
                        "methods": ["GET", "POST"],
                        "addressRanges": [address + "/32"],
                    }
                    for origin in origins
                ],
                "exclusions": [],
                "identities": [
                    {"id": "anonymous", "kind": "anonymous", "origins": origins},
                    {
                        "id": "signed",
                        "kind": "bearer",
                        "origins": origins,
                        "credential": {"credentialId": "fixture-bearer", "kind": "target-bearer@1"},
                    },
                ],
                "protectedDestinations": [
                    {"hostname": "control.example", "addresses": ["10.255.255.1"], "port": 443}
                ],
            }
        )
    )
    material = HTTPIdentityMaterial.model_validate_json(
        json.dumps(
            {
                "credential": {"credentialId": "fixture-bearer", "kind": "target-bearer@1"},
                "fields": {"token": "thin-fixture-canary"},
            }
        )
    )
    settings = RuntimeSettings(
        artifact_api_url="https://control.example",
        request_timeout_seconds=10,
        http_identities={"signed": material},
    )
    dns = {"mixed": False, "calls": 0}

    async def resolver(host, port):
        if host in {"target.example", "localhost"}:
            dns["calls"] += 1
            values = [address, "127.0.0.1"] if dns["mixed"] else [address]
        else:
            values = ["10.255.255.1"]
        return tuple(ipaddress.ip_address(value) for value in values)

    return Authority(assignment, settings, resolver), origins, dns


async def scanner_probe(engine, directory, url, path="/allowed/scanner"):
    task = asyncio.current_task()
    engine.scanner_tasks.add(task)
    try:
        return await _scanner_probe(engine, directory, url, path)
    finally:
        engine.scanner_tasks.discard(task)


async def _scanner_probe(engine, directory, url, path):
    binary = shutil.which("nuclei")
    require(binary is not None, "installed Nuclei is required")
    template = directory / "thin-fixture.yaml"
    template.write_text(
        "id: thin-transport-fixture\ninfo:\n  name: Thin transport fixture\n"
        "  author: contractor\n  severity: info\nhttp:\n  - method: GET\n"
        "    path: [" + json.dumps("{{BaseURL}}" + path) + "]\n    matchers:\n"
        '      - type: word\n        words: ["thin-fixture"]\n'
    )
    config = directory / "scanner-config.yaml"
    config.write_text("proxy: [" + json.dumps(engine.proxy_url()) + "]\nproxy-internal: true\n")
    config.chmod(0o600)
    (directory / "nuclei-templates").mkdir(exist_ok=True)
    command = [
        binary,
        "-config",
        str(config),
        "-t",
        str(template),
        "-u",
        url,
        "-jsonl",
        "-silent",
        "-no-color",
        "-disable-update-check",
        "-no-interactsh",
        "-no-httpx",
        "-disable-redirects",
        "-retries",
        "0",
        "-timeout",
        "5",
        "-concurrency",
        "1",
        "-payload-concurrency",
        "1",
        "-rate-limit",
        "0",
        "-omit-raw",
        "-omit-template",
    ]
    process = await asyncio.create_subprocess_exec(
        *command, cwd=directory, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
    finally:
        if process.returncode is None:
            process.kill()
        join = asyncio.create_task(process.wait())
        await join_owned_task(join)
        join.result()
        engine.scanner_processes_joined += 1
    require(process.returncode == 0, "Nuclei failed against local fixture")
    matches = [json.loads(line) for line in stdout.decode().splitlines() if line.startswith("{")]
    require(
        any(row.get("template-id") == "thin-transport-fixture" for row in matches),
        "Nuclei fixture match missing",
    )
    return {
        "exit_code": process.returncode,
        "fixture_matches": len(matches),
        "stderr_bytes": len(stderr),
    }


async def assess(directory):
    address = fixture_address()
    ca, _, tls, _ = certificates(directory)
    origin = Origin(address)
    http_port = await origin.listen()
    https_port = await origin.listen(tls)
    owner, origins, dns = owner_for(address, http_port, https_port)
    engine = Engine(owner, directory / "proxy", ca)
    results = {}
    try:
        facade = await engine.start()
        status, body = await facade.request(origins[0] + "/allowed/python")
        require(status == 200 and body == b"thin-fixture", "programmatic HTTP failed")
        status, _ = await facade.request(
            origins[1] + "/allowed/tls", method="POST", body=b"payload"
        )
        require(status == 200 and origin.sni[-1] == "localhost", "verified HTTPS/SNI failed")
        require(origin.trace[-1]["host"] == f"localhost:{https_port}", "Host changed after pinning")
        results["programmatic_ingress"] = {
            "http": 200,
            "verified_https": 200,
            "host_sni_preserved": True,
        }
        await facade.request(origins[0] + "/allowed/identity", identity="signed")
        require(origin.trace[-1]["bearer"], "assigned bearer missing")
        await facade.request(origins[0] + "/allowed/identity")
        require(not origin.trace[-1]["bearer"], "bearer crossed identity")
        await facade.request(origins[0] + "/allowed/cookie")
        await facade.request(origins[0] + "/allowed/cookie-check")
        require(origin.trace[-1]["cookie"], "Runtime cookie session was not applied")
        results["identity_sessions"] = {"bearer_isolated": True, "cookie_updated": True}
        await facade.request(origins[1] + "/allowed/other-origin")
        require(not origin.trace[-1]["cookie"], "cookie crossed origin")
        results["identity_sessions"]["cookie_origin_isolated"] = True
        before = len(origin.trace)
        for path, method in [("/denied", "GET"), ("/allowed/method", "DELETE")]:
            status, _ = await facade.request(origins[0] + path, method=method)
            require(status == 403, "scope did not deny")
        dns["mixed"] = True
        status, _ = await facade.request(origins[0] + "/allowed/mixed-dns")
        dns["mixed"] = False
        require(status == 403, "mixed DNS was not denied")
        owner.callback_fault = True
        status, _ = await facade.request(origins[0] + "/allowed/callback-error")
        owner.callback_fault = False
        require(status == 403, "callback failure did not fail closed")
        require(len(origin.trace) == before, "denied request reached target")
        results["admission"] = {
            "denied_path": 403,
            "denied_method": 403,
            "mixed_dns": 403,
            "callback_error": 403,
            "target_requests": 0,
        }
        owner.peer_fault = True
        try:
            await facade.request(origins[0] + "/allowed/wrong-peer")
        except httpx.HTTPError:
            pass
        finally:
            owner.peer_fault = False
        require(len(origin.trace) == before, "unverified peer received application request")
        results["peer_fault_injection"] = {"target_requests": 0, "send_guard_rejected": True}
        scanner_before = len(origin.trace)
        results["nuclei"] = await scanner_probe(engine, directory, origins[0])
        require(len(origin.trace) == scanner_before + 1, "scanner sent unexpected request count")
        require(origin.trace[-1]["path"] == "/allowed/scanner", "scanner bypassed intended request")
        results["nuclei"]["target_requests"] = 1
        results["lease_after_setup"] = {}
        for setup, origin_url in [("tcp", origins[0]), ("tls", origins[1])]:
            entered, release = asyncio.Event(), asyncio.Event()
            owner.gates[setup] = (entered, release)
            before = len(origin.trace)
            blocked_before = owner.blocked_writes
            task = asyncio.create_task(facade.request(origin_url + "/allowed/lease"))
            await asyncio.wait_for(entered.wait(), 10)
            owner.clock = owner.deadline + 1
            release.set()
            with contextlib.suppress(httpx.HTTPError):
                await task
            owner.gates.pop(setup)
            owner.clock = 10.0
            require(len(origin.trace) == before, "expired authority sent target HTTP")
            require(owner.blocked_writes > blocked_before, "write guard did not observe expiry")
            results["lease_after_setup"][setup] = {
                "target_requests": 0,
                "write_guard_rejected": True,
            }
        task = asyncio.create_task(facade.request(origins[0] + "/allowed/stall-client"))
        await asyncio.wait_for(origin.stall_started.wait(), 10)
        scanner = asyncio.create_task(
            scanner_probe(engine, directory, origins[0], "/allowed/stall-scanner")
        )
        await asyncio.wait_for(origin.both_stalls_started.wait(), 10)
        await engine.close()
        with contextlib.suppress(asyncio.CancelledError, httpx.HTTPError):
            await task
        with contextlib.suppress(asyncio.CancelledError):
            await scanner
        await asyncio.wait_for(origin.stall_closed.wait(), 5)
        require(not engine.server.connections and not facade.tasks, "engine work survived stop")
        require(
            not engine.scanner_tasks and engine.scanner_processes_joined == 2,
            "scanner survived stop",
        )
        results["cleanup"] = {
            "engine_connections": 0,
            "facade_tasks": 0,
            "target_saw_eof": True,
            "proxy_capability_erased": not owner.token,
            "scanner_processes_joined": engine.scanner_processes_joined,
            "target_stalled_connections_closed": origin.closed_stalls,
        }
        results["policy_trace"] = {
            "dns_lookups": dns["calls"],
            "peer_checks": owner.peer_checks,
            "target_send_checks": owner.target_write_checks,
            "blocked_writes": owner.blocked_writes,
            "target_requests": len(origin.trace),
        }
        return results
    finally:
        await engine.close()
        await origin.close()


async def main(output):
    require(importlib.metadata.version("mitmproxy") == VERSION, "unexpected mitmproxy version")
    production = [ROOT / "runtime" / name for name in ("pyproject.toml", "uv.lock")]
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in production]
    binary = shutil.which("nuclei")
    require(binary is not None, "Nuclei is required")
    version_process = await asyncio.create_subprocess_exec(
        binary, "-version", "-duc", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(version_process.communicate(), 10)
    require(f"v{NUCLEI_VERSION}".encode() in stdout + stderr, "unexpected Nuclei version")
    result = {
        "schema": "mitmproxy-thin-transport-spike@1",
        "mitmproxy": VERSION,
        "nuclei": NUCLEI_VERSION,
        "nuclei_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "facade_lines": len(inspect.getsource(ThinHTTPTransport).splitlines()),
        "io_guard_lines": len(inspect.getsource(GuardedLayer).splitlines()),
        "fixture_address_class": "owned private non-loopback",
        "repetitions": [],
    }
    with tempfile.TemporaryDirectory(prefix="contractor-thin-proxy-") as temporary:
        async with asyncio.timeout(120):
            for index in range(3):
                directory = Path(temporary) / str(index)
                directory.mkdir()
                result["repetitions"].append(await assess(directory))
                print(json.dumps({"repetition": index + 1, "completed": True}), flush=True)
    require(
        before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in production],
        "production dependencies changed",
    )
    result["production_dependencies_unchanged"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
