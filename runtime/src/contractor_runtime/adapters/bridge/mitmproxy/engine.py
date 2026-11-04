"""One pinned native HTTP engine owned by an allocation's HTTP authority."""

from __future__ import annotations

import asyncio
import collections
import contextvars
import importlib.metadata
import ipaddress
import logging
import ssl
import sys
import tempfile
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from mitmproxy import ctx, http, master, options
from mitmproxy.addons import next_layer, proxyserver, tlsconfig
from mitmproxy.connection import Client, ConnectionState, Server
from mitmproxy.net import tls as native_tls
from mitmproxy.proxy import commands, events, server
from mitmproxy.proxy.context import Context
from mitmproxy.proxy.layers.http import HttpErrorHook, HttpResponseHook

from contractor_runtime.adapters.bridge import BridgeFailure
from contractor_runtime.adapters.bridge.mitmproxy.lifecycle import OwnedHandler
from contractor_runtime.adapters.bridge.mitmproxy.messages import native_url, request_message
from contractor_runtime.adapters.bridge.mitmproxy.protocol import (
    Abort,
    BoundedHttp1Client,
    BoundedHttp2Client,
    RoutedHttpLayer,
    Submit,
)
from contractor_runtime.adapters.bridge.mitmproxy.state import Exchange
from contractor_runtime.adapters.http_session import PrivateRequest, PrivateResponse
from contractor_runtime.adapters.http_stream import RequestBoundary
from contractor_runtime.adapters.http_transport import HTTPTransportError
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.http_limits import MAX_CONTROLLED_HEADER_BYTES, MAX_HEADERS
from contractor_runtime.core.target_policy import TargetDenied
from contractor_runtime.threads import to_thread_until_done

if TYPE_CHECKING:
    from contractor_runtime.adapters.bridge import HTTPProxyBridge
    from contractor_runtime.adapters.http_policy import _RequestPolicy

VERSION = "12.3.1"
_active: MitmproxyEngine | None = None
_MISSING = object()


class _ProcessState:
    """Exclusive process state, restored only after joined resource erasure."""

    def __init__(self):
        self.owned = False
        self.previous: dict[str, object] = {}
        self.logger = logging.getLogger("mitmproxy")
        self.handlers: list[logging.Handler] = []
        self.propagate = True
        self.keylog: native_tls.MasterSecretLogger | None = None

    @staticmethod
    def erase_tls_contexts():
        native_tls.create_proxy_server_context.cache_clear()
        native_tls.create_client_proxy_context.cache_clear()

    def capture(self, owner: MitmproxyEngine):
        global _active
        if _active is not None:
            raise RuntimeError("native HTTP engine already belongs to an allocation")
        _active = owner
        self.owned = True
        self.previous = {key: getattr(ctx, key, _MISSING) for key in ("master", "log", "options")}
        self.handlers, self.propagate = self.logger.handlers[:], self.logger.propagate
        self.keylog = native_tls.log_master_secret
        native_tls.log_master_secret = None
        self.logger.handlers, self.logger.propagate = [logging.NullHandler()], False
        self.erase_tls_contexts()

    def restore(self):
        global _active
        if not self.owned:
            return
        self.erase_tls_contexts()
        for key, previous in self.previous.items():
            if previous is _MISSING:
                ctx.__dict__.pop(key, None)
            else:
                setattr(ctx, key, previous)
        self.logger.handlers, self.logger.propagate = self.handlers, self.propagate
        native_tls.log_master_secret = self.keylog
        self.previous.clear()
        self.handlers = []
        self.keylog = None
        self.owned = False
        _active = None


_dial: contextvars.ContextVar[Exchange] = contextvars.ContextVar("bridge_mitmproxy_dial")


@dataclass
class _Wakeup:
    connection: Server
    request: commands.RequestWakeup
    task: asyncio.Task[None] | None = None


class _Permit:
    """Native connection backpressure with a second check after its queue."""

    def __init__(self, semaphore: asyncio.Semaphore):
        self.semaphore = semaphore
        self.users = 0

    async def __aenter__(self):
        self.users += 1
        acquired = False
        try:
            await self.semaphore.acquire()
            acquired = True
            _dial.get().boundary.require_time()
        except BaseException:
            if acquired:
                self.semaphore.release()
            self.users -= 1
            raise

    async def __aexit__(self, *_):
        self.semaphore.release()
        self.users -= 1


class _Reader:
    def __init__(self, inner: asyncio.StreamReader, pool: _Pool, connection: Server):
        self.inner, self.pool, self.connection = inner, pool, connection

    async def read(self, maximum):
        self.pool.check(self.connection)
        return await self.inner.read(maximum)


class _Writer:
    def __init__(self, inner: asyncio.StreamWriter, pool: _Pool, connection: Server):
        self.inner, self.pool, self.connection = inner, pool, connection

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def write(self, data):
        self.pool.check(self.connection)
        self.inner.write(data)

    async def drain(self):
        self.pool.check(self.connection)
        await self.inner.drain()


class _Pool(OwnedHandler, server.ConnectionHandler):
    """A native HTTP layer shared only within one identity and exact origin."""

    def __init__(self, engine: MitmproxyEngine, url: str):
        self.engine = engine
        parsed = urlsplit(url)
        self.host, self.port = (
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
        client = Client(
            peername=("127.0.0.1", 0), sockname=("127.0.0.1", 0), state=ConnectionState.OPEN
        )
        context = Context(client, engine.options)
        super().__init__(context)
        self.layer = RoutedHttpLayer(context, self)
        self.exchanges: dict[int, Exchange] = {}
        self.parsers: dict[Server, BoundedHttp1Client | BoundedHttp2Client] = {}
        self.connecting: dict[Server, Exchange] = {}
        self.verified: set[Server] = set()
        self.next_stream = 1
        self.closed = False
        self.hooks: set[asyncio.Task[None]] = set()
        self.writers: set[asyncio.StreamWriter | _Writer] = set()
        self._wakeups: dict[Coroutine[object, object, None], _Wakeup] = {}
        native_limit = self.max_conns.default_factory
        self.max_conns = collections.defaultdict(lambda: _Permit(native_limit()))

    def busy(self, connection):
        return any(exchange.connection == connection for exchange in self.exchanges.values())

    def check(self, connection):
        try:
            for exchange in self.exchanges.values():
                if exchange.connection == connection:
                    exchange.boundary.require_time()
            self.engine.require_authority()
            if connection not in self.verified:
                raise TargetDenied
        except (TimeoutError, TargetDenied, HTTPTransportError) as error:
            for exchange in self.exchanges.values():
                if exchange.connection == connection:
                    exchange.reason = (
                        "policy_denied" if isinstance(error, TargetDenied) else "timeout"
                    )
            # Native read/drain paths classify socket termination as OSError.
            # Never hand an authority exception to an unobserved native task.
            raise OSError("HTTP authority ended") from None

    def fail(self, error: BaseException | None, *, callback: bool):
        for exchange in self.exchanges.values():
            if isinstance(error, TargetDenied):
                exchange.reason = "policy_denied"
            elif isinstance(error, TimeoutError) or (
                callback and isinstance(error, HTTPTransportError)
            ):
                exchange.reason = "timeout"
            elif isinstance(error, OSError) and not callback:
                exchange.reason = exchange.reason or "transport_failed"
            else:
                source = "callback" if callback else "dispatch"
                exchange.exception = RuntimeError(f"native HTTP engine {source} failed")
            exchange.finish()
        for connection in tuple(self.transports):
            self.close_connection(connection)

    def log(self, message, level=logging.INFO, exc_info=None):
        # Native protocol errors can contain raw bytes. Preserve classification
        # and bug signals without forwarding diagnostics or exception content.
        if exc_info:
            self.fail(sys.exc_info()[1], callback=False)

    async def handle_hook(self, hook):
        try:
            await self.engine.handle_hook(self, hook)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.fail(error, callback=True)

    async def server_event(self, event):
        await super().server_event(event)
        if isinstance(event, events.HookCompleted) and isinstance(
            event.command, (HttpResponseHook, HttpErrorHook)
        ):
            (flow,) = event.command.args()
            exchange = self.engine.flows.get(flow.id)
            if exchange is not None:
                exchange.finish()

    def wakeup(self, request: commands.RequestWakeup):
        # Native dispatch calls this synchronously before creating its Task.
        # Bind now so retirement also owns timers cancelled before first execution.
        source = self.layer.command_sources[request]
        owned = _Wakeup(source.context.server, request)
        coroutine = self._run_wakeup(owned)
        self._wakeups[coroutine] = owned
        return coroutine

    async def _run_wakeup(self, owned: _Wakeup) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("native HTTP timer has no task")
        owned.task = task
        try:
            await super().wakeup(owned.request)
        finally:
            self.wakeup_timer.discard(task)
            self._wakeups.pop(task.get_coro(), None)
            self.layer.command_sources.pop(owned.request, None)

    async def _retire_timers(self, connection: Server) -> None:
        owned = {c: w for c, w in self._wakeups.items() if w.connection is connection}
        # Native wakeup_timer drops a task before dispatch, so include tasks
        # already waiting to deliver Wakeup as well as those not yet started.
        tasks = {w.task for w in owned.values() if w.task is not None}
        tasks.update(task for task in self.wakeup_timer if task.get_coro() in owned)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for coroutine, wakeup in owned.items():
            self._wakeups.pop(coroutine, None)
            self.layer.command_sources.pop(wakeup.request, None)
        self.wakeup_timer.difference_update(tasks)
        if any(isinstance(result, Exception) for result in results):
            raise RuntimeError("native HTTP engine timer failed")

    async def open_connection(self, command):
        exchange = self.connecting[command.connection]
        token = _dial.set(exchange)
        try:
            exchange.boundary.require_time()
            await super().open_connection(command)
        except (TimeoutError, TargetDenied) as error:
            exchange.reason = "timeout" if isinstance(error, TimeoutError) else "policy_denied"
            await self.server_event(events.OpenConnectionCompleted(command, exchange.reason))
        finally:
            _dial.reset(token)
            cleanup = asyncio.create_task(
                self._release_connection(command.connection), name="http-engine-connection-close"
            )
            cancelled = await join_owned_task(cleanup)
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _release_connection(self, connection: Server) -> None:
        await self._retire_timers(connection)
        io = self.transports.get(connection)
        if io is not None and io.writer is not None:
            self.writers.add(io.writer)
            await self.close_writer(io.writer)
        self.transports.pop(connection, None)
        self.parsers.pop(connection, None)
        self.connecting.pop(connection, None)
        self.verified.discard(connection)
        self.layer.connections.pop(connection, None)
        self.layer.waiting_for_establishment.pop(connection, None)
        permit = self.max_conns.get(connection.address)
        if permit is not None and not permit.users:
            self.max_conns.pop(connection.address, None)

    async def submit(self, exchange: Exchange):
        if self.closed:
            raise BridgeFailure("transport_failed")
        exchange.stream_id = self.next_stream
        self.next_stream += 2
        exchange.flow = http.HTTPFlow(self.client, Server(address=(self.host, self.port)))
        if exchange.request is None:
            raise RuntimeError("native HTTP exchange was released")
        exchange.flow.request = request_message(exchange.url, exchange.request)
        self.exchanges[exchange.stream_id] = exchange
        self.engine.flows[exchange.flow.id] = exchange
        await self.server_event(Submit(self.client, exchange))

    async def abort(self, exchange):
        if exchange.stream_id in self.layer.streams:
            await self.server_event(Abort(exchange))

    async def release_idle(self, connection):
        # Native connect permits last for a socket's entire lifetime. A dial
        # queued before any socket was ready cannot migrate to a now-idle one.
        # Free an idle permit so that such a waiter cannot starve behind reuse.
        if (
            connection in self.transports
            and not self.busy(connection)
            and any(io.writer is None for io in self.transports.values())
        ):
            io = self.transports[connection]
            self.close_connection(connection)
            if io.handler is not None:
                await join_owned_task(io.handler)
                if not io.handler.cancelled():
                    io.handler.result()

    async def close(self):
        self.closed = True
        await self.join()
        for exchange in self.exchanges.values():
            exchange.reason = exchange.reason or "transport_failed"
            exchange.finish()
        self.layer.connections.clear()
        self.layer.streams.clear()
        self.layer.command_sources.clear()
        self._wakeups.clear()
        self.layer.waiting_for_establishment.clear()
        self.parsers.clear()
        self.connecting.clear()
        self.verified.clear()


class MitmproxyEngine:
    """Native engine resource; HTTPTransport remains the sole domain owner."""

    def __init__(
        self, *, require_authority: Callable[[], object], ssl_context: ssl.SSLContext | None = None
    ):
        self.require_authority = require_authority
        self.ssl_context = ssl_context
        self.master: master.Master | None = None
        self.options: options.Options | None = None
        self.tls: tlsconfig.TlsConfig | None = None
        self.next_layer: next_layer.NextLayer | None = None
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.close_task: asyncio.Task[None] | None = None
        self.pools: dict[tuple[str, str], _Pool] = {}
        self.flows: dict[str, Exchange] = {}
        self.ingresses: set[HTTPProxyBridge] = set()
        self.closed = False
        self._process = _ProcessState()
        self.started = self.starting = False

    async def start(self):
        if self.started or self.starting or self.closed:
            raise RuntimeError("native HTTP engine cannot be restarted")
        self.starting = True
        if importlib.metadata.version("mitmproxy") != VERSION:
            raise RuntimeError("unsupported native HTTP engine")
        self._process.capture(self)
        self.directory = await to_thread_until_done(
            lambda: tempfile.TemporaryDirectory(prefix="contractor-http-engine-"),
            name="http-engine-directory",
        )
        trusted = None
        if self.ssl_context is not None:
            if (
                self.ssl_context.verify_mode != ssl.CERT_REQUIRED
                or not self.ssl_context.check_hostname
            ):
                raise ValueError("controlled HTTP requires verified TLS")

            def trust_bundle():
                path = Path(self.directory.name) / "trusted.pem"
                path.write_text(
                    "".join(
                        ssl.DER_cert_to_PEM_cert(cert)
                        for cert in self.ssl_context.get_ca_certs(binary_form=True)
                    )
                )
                return str(path)

            trusted = await to_thread_until_done(trust_bundle, name="http-engine-trust")
        self.options = options.Options(
            confdir=self.directory.name,
            listen_host="127.0.0.1",
            listen_port=0,
            upstream_cert=False,
            rawtcp=False,
            http2=True,
            ssl_verify_upstream_trusted_ca=trusted,
        )
        self.master = master.Master(self.options, with_termlog=False)
        self.tls = tlsconfig.TlsConfig()
        self.next_layer = next_layer.NextLayer()
        self.master.addons.add(proxyserver.Proxyserver(), self.next_layer, self.tls)
        self.options.update(server=False, connection_strategy="lazy", store_streamed_bodies=False)
        await to_thread_until_done(
            self.tls.configure, set(self.options.keys()), name="http-engine-tls-config"
        )
        # Embedding must not install Master.run's process-wide task factory or
        # event-loop exception handler. Native resources are owned here.
        self.next_layer.configure(set(self.options.keys()))
        self.started = True
        self.starting = False

    def create_ingress(self, bridge, reader, writer):
        """Construct this implementation's local scanner protocol handler."""
        from contractor_runtime.adapters.bridge.mitmproxy.ingress import Ingress

        return Ingress(bridge, reader, writer, self)

    async def request(
        self,
        url: str,
        boundary: RequestBoundary,
        request: PrivateRequest,
        policy: _RequestPolicy,
        identity: str,
        body_limit: int,
    ) -> PrivateResponse:
        parsed = urlsplit(url)
        addresses = await policy.resolve(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
        boundary.require_time()
        origin = parsed.scheme + "://" + parsed.netloc
        key = (identity, origin)
        pool = self.pools.get(key)
        if pool is None:
            pool = _Pool(self, url)
            self.pools[key] = pool
            await pool.server_event(events.Start())
        future: asyncio.Future[PrivateResponse] = asyncio.get_running_loop().create_future()
        exchange = Exchange(
            url,
            boundary,
            request,
            addresses,
            body_limit,
            future,
        )
        try:
            await pool.submit(exchange)
            return await asyncio.shield(future)
        finally:
            cleanup = asyncio.create_task(
                self._release_exchange(pool, exchange), name="http-engine-exchange-close"
            )
            cancelled = await join_owned_task(cleanup)
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _release_exchange(self, pool: _Pool, exchange: Exchange) -> None:
        if exchange.flow is not None:
            await pool.abort(exchange)
            # Join callbacks advancing this exact stream before erasing private state.
            while pending := [task for task in exchange.hooks if not task.done()]:
                await asyncio.gather(*pending, return_exceptions=True)
            self.flows.pop(exchange.flow.id, None)
        parser = pool.parsers.get(exchange.connection)
        if isinstance(parser, BoundedHttp2Client):
            parser.forget(exchange.stream_id)
        pool.exchanges.pop(exchange.stream_id, None)
        await pool.release_idle(exchange.connection)
        if exchange.future is not None:
            if not exchange.future.done():
                exchange.future.cancel()
            elif not exchange.future.cancelled():
                exchange.future.exception()
        exchange.hooks.clear()
        exchange.clear()

    async def handle_hook(self, pool, hook):
        data = hook.args()[0]
        if hook.name == "server_connect":
            exchange = pool.connecting[data.server]
            exchange.connection = data.server
            exchange.boundary.require_time()
        elif hook.name == "server_connected":
            exchange = pool.connecting[data.server]
            io = pool.transports[data.server]
            io.reader = _Reader(io.reader, pool, data.server)
            io.writer = _Writer(io.writer, pool, data.server)
            peer = data.server.peername
            if peer is None or (ipaddress.ip_address(peer[0]), peer[1]) != (
                exchange.addresses[0],
                pool.port,
            ):
                exchange.reason = "policy_denied"
                data.server.error = "policy_denied"
                # Install the guard even for rejected peers. Native dispatch may
                # still resume connection setup after this callback returns.
                return
            pool.verified.add(data.server)
        elif hook.name == "server_connect_error":
            exchange = pool.connecting.get(data.server)
            if exchange is not None:
                exchange.reason = exchange.reason or "connect_failed"
        elif hook.name == "tls_start_server":
            await to_thread_until_done(
                self.tls.tls_start_server, data, name="http-engine-target-tls"
            )
        elif hook.name == "tls_failed_server":
            exchange = pool.connecting.get(data.conn)
            if exchange is not None:
                exchange.reason = "tls_failed"
        elif isinstance(data, http.HTTPFlow):
            exchange = self.flows[data.id]
            if hook.name == "requestheaders":
                exchange.boundary.require_time()
                if native_url(data.request) != exchange.url:
                    exchange.reason = "policy_denied"
                    data.kill()
                data.request.stream = True
            elif hook.name == "responseheaders":
                headers = list(data.response.headers.fields)
                if (
                    len(headers) > MAX_HEADERS
                    or sum(len(k) + len(v) + 4 for k, v in headers) > MAX_CONTROLLED_HEADER_BYTES
                ):
                    exchange.reason = "response_invalid"
                    data.kill()
                    return
                exchange.response = PrivateResponse(
                    exchange.url, data.response.status_code, headers
                )
                data.response.stream = lambda _: b""
            elif hook.name == "error":
                exchange.reason = exchange.reason or "response_invalid"

    async def close(self):
        if self.closed:
            return
        if self.close_task is None or self.close_task.done():
            self.close_task = asyncio.create_task(self._close(), name="http-engine-close")
        cancelled = await join_owned_task(self.close_task)
        self.close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        for ingress in tuple(self.ingresses):
            await ingress.__aexit__()
        for pool in self.pools.values():
            await pool.close()
        self.pools.clear()
        self.flows.clear()
        if self.master is not None:
            await self.master.done()
            self.master = None
        if self._process.owned:
            self._process.erase_tls_contexts()
        self.options = self.tls = self.next_layer = self.ssl_context = None
        if self.directory is not None:
            await to_thread_until_done(self.directory.cleanup, name="http-engine-files-close")
            self.directory = None
        # A failed writer join or file deletion keeps exclusive process ownership
        # and its exact cleanup handles until a later close succeeds.
        self._process.restore()
        self.closed = True
