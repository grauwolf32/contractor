"""One Runtime-owned native Master with allocation-scoped library options."""

from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
import logging
import tempfile
from contextlib import contextmanager

from mitmproxy import ctx, master, options
from mitmproxy.addons import next_layer, proxyserver, tlsconfig
from mitmproxy.net import tls as native_tls

from contractor_runtime.adapters.bridge.errors import BridgeInfrastructureError
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.threads import to_thread_until_done

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
            raise RuntimeError("native HTTP engine already belongs to a Runtime")
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


_options = contextvars.ContextVar("contractor_native_http_options")


class _ContextOptions:
    def __getattr__(self, name):
        return getattr(_options.get(), name)


@contextmanager
def native_options(value):
    token = _options.set(value)
    try:
        yield
    finally:
        _options.reset(token)


class MitmproxyEngine:
    """Own process integration once and retain sessions through confirmed cleanup."""

    def __init__(self):
        if importlib.metadata.version("mitmproxy") != VERSION:
            raise RuntimeError("unsupported native HTTP engine")
        self.options = self.master = self.next_layer = self.directory = None
        self.sessions = set()
        self._lock = asyncio.Lock()
        self._process = _ProcessState()
        self._closing = self._failed = False
        self._close_task = None

    def require_available(self):
        if self._closing or self._failed:
            raise BridgeInfrastructureError()

    def create_session(self, *, require_authority, ssl_context=None):
        from contractor_runtime.adapters.bridge.mitmproxy.engine import MitmproxySession

        self.require_available()
        session = MitmproxySession(
            self, require_authority=require_authority, ssl_context=ssl_context
        )
        self.sessions.add(session)
        return session

    async def start(self):
        async with self._lock:
            self.require_available()
            if self.master is not None:
                return
            try:
                self._process.capture(self)
                self.directory = await to_thread_until_done(
                    lambda: tempfile.TemporaryDirectory(prefix="contractor-http-engine-"),
                    name="http-engine-directory",
                )
                self.options = options.Options(
                    confdir=self.directory.name,
                    listen_host="127.0.0.1",
                    listen_port=0,
                    upstream_cert=False,
                    rawtcp=False,
                    http2=True,
                )
                self.master = master.Master(self.options, with_termlog=False)
                self.next_layer = next_layer.NextLayer()
                self.master.addons.add(
                    proxyserver.Proxyserver(), self.next_layer, tlsconfig.TlsConfig()
                )
                self.options.update(
                    server=False, connection_strategy="lazy", store_streamed_bodies=False
                )
                self.next_layer.configure(set(self.options.keys()))
                ctx.options = _ContextOptions()
            except BaseException:
                self._failed = True
                raise

    async def configure_tls(self, config, session_options):
        # Native CA creation changes the process umask, so session setup must join
        # before another session can enter it. Requests keep their private locks.
        async with self._lock:
            self.require_available()
            with native_options(session_options):
                await to_thread_until_done(
                    config.configure, set(session_options.keys()), name="http-session-tls"
                )

    @staticmethod
    def erase_tls_contexts():
        _ProcessState.erase_tls_contexts()

    async def close(self):
        self._closing = True
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close(), name="http-runtime-close")
        cancelled = await join_owned_task(self._close_task)
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self):
        for session in tuple(self.sessions):
            await session.close()
        async with self._lock:
            if self.master is not None:
                with native_options(self.options):
                    await self.master.done()
                self.master = None
            if self._process.owned:
                self.erase_tls_contexts()
            if self.directory is not None:
                await to_thread_until_done(self.directory.cleanup, name="http-engine-files-close")
                self.directory = None
            self.options = self.next_layer = None
            self._process.restore()
