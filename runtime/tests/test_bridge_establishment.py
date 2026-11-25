"""Cancellation and timeout own real pending TCP/TLS work until it is joined."""

import asyncio
import threading
import time

import pytest
from fakes.engagement_socket import preflight, tls_contexts
from fakes.native_http import NativeTarget


def run(scenario):
    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


class TimeoutProbe:
    """Reschedule the real inner request timeout after establishment is suspended."""

    def __init__(self):
        self.scopes = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    def timeout(self, delay):
        scope = asyncio.timeout(delay)
        self.scopes.append(scope)
        return scope


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
@pytest.mark.parametrize("stop", ["cancel", "timeout"])
@pytest.mark.parametrize("repeated", [False, True], ids=["once", "repeated"])
def test_cancelled_tls_setup_is_joined_before_call_returns(
    tmp_path, monkeypatch, http2, stop, repeated
):
    from mitmproxy.addons.tlsconfig import TlsConfig

    from contractor_runtime.adapters import http_transport
    from contractor_runtime.adapters.bridge.mitmproxy.engine import (
        MitmproxySession as MitmproxyEngine,
    )

    release = threading.Event()
    finished = threading.Event()
    original_tls = TlsConfig.tls_start_server
    original_release = MitmproxyEngine._release_exchange
    probe = TimeoutProbe()
    monkeypatch.setattr(http_transport, "asyncio", probe)

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        loop = asyncio.get_running_loop()
        entered, cleaning, cleaned = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = 0

        def held_tls(config, data):
            nonlocal calls
            calls += 1
            if calls == 1:
                loop.call_soon_threadsafe(entered.set)
                try:
                    assert release.wait(10), "TLS test gate was not released"
                    return original_tls(config, data)
                finally:
                    finished.set()
            return original_tls(config, data)

        async def observe_cleanup(engine, pool, exchange):
            if not release.is_set():
                cleaning.set()
            try:
                await original_release(engine, pool, exchange)
            finally:
                cleaned.set()

        monkeypatch.setattr(TlsConfig, "tls_start_server", held_tls)
        monkeypatch.setattr(MitmproxyEngine, "_release_exchange", observe_cleanup)
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.open_proxy()  # Initialize without opening a target connection.
            call = asyncio.create_task(
                owner.request(target.origin + "/allowed", method="POST", body=b"mutate", timeout=30)
            )
            try:
                await entered.wait()
                pool = next(iter(owner._engine.pools.values()))
                tcp = next(io.handler for io in pool.transports.values())
                callbacks = [task for task in pool.hooks if not task.done()]
                assert callbacks and target.trace == []
                if stop == "cancel":
                    call.cancel()
                else:
                    probe.scopes[-1].reschedule(loop.time())
                await cleaning.wait()
                if repeated:
                    call.cancel()
                    await asyncio.sleep(0)
                    call.cancel()
                assert not call.done() and not cleaned.is_set() and not finished.is_set()
                release.set()
                if stop == "cancel" or repeated:
                    with pytest.raises(asyncio.CancelledError):
                        await call
                else:
                    outcome, response = await call
                    assert (
                        outcome.kind == "failed"
                        and outcome.reason == "timeout"
                        and response is None
                    )
                assert finished.is_set() and all(task.done() for task in callbacks)
                assert tcp.done() and not pool.transports and not pool.connecting
                assert (
                    not pool.exchanges and not pool.layer.streams and not pool.layer.command_sources
                )
                assert not owner._engine.flows and target.trace == []
                outcome, response = await owner.request(target.origin + "/allowed")
                assert outcome.kind == "completed" and response.body == b"ok"
                assert [row[0] for row in target.trace] == [b"GET"]
                assert owner._request_count == 2
            finally:
                release.set()
                await asyncio.gather(call, return_exceptions=True)

    try:
        run(scenario)
    finally:
        release.set()


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_cancelled_tcp_dial_cannot_connect_after_call_returns(tmp_path, monkeypatch, stop):
    from contractor_runtime.adapters import http_transport

    probe = TimeoutProbe()
    monkeypatch.setattr(http_transport, "asyncio", probe)
    original = asyncio.open_connection

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        entered, resume, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
        first = True
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):

            async def connect(host, port, **options):
                nonlocal first
                if port == target.port and first:
                    first = False
                    entered.set()
                    try:
                        await resume.wait()
                    finally:
                        exited.set()
                return await original(host, port, **options)

            monkeypatch.setattr(asyncio, "open_connection", connect)
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.open_proxy()
            call = asyncio.create_task(
                owner.request(target.origin + "/allowed", method="POST", body=b"mutate", timeout=30)
            )
            try:
                await entered.wait()
                pool = next(iter(owner._engine.pools.values()))
                tcp = next(io.handler for io in pool.transports.values())
                if stop == "cancel":
                    call.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await call
                else:
                    probe.scopes[-1].reschedule(asyncio.get_running_loop().time())
                    outcome, _ = await call
                    assert outcome.reason == "timeout"
                assert exited.is_set() and tcp.done() and not pool.transports
                assert not pool.hooks and not pool.exchanges and not pool.layer.streams
                assert not pool.layer.command_sources and target.trace == []
                resume.set()
                assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
                assert [row[0] for row in target.trace] == [b"GET"]
            finally:
                resume.set()
                await asyncio.gather(call, return_exceptions=True)

    run(scenario)


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
@pytest.mark.parametrize("state", ["missing", "closing"])
def test_native_encoders_and_command_filter_drop_unowned_request_messages(tmp_path, http2, state):
    from mitmproxy import http
    from mitmproxy.proxy import commands
    from mitmproxy.proxy.layers import http as native_http

    from contractor_runtime.adapters.bridge.mitmproxy.messages import request_message
    from contractor_runtime.adapters.bridge.mitmproxy.protocol import Child
    from contractor_runtime.adapters.bridge.mitmproxy.state import Exchange
    from contractor_runtime.adapters.http_session import PrivateRequest
    from contractor_runtime.adapters.http_stream import RequestBoundary

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            pool = next(iter(owner._engine.pools.values()))
            connection, parser = next(iter(pool.parsers.items()))
            request = PrivateRequest(
                [(b"Host", f"target.example:{target.port}".encode())], "POST", b"mutate"
            )
            exchange = Exchange(
                target.origin + "/allowed",
                RequestBoundary(
                    time.monotonic() + 60, authority_deadline=lambda: time.monotonic() + 60
                ),
                request,
                (),
                100,
                asyncio.get_running_loop().create_future(),
                stream_id=999,
                closing=True,
            )
            if state == "closing":
                pool.exchanges[999] = exchange
            messages = [
                native_http.RequestHeaders(999, request_message(exchange.url, request), False),
                native_http.RequestData(999, b"mutate"),
                native_http.RequestTrailers(999, http.Headers([(b"x-trailer", b"v")])),
                native_http.RequestEndOfMessage(999),
            ]
            try:
                # Gate the actual native encoders independently of the parent
                # filter, including late DATA/END events, without emitting bytes.
                for message in messages:
                    assert list(parser.handle_event(message)) == []

                class Emit:
                    def __init__(self, command):
                        self.command = command

                    def handle_event(self, _):
                        yield self.command

                for message in messages:
                    assert (
                        list(
                            Child(
                                Emit(native_http.SendHttp(message, connection)), pool
                            ).handle_event(None)
                        )
                        == []
                    )
                assert (
                    list(Child(Emit(commands.OpenConnection(connection)), pool).handle_event(None))
                    == []
                )
                assert len(target.trace) == 1
            finally:
                pool.exchanges.pop(999, None)
                exchange.future.cancel()
                exchange.clear()
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            assert len(target.connections) == 1 and len(target.trace) == 2

    run(scenario)


def test_failed_establishment_close_still_joins_tls_and_retains_cleanup_ownership(
    tmp_path, monkeypatch
):
    from mitmproxy.addons.tlsconfig import TlsConfig

    from contractor_runtime.adapters.bridge.mitmproxy.engine import _Pool

    release = threading.Event()
    original_tls, original_close = TlsConfig.tls_start_server, _Pool.close_writer

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        loop = asyncio.get_running_loop()
        entered, refused = asyncio.Event(), asyncio.Event()
        refuse = True

        def held(config, data):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "TLS test gate was not released"
            return original_tls(config, data)

        async def close(pool, writer):
            if refuse:
                refused.set()
                raise RuntimeError("injected establishment close refusal")
            await original_close(pool, writer)

        monkeypatch.setattr(TlsConfig, "tls_start_server", held)
        monkeypatch.setattr(_Pool, "close_writer", close)
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            call = asyncio.create_task(
                owner.request(target.origin + "/allowed", method="POST", body=b"mutate")
            )
            try:
                await entered.wait()
                pool = next(iter(owner._engine.pools.values()))
                exchange = next(iter(pool.exchanges.values()))
                future = exchange.future
                call.cancel()
                await refused.wait()
                assert not call.done()
                release.set()
                with pytest.raises(RuntimeError, match="injected establishment close refusal"):
                    await call
                assert all(task.done() for task in exchange.hooks)
                assert exchange.closing and pool.exchanges and target.trace == []
                refuse = False
                await owner.close()
                assert not pool.exchanges and not pool.writers
                assert not future._log_traceback
                assert exchange.request is exchange.flow is exchange.future is None
            finally:
                release.set()
                refuse = False
                await asyncio.gather(call, return_exceptions=True)

    try:
        run(scenario)
    finally:
        release.set()
