"""Connection-local native timer retirement on actual TLS/HTTP/2 sockets."""

import asyncio
import gc
import time
import weakref

import pytest
from fakes.engagement_socket import preflight, tls_contexts
from fakes.native_http import NativeTarget


def run(scenario):
    async def bounded():
        async with asyncio.timeout(30):
            await scenario()

    asyncio.run(bounded())


def observe_connections(monkeypatch):
    from contractor_runtime.adapters.bridge.mitmproxy.engine import MitmproxyEngine

    original = MitmproxyEngine.handle_hook
    opened, parsers = [], []

    async def observe(engine, pool, hook):
        await original(engine, pool, hook)
        if hook.name == "server_connected":
            connection = hook.args()[0].server
            opened.append((connection, pool.transports[connection].handler))
        elif hook.name == "responseheaders":
            exchange = engine.flows[hook.args()[0].id]
            parsers.append(weakref.ref(pool.parsers[exchange.connection]))

    monkeypatch.setattr(MitmproxyEngine, "handle_hook", observe)
    return opened, parsers


@pytest.mark.parametrize("http2", [False, True], ids=["http11", "http2"])
def test_closed_target_connections_release_timers_and_protocol_layers(tmp_path, monkeypatch, http2):
    opened, parsers = observe_connections(monkeypatch)

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if http2 else ["http/1.1"])
        async with (
            NativeTarget(ssl_context=server, close_paths=[b"/rotate"]) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            counts = []
            for _ in range(50):
                outcome, response = await owner.request(target.origin + "/rotate")
                assert outcome.kind == "completed" and response.body == b"ok"
                await asyncio.gather(opened[-1][1], return_exceptions=True)
                pool = next(iter(owner._engine.pools.values()))
                counts.append(
                    (
                        len(pool.transports),
                        len(owner._engine.flows),
                        len(pool.wakeup_timer),
                        len(pool.layer.command_sources),
                        len(pool._connection_tasks),
                        len(pool._writer_closes),
                    )
                )
            assert len(target.connections) == len(opened) == 50
            assert counts == [(0, 0, 0, 0, 0, 0)] * 50
            assert not pool.parsers and not pool.connecting and not pool.exchanges
            assert set(pool.layer.connections) == {pool.client}
            # Completed cancelled tasks may retain their cancellation traceback.
            # Drop the observer's handles before checking engine reachability.
            opened.clear()
            await asyncio.sleep(0)
            gc.collect()
            assert len(parsers) == 50 and all(reference() is None for reference in parsers)

    run(scenario)


def test_retiring_one_http2_connection_keeps_another_stream_and_keepalive(tmp_path, monkeypatch):
    opened, _ = observe_connections(monkeypatch)
    original = asyncio.open_connection

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        async with (
            NativeTarget(ssl_context=server, close_paths=[b"/rotate"]) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            dials, both = [], asyncio.Event()

            async def connect(host, port, **options):
                if port == target.port:
                    dials.append(True)
                    if len(dials) == 2:
                        both.set()
                    await both.wait()
                return await original(host, port, **options)

            monkeypatch.setattr(asyncio, "open_connection", connect)
            held = asyncio.create_task(owner.request(target.origin + "/hold"))
            rotated = asyncio.create_task(owner.request(target.origin + "/rotate"))
            await target.held.wait()
            outcome, response = await rotated
            assert outcome.kind == "completed" and response.body == b"ok"
            pool = next(iter(owner._engine.pools.values()))
            closed = next(handler for connection, handler in opened if not connection.connected)
            await asyncio.gather(closed, return_exceptions=True)
            assert not held.done() and len(pool.transports) == 1
            assert len(pool.wakeup_timer) == 1
            from mitmproxy.proxy import commands

            timers = [
                command
                for command in pool.layer.command_sources
                if isinstance(command, commands.RequestWakeup)
            ]
            assert len(timers) == 1
            assert pool.layer.command_sources[timers[0]].context.server in pool.transports
            target.release.set()
            assert (await held)[0].kind == "completed"
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            assert len(target.connections) == 2 and len(pool.wakeup_timer) == 1

    run(scenario)


@pytest.mark.parametrize("phase", ["pending", "dispatch"])
def test_connection_retirement_joins_wakeup_before_start_or_during_dispatch(
    tmp_path, monkeypatch, phase
):
    from mitmproxy.proxy import commands, events

    opened, _ = observe_connections(monkeypatch)

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            pool = next(iter(owner._engine.pools.values()))
            connection, handler = opened[0]
            source = next(
                source
                for command, source in pool.layer.command_sources.items()
                if isinstance(command, commands.RequestWakeup)
            )
            request = commands.RequestWakeup(0 if phase == "dispatch" else 58)
            pool.layer.command_sources[request] = source
            entered, finished = asyncio.Event(), asyncio.Event()
            original = pool.server_event

            async def dispatch(event):
                if isinstance(event, events.Wakeup) and event.command is request:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        finished.set()
                await original(event)

            monkeypatch.setattr(pool, "server_event", dispatch)
            coroutine = pool.wakeup(request)
            record = pool._wakeups[coroutine]
            timer = asyncio.create_task(coroutine)
            pool.wakeup_timer.add(timer)
            if phase == "dispatch":
                await entered.wait()
                assert timer not in pool.wakeup_timer and record.task is timer
            pool.close_connection(connection)
            await pool._retire_timers(connection)
            await asyncio.gather(handler, return_exceptions=True)
            assert timer.cancelled() and not pool.wakeup_timer and not pool._wakeups
            assert not pool.layer.command_sources and not pool.transports
            if phase == "pending":
                assert record.task is None and not entered.is_set()
            else:
                assert finished.is_set()

    run(scenario)


def test_connection_cleanup_joins_retirement_through_repeated_cancellation(tmp_path, monkeypatch):
    opened, _ = observe_connections(monkeypatch)

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            pool = next(iter(owner._engine.pools.values()))
            connection, handler = opened[0]
            entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
            original = pool._retire_timers

            async def held(connection):
                await original(connection)
                entered.set()
                await release.wait()
                finished.set()

            monkeypatch.setattr(pool, "_retire_timers", held)
            pool.close_connection(connection)
            await entered.wait()
            handler.cancel()
            await asyncio.sleep(0)
            handler.cancel()
            await asyncio.sleep(0)
            assert not handler.done() and not finished.is_set()
            release.set()
            await asyncio.gather(handler, return_exceptions=True)
            assert finished.is_set() and not pool._wakeups and not pool.layer.command_sources
            assert not pool.transports and not pool.parsers

    run(scenario)


def test_live_native_wakeup_dispatch_reschedules_and_releases_its_old_route(tmp_path, monkeypatch):
    from mitmproxy.proxy import commands

    opened, _ = observe_connections(monkeypatch)

    async def scenario():
        server, trust, _ = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"])
        async with (
            NativeTarget(ssl_context=server) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            pool = next(iter(owner._engine.pools.values()))
            connection, handler = opened[0]
            source = next(
                source
                for command, source in pool.layer.command_sources.items()
                if isinstance(command, commands.RequestWakeup)
            )
            request = commands.RequestWakeup(0)
            pool.layer.command_sources[request] = source
            timer = asyncio.create_task(pool.wakeup(request))
            pool.wakeup_timer.add(timer)
            await timer
            assert request not in pool.layer.command_sources
            assert owner._engine.options.http2_ping_keepalive > 0
            assert len(pool._wakeups) == len(pool.wakeup_timer) == 2
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            assert len(target.connections) == 1
            pool.close_connection(connection)
            await asyncio.gather(handler, return_exceptions=True)
            assert not pool._wakeups and not pool.layer.command_sources and not pool.wakeup_timer

    run(scenario)
