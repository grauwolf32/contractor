"""A closed peer reset is a request outcome, not unconfirmed resource cleanup."""

import asyncio
import gc
import time
from pathlib import Path

import pytest
from fakes.bridge_reset import EarlyDenial, ResetTarget
from fakes.engagement_socket import preflight, tls_contexts


def run(scenario):
    async def bounded():
        async with asyncio.timeout(20):
            await scenario()

    asyncio.run(bounded())


def assert_erased(engine, pool, directory):

    from contractor_runtime.adapters.bridge.mitmproxy import runtime

    assert engine.closed and runtime._active is None and not directory.exists()
    assert not engine.pools and not engine.flows and not engine.ingresses
    assert not pool.writers and not pool.transports and not pool.parsers
    assert not pool._connection_tasks and not pool._writer_closes
    assert not pool.hooks and not pool.wakeup_timer and not pool._wakeups
    assert not pool.connecting and not pool.verified and not pool.exchanges
    assert not pool.layer.connections and not pool.layer.command_sources
    assert engine.options is engine.tls is engine.ssl_context is None


def test_early_404_with_unread_post_body_closes_and_reuses_process():
    async def scenario():
        failures = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: failures.append(context))
        for _ in range(2):
            async with EarlyDenial() as target, preflight(target) as (owner, _):
                owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                outcome, response = await owner.request(
                    target.origin + "/allowed", method="POST", body=b'{"amount":1}'
                )
                assert outcome.kind == "completed" and response.status == 404
                assert response.body == b"{}"
                await target.task
                assert target.unread_body and owner._request_count == 1
                engine = owner._engine
                pool = next(iter(engine.pools.values()))
                directory = Path(engine.directory.name)
                await asyncio.gather(owner.close(), owner.close())
                await owner.close()
                assert_erased(engine, pool, directory)
        gc.collect()
        assert failures == []

    run(scenario)


@pytest.mark.parametrize("protocol", ["http", "tls", "h2"])
@pytest.mark.parametrize("partial", [False, True], ids=["complete", "partial"])
def test_target_tcp_reset_preserves_response_outcome_and_never_replays(
    tmp_path, monkeypatch, protocol, partial
):
    async def scenario():
        from contractor_runtime.adapters.bridge.mitmproxy import engine as module

        server, trust, names = tls_contexts(tmp_path)
        server.set_alpn_protocols(["h2"] if protocol == "h2" else ["http/1.1"])
        headers = asyncio.Event()
        handlers = []
        original = module.MitmproxySession.handle_hook

        async def observe(engine, pool, hook):
            await original(engine, pool, hook)
            if hook.name == "server_connected":
                handlers.append(pool.transports[hook.args()[0].server].handler)
            if hook.name == "responseheaders":
                headers.set()

        monkeypatch.setattr(module.MitmproxySession, "handle_hook", observe)
        async with (
            ResetTarget(
                ssl_context=server if protocol != "http" else None, partial=partial
            ) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            request = asyncio.create_task(owner.request(target.origin + "/allowed"))
            await headers.wait()
            if not partial:
                outcome, response = await request
                assert outcome.kind == "completed" and response.status == 404
                assert response.body == b"nope"
            target.reset.set()
            outcome, response = await request
            if partial:
                assert outcome.kind == "failed"
                assert outcome.reason in {"response_invalid", "transport_failed"}
                assert outcome.request_may_have_been_sent
            results = await asyncio.gather(*handlers, return_exceptions=True)
            assert all(
                value is None or isinstance(value, asyncio.CancelledError) for value in results
            )
            assert len(target.requests) == owner._request_count == 1
            if protocol != "http":
                assert names == ["target.example"]
            engine = owner._engine
            pool = next(iter(engine.pools.values()))
            directory = Path(engine.directory.name)
            await owner.close()
            assert_erased(engine, pool, directory)

    run(scenario)


@pytest.mark.parametrize("failure", [ConnectionResetError, BrokenPipeError, RuntimeError, OSError])
@pytest.mark.parametrize("closed", [False, True], ids=["live", "closed"])
def test_close_error_requires_both_expected_error_and_closed_descriptor(
    monkeypatch, failure, closed
):
    from fakes.engagement_socket import SocketTarget

    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            pool = next(iter(owner._engine.pools.values()))
            writer = next(io.writer for io in pool.transports.values() if io.writer)
            pool.writers.add(writer)
            if closed:
                writer.close()
                await writer.wait_closed()

            async def rejected():
                raise failure("injected close failure")

            with monkeypatch.context() as patch:
                if not closed:
                    patch.setattr(writer.inner, "close", lambda: None)
                patch.setattr(writer.inner, "wait_closed", rejected)
                if closed and failure in (ConnectionResetError, BrokenPipeError):
                    await pool.close_writer(writer)
                    assert writer not in pool.writers
                else:
                    with pytest.raises(failure):
                        await pool.close_writer(writer)
                    assert writer in pool.writers
            await owner.close()
            assert not pool.writers

    run(scenario)


def test_detached_native_failure_is_observed_reported_and_cleanup_can_retry(monkeypatch):
    from contractor_runtime.adapters.bridge.mitmproxy import runtime

    async def scenario():

        failures = []
        asyncio.get_running_loop().set_exception_handler(lambda _, value: failures.append(value))
        async with ResetTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            engine = owner._engine
            pool = next(iter(engine.pools.values()))
            handler = next(io.handler for io in pool.transports.values())
            finished = asyncio.Event()
            handler.add_done_callback(lambda _: finished.set())
            original = pool._release_connection

            async def broken(connection):
                await original(connection)
                raise RuntimeError("injected native retirement defect")

            monkeypatch.setattr(pool, "_release_connection", broken)
            target.reset.set()
            await finished.wait()
            assert not pool.transports and handler in pool._connection_tasks
            assert not handler._log_traceback
            with pytest.raises(RuntimeError, match="injected native retirement defect"):
                await owner.close()
            assert runtime._active is engine.runtime and not engine.closed
            assert not pool._connection_tasks
            await owner.close()
            gc.collect()
            assert failures == []

    run(scenario)


@pytest.mark.parametrize("protocol", ["http", "tls", "h2"])
@pytest.mark.parametrize("during", [False, True], ids=["after-response", "during-response"])
def test_scanner_reset_joins_private_ingress(tmp_path, protocol, during):
    import ssl
    from contextlib import suppress

    import h2.config
    import h2.connection
    import h2.events
    from fakes.bridge_reset import reset_socket
    from fakes.engagement_socket import SocketTarget

    from contractor_runtime.adapters.bridge import HTTPProxyBridge
    from contractor_runtime.adapters.bridge.mitmproxy import runtime

    async def scenario():

        release = asyncio.Event()

        async def respond(target, reader, writer, message):
            await release.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            await writer.drain()

        server, trust, _ = tls_contexts(tmp_path)
        async with (
            SocketTarget(respond, ssl_context=server if protocol != "http" else None) as target,
            preflight(target, ssl_context=trust) as (owner, _),
        ):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 15
            ) as bridge:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", bridge._server.sockets[0].getsockname()[1]
                )
                try:
                    authority = f"target.example:{target.port}".encode()
                    if protocol != "http":
                        writer.write(
                            b"CONNECT "
                            + authority
                            + b" HTTP/1.1\r\nHost: "
                            + authority
                            + b"\r\nProxy-Authorization: "
                            + bridge._authorization
                            + b"\r\n\r\n"
                        )
                        await writer.drain()
                        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
                        # Trust the bridge's actual per-allocation CA.
                        tls = ssl.create_default_context(
                            cafile=Path(bridge._engine.directory.name) / "mitmproxy-ca-cert.pem"
                        )
                        tls.set_alpn_protocols(["h2" if protocol == "h2" else "http/1.1"])
                        await writer.start_tls(tls, server_hostname="target.example")
                    if protocol == "h2":
                        connection = h2.connection.H2Connection(
                            h2.config.H2Configuration(client_side=True)
                        )
                        connection.initiate_connection()
                        connection.send_headers(
                            1,
                            [
                                (b":method", b"GET"),
                                (b":scheme", b"https"),
                                (b":authority", authority),
                                (b":path", b"/allowed"),
                            ],
                            end_stream=True,
                        )
                        writer.write(connection.data_to_send())
                    else:
                        path = (
                            target.origin.encode() + b"/allowed"
                            if protocol == "http"
                            else b"/allowed"
                        )
                        writer.write(
                            b"GET "
                            + path
                            + b" HTTP/1.1\r\nHost: "
                            + authority
                            + b"\r\nProxy-Authorization: "
                            + bridge._authorization
                            + b"\r\n\r\n"
                        )
                    await writer.drain()
                    await target.request_received.wait()
                    handlers = tuple(bridge._handlers)
                    tasks = tuple(bridge._tasks)
                    if not during:
                        release.set()
                        if protocol == "h2":
                            ended = False
                            while not ended:
                                data = await reader.read(65536)
                                assert data
                                ended = any(
                                    isinstance(event, h2.events.StreamEnded)
                                    for event in connection.receive_data(data)
                                )
                        else:
                            assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
                            assert await reader.readexactly(2) == b"ok"
                    reset_socket(writer)
                    release.set()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    assert all(
                        result is None or isinstance(result, asyncio.CancelledError)
                        for result in results
                    )
                    assert all(
                        not handler.writers and not handler._connection_tasks
                        for handler in handlers
                    )
                finally:
                    release.set()
                    writer.close()
                    with suppress(OSError):
                        await writer.wait_closed()
            assert not bridge._handlers and not bridge._writers and not bridge._tasks
            assert not bridge._private_values and bridge._authorization == b""
            assert owner._request_count == 1
            await owner.close()
            assert runtime._active is None

    run(scenario)


def test_reset_close_is_joined_through_concurrent_repeated_cancellation(monkeypatch):
    from contractor_runtime.adapters.bridge.mitmproxy import runtime

    async def scenario():

        async with ResetTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            engine = owner._engine
            pool = next(iter(engine.pools.values()))
            writer = next(io.writer for io in pool.transports.values() if io.writer)
            original = writer.wait_closed
            entered, finish = asyncio.Event(), asyncio.Event()
            waits = []

            async def held():
                waits.append(True)
                entered.set()
                await finish.wait()
                await original()

            monkeypatch.setattr(writer.inner, "wait_closed", held)
            target.reset.set()
            await entered.wait()
            closing = asyncio.create_task(engine.close())
            concurrent = asyncio.create_task(pool.close_writer(writer))
            try:
                for _ in range(2):
                    await asyncio.sleep(0)
                    closing.cancel()
                    concurrent.cancel()
                await asyncio.sleep(0)
                assert not closing.done() and not concurrent.done()
                assert runtime._active is engine.runtime and writer in pool.writers
                assert not engine.closed and waits == [True]
            finally:
                finish.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(asyncio.CancelledError):
                await concurrent
            assert waits == [True]
            assert engine.closed and not pool.writers and not pool._writer_closes
            await owner.close()

    run(scenario)


def test_scanner_reset_before_handler_start_retires_the_exact_writer(monkeypatch):
    from contextlib import suppress

    from fakes.bridge_reset import reset_socket
    from fakes.engagement_socket import SocketTarget

    from contractor_runtime.adapters.bridge import HTTPProxyBridge

    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            async with HTTPProxyBridge(
                owner, credential_id=None, deadline=time.monotonic() + 15
            ) as bridge:
                entered = asyncio.Event()

                async def before_start(handler, writer):
                    entered.set()
                    await asyncio.Event().wait()

                monkeypatch.setattr(bridge, "_handle", before_start)
                _, writer = await asyncio.open_connection(
                    "127.0.0.1", bridge._server.sockets[0].getsockname()[1]
                )
                try:
                    await entered.wait()
                    handler = next(iter(bridge._handlers))
                    io = handler.transports[handler.client]
                    reset_socket(writer)
                    # Observe actual kernel reset before cancellation; no native
                    # handler has run or taken responsibility for this socket yet.
                    with pytest.raises(ConnectionResetError):
                        await io.reader.read(1)
                    await bridge.__aexit__()
                    assert not bridge._writers and not bridge._handlers and not bridge._tasks
                    assert not handler.writers and not handler.transports
                    assert not target.connections and owner._request_count == 0
                finally:
                    writer.close()
                    with suppress(OSError):
                        await writer.wait_closed()
            await owner.close()

    run(scenario)
