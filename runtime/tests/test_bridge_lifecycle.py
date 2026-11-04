"""Native resource ownership, retry and partial-write behavior on real sockets."""

import asyncio
import logging
import time
from pathlib import Path

import pytest
from fakes.engagement_socket import SocketTarget, preflight


def run(scenario):
    async def bounded():
        async with asyncio.timeout(15):
            await scenario()

    asyncio.run(bounded())


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"])
def test_native_partial_application_write_is_never_replayed(monkeypatch, method):
    from contractor_runtime.adapters.bridge.mitmproxy.engine import _Writer

    original = _Writer.write
    sent = []

    def partial(writer, data):
        sent.append(data[:16])
        original(writer, data[:16])
        raise OSError("injected partial application write")

    async def scenario():
        monkeypatch.setattr(_Writer, "write", partial)
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, response = await owner.request(target.origin + "/allowed", method=method)
            await target.first_byte.wait()
            assert outcome.kind == "failed" and response is None
            assert owner._request_count == 1 and len(target.connections) == 1
            assert target.written == b"".join(sent) and len(sent) == 1
            assert not owner._engine.flows

    run(scenario)


@pytest.mark.parametrize("method,attempts", [("GET", 2), ("HEAD", 2), ("OPTIONS", 2), ("POST", 1)])
def test_native_presend_connect_failure_retry_is_method_bounded(monkeypatch, method, attempts):
    original = asyncio.open_connection
    dials = []

    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):

            async def refuse(host, port, **options):
                if port == target.port:
                    dials.append((host, port))
                    raise ConnectionRefusedError("injected connect failure")
                return await original(host, port, **options)

            monkeypatch.setattr(asyncio, "open_connection", refuse)
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            outcome, _ = await owner.request(target.origin + "/allowed", method=method)
            assert outcome.reason == "connect_failed"
            assert len(dials) == owner._request_count == attempts
            assert target.written == b""

    run(scenario)


@pytest.mark.parametrize("failure", ["writer", "files"])
def test_failed_native_close_keeps_ownership_until_retry(monkeypatch, failure):
    from mitmproxy import ctx
    from mitmproxy.net import tls

    from contractor_runtime.adapters.bridge.mitmproxy import engine as module

    previous = {key: getattr(ctx, key, None) for key in ("master", "options", "log")}
    logger = logging.getLogger("mitmproxy")
    handlers, propagate, keylog = logger.handlers[:], logger.propagate, tls.log_master_secret
    rejected = True

    async def scenario():
        nonlocal rejected
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            assert (await owner.request(target.origin + "/allowed"))[0].kind == "completed"
            engine = owner._engine
            directory = Path(engine.directory.name)
            if failure == "writer":
                pool = next(iter(engine.pools.values()))
                writer = next(io.writer.inner for io in pool.transports.values() if io.writer)
                original = writer.wait_closed

                async def wait_closed():
                    if rejected:
                        raise RuntimeError("injected writer close failure")
                    await original()

                monkeypatch.setattr(writer, "wait_closed", wait_closed)
            else:
                original = engine.directory.cleanup

                def cleanup():
                    if rejected:
                        raise RuntimeError("injected file cleanup failure")
                    original()

                monkeypatch.setattr(engine.directory, "cleanup", cleanup)
            with pytest.raises(RuntimeError, match=r"injected .* failure"):
                await owner.close()
            assert owner._engine is engine and module._active is engine and not engine.closed
            assert directory.exists() and tls.log_master_secret is None
            contender = module.MitmproxyEngine(require_authority=lambda: None)
            with pytest.raises(RuntimeError, match="already belongs"):
                await contender.start()
            await contender.close()
            assert module._active is engine and tls.log_master_secret is None
            rejected = False
            await owner.close()
            assert engine.closed and module._active is None and not directory.exists()
            assert logger.handlers == handlers and logger.propagate == propagate
            assert tls.log_master_secret is keylog
            assert all(getattr(ctx, key, None) is value for key, value in previous.items())
            replacement = module.MitmproxyEngine(require_authority=lambda: None)
            await replacement.start()
            await replacement.close()

    run(scenario)


def test_repeated_cancellation_joins_native_close_before_restoring_state(monkeypatch):
    from contractor_runtime.adapters.bridge.mitmproxy import engine as module

    async def scenario():
        async with SocketTarget() as target, preflight(target) as (owner, _):
            owner.configure(authority_deadline=lambda: time.monotonic() + 60)
            await owner.request(target.origin + "/allowed")
            engine = owner._engine
            pool = next(iter(engine.pools.values()))
            original = pool.close
            entered, release = asyncio.Event(), asyncio.Event()

            async def held():
                entered.set()
                await release.wait()
                await original()

            monkeypatch.setattr(pool, "close", held)
            closing = asyncio.create_task(engine.close())
            await entered.wait()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not engine.closed and module._active is engine
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert engine.closed and module._active is None
            assert not engine.pools and not engine.flows

    run(scenario)
