"""Production-owner isolation matrix through ordinary HTTP and scanner ingress."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path


def matrix_module():
    path = Path(__file__).parents[2] / "scripts/experiments/mitmproxy_shared_sessions.py"
    spec = importlib.util.spec_from_file_location("shared_http_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_native_engine_keeps_allocation_sessions_isolated():
    original = os.umask(0)
    os.umask(original)
    try:
        report = asyncio.run(matrix_module().assess())
    finally:
        restored = os.umask(original)
    assert restored == original
    assert report["verdict"] == "supported"
    assert len(report["matrix"]) == 9
    assert all(case["native_master_creations"] == 1 for case in report["matrix"].values())
    assert report["matrix"]["churn"]["sessions"] == 32


def test_shared_engine_fault_fences_every_session_and_retains_failed_cleanup(monkeypatch):
    from contextlib import AsyncExitStack

    import pytest
    from fakes.engagement_socket import SocketTarget

    from contractor_runtime.adapters.bridge.errors import BridgeInfrastructureError

    module = matrix_module()

    async def scenario():
        async with SocketTarget() as target, AsyncExitStack() as stack:
            engine = await module.open_engine(None)
            stack.push_async_callback(engine.close)
            owners = [await module.owner_for(stack, engine, target, None, str(i)) for i in range(4)]
            for owner in owners:
                assert (await module.call(owner, target))[0].kind == "completed"
            sessions = [owner._engine for owner in owners]
            directory = sessions[0].directory
            cleanup = directory.cleanup
            failed = True

            def remove_files():
                if failed:
                    raise OSError("injected session cleanup failure")
                cleanup()

            monkeypatch.setattr(directory, "cleanup", remove_files)
            engine._failed = True
            for owner in owners:
                with pytest.raises(BridgeInfrastructureError):
                    await module.call(owner, target)
            assert len(target.requests) == 4
            with pytest.raises(OSError, match="injected session"):
                await owners[0].close()
            assert sessions[0] in engine.sessions and sessions[0].directory is directory
            failed = False
            await owners[0].close()
            assert sessions[0] not in engine.sessions
            await engine.close()
            assert not engine.sessions and all(session.closed for session in sessions)
            assert all(session.directory is None for session in sessions)

    asyncio.run(scenario())


def test_factory_startup_probe_guards_version_and_shares_one_lazy_engine(monkeypatch):
    import time

    import pytest
    from fakes.engagement_socket import SocketTarget, resolver_for, socket_spec

    from contractor_runtime.adapters.bridge.mitmproxy import runtime
    from contractor_runtime.adapters.http_transport import HTTPTransportFactory

    async def scenario():
        factory = HTTPTransportFactory()
        version = runtime.importlib.metadata.version
        monkeypatch.setattr(runtime.importlib.metadata, "version", lambda _: "unsupported")
        with pytest.raises(RuntimeError, match="unsupported native HTTP engine"):
            await factory.probe()
        assert factory._engine is None
        monkeypatch.setattr(runtime.importlib.metadata, "version", version)
        assert await factory.probe()
        assert factory._engine.master is None and runtime._active is None
        async with SocketTarget() as target:
            factory._resolver = resolver_for(target).resolver
            owners = [
                factory.create(
                    socket_spec(target).model_copy(
                        update={"allocation_id": f"allocation-{i}"},
                        deep=True,
                    )
                )
                for i in range(4)
            ]
            try:
                for owner in owners:
                    owner.configure(authority_deadline=lambda: time.monotonic() + 60)
                results = await asyncio.gather(
                    *(owner.request(target.origin + "/allowed") for owner in owners)
                )
                assert all(outcome.kind == "completed" for outcome, _ in results)
                assert len(factory._engine.sessions) == 4
                assert all(owner._engine.runtime is factory._engine for owner in owners)
                await owners[0].close()
                assert (await owners[1].request(target.origin + "/allowed"))[0].kind == "completed"
                assert len(factory._engine.sessions) == 3
            finally:
                await asyncio.gather(*(owner.close() for owner in owners))
                await factory.close()
        assert factory._engine.master is None and not factory._engine.sessions
        assert runtime._active is None

    asyncio.run(scenario())


def test_shared_tls_configuration_joins_before_next_session(monkeypatch):
    from types import SimpleNamespace

    from contractor_runtime.adapters.bridge.mitmproxy import runtime

    async def scenario():
        engine = runtime.MitmproxyEngine()
        entered, release, attempted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active = maximum = calls = 0

        async def configure_in_thread(configure, keys, *, name):
            nonlocal active, maximum
            assert name == "http-session-tls" and keys == {"owner"}
            active += 1
            maximum = max(maximum, active)
            entered.set()
            await release.wait()
            configure(keys)
            active -= 1

        monkeypatch.setattr(runtime, "to_thread_until_done", configure_in_thread)

        async def session(owner):
            nonlocal calls
            calls += 1
            if calls == 4:
                attempted.set()

            def configure(keys):
                assert runtime._options.get() == {"owner": owner}

            await engine.configure_tls(SimpleNamespace(configure=configure), {"owner": owner})

        tasks = [asyncio.create_task(session(owner)) for owner in range(4)]
        try:
            await entered.wait()
            await attempted.wait()
            release.set()
            await asyncio.gather(*tasks)
            assert maximum == 1
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await engine.close()

    asyncio.run(scenario())
