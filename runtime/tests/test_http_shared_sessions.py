"""Production-owner isolation matrix through ordinary HTTP and scanner ingress."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


def matrix_module():
    path = Path(__file__).parents[2] / "scripts/experiments/mitmproxy_shared_sessions.py"
    spec = importlib.util.spec_from_file_location("shared_http_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_native_engine_keeps_allocation_sessions_isolated():
    report = asyncio.run(matrix_module().assess())
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
