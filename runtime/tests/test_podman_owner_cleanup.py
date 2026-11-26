"""Exercise owner EOF and backend teardown with exact, observable removals."""

import asyncio
import socket
import time
from pathlib import Path

import pytest

from contractor_runtime.sandbox.podman.lifecycle_backend import Entry, LifecycleBackend
from contractor_runtime.sandbox.podman.owner import send_rpc, serve_owner
from contractor_runtime.sandbox.podman.settings import PodmanSettings


class Engine:
    def __init__(self):
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()
        self.removals = []

    async def discover(self, **kwargs):
        return ()

    async def confirm_removed(self, allocation_id, **kwargs):
        self.removals.append(allocation_id)
        self.entered.set()
        await self.resume.wait()

    async def close(self, **kwargs):
        pass


def test_health_eof_joins_inflight_close_before_cleanup_retry():
    async def scenario():
        engine = Engine()
        backend = LifecycleBackend(PodmanSettings(), engine=engine)
        backend.opened = True
        backend.entries["allocation-review"] = Entry(
            "allocation-review", Path("/unused-review"), stopped=True
        )
        close = backend.close
        disconnect = backend.disconnect
        closing, errors = [], []
        disconnected = asyncio.Event()
        disconnects = 0

        async def observed_close(**kwargs):
            closing.append(asyncio.current_task())
            try:
                await close(**kwargs)
            except Exception as error:
                errors.append(type(error).__name__)
                raise

        def observed_disconnect():
            nonlocal disconnects
            disconnects += 1
            if disconnects == 3:
                disconnected.set()
            disconnect()

        backend.close = observed_close
        backend.disconnect = observed_disconnect
        control, child_control = socket.socketpair()
        # This test only uses health EOF, which stream sockets also expose on macOS.
        health, child_health = socket.socketpair()
        for channel in (control, child_control, health, child_health):
            channel.setblocking(False)
        owner = asyncio.create_task(serve_owner(child_control, child_health, backend, ()))
        try:
            await send_rpc(control, {"op": "close", "deadline": time.monotonic() + 10})
            await engine.entered.wait()
            health.close()
            await disconnected.wait()
            # Let the cancelled channel workers and their completion callbacks run.
            for _ in range(8):
                await asyncio.sleep(0)
        finally:
            engine.resume.set()
            await owner
            await asyncio.gather(*closing, return_exceptions=True)
            control.close()
            health.close()
        assert len(closing) == 1
        assert engine.removals == ["allocation-review"]
        assert not errors and backend.closed and not backend.entries

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_backend_close_joins_other_removals_after_failure(cancel):
    async def scenario():
        engine = Engine()
        backend = LifecycleBackend(PodmanSettings(), engine=engine)
        for identity in ("allocation-failing", "allocation-blocked"):
            backend.entries[identity] = Entry(identity, Path("/unused-review"), stopped=True)
        first_settled = asyncio.Event()

        async def remove(identity, **kwargs):
            if identity == "allocation-failing":
                await engine.entered.wait()
                asyncio.current_task().add_done_callback(lambda _: first_settled.set())
                raise OSError("injected removal failure")
            await Engine.confirm_removed(engine, identity, **kwargs)

        engine.confirm_removed = remove
        closing = asyncio.create_task(backend.close(deadline=time.monotonic() + 10))
        try:
            await first_settled.wait()
            assert not closing.done()
            if cancel:
                closing.cancel()
            engine.resume.set()
            result = (await asyncio.gather(closing, return_exceptions=True))[0]
            assert isinstance(result, asyncio.CancelledError if cancel else OSError)
            assert tuple(backend.entries) == ("allocation-failing",)
        finally:
            engine.resume.set()
            await asyncio.gather(closing, return_exceptions=True)
            engine.confirm_removed = lambda identity, **kw: Engine.confirm_removed(
                engine, identity, **kw
            )
            await backend.close(deadline=time.monotonic() + 10)

    asyncio.run(scenario())
