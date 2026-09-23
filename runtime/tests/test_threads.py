from __future__ import annotations

import asyncio
import threading

import pytest

from contractor_runtime.threads import to_thread_until_done


def test_returns_result_and_propagates_thread_errors() -> None:
    def fail() -> None:
        raise OSError("disk")

    async def scenario() -> None:
        assert await to_thread_until_done(pow, 2, 5, name="test") == 32
        with pytest.raises(OSError, match="disk"):
            await to_thread_until_done(fail, name="test")

    asyncio.run(scenario())


@pytest.mark.parametrize("thread_fails", [False, True])
@pytest.mark.parametrize("cancellations", [1, 2, 3])
def test_repeated_cancellation_waits_for_thread(cancellations: int, thread_fails: bool) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def work() -> str:
        started.set()
        release.wait(5)
        finished.set()
        if thread_fails:
            raise OSError("late failure")
        return "done"

    async def scenario() -> None:
        lock = asyncio.Lock()

        async def owner() -> None:
            async with lock:
                await to_thread_until_done(work, name="test")

        task = asyncio.create_task(owner())
        await asyncio.to_thread(started.wait, 5)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0.01)
            # The lock must stay held while the thread still runs.
            assert not task.done()
            assert lock.locked()
            assert not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not lock.locked()

    asyncio.run(scenario())
