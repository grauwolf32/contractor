from __future__ import annotations

import asyncio
import gc
import threading
from typing import Any

import pytest

from contractor_runtime.threads import consume_task_exception, to_thread_until_done


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


def test_consumed_task_exception_is_never_reported_as_unretrieved() -> None:
    reports: list[dict[str, Any]] = []

    async def fail() -> None:
        raise OSError("private detail")

    async def scenario() -> None:
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: reports.append(context)
        )
        consumed = asyncio.create_task(fail())
        consumed.add_done_callback(consume_task_exception)
        cancelled = asyncio.create_task(asyncio.Event().wait())
        cancelled.add_done_callback(consume_task_exception)
        cancelled.cancel()
        # The control proves that an unconsumed failure would be reported.
        abandoned = asyncio.create_task(fail())
        await asyncio.wait({consumed, cancelled, abandoned})
        del consumed, cancelled, abandoned
        gc.collect()

    asyncio.run(scenario())

    assert [context["message"] for context in reports] == ["Task exception was never retrieved"]


@pytest.mark.parametrize("outcome", ["value", "failure", "cancelled"])
def test_owned_join_leaves_outcome_and_cancellation_policy_to_caller(outcome: str) -> None:
    from contractor_runtime.core.async_tasks import join_owned_task

    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def child() -> str:
            entered.set()
            await release.wait()
            if outcome == "failure":
                raise ValueError("child failure")
            if outcome == "cancelled":
                raise asyncio.CancelledError
            return "value"

        task = asyncio.create_task(child())
        owner = asyncio.create_task(join_owned_task(task))
        await entered.wait()
        for _ in range(3):
            owner.cancel()
            await asyncio.sleep(0)
            assert not owner.done()
            assert not task.cancelled()
        release.set()
        assert await owner is True
        if outcome == "failure":
            with pytest.raises(ValueError, match="child failure"):
                task.result()
        elif outcome == "cancelled":
            assert task.cancelled()
        else:
            assert task.result() == "value"
        assert await join_owned_task(task) is False

    asyncio.run(scenario())
