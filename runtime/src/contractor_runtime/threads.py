"""Run blocking work in a thread without abandoning it on cancellation.

The module also owns retrieval of the outcome of a task that nobody awaits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from contractor_runtime.core.async_tasks import join_owned_task


async def to_thread_until_done[T](function: Callable[..., T], *args: Any, name: str) -> T:
    """Run ``function`` in a thread and wait for it even when cancelled.

    Python cannot stop a thread that is already inside a syscall or native
    parser. Callers that hold a lock or own a directory must not release it
    while the thread still runs, so every cancellation, including repeated
    ones, is absorbed until the thread returns. The cancellation is then
    re-raised and the thread's own result or exception is discarded. An
    allocation-wide stop deadline fences the Runtime if the thread never
    returns.
    """

    task = asyncio.create_task(asyncio.to_thread(function, *args), name=name)
    cancelled = await join_owned_task(task)
    if cancelled:
        consume_task_exception(task)
        raise asyncio.CancelledError
    return task.result()


def consume_task_exception(task: asyncio.Future[Any]) -> None:
    """Retrieve a done task's exception without raising or exposing it.

    Use it as a done callback, or directly on a done task, whenever the
    outcome is deliberately discarded: asyncio otherwise logs an exception
    that was never retrieved, together with its content. A cancelled task has
    no exception to retrieve.
    """

    if not task.cancelled():
        task.exception()
