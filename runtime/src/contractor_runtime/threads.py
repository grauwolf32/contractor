"""Run blocking work in a thread without abandoning it on cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


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
    cancelled = False
    while not task.done():
        try:
            # asyncio.wait neither cancels the task nor raises its exception,
            # so a failure after a cancellation cannot mask that cancellation.
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()
