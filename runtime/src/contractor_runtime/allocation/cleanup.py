"""Deadline-bounded cleanup primitives; lifecycle ownership stays in the service."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from datetime import datetime

from contractor_runtime.factories import (
    ToolInstance,
)
from contractor_runtime.threads import consume_task_exception


async def _close_tools(tools: Mapping[str, ToolInstance]) -> None:
    for name in reversed(tuple(tools)):
        await tools[name].close()
        if isinstance(tools, MutableMapping):
            del tools[name]


async def _await_before_deadline(
    operation: Callable[[], Awaitable[None]],
    *,
    deadline: datetime,
    now: Callable[[], datetime],
) -> None:
    remaining = (deadline - now()).total_seconds()
    if remaining <= 0:
        raise TimeoutError
    task = asyncio.create_task(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(consume_task_exception)
        raise
    if not done:
        task.cancel()
        task.add_done_callback(consume_task_exception)
        raise TimeoutError
    await task
