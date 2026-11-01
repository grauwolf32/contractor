"""Failure-isolated notifications joined before producer cancellation propagates."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from contractor_runtime.core.async_tasks import join_owned_task


async def notify_subscriber(operation: Callable[[], Awaitable[None]], *, name: str) -> None:
    async def invoke() -> None:
        await operation()

    task = asyncio.create_task(invoke(), name=name)
    cancelled = await join_owned_task(task)
    if not task.cancelled():
        task.exception()
    if cancelled:
        raise asyncio.CancelledError
