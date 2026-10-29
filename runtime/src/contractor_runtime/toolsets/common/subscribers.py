"""Failure-isolated notifications joined before producer cancellation propagates."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def notify_subscriber(operation: Callable[[], Awaitable[None]], *, name: str) -> None:
    async def invoke() -> None:
        await operation()

    task = asyncio.create_task(invoke(), name=name)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            cancelled = cancelled or bool(current and current.cancelling())
        except Exception:
            break
    if not task.cancelled():
        task.exception()
    if cancelled:
        raise asyncio.CancelledError
