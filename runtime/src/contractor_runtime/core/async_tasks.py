"""Joining work whose lifetime belongs to the calling operation."""

from __future__ import annotations

import asyncio


async def join_owned_task(task: asyncio.Future[object]) -> bool:
    """Wait through repeated caller cancellation, returning whether interrupted.

    Never cancel the child or inspect its outcome. The owner must retrieve that
    outcome and choose error/cancellation precedence. Deadlines belong inside
    the owned operation or to the allocation's fencing owner.
    """

    interrupted = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            interrupted = True
    return interrupted
