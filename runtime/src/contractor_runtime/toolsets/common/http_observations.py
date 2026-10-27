"""Optional transport observation subscriber; independent of persistence owners."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol


class HTTPObservationSink(Protocol):
    async def receive(self, observations: Sequence[dict]) -> None: ...


async def notify_http_observations(
    sink: HTTPObservationSink | None,
    observations: Sequence[dict],
) -> None:
    """Isolate subscriber failure and join it before propagating caller cancellation.

    The subscriber owns its independent deadline. A child cancellation is a
    subscriber failure; cancellation of the producer still reaches its caller.
    """
    if sink is None or not observations:
        return
    task = asyncio.create_task(sink.receive(observations), name="http-observation")
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
        task.exception()  # consume a subscriber failure without changing target data
    if cancelled:
        raise asyncio.CancelledError
