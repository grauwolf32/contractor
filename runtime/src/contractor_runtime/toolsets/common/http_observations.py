"""Optional transport observation subscriber; independent of persistence owners."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from contractor_runtime.toolsets.common.subscribers import notify_subscriber


class HTTPObservationSink(Protocol):
    async def receive(self, observations: Sequence[dict], *, gaps: Sequence[dict] = ()) -> None: ...


async def notify_http_observations(
    sink: HTTPObservationSink | None,
    observations: Sequence[dict],
    *,
    gaps: Sequence[dict] = (),
) -> None:
    """Isolate subscriber failure and join it before propagating caller cancellation.

    The subscriber owns its independent deadline. A child cancellation is a
    subscriber failure; cancellation of the producer still reaches its caller.
    """
    if sink is None or not (observations or gaps):
        return
    await notify_subscriber(
        lambda: sink.receive(observations, gaps=gaps) if gaps else sink.receive(observations),
        name="http-observation",
    )
