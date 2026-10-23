"""An asyncio event loop whose monotonic clock a test moves forward explicitly."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


class AdvanceableClockLoop(asyncio.SelectorEventLoop):
    """Loop time is real monotonic time plus an offset that only the test moves.

    Loop timers (``asyncio.timeout_at``, ``asyncio.timeout``, ``wait_for`` and
    ``sleep``) follow ``time()``. Advancing past a deadline therefore fires its
    timer on the next loop iteration, at a point the test chose with a
    synchronization gate, without waiting in real time. Real time still flows,
    so short ``wait_for`` guards keep a broken scenario from hanging.
    """

    def __init__(self) -> None:
        super().__init__()
        self._offset = 0.0

    def time(self) -> float:
        return super().time() + self._offset

    def advance(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("the loop clock only moves forward")
        self._offset += seconds


def advance_clock(seconds: float) -> None:
    """Move the running loop's clock forward; due timers fire on its next iteration."""

    loop = asyncio.get_running_loop()
    if not isinstance(loop, AdvanceableClockLoop):
        raise RuntimeError("the scenario does not run on an AdvanceableClockLoop")
    loop.advance(seconds)


def run_with_advanceable_clock[T](scenario: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(scenario, loop_factory=AdvanceableClockLoop)
