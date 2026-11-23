"""Deterministic probes for document preparation that runs in worker threads.

The probes use thread identity, events and completion order, never elapsed
time, to show where document work runs and what waits for it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import pytest


class PreparationGate:
    """Hold a wrapped function in its worker thread until the test releases it.

    A call on the event-loop thread fails at once: that thread could never
    deliver the release it would wait for. Construct the gate inside the loop.
    """

    def __init__(self, function: Callable[..., Any], label: str, order: list[str]) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._loop_thread = threading.get_ident()
        self._function = function
        self._label = label
        self._order = order

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        if threading.get_ident() == self._loop_thread:
            raise AssertionError(f"{self._label} work ran on the event-loop thread")
        self.release.wait()
        try:
            return self._function(*args, **kwargs)
        finally:
            self._order.append(self._label)

    async def wait_entered(self, task: asyncio.Task[Any]) -> None:
        """Return once the gated thread waits, or raise what ended the task first."""

        waiter = asyncio.ensure_future(asyncio.to_thread(self.entered.wait))
        await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
        self.entered.set()
        await waiter
        if task.done():
            task.result()
            raise AssertionError("the operation ended without reaching its gated work")


def observe_threads(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    functions: dict[str, tuple[str, Callable[..., Any] | None]],
) -> dict[str, set[int]]:
    """Record the threads that call module functions, optionally via a replacement."""

    threads: dict[str, set[int]] = {}
    for name, (stage, replacement) in functions.items():
        function = replacement if replacement is not None else getattr(module, name)

        def run(
            *args: Any, _stage: str = stage, _function: Callable[..., Any] = function, **kwargs: Any
        ) -> Any:
            threads.setdefault(_stage, set()).add(threading.get_ident())
            return _function(*args, **kwargs)

        monkeypatch.setattr(module, name, run)
    return threads


async def loop_callback(order: list[str]) -> None:
    """Schedule an independent callback on the running loop and wait until it ran."""

    ran = asyncio.Event()

    def callback() -> None:
        order.append("loop callback")
        ran.set()

    asyncio.get_running_loop().call_soon(callback)
    await ran.wait()


async def yield_to_loop(turns: int = 20) -> None:
    """Let ready tasks run; a task waiting on a held lock or thread stays pending."""

    for _ in range(turns):
        await asyncio.sleep(0)
