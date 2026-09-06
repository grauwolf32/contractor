"""Retain filesystem ownership after cancellation of its awaiting caller."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from contractor_runtime.projectfs.errors import WorkspaceStorageError

T = TypeVar("T")


class WorkspaceOperationGuard:
    """One serialization boundary, including worker threads and disposal.

    The owning task, not its caller, releases the lock. Cancelling a caller
    therefore cannot race another edit or cleanup against a running syscall.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owners: set[asyncio.Task[object]] = set()
        self._fenced = False
        self._closing = False
        self._cleanup: asyncio.Task[None] | None = None

    @property
    def fenced(self) -> bool:
        return self._fenced

    def fence(self) -> None:
        self._fenced = True

    async def run(self, operation: Callable[[], T], *, deadline: float) -> T:
        """Run blocking work without releasing ownership on caller timeout."""

        async def threaded() -> T:
            return await asyncio.to_thread(operation)

        return await self.run_async(threaded, deadline=deadline)

    async def run_async(self, operation: Callable[[], Awaitable[T]], *, deadline: float) -> T:
        """Same owner for async sandbox execution and synchronous filesystem I/O."""
        remaining = _remaining(deadline)
        if self._fenced or self._closing:
            raise WorkspaceStorageError("workspace_unavailable")
        # Acquire in the caller: a cancelled waiter must never start its work.
        try:
            await asyncio.wait_for(self._lock.acquire(), remaining)
        except TimeoutError:
            raise WorkspaceStorageError("workspace_unavailable") from None
        if self._fenced or self._closing or time.monotonic() >= deadline:
            self._lock.release()
            raise WorkspaceStorageError("workspace_unavailable")

        async def owned() -> T:
            try:
                return await operation()
            finally:
                self._lock.release()

        task = asyncio.create_task(owned(), name="workspace-operation")
        self._owners.add(task)
        task.add_done_callback(self._completed)
        # A zero timeout also fences ownership if the deadline expires between
        # launching the owner and starting the wait.
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), max(0.0, deadline - time.monotonic())
            )
        except TimeoutError:
            self.fence()
            raise WorkspaceStorageError("workspace_unavailable") from None
        except asyncio.CancelledError:
            # The operation may already have effects. Keep it owned and deny
            # further work until allocation teardown, even if it later succeeds.
            self.fence()
            raise

    async def close(self, cleanup: Callable[[], None], *, deadline: float) -> None:
        """Join a single disposal task after outstanding operations settle."""
        remaining = _remaining(deadline)
        self._closing = True
        if self._cleanup is None:

            async def dispose() -> None:
                async with self._lock:
                    await asyncio.to_thread(cleanup)

            self._cleanup = asyncio.create_task(dispose(), name="workspace-disposal")
            self._cleanup.add_done_callback(_consume_exception)
        try:
            await asyncio.wait_for(asyncio.shield(self._cleanup), remaining)
        except TimeoutError:
            self.fence()
            raise WorkspaceStorageError("workspace_unavailable") from None
        except asyncio.CancelledError:
            self.fence()
            raise
        except Exception:
            self.fence()
            raise WorkspaceStorageError("workspace_unavailable") from None

    def _completed(self, task: asyncio.Task[object]) -> None:
        self._owners.discard(task)
        _consume_exception(task)


def _consume_exception(task: asyncio.Task[object]) -> None:
    if not task.cancelled():
        task.exception()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise WorkspaceStorageError("workspace_unavailable")
    return remaining
