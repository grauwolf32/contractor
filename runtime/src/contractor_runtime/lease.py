"""Monotonic confirmed-control-lease watchdog for the Runtime Agent."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class LeaseWatchdog:
    """Expires authority independently from heartbeat transport progress.

    ``arm`` is reserved for a successful registration handshake. Normal
    heartbeat responses may only extend a still-live lease with a strictly new
    acknowledgement. Once expiry is observed, late or replayed responses
    cannot revive the lease.
    """

    def __init__(
        self,
        on_expired: Callable[[], Awaitable[None]],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_expired = on_expired
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._changed = asyncio.Event()
        self._deadline: float | None = None
        self._last_ack = 0
        self._expired = False
        self._expiry_started = False

    @property
    def expired(self) -> bool:
        return self._expired

    @property
    def last_ack(self) -> int:
        return self._last_ack

    async def arm(self, lease_seconds: float) -> None:
        """Start a new lease generation after successful registration."""

        if lease_seconds <= 0:
            raise ValueError("confirmed lease must be positive")
        async with self._lock:
            self._deadline = self._monotonic() + lease_seconds
            self._expired = False
            self._expiry_started = False
            self._changed.set()

    async def acknowledge(self, ack_sequence: int, lease_seconds: float) -> bool:
        """Renew a live generation for a strictly newer valid acknowledgement."""

        if ack_sequence <= 0 or lease_seconds <= 0:
            raise ValueError("ack sequence and confirmed lease must be positive")
        async with self._lock:
            now = self._monotonic()
            if self._deadline is None or self._expired or now >= self._deadline:
                if self._deadline is not None and now >= self._deadline:
                    self._expired = True
                    self._changed.set()
                return False
            if ack_sequence <= self._last_ack:
                return False
            self._last_ack = ack_sequence
            self._deadline = now + lease_seconds
            self._changed.set()
            return True

    async def expire_if_due(self) -> bool:
        """Observe one expiry edge and invoke the shutdown callback once."""

        invoke = False
        async with self._lock:
            if (
                self._deadline is not None
                and self._monotonic() >= self._deadline
                and not self._expired
            ):
                self._expired = True
                self._changed.set()
            if self._expired and not self._expiry_started:
                self._expiry_started = True
                invoke = True
        if invoke:
            await self._on_expired()
        return invoke

    async def run(self, stop: asyncio.Event) -> None:
        """Watch the deadline until process shutdown, including across re-arm."""

        while not stop.is_set():
            await self.expire_if_due()
            async with self._lock:
                deadline = self._deadline
                expired = self._expired
                self._changed.clear()
            delay = None if deadline is None or expired else max(0.0, deadline - self._monotonic())
            changed = asyncio.create_task(self._changed.wait(), name="lease-watchdog-change")
            stopped = asyncio.create_task(stop.wait(), name="lease-watchdog-stop")
            try:
                done, _ = await asyncio.wait(
                    {changed, stopped}, timeout=delay, return_when=asyncio.FIRST_COMPLETED
                )
                if stopped in done:
                    return
            finally:
                for task in (changed, stopped):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(changed, stopped, return_exceptions=True)
