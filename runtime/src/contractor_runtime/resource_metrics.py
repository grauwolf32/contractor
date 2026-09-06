"""Allocation-local observations of the Runtime process, never its children.

Reads are synchronous, bounded kernel operations on the event-loop thread: no
executor, overlapping reads, GC, lifetime high-water mark or retained time series.
The sole async owner only waits for the next RSS observation and is cancellable.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from contractor_runtime.contracts import ResourceReason, RuntimeResources

SUPPORTED_PERFORMANCE_METRICS_VERSIONS = (1,)
INTERVAL_SECONDS = 15
MAX_INTEGER = 2**53 - 1


@dataclass(frozen=True, slots=True)
class ProcessReading:
    cpu_user_seconds: float | None = None
    cpu_system_seconds: float | None = None
    rss_bytes: int | None = None
    reason: ResourceReason | None = None


def read_process(boundary: bool) -> ProcessReading:
    """Linux current RSS; only boundaries need process CPU counters.

    /proc/self/statm is a small kernel-generated record, not an operator path.
    Failure of one measurement does not erase independently readable fields.
    Unsupported hosts can still implement the protocol with an unavailable result.
    """
    if sys.platform != "linux":
        return ProcessReading(reason="unsupported_platform")
    user = system = rss = None
    reason: ResourceReason | None = None
    if boundary:
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            user, system = usage.ru_utime, usage.ru_stime
        except (OSError, ValueError):
            reason = "read_failed"
    try:
        with open("/proc/self/statm", "rb", buffering=0) as source:
            raw = source.read(257)
        if len(raw) > 256:
            raise ValueError("oversized process record")
        rss = int(raw.split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        reason = "read_failed"
    return ProcessReading(user, system, rss, reason)


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


class ResourceCollector:
    """One constant-size accumulator; construct/start only for opted-in allocations."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        reader: Callable[[bool], ProcessReading] = read_process,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._clock = clock
        self._reader = reader
        self._sleep = sleep
        self._task: asyncio.Task[None] | None = None
        self._started: float | None = None
        self._next_at = 0.0
        self._last_rss_at: float | None = None
        self._gap = 0.0
        self._count = 0
        self._peak: int | None = None
        self._start = ProcessReading()
        self._reason: ResourceReason | None = None
        self._closed = False
        self._result: RuntimeResources | None = None

    def start(self) -> None:
        if self._started is not None or self._closed:
            return
        try:
            self._started = self._time()
            self._next_at = self._started + INTERVAL_SECONDS
            self._start = self._observe(self._started, boundary=True)
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run(), name="allocation-resource-sampler")
        except Exception:
            self._reason = "read_failed"

    def close(self) -> None:
        """Discard collection without inventing an endpoint after an unconfirmed stop.

        No join or resource I/O can extend a finalization deadline. The task has
        no async cleanup and cannot read after this synchronous close operation.
        """
        self._closed = True
        if self._task is not None:
            self._task.cancel()

    def finish(self) -> RuntimeResources:
        if self._result is not None:
            return self._result
        was_closed = self._closed
        self.close()
        try:
            if was_closed or self._started is None:
                raise ValueError("collection has no confirmed interval")
            ended = self._time()
            if ended > self._next_at:
                self._mark("sampling_gap")
            end = self._observe(ended, boundary=True)
            duration = ended - self._started
            last_rss = self._last_rss_at if self._last_rss_at is not None else self._started
            self._gap = max(self._gap, ended - last_rss)
            if self._gap > 30:
                self._mark("sampling_gap")
            values: dict[str, object] = {
                "durationSeconds": duration,
                "maxSampleGapSeconds": self._gap,
                "rssSampleCount": self._count,
            }
            for name, value in (
                ("rssStartBytes", self._start.rss_bytes),
                ("rssEndBytes", end.rss_bytes),
                ("rssPeakObservedBytes", self._peak),
            ):
                if value is not None:
                    values[name] = value
            for name in ("cpu_user_seconds", "cpu_system_seconds"):
                first, last = getattr(self._start, name), getattr(end, name)
                if first is not None and last is not None:
                    if last < first:
                        self._mark("counter_reset")
                    else:
                        values[name] = last - first
            available = self._count > 0 or any(name.startswith("cpu_") for name in values)
            status = (
                "complete" if self._reason is None else ("partial" if available else "unavailable")
            )
            if self._reason is not None:
                values["reason"] = self._reason
            self._result = RuntimeResources(
                version=1, scope="runtime_process", status=status, **values
            )
        except Exception:
            # Observation/clock/validation failures are never Worker failures.
            self._result = RuntimeResources(
                version=1, scope="runtime_process", status="unavailable", reason="read_failed"
            )
        return self._result

    def _time(self) -> float:
        value = self._clock()
        if not _number(value) or (self._started is not None and value < self._started):
            raise ValueError("invalid monotonic observation")
        return float(value)

    def _mark(self, reason: ResourceReason) -> None:
        if self._reason is None:
            self._reason = reason

    def _observe(self, at: float, *, boundary: bool) -> ProcessReading:
        try:
            value = self._reader(boundary)
            user, system, rss = value.cpu_user_seconds, value.cpu_system_seconds, value.rss_bytes
            if value.reason is not None:
                self._mark(value.reason)
            if boundary:
                if not _number(user):
                    user = None
                    self._mark("read_failed")
                if not _number(system):
                    system = None
                    self._mark("read_failed")
            if type(rss) is not int or not 0 <= rss <= MAX_INTEGER:
                rss = None
                self._mark("read_failed")
            else:
                previous = self._last_rss_at if self._last_rss_at is not None else self._started
                assert previous is not None
                self._gap = max(self._gap, at - previous)
                self._last_rss_at = at
                self._count += 1
                self._peak = max(self._peak or 0, rss)
            return ProcessReading(user, system, rss)
        except Exception:
            self._mark("read_failed")
            return ProcessReading()

    async def _run(self) -> None:
        try:
            while not self._closed:
                await self._sleep(max(0.0, self._next_at - self._time()))
                if self._closed:
                    return
                at = self._time()
                if at < self._next_at:
                    continue
                skipped = int((at - self._next_at) // INTERVAL_SECONDS)
                if skipped:
                    self._mark("sampling_gap")
                self._observe(at, boundary=False)
                self._next_at += (skipped + 1) * INTERVAL_SECONDS
        except asyncio.CancelledError:
            raise
        except Exception:
            self._mark("read_failed")
