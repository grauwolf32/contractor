"""Best-effort allocation-private HTTP batching; Go owns sitemap semantics."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from uuid import uuid4

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.sitemap import SitemapClient

_LOGGER = logging.getLogger(__name__)
MAX_ITEMS = 100
MAX_BYTES = 256 << 10
FLUSH_INTERVAL_SECONDS = 5.0
FLUSH_BUDGET_SECONDS = 15.0
MAX_ATTEMPTS = 3
_LOG_FAILURE_CODES = frozenset(
    {
        "artifact_history_protected",
        "artifact_history_maintenance_pending",
        "storage_transaction_conflict",
        "invalid_request",
        "allocation_not_active",
        "forbidden",
        "not_found",
        "sitemap_unavailable",
    }
)


def _wire_bytes(value: dict) -> int:
    # This is only private transport sizing, never a sitemap identity/hash.
    return len(
        json.dumps(
            {"apiVersion": "contractor/v1alpha1", "input": value},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )


class SitemapCollector:
    def __init__(
        self,
        client: SitemapClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client, self._clock, self._sleep, self._wait = client, clock, sleep, wait
        self._pending: list[dict] = []
        self._bytes = 0
        self._since = clock()
        self._timer: asyncio.Task | None = None
        self._publisher: asyncio.Task | None = None
        self._checkpoint_deadlines: set[float] = set()
        self._closed = False
        self._closing = False
        self._disabled = False
        self._failure: ArtifactAPIError | None = None
        self._buffer_warning = False

    async def receive(self, observations: Sequence[dict]) -> None:
        """Admit one small private frame without awaiting any storage/network I/O.

        No headers/bodies, durable queue, URL keys or sitemap hashes are built.
        The existing Go projector runs before receipt hashing and Artifact
        publication.
        """
        if self._closed or self._closing or self._disabled or not observations:
            return
        try:
            rows = [*self._pending, *observations]
            size = _wire_bytes({"observations": rows})
            if len(rows) > MAX_ITEMS or size > MAX_BYTES:
                self._kick()
                rows = [*self._pending, *observations]
                size = _wire_bytes({"observations": rows})
            if len(rows) > MAX_ITEMS or size > MAX_BYTES:
                if not self._buffer_warning:
                    _LOGGER.warning("sitemap collection buffer full")
                    self._buffer_warning = True
                return
            if not self._pending:
                self._since = self._clock()
            self._pending, self._bytes = rows, size
            if len(rows) >= MAX_ITEMS or self._clock() - self._since >= FLUSH_INTERVAL_SECONDS:
                self._kick()
            self._arm()
        except Exception as error:
            self._record_failure(error)

    def _record_failure(self, error: BaseException) -> None:
        failure = (
            error
            if isinstance(error, ArtifactAPIError)
            else ArtifactAPIError(503, "sitemap_unavailable", True)
        )
        if self._failure is None or self._failure.code != failure.code:
            # The JSON logger deliberately drops arbitrary extras. Render only
            # reviewed constant codes, never remote text or exception contents.
            code = failure.code if failure.code in _LOG_FAILURE_CODES else "sitemap_unavailable"
            _LOGGER.warning(
                "sitemap collection failed (%s)",
                code,
                extra={"code": code, "errorType": type(error).__name__},
            )
        self._failure = failure

    def _arm(self) -> None:
        if (
            self._timer is None
            and self._pending
            and not self._closed
            and not self._closing
            and not self._disabled
        ):
            self._timer = asyncio.create_task(self._tick(), name="sitemap-collection-timer")

    async def _tick(self) -> None:
        try:
            await self._wait(max(0.0, FLUSH_INTERVAL_SECONDS - (self._clock() - self._since)))
            self._kick()
        finally:
            self._timer = None

    def _kick(self) -> None:
        if self._publisher is not None or self._disabled or not self._pending:
            return
        payload = {"observations": self._pending}
        deadline = min(
            [
                self._clock() + FLUSH_BUDGET_SECONDS,
                *self._checkpoint_deadlines,
            ]
        )
        self._pending, self._bytes = [], 0
        self._publisher = asyncio.create_task(
            self._publish(payload, deadline), name="sitemap-publisher"
        )

    async def _publish(self, payload: dict, deadline: float) -> None:
        key = str(uuid4())
        try:
            async with asyncio.timeout(max(0.0, deadline - self._clock())):
                for attempt in range(MAX_ATTEMPTS):
                    if self._clock() >= deadline:
                        raise TimeoutError
                    try:
                        await self._client.collect(payload, key)
                        self._failure = None
                        return
                    except (ArtifactAPIError, ArtifactTransportError) as error:
                        retryable = not isinstance(error, ArtifactAPIError) or (
                            error.retryable and error.code != "storage_transaction_conflict"
                        )
                        if not retryable or attempt == MAX_ATTEMPTS - 1:
                            raise
                        delay = 0.25 * 2**attempt
                        if (
                            isinstance(error, ArtifactAPIError)
                            and error.retry_after_seconds is not None
                        ):
                            delay = max(delay, error.retry_after_seconds)
                        if delay >= deadline - self._clock():
                            raise
                        await self._sleep(delay)
        except (Exception, asyncio.CancelledError) as error:
            self._disabled = True
            self._record_failure(error)
        finally:
            self._publisher = None
            if not self._closed and not self._disabled:
                if self._pending and (
                    len(self._pending) >= MAX_ITEMS
                    or self._clock() - self._since >= FLUSH_INTERVAL_SECONDS
                ):
                    self._kick()
                self._arm()

    async def _join_publisher(self) -> None:
        task = self._publisher
        if task is None:
            return
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        if not task.cancelled():
            task.exception()
        if cancelled:
            raise asyncio.CancelledError

    async def checkpoint(self) -> None:
        """Attempt pending publication before a live read or allocation fencing.

        Failure is a collector log diagnostic. Readers still fetch committed
        state and Worker/Run results keep their existing contracts.
        """
        if self._closed:
            return
        deadline = self._clock() + FLUSH_BUDGET_SECONDS
        self._checkpoint_deadlines.add(deadline)
        try:
            async with asyncio.timeout(FLUSH_BUDGET_SECONDS):
                while not self._disabled and (self._pending or self._publisher):
                    self._kick()
                    await self._join_publisher()
        except TimeoutError as error:
            self._disabled = True
            self._record_failure(error)
        finally:
            self._checkpoint_deadlines.discard(deadline)

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        timer = self._timer
        if timer is not None:
            timer.cancel()
            with suppress(asyncio.CancelledError):
                await timer
            self._timer = None
        try:
            await self.checkpoint()
        finally:
            # Producers are closed by composition first. Join the publisher
            # before dropping its private input, even on caller cancellation.
            await self._join_publisher()
            self._closed = True
            self._pending, self._bytes = [], 0
