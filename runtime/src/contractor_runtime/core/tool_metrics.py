"""Tool-call metrics interface and timed recording shared by built-in toolsets."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any, Protocol


class ToolMetrics(Protocol):
    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        secrets: tuple[str, ...] = (),
        duration_ms: int | None = None,
    ) -> None: ...


class ToolCallCancelled(Exception):
    """Recorded outcome of a tool call whose task was cancelled."""

    code = "tool_call_cancelled"
    retryable = True

    def __init__(self) -> None:
        super().__init__("Tool call was cancelled")


def elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)


class RecordedToolCall:
    """Time one tool call and record exactly one metrics outcome on exit.

    Use it as a context manager around the call body and pass the metric
    projection of a successful result to ``succeed``. An exception is recorded
    and propagates unchanged unless ``bound_error`` maps it to another error,
    which then replaces it. Task cancellation is recorded as ``cancelled()``
    and propagates.
    """

    def __init__(
        self,
        metrics: ToolMetrics,
        name: str,
        arguments: Mapping[str, Any],
        *,
        secrets: tuple[str, ...] = (),
        bound_error: Callable[[Exception], Exception] | None = None,
        cancelled: Callable[[], Exception] = ToolCallCancelled,
        started_ns: int | None = None,
    ) -> None:
        self._metrics = metrics
        self._name = name
        self._arguments = arguments
        self._secrets = secrets
        self._bound_error = bound_error
        self._cancelled = cancelled
        self._result: Mapping[str, Any] | None = None
        self._started_ns = time.perf_counter_ns() if started_ns is None else started_ns

    def succeed(
        self,
        result: Mapping[str, Any] | None,
        *,
        arguments: Mapping[str, Any] | None = None,
    ) -> None:
        self._result = result
        if arguments is not None:
            self._arguments = arguments

    def fail(self, error: Exception) -> None:
        self._record(error=error)

    def __enter__(self) -> RecordedToolCall:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self._record(result=self._result)
            return
        if isinstance(error, asyncio.CancelledError):
            self._record(error=self._cancelled())
            return
        if not isinstance(error, Exception):
            return
        bounded = error if self._bound_error is None else self._bound_error(error)
        self._record(error=bounded)
        if bounded is not error:
            raise bounded from None

    def _record(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._metrics.record_tool_call(
            self._name,
            arguments=self._arguments,
            result=result,
            error=error,
            secrets=self._secrets,
            duration_ms=elapsed_ms(self._started_ns),
        )


async def instrumented_call[T](
    call: RecordedToolCall,
    operation: Awaitable[T],
    summarize: Callable[[T], Mapping[str, Any]],
) -> T:
    """Await ``operation`` inside ``call`` and record ``summarize(result)``."""

    with call:
        result = await operation
        call.succeed(summarize(result))
        return result
