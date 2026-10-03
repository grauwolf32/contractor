from __future__ import annotations

import asyncio
from typing import Any

import pytest

from contractor_runtime.toolsets.common.metrics import (
    RecordedToolCall,
    ToolCallCancelled,
    instrumented_call,
)


class RecordingMetrics:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_tool_call(self, name: str, **values: Any) -> None:
        self.calls.append({"name": name, **values})


class DomainError(Exception):
    code = "domain_failed"


def test_success_records_projection_and_replacement_arguments() -> None:
    metrics = RecordingMetrics()

    with RecordedToolCall(metrics, "tool", {"a": 1}, secrets=("s",)) as call:
        call.succeed({"count": 2}, arguments={"a": 1, "result_count": 2})

    [recorded] = metrics.calls
    assert recorded["name"] == "tool"
    assert recorded["arguments"] == {"a": 1, "result_count": 2}
    assert recorded["result"] == {"count": 2}
    assert recorded["error"] is None
    assert recorded["secrets"] == ("s",)
    assert recorded["duration_ms"] >= 0


def test_failure_is_recorded_and_propagates_unchanged() -> None:
    metrics = RecordingMetrics()
    error = ValueError("bad")

    with pytest.raises(ValueError) as raised, RecordedToolCall(metrics, "tool", {}):
        raise error

    assert raised.value is error
    assert metrics.calls[0]["error"] is error
    assert metrics.calls[0]["result"] is None


def test_bounded_failure_replaces_the_raised_error() -> None:
    metrics = RecordingMetrics()
    bounded = DomainError()

    with (
        pytest.raises(DomainError) as raised,
        RecordedToolCall(metrics, "tool", {}, bound_error=lambda _: bounded),
    ):
        raise RuntimeError("private detail")

    assert raised.value is bounded
    assert raised.value.__suppress_context__
    assert metrics.calls[0]["error"] is bounded


def test_cancellation_records_cancelled_error_and_propagates() -> None:
    metrics = RecordingMetrics()

    async def scenario() -> None:
        async def operation() -> dict[str, Any]:
            await asyncio.sleep(10)
            return {}

        task = asyncio.create_task(
            instrumented_call(RecordedToolCall(metrics, "tool", {}), operation(), dict)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    [recorded] = metrics.calls
    assert isinstance(recorded["error"], ToolCallCancelled)
    assert recorded["error"].code == "tool_call_cancelled"
    assert recorded["error"].retryable is True


def test_domain_cancellation_error_is_used_when_supplied() -> None:
    metrics = RecordingMetrics()
    cancelled = DomainError()

    with (
        pytest.raises(asyncio.CancelledError),
        RecordedToolCall(metrics, "tool", {}, cancelled=lambda: cancelled),
    ):
        raise asyncio.CancelledError

    assert metrics.calls[0]["error"] is cancelled


def test_fail_records_without_a_body() -> None:
    metrics = RecordingMetrics()
    error = DomainError()

    RecordedToolCall(metrics, "tool", {"raw": True}).fail(error)

    assert metrics.calls == [
        {
            "name": "tool",
            "arguments": {"raw": True},
            "result": None,
            "error": error,
            "secrets": (),
            "duration_ms": metrics.calls[0]["duration_ms"],
        }
    ]
