"""Framework-neutral, content-free Runtime instrumentation hooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

type ScalarAttribute = str | int | bool
type TelemetryAttribute = ScalarAttribute | Sequence[ScalarAttribute]


class RuntimeSpan(Protocol):
    def end(
        self,
        *,
        outcome: str,
        attributes: Mapping[str, TelemetryAttribute] | None = None,
    ) -> None: ...


class RuntimeInstrumentation(Protocol):
    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, TelemetryAttribute] | None = None,
    ) -> RuntimeSpan: ...


def start_isolated_span(
    instrumentation: RuntimeInstrumentation | None,
    name: str,
    attributes: Mapping[str, TelemetryAttribute] | None = None,
) -> RuntimeSpan | None:
    """Start a span unless instrumentation is absent or fails.

    Telemetry is optional: a failing sink yields no span and never changes the
    operation it observes.
    """

    if instrumentation is None:
        return None
    try:
        return instrumentation.start_span(name, attributes=attributes)
    except Exception:
        return None


def end_isolated_span(
    span: RuntimeSpan | None,
    *,
    outcome: str,
    attributes: Mapping[str, TelemetryAttribute] | None = None,
) -> None:
    """End a span from ``start_isolated_span``; a failing sink is ignored."""

    if span is None:
        return
    try:
        span.end(outcome=outcome, attributes=attributes)
    except Exception:
        return
