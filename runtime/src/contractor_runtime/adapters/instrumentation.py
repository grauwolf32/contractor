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
