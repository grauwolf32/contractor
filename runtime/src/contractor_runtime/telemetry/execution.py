"""Execution transcripts stay in authorized observations, not telemetry sinks."""

from __future__ import annotations

from contextlib import nullcontext


class _ContentFreeSpan:
    def __init__(self, span):
        self._span = span

    def end(self, *, outcome, attributes=None):
        self._span.end(outcome=outcome, attributes=attributes)

    def activate(self):
        activate = getattr(self._span, "activate", None)
        return activate() if callable(activate) else nullcontext()


class ContentFreeInstrumentation:
    """Preserve the published metadata-only protocol even for opt-in sinks.

    Suppress content across the entire execution-enabled Worker: subsequent
    model requests, summaries and finalizers can contain previous tool output.
    """

    def __init__(self, instrumentation):
        self._instrumentation = instrumentation

    def start_span(self, name, *, attributes=None):
        return _ContentFreeSpan(self._instrumentation.start_span(name, attributes=attributes))
