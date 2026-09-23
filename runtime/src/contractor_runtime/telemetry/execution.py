"""Sensitive tool output stays in authorized observations, not telemetry sinks."""

from __future__ import annotations

from contextlib import nullcontext

# A selected tool sets this attribute to True when its results can carry
# material that must never reach an opt-in content-capturing telemetry sink:
# command output, captured HTTP traffic or session values.
SENSITIVE_OUTPUT_ATTRIBUTE = "contractor_sensitive_output"


def declares_sensitive_output(tool: object) -> bool:
    """Whether a tool (or the callable behind an ADK tool) declares sensitive output."""

    owner = getattr(tool, "func", tool)
    return getattr(owner, SENSITIVE_OUTPUT_ATTRIBUTE, False) is True


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

    Suppress content across the entire Worker that selected a sensitive-output
    tool: subsequent model requests, summaries and finalizers can contain
    previous tool output.
    """

    def __init__(self, instrumentation):
        self._instrumentation = instrumentation

    def start_span(self, name, *, attributes=None):
        return _ContentFreeSpan(self._instrumentation.start_span(name, attributes=attributes))
