"""Bounded counters for one Worker invocation, independent of ADK callbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from contractor_runtime.llm.usage import project_token_usage

_MAX_UINT64 = 2**64 - 1
_MAX_INVOCATION_TOOL_NAMES = 256


@dataclass(slots=True)
class InvocationMetricsReducer:
    """Content-free counters for exactly one Worker invocation."""

    model_calls: int = 0
    model_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    token_usage_unavailable: int = 0
    latest_prompt_tokens: int | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    tools: dict[str, dict[str, int]] = field(default_factory=dict)
    truncated: bool = False

    def record_model_call(self) -> None:
        self.model_calls = _saturating_add(self.model_calls, 1)

    def record_model_usage(self, usage: Any | None) -> None:
        projected = project_token_usage(usage)
        self.latest_prompt_tokens = projected.prompt_tokens
        if projected.total_unavailable:
            self.token_usage_unavailable = _saturating_add(self.token_usage_unavailable, 1)
        for value, field_name in (
            (projected.prompt_tokens, "input_tokens"),
            (projected.output_tokens, "output_tokens"),
            (projected.total_tokens, "total_tokens"),
            (projected.cached_input_tokens, "cached_input_tokens"),
        ):
            if value is not None:
                setattr(self, field_name, _saturating_add(getattr(self, field_name), value))

    def record_model_error(self) -> None:
        self.model_errors = _saturating_add(self.model_errors, 1)

    def record_tool_call(self, name: str, *, failed: bool) -> None:
        self.tool_calls = _saturating_add(self.tool_calls, 1)
        if failed:
            self.tool_errors = _saturating_add(self.tool_errors, 1)
        identifier = _metric_identifier(name)
        aggregate = self.tools.get(identifier)
        if aggregate is None:
            if len(self.tools) >= _MAX_INVOCATION_TOOL_NAMES:
                self.truncated = True
                return
            aggregate = {"calls": 0, "failures": 0}
            self.tools[identifier] = aggregate
        aggregate["calls"] = _saturating_add(aggregate["calls"], 1)
        if failed:
            aggregate["failures"] = _saturating_add(aggregate["failures"], 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "modelCalls": self.model_calls,
            "modelErrors": self.model_errors,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "tokenUsageUnavailable": self.token_usage_unavailable,
            "latestPromptTokens": self.latest_prompt_tokens,
            "toolCalls": self.tool_calls,
            "toolErrors": self.tool_errors,
            "tools": {name: dict(values) for name, values in sorted(self.tools.items())},
            "truncated": self.truncated,
        }


def _metric_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized[:64] or "unknown"


def _saturating_add(current: int, increment: int) -> int:
    return min(_MAX_UINT64, current + increment)
