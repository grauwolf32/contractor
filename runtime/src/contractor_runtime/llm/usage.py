"""Closed projection of provider token-usage metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_TOKEN_COUNT = 2**64 - 1


@dataclass(frozen=True, slots=True)
class TokenUsageProjection:
    prompt_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None

    @property
    def total_unavailable(self) -> bool:
        return self.total_tokens is None


def project_token_usage(usage: Any | None) -> TokenUsageProjection:
    """Keep only non-negative, arithmetically consistent provider counters.

    A positive total is the only evidence that cumulative token accounting is
    available. This deliberately treats LiteLLM's zero-valued representation
    of an omitted OpenAI ``usage`` member as unavailable. Partial prompt usage
    remains useful for the independent context-window rule, unless the
    counters contradict one another.
    """

    prompt = _counter(usage, "prompt_token_count", positive=False)
    output = _counter(usage, "candidates_token_count", positive=False)
    total = _counter(usage, "total_token_count", positive=True)
    cached = _counter(usage, "cached_content_token_count", positive=False)
    inconsistent = (
        total is not None and prompt is not None and output is not None and prompt + output > total
    ) or (prompt is not None and cached is not None and cached > prompt)
    if inconsistent:
        return TokenUsageProjection(None, None, None, None)
    return TokenUsageProjection(
        _bounded(prompt), _bounded(output), _bounded(total), _bounded(cached)
    )


def _counter(usage: Any | None, name: str, *, positive: bool) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    if type(value) is not int or value < (1 if positive else 0):
        return None
    return value


def _bounded(value: int | None) -> int | None:
    return min(value, MAX_TOKEN_COUNT) if value is not None else None
