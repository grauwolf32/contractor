"""Invocation-local model, tool, and token budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contractor_runtime.llm.usage import project_token_usage


class WorkerBudgetExceeded(RuntimeError):
    """Safe internal control signal for one exhausted invocation dimension."""

    def __init__(self, dimension: str, limit: int, observed: int) -> None:
        self.dimension = dimension
        self.limit = limit
        self.observed = observed
        super().__init__(f"Worker invocation budget exhausted ({dimension})")


@dataclass(slots=True)
class _InvocationBudget:
    max_model_calls: int
    max_tool_calls: int
    max_total_tokens: int
    metrics: Any
    cumulative_budget: int | None = None
    summary_prompt_boundary: int | None = None
    model_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    token_usage_unavailable: int = 0
    latest_prompt_tokens: int | None = None
    failure: WorkerBudgetExceeded | None = None

    def _exhausted(self, dimension, limit, observed):
        self.failure = WorkerBudgetExceeded(dimension, limit, observed)
        return self.failure

    def start(self) -> None:
        self.metrics.start_worker_budget(
            max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls,
            max_total_tokens=self.max_total_tokens,
        )
        self._sync()

    def before_model_call(self) -> None:
        self._require_token_capacity()
        if self.model_calls >= self.max_model_calls:
            raise self._exhausted("model_calls", self.max_model_calls, self.model_calls)
        self.model_calls += 1
        self._sync()

    def before_tool_call(self) -> None:
        self._require_token_capacity()
        if self.tool_calls >= self.max_tool_calls:
            raise self._exhausted("tool_calls", self.max_tool_calls, self.tool_calls)
        self.tool_calls += 1
        self._sync()

    def after_model_response(self, usage: Any | None) -> None:
        projected = project_token_usage(usage)
        self.latest_prompt_tokens = projected.prompt_tokens
        if projected.total_tokens is None:
            self.token_usage_unavailable += 1
            self._sync()
            return
        self.total_tokens += projected.total_tokens
        self._sync()
        if self.total_tokens > self.max_total_tokens:
            raise self._exhausted("total_tokens", self.max_total_tokens, self.total_tokens)

    def _require_token_capacity(self) -> None:
        if self.total_tokens >= self.max_total_tokens:
            raise self._exhausted("total_tokens", self.max_total_tokens, self.total_tokens)

    def should_summarize(self) -> bool:
        return (
            self.cumulative_budget is not None and self.total_tokens >= self.cumulative_budget
        ) or (
            self.summary_prompt_boundary is not None
            and self.latest_prompt_tokens is not None
            and self.latest_prompt_tokens >= self.summary_prompt_boundary
        )

    def _sync(self) -> None:
        self.metrics.observe_worker_budget(
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            total_tokens=self.total_tokens,
            token_usage_unavailable=self.token_usage_unavailable,
        )
