"""Bounded, secret-safe in-memory metrics for one Worker allocation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts import (
    ExecutionError,
    ExecutionMetrics,
    ExecutionReport,
    TerminationError,
    ToolCallOutcome,
    ToolCallRecord,
    ToolMetrics,
    WorkerBudgetMetrics,
    WorkerCompletionDiagnostics,
    WorkerSummarizerMetrics,
)
from contractor_runtime.token_usage import project_token_usage

MAX_METRIC_TOOL_CALLS = 1000
MAX_METRIC_ERRORS = 100
MAX_ARGUMENT_SUMMARY_BYTES = 4096
MAX_ERROR_MESSAGE_BYTES = 4096
MAX_REPORT_JSON_BYTES = 1024 * 1024
MAX_METRIC_ITEMS = 32
MAX_METRIC_DEPTH = 4
MAX_METRIC_TEXT_BYTES = 4096
MAX_METRIC_COUNTER = 2**64 - 1
_ALLOCATION_ENVELOPE_RESERVE_BYTES = 4096
SENSITIVE_METRIC_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "body",
        "bytes",
        "content",
        "command",
        "cwd",
        "stdout",
        "stderr",
        "cookie",
        "data",
        "data_base64",
        "llm_gateway_token",
        "password",
        "payload",
        "proxy_authorization",
        "secret",
        "set_cookie",
        "token",
    }
)
SAFE_DERIVED_SIZE_METRIC_KEYS = frozenset(
    {
        "content_bytes",
        "description_bytes",
        "result_content_bytes",
        "result_description_bytes",
    }
)
_TOOL_METRIC_CORRELATION: ContextVar[str | None] = ContextVar(
    "contractor_tool_metric_correlation", default=None
)


def bind_tool_metric_correlation(correlation_id: str) -> Token[str | None]:
    """Bind one Runtime-owned tool attempt to existing typed metric producers."""

    return _TOOL_METRIC_CORRELATION.set(correlation_id)


def reset_tool_metric_correlation(token: Token[str | None]) -> None:
    _TOOL_METRIC_CORRELATION.reset(token)


@dataclass(slots=True)
class MetricsState:
    """Allocation-wide counters and bounded detail accumulated across A2A calls."""

    counters: dict[str, int] = field(default_factory=dict)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    errors: list[ExecutionError] = field(default_factory=list)
    final_outcome: str | None = None
    truncated: bool = False
    _tool_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_call_number: int = 1
    _worker_budget: WorkerBudgetMetrics | None = None
    _summarizer: WorkerSummarizerMetrics | None = None
    _completion: WorkerCompletionDiagnostics | None = None
    _tool_correlation_outcomes: dict[str, bool] = field(default_factory=dict, repr=False)

    def record_completion(self, value: WorkerCompletionDiagnostics) -> None:
        self._completion = WorkerCompletionDiagnostics.model_validate(
            value.model_dump(by_alias=True)
        )

    def start_worker_budget(
        self, *, max_model_calls: int, max_tool_calls: int, max_total_tokens: int
    ) -> None:
        self._worker_budget = WorkerBudgetMetrics(
            maxModelCalls=max_model_calls,
            maxToolCalls=max_tool_calls,
            maxTotalTokens=max_total_tokens,
            observedModelCalls=0,
            observedToolCalls=0,
            observedTotalTokens=0,
            tokenUsageUnavailable=0,
        )

    def observe_worker_budget(
        self,
        *,
        model_calls: int,
        tool_calls: int,
        total_tokens: int,
        token_usage_unavailable: int,
    ) -> None:
        budget = self._worker_budget
        if budget is None:
            return
        self._worker_budget = budget.model_copy(
            update={
                "observed_model_calls": model_calls,
                "observed_tool_calls": tool_calls,
                "observed_total_tokens": total_tokens,
                "token_usage_unavailable": token_usage_unavailable,
            }
        )

    def record_worker_budget_exhausted(self, dimension: str) -> None:
        normalized = _metric_identifier(dimension)
        budget = self._worker_budget
        if budget is not None:
            self._worker_budget = budget.model_copy(update={"exhausted": normalized})
        self._append_error(
            ExecutionError(
                code="worker_budget_exhausted",
                message=f"Worker invocation budget exhausted ({normalized})",
                retryable=True,
            )
        )

    def record_summarizer_attempt(
        self,
        *,
        succeeded: bool,
        model_calls: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        token_usage_unavailable: int,
        failure_code: str | None = None,
    ) -> None:
        if model_calls not in {0, 1} or token_usage_unavailable not in {0, 1}:
            raise ValueError("summarizer call counters are invalid")
        if token_usage_unavailable > model_calls:
            raise ValueError("summarizer missing usage exceeds model calls")
        counters = (input_tokens, output_tokens, total_tokens)
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("summarizer token counters are invalid")
        if succeeded:
            if model_calls != 1 or failure_code is not None:
                raise ValueError("successful summarizer attempt is invalid")
        elif (
            not isinstance(failure_code, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", failure_code) is None
        ):
            raise ValueError("summarizer failure code is invalid")

        current = self._summarizer
        attempts = (current.attempts if current is not None else 0) + 1
        succeeded_count = (current.succeeded if current is not None else 0) + int(succeeded)
        failed_count = (current.failed if current is not None else 0) + int(not succeeded)
        failure_codes = dict(current.failure_codes) if current is not None else {}
        if failure_code is not None:
            failure_codes[failure_code] = min(
                MAX_METRIC_COUNTER, failure_codes.get(failure_code, 0) + 1
            )
        self._summarizer = WorkerSummarizerMetrics(
            attempts=min(MAX_METRIC_COUNTER, attempts),
            succeeded=min(MAX_METRIC_COUNTER, succeeded_count),
            failed=min(MAX_METRIC_COUNTER, failed_count),
            modelCalls=min(
                MAX_METRIC_COUNTER,
                (current.model_calls if current is not None else 0) + model_calls,
            ),
            inputTokens=min(
                MAX_METRIC_COUNTER,
                (current.input_tokens if current is not None else 0) + input_tokens,
            ),
            outputTokens=min(
                MAX_METRIC_COUNTER,
                (current.output_tokens if current is not None else 0) + output_tokens,
            ),
            totalTokens=min(
                MAX_METRIC_COUNTER,
                (current.total_tokens if current is not None else 0) + total_tokens,
            ),
            tokenUsageUnavailable=min(
                MAX_METRIC_COUNTER,
                (current.token_usage_unavailable if current is not None else 0)
                + token_usage_unavailable,
            ),
            failureCodes=failure_codes,
        )

    def record_tool_call(
        self,
        name: str,
        *,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any] | None = None,
        error: Exception | None = None,
        secrets: tuple[str, ...] = (),
        duration_ms: int | None = None,
        result_size_bytes: int | None = None,
    ) -> None:
        correlation_id = _TOOL_METRIC_CORRELATION.get()
        if correlation_id is not None and correlation_id in self._tool_correlation_outcomes:
            # A tool implementation and an adapter wrapper may both observe the
            # same logical call. The ADK plugin owns the correlation and admits
            # only the first already-sanitized projection.
            return
        identifier = _metric_identifier(name)
        self._increment("tool_calls")
        self._increment(f"tool_calls.{identifier}")
        aggregate = self._tool_metrics.setdefault(
            identifier, {"calls": 0, "succeeded": 0, "failed": 0}
        )
        aggregate["calls"] = min(MAX_METRIC_COUNTER, aggregate["calls"] + 1)

        sanitized_arguments, value_truncated = _sanitize_metric_value(arguments, secrets=secrets)
        arguments_summary, size_truncated = _bound_arguments(sanitized_arguments)
        call_error: ExecutionError | None = None
        if error is None:
            outcome = ToolCallOutcome.SUCCEEDED
            aggregate["succeeded"] = min(MAX_METRIC_COUNTER, aggregate["succeeded"] + 1)
        else:
            outcome = ToolCallOutcome.FAILED
            aggregate["failed"] = min(MAX_METRIC_COUNTER, aggregate["failed"] + 1)
            self._increment("tool_errors")
            code = _bounded_text(str(getattr(error, "code", "tool_call_failed")), secrets)
            retryable = bool(getattr(error, "retryable", False))
            call_error = ExecutionError(
                code=code,
                message=_bounded_error(f"Tool {name} failed ({type(error).__name__})", secrets),
                retryable=retryable,
            )
            self._append_error(
                ExecutionError(
                    code=f"tool_{identifier}_failed",
                    message=_bounded_error(f"Tool {name} failed ({type(error).__name__})", secrets),
                    retryable=retryable,
                )
            )

        if result_size_bytes is not None:
            if type(result_size_bytes) is not int or result_size_bytes < 0:
                raise ValueError("tool result metric size must not be negative")
            result_size = result_size_bytes
        else:
            result_size = _json_size(result) if result is not None else None
        record = ToolCallRecord(
            callId=f"tool-{self._next_call_number:08d}",
            tool=_bounded_text(name, secrets),
            arguments=arguments_summary,
            argumentsTruncated=value_truncated or size_truncated,
            outcome=outcome,
            durationMs=max(0, duration_ms) if duration_ms is not None else None,
            resultSizeBytes=result_size,
            error=call_error,
        )
        self._next_call_number += 1
        self._append_tool_call(record)
        if correlation_id is not None:
            self._tool_correlation_outcomes[correlation_id] = error is not None

    def correlated_tool_outcome(self, correlation_id: str) -> bool | None:
        """Return False/True for recorded success/failure, or None if absent."""

        return self._tool_correlation_outcomes.get(correlation_id)

    def release_tool_correlation(self, correlation_id: str) -> None:
        self._tool_correlation_outcomes.pop(correlation_id, None)

    def record_sandbox_execution(
        self,
        *,
        stdout_bytes: int,
        stderr_bytes: int,
        stdout_truncated: bool,
        stderr_truncated: bool,
        exit_code: int | None = None,
    ) -> None:
        for key, value in (
            ("stdout_bytes", stdout_bytes),
            ("stderr_bytes", stderr_bytes),
            ("stdout_truncated", int(stdout_truncated)),
            ("stderr_truncated", int(stderr_truncated)),
        ):
            self._increment(f"sandbox.{key}", value)
        if exit_code is not None:
            self._increment("sandbox.completed_nonzero" if exit_code else "sandbox.completed_zero")

    def record_model_call(self) -> None:
        self._increment("llm_calls")

    def record_model_usage(self, usage: Any) -> None:
        projected = project_token_usage(usage)
        for value, counter in (
            (projected.prompt_tokens, "input_tokens"),
            (projected.output_tokens, "output_tokens"),
            (projected.total_tokens, "total_tokens"),
            (projected.cached_input_tokens, "cached_input_tokens"),
        ):
            if value is not None:
                self._increment(counter, value)

    def record_model_error(self, error: Exception) -> None:
        error_type = getattr(error, "provider_error_type", type(error).__name__)
        if (
            not isinstance(error_type, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,127}", error_type) is None
        ):
            error_type = type(error).__name__
        self._increment("llm_errors")
        self._append_error(
            ExecutionError(
                code="model_call_failed",
                message=_bounded_error(f"Model call failed ({error_type})", ()),
                retryable=True,
            )
        )

    def record_workspace_export(
        self,
        *,
        state_bytes: int | None = None,
        diff_bytes: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self._increment("workspace_exports")
        if error is None:
            if (
                type(state_bytes) is not int
                or state_bytes < 0
                or type(diff_bytes) is not int
                or diff_bytes < 0
            ):
                raise ValueError("workspace export byte metrics must be non-negative")
            self._increment("workspace_exports.succeeded")
            self._increment("workspace_export_state_bytes", state_bytes)
            self._increment("workspace_export_diff_bytes", diff_bytes)
            return
        self._increment("workspace_exports.failed")
        cause = _metric_identifier(str(getattr(error, "cause", "unknown")))
        retryable = bool(getattr(error, "retryable", False))
        self._append_error(
            ExecutionError(
                code="workspace_export_failed",
                message=f"Workspace export failed ({cause})",
                retryable=retryable,
            )
        )

    def record_outcome(self, outcome: str) -> None:
        normalized = _metric_identifier(outcome)
        self.final_outcome = normalized
        self._increment(f"outcomes.{normalized}")

    def snapshot(self) -> dict[str, Any]:
        """Return the ADK State projection; it intentionally contains no raw results."""

        result = {
            "counters": dict(sorted(self.counters.items())),
            "toolCalls": [
                call.model_dump(mode="json", by_alias=True, exclude_none=True)
                for call in self.tool_calls
            ],
            "errors": [
                error.model_dump(mode="json", by_alias=True, exclude_none=True)
                for error in self.errors
            ],
            "finalOutcome": self.final_outcome,
            "truncated": self.truncated,
        }
        if self._worker_budget is not None:
            result["workerBudget"] = self._worker_budget.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        if self._completion is not None:
            result["completion"] = self._completion.model_dump(by_alias=True, exclude_none=True)
        return result

    def build_report(
        self,
        *,
        report_id: str,
        duration_ms: int,
        complete: bool = True,
        extra_errors: tuple[TerminationError, ...] = (),
    ) -> ExecutionReport:
        errors = list(self.errors)
        truncated = self.truncated
        for item in extra_errors:
            errors.append(
                ExecutionError(
                    code=item.code,
                    message=_bounded_error(item.message, ()),
                    retryable=item.retryable,
                )
            )
        if len(errors) > MAX_METRIC_ERRORS:
            errors = errors[-MAX_METRIC_ERRORS:]
            truncated = True

        tools = {
            name: ToolMetrics(
                calls=values["calls"],
                succeeded=values["succeeded"],
                failed=values["failed"],
            )
            for name, values in sorted(self._tool_metrics.items())
        }
        metrics = ExecutionMetrics(
            durationMs=max(0, duration_ms),
            modelCalls=self.counters.get("llm_calls", 0),
            inputTokens=self.counters.get("input_tokens"),
            outputTokens=self.counters.get("output_tokens"),
            totalTokens=self.counters.get("total_tokens"),
            tools=tools,
            workerBudget=self._worker_budget,
            summarizer=self._summarizer,
        )
        tool_calls = list(self.tool_calls)
        report = ExecutionReport(
            reportId=report_id,
            complete=complete,
            metrics=metrics,
            toolCalls=tool_calls,
            errors=errors,
            truncated=truncated,
            completion=self._completion.model_copy(deep=True) if self._completion else None,
        )
        maximum = MAX_REPORT_JSON_BYTES - _ALLOCATION_ENVELOPE_RESERVE_BYTES
        while (
            _json_size(report.model_dump(mode="json", by_alias=True, exclude_none=True)) > maximum
        ):
            if tool_calls:
                tool_calls.pop(0)
            elif errors:
                errors.pop(0)
            else:
                raise ValueError("aggregate execution report exceeds the 1 MiB limit")
            report = report.model_copy(
                update={"tool_calls": list(tool_calls), "errors": list(errors), "truncated": True}
            )
        return report

    def _increment(self, name: str, value: int = 1) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("metric counter increment must be a non-negative integer")
        self.counters[name] = min(MAX_METRIC_COUNTER, self.counters.get(name, 0) + value)

    def _append_tool_call(self, record: ToolCallRecord) -> None:
        if len(self.tool_calls) >= MAX_METRIC_TOOL_CALLS:
            self.tool_calls.pop(0)
            self.truncated = True
        self.tool_calls.append(record)

    def _append_error(self, error: ExecutionError) -> None:
        if len(self.errors) >= MAX_METRIC_ERRORS:
            self.errors.pop(0)
            self.truncated = True
        self.errors.append(error)


def _sanitize_metric_value(
    value: Any,
    *,
    secrets: tuple[str, ...],
    key: str | None = None,
    depth: int = 0,
) -> tuple[Any, bool]:
    if (
        key is not None
        and _metric_identifier(key) in SAFE_DERIVED_SIZE_METRIC_KEYS
        and type(value) is int
        and value >= 0
    ):
        return value, False
    if key is not None and _is_sensitive_metric_key(key):
        return {"redacted": True, "size": _value_size(value)}, False
    if depth >= MAX_METRIC_DEPTH:
        return "[TRUNCATED]", True
    if isinstance(value, bytes | bytearray | memoryview):
        return {"bytes": len(value)}, False
    if isinstance(value, str):
        bounded = _bounded_text(value, secrets)
        return bounded, bounded != value
    if value is None or isinstance(value, bool | int):
        return value, False
    if isinstance(value, float):
        return (value, False) if math.isfinite(value) else ("[NON_FINITE]", True)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        truncated = False
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= MAX_METRIC_ITEMS:
                result["__truncated__"] = True
                truncated = True
                break
            normalized_key = _bounded_text(str(item_key), secrets)
            sanitized, child_truncated = _sanitize_metric_value(
                item_value,
                secrets=secrets,
                key=str(item_key),
                depth=depth + 1,
            )
            result[normalized_key] = sanitized
            truncated = truncated or child_truncated or normalized_key != str(item_key)
        return result, truncated
    if isinstance(value, list | tuple):
        result: list[Any] = []
        truncated = len(value) > MAX_METRIC_ITEMS
        for item in value[:MAX_METRIC_ITEMS]:
            sanitized, child_truncated = _sanitize_metric_value(
                item, secrets=secrets, depth=depth + 1
            )
            result.append(sanitized)
            truncated = truncated or child_truncated
        if len(value) > MAX_METRIC_ITEMS:
            result.append("[TRUNCATED]")
        return result, truncated
    return f"<{type(value).__name__}>", True


def _bound_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, dict):
        return {"value": value}, False
    if _json_size(value) <= MAX_ARGUMENT_SUMMARY_BYTES:
        return value, False
    return {"summary": "[TRUNCATED]", "originalSizeBytes": _json_size(value)}, True


def _bounded_text(value: str, secrets: tuple[str, ...]) -> str:
    if "://" in value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            # A URL-shaped value which cannot be parsed safely is not useful
            # diagnostic detail and may still contain credentials.
            return "[REDACTED_URL]"
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return "[REDACTED_URL]"
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return _truncate_utf8(result, MAX_METRIC_TEXT_BYTES)


def _bounded_error(value: str, secrets: tuple[str, ...]) -> str:
    return _truncate_utf8(_bounded_text(value, secrets), MAX_ERROR_MESSAGE_BYTES)


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    suffix = "…".encode()
    return encoded[: limit - len(suffix)].decode("utf-8", errors="ignore") + "…"


def _json_size(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        )
    except (TypeError, ValueError):
        return 0


def _value_size(value: Any) -> int | None:
    if isinstance(value, str | bytes | bytearray | memoryview | list | tuple | Mapping):
        return len(value)
    return None


def _metric_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized[:64] or "unknown"


def _is_sensitive_metric_key(value: str) -> bool:
    normalized = _metric_identifier(value)
    compact = normalized.replace("_", "")
    if normalized in SENSITIVE_METRIC_KEYS or compact in {
        item.replace("_", "") for item in SENSITIVE_METRIC_KEYS
    }:
        return True
    if any(
        fragment in compact
        for fragment in (
            "apikey",
            "authorization",
            "base64",
            "cookie",
            "password",
            "secret",
            "token",
        )
    ):
        return True
    return compact.endswith(("body", "bytes", "content", "data", "payload"))
