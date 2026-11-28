"""Bounded, secret-safe in-memory metrics for one Worker allocation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.contracts import (
    MAX_EXACT_INTEGER,
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
from contractor_runtime.contracts.base import ARGUMENT_NUMBER_MARKER, argument_number_in_range
from contractor_runtime.contracts.compaction import (
    WorkerCompactionProgress,
    WorkerModelPhases,
    WorkerModelUsage,
)
from contractor_runtime.contracts.llm_cost import CostPhase
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.llm.cost import CostCollector
from contractor_runtime.llm.errors import GatewayFailure
from contractor_runtime.llm.usage import project_token_usage
from contractor_runtime.redaction import redact_private_values, substring_values

MAX_METRIC_TOOL_CALLS = 1000
MAX_METRIC_ERRORS = 100
MAX_ARGUMENT_SUMMARY_BYTES = 4096
MAX_ERROR_MESSAGE_BYTES = 4096
MAX_REPORT_JSON_BYTES = 1024 * 1024
MAX_METRIC_ITEMS = 32
MAX_METRIC_DEPTH = 4
MAX_METRIC_TEXT_BYTES = 4096
# Allocation counters saturate at the Worker State bound; their report
# projection saturates at the lower report bound, MAX_EXACT_INTEGER (OBS-R111).
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
        "database_url",
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
# Tool error codes are Runtime vocabulary and are never redacted. Text in a code
# position that is not such an identifier is replaced by the generic code.
SAFE_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TOOL_METRIC_CORRELATION: ContextVar[str | None] = ContextVar(
    "contractor_tool_metric_correlation", default=None
)


def safe_error_code(error: Exception | None) -> str:
    code = getattr(error, "code", "tool_call_failed")
    if not isinstance(code, str) or SAFE_ERROR_CODE.fullmatch(code) is None:
        return "tool_call_failed"
    return code


def bind_tool_metric_correlation(correlation_id: str) -> Token[str | None]:
    """Bind one Runtime-owned tool attempt to existing typed metric producers."""

    return _TOOL_METRIC_CORRELATION.set(correlation_id)


def reset_tool_metric_correlation(token: Token[str | None]) -> None:
    _TOOL_METRIC_CORRELATION.reset(token)


@dataclass(slots=True)
class MetricsState:
    """Allocation-wide counters and bounded detail accumulated across A2A calls.

    Retained tool-call and error records change only through the recording
    methods. Each admitted record is serialized once, there, into the JSON
    projection that every later State snapshot reuses (WRK-R169).
    """

    counters: dict[str, int] = field(default_factory=dict)
    llm_cost: CostCollector = field(default_factory=CostCollector, repr=False)
    final_outcome: str | None = None
    truncated: bool = False
    _tool_calls: list[ToolCallRecord] = field(default_factory=list)
    _errors: list[ExecutionError] = field(default_factory=list)
    # Index-aligned with the record lists and bounded by the same retention
    # limits. The dictionaries are never mutated after admission.
    _tool_call_projections: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _error_projections: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _tool_metrics: dict[str, dict[str, int]] = field(default_factory=dict)
    _next_call_number: int = 1
    _worker_budget: WorkerBudgetMetrics | None = None
    _summarizer: WorkerSummarizerMetrics | None = None
    _completion: WorkerCompletionDiagnostics | None = None
    _model_phases: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            phase: dict.fromkeys(WorkerModelUsage.model_fields, 0)
            for phase in ("main", "compaction", "finalization")
        }
    )
    _tool_correlation_outcomes: dict[str, bool] = field(default_factory=dict, repr=False)

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        """Retained tool-call records, oldest first."""

        return tuple(self._tool_calls)

    @property
    def errors(self) -> tuple[ExecutionError, ...]:
        """Retained errors, oldest first."""

        return tuple(self._errors)

    def record_completion(self, value: WorkerCompletionDiagnostics) -> None:
        self._completion = WorkerCompletionDiagnostics.model_validate(
            value.model_dump(by_alias=True)
        )

    def model_phase_usage(self, phase: CostPhase) -> WorkerModelUsage:
        if phase == "summarization":
            summary = self._summarizer
            return WorkerModelUsage(
                **{
                    key: getattr(summary, key) if summary is not None else 0
                    for key in WorkerModelUsage.model_fields
                    if key != "duration_ms"
                }
            )
        return WorkerModelUsage(
            **{key: _report_integer(value) for key, value in self._model_phases[phase].items()}
        )

    def model_phases(self) -> WorkerModelPhases:
        return WorkerModelPhases(
            **{phase: self.model_phase_usage(phase) for phase in WorkerModelPhases.model_fields}
        )

    def record_model_duration(self, phase: CostPhase, duration_ms: int) -> None:
        self._increment_phase(phase, "duration_ms", max(0, duration_ms))

    def _increment_phase(self, phase: CostPhase, counter: str, amount: int = 1) -> None:
        values = self._model_phases[phase]
        values[counter] = min(MAX_METRIC_COUNTER, values[counter] + amount)

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
                "observed_total_tokens": _report_integer(total_tokens),
                "token_usage_unavailable": _report_integer(token_usage_unavailable),
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
                MAX_EXACT_INTEGER, failure_codes.get(failure_code, 0) + 1
            )
        self._summarizer = WorkerSummarizerMetrics(
            attempts=min(MAX_EXACT_INTEGER, attempts),
            succeeded=min(MAX_EXACT_INTEGER, succeeded_count),
            failed=min(MAX_EXACT_INTEGER, failed_count),
            modelCalls=min(
                MAX_EXACT_INTEGER,
                (current.model_calls if current is not None else 0) + model_calls,
            ),
            inputTokens=min(
                MAX_EXACT_INTEGER,
                (current.input_tokens if current is not None else 0) + input_tokens,
            ),
            outputTokens=min(
                MAX_EXACT_INTEGER,
                (current.output_tokens if current is not None else 0) + output_tokens,
            ),
            totalTokens=min(
                MAX_EXACT_INTEGER,
                (current.total_tokens if current is not None else 0) + total_tokens,
            ),
            tokenUsageUnavailable=min(
                MAX_EXACT_INTEGER,
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
            code = _metric_error_code(error, secrets)
            retryable = bool(getattr(error, "retryable", False))
            message = (
                error.diagnostic_message
                if isinstance(error, ToolInputError)
                else f"Tool {name} failed ({type(error).__name__})"
            )
            call_error = ExecutionError(
                code=code,
                message=_bounded_error(message, secrets),
                retryable=retryable,
            )
            self._append_error(
                ExecutionError(
                    code=f"tool_{identifier}_failed",
                    message=_bounded_error(message, secrets),
                    retryable=retryable,
                )
            )

        if result_size_bytes is not None:
            if type(result_size_bytes) is not int or result_size_bytes < 0:
                raise ValueError("tool result metric size must not be negative")
            result_size = result_size_bytes
        else:
            result_size = _json_size(result) if result is not None else None
        optional = {
            "arguments": arguments_summary,
            "durationMs": max(0, duration_ms) if duration_ms is not None else None,
            "resultSizeBytes": result_size,
            "error": call_error,
        }
        record = ToolCallRecord(
            callId=f"tool-{self._next_call_number:08d}",
            # A Runtime-registered tool name is bounded but never redacted.
            tool=_truncate_utf8(name, MAX_METRIC_TEXT_BYTES),
            argumentsTruncated=value_truncated or size_truncated,
            outcome=outcome,
            **{key: value for key, value in optional.items() if value is not None},
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

    def record_model_call(self, *, phase: CostPhase = "main") -> None:
        self._increment("llm_calls")
        self._increment_phase(phase, "model_calls")
        # An in-flight or rejected response is unavailable until actual usage arrives.
        self._increment_phase(phase, "token_usage_unavailable")

    def record_model_usage(self, usage: Any, *, phase: CostPhase = "main") -> None:
        projected = project_token_usage(usage)
        for value, counter in (
            (projected.prompt_tokens, "input_tokens"),
            (projected.output_tokens, "output_tokens"),
            (projected.total_tokens, "total_tokens"),
            (projected.cached_input_tokens, "cached_input_tokens"),
        ):
            if value is not None:
                self._increment(counter, value)
                if counter != "cached_input_tokens":
                    self._increment_phase(phase, counter, value)
        if projected.total_tokens is not None:
            values = self._model_phases[phase]
            values["token_usage_unavailable"] = max(0, values["token_usage_unavailable"] - 1)

    def record_recovery_authority_retry(self, cause: str) -> None:
        """Count one reconnect to the model recovery authority; cause is a safe token."""
        self._increment("llm_recovery_authority_retries")
        if re.fullmatch(r"[a-z0-9_]{1,32}", cause):
            self._increment(f"llm_recovery_authority_retries.{cause}")

    def record_model_error(self, error: Exception) -> None:
        error_type = getattr(error, "provider_error_type", type(error).__name__)
        if (
            not isinstance(error_type, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,127}", error_type) is None
        ):
            error_type = type(error).__name__
        failure = getattr(error, "failure", None)
        detail = error_type
        if isinstance(failure, GatewayFailure):
            detail += f"; {failure.code}"
            if failure.status is not None:
                detail += f"; HTTP {failure.status}"
        self._increment("llm_errors")
        self._append_error(
            ExecutionError(
                code="model_call_failed",
                message=_bounded_error(f"Model call failed ({detail})", ()),
                retryable=bool(getattr(error, "retryable", True)),
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

    def record_tool_report_failure(self) -> None:
        # Publishing is a Worker operation, separate from the scanner's outcome.
        # Never attach transport exception text, paths, or report payloads here.
        self._append_error(
            ExecutionError(
                code="tool_report_failed",
                message="Tool report publication failed",
                retryable=False,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        """Return the ADK State projection; it intentionally contains no raw results.

        The returned dictionary and lists are new, but the retained record
        projections inside them are shared with this reducer and with later
        snapshots, so the caller must not mutate them. WorkerStateStore, the
        only production caller, deep-copies every State that leaves it.
        """

        result = {
            "counters": dict(sorted(self.counters.items())),
            # Copies of the reference lists: linear in retained records, but
            # no record is serialized again.
            "toolCalls": list(self._tool_call_projections),
            "errors": list(self._error_projections),
            "finalOutcome": self.final_outcome,
            "truncated": self.truncated,
        }
        if (
            any(v["model_calls"] for v in self._model_phases.values())
            or self._summarizer is not None
        ):
            result["modelPhases"] = self.model_phases().model_dump(by_alias=True)
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
        compaction: WorkerCompactionProgress | None = None,
    ) -> ExecutionReport:
        errors = list(self._errors)
        error_projections = list(self._error_projections)
        truncated = self.truncated
        for item in extra_errors:
            error = ExecutionError(
                code=item.code,
                message=_bounded_error(item.message, ()),
                retryable=item.retryable,
            )
            errors.append(error)
            error_projections.append(_record_projection(error))
        if len(errors) > MAX_METRIC_ERRORS:
            errors = errors[-MAX_METRIC_ERRORS:]
            error_projections = error_projections[-MAX_METRIC_ERRORS:]
            truncated = True

        tools = {
            name: ToolMetrics(
                calls=_report_integer(values["calls"]),
                succeeded=_report_integer(values["succeeded"]),
                failed=_report_integer(values["failed"]),
            )
            for name, values in sorted(self._tool_metrics.items())
        }
        model_calls = self._report_counter("llm_calls") or 0

        def token_counter(name: str) -> int | None:
            value = self._report_counter(name)
            # This producer knows no direct call started. Delegated Planner
            # usage belongs to its Planner report, not this allocation report.
            # A started call with no usage still leaves its counters absent.
            return 0 if value is None and model_calls == 0 else value

        optional_metrics = {
            "modelPhases": self.model_phases()
            if any(v["model_calls"] for v in self._model_phases.values())
            or self._summarizer is not None
            else None,
            "inputTokens": token_counter("input_tokens"),
            "outputTokens": token_counter("output_tokens"),
            "totalTokens": token_counter("total_tokens"),
            "workerBudget": self._worker_budget,
            "summarizer": self._summarizer,
        }
        metrics = ExecutionMetrics(
            llmCost=self.llm_cost.snapshot(self._report_counter("llm_calls") or 0),
            durationMs=_report_integer(max(0, duration_ms)),
            modelCalls=model_calls,
            tools=tools,
            **{key: value for key, value in optional_metrics.items() if value is not None},
        )
        report = ExecutionReport(
            reportId=report_id,
            complete=complete,
            metrics=metrics,
            toolCalls=list(self._tool_calls),
            errors=errors,
            truncated=truncated,
            **({"completion": self._completion.model_copy(deep=True)} if self._completion else {}),
            **({"compaction": compaction} if compaction is not None else {}),
        )
        return trim_report_detail(
            report,
            MAX_REPORT_JSON_BYTES - _ALLOCATION_ENVELOPE_RESERVE_BYTES,
            tool_call_projections=self._tool_call_projections,
            error_projections=error_projections,
        )

    def _report_counter(self, name: str) -> int | None:
        value = self.counters.get(name)
        return _report_integer(value) if value is not None else None

    def _increment(self, name: str, value: int = 1) -> None:
        if type(value) is not int or value < 0:
            raise ValueError("metric counter increment must be a non-negative integer")
        self.counters[name] = min(MAX_METRIC_COUNTER, self.counters.get(name, 0) + value)

    def _append_tool_call(self, record: ToolCallRecord) -> None:
        projection = _record_projection(record)
        if len(self._tool_calls) >= MAX_METRIC_TOOL_CALLS:
            self._tool_calls.pop(0)
            self._tool_call_projections.pop(0)
            self.truncated = True
        self._tool_calls.append(record)
        self._tool_call_projections.append(projection)

    def _append_error(self, error: ExecutionError) -> None:
        projection = _record_projection(error)
        if len(self._errors) >= MAX_METRIC_ERRORS:
            self._errors.pop(0)
            self._error_projections.pop(0)
            self.truncated = True
        self._errors.append(error)
        self._error_projections.append(projection)


def _report_integer(value: int) -> int:
    """Saturate an allocation counter at the execution report bound (OBS-R111)."""

    return min(MAX_EXACT_INTEGER, value)


def trim_report_detail(
    report: ExecutionReport,
    maximum: int,
    *,
    tool_call_projections: Sequence[Mapping[str, Any]],
    error_projections: Sequence[Mapping[str, Any]],
) -> ExecutionReport:
    """Bound a report's JSON size by removing its oldest detail (OBS-R41, OBS-R110).

    Overflow removes the oldest tool-call record first and the oldest error
    only after no tool-call record remains, and marks the report truncated: the
    result is the first report of that removal sequence whose compact JSON fits
    ``maximum``. The projections are the records' JSON projections, index-aligned
    with ``report.tool_calls`` and ``report.errors``. The JSON of a list is its
    elements' JSON joined by commas, so each record is measured once and every
    removal is subtracted from the size; the complete report is measured at most
    three times, for the candidate, the envelope without detail and the retained
    result, however many records are removed. The Server's telemetry.trimDetail
    applies the same rule, and api/testdata/v1alpha1/telemetry-report-trim-cases.json
    pins both.
    """

    if len(tool_call_projections) != len(report.tool_calls) or len(error_projections) != len(
        report.errors
    ):
        raise RuntimeError("report record projections are not aligned with the report")
    if _report_size(report) <= maximum:
        return report
    envelope = report.model_copy(update={"tool_calls": [], "errors": [], "truncated": True})
    call_sizes = [_json_size(projection) for projection in tool_call_projections]
    error_sizes = [_json_size(projection) for projection in error_projections]
    size = _report_size(envelope) + _list_size(call_sizes) + _list_size(error_sizes)
    # The candidate does not fit, so at least one record goes even when the
    # shorter "truncated":true alone would make it fit.
    calls = errors = 0
    removed = False
    while not removed or size > maximum:
        if calls < len(call_sizes):
            size -= _removal_size(call_sizes, calls)
            calls += 1
        elif errors < len(error_sizes):
            size -= _removal_size(error_sizes, errors)
            errors += 1
        else:
            raise ValueError("aggregate execution report exceeds the 1 MiB limit")
        removed = True
    result = report.model_copy(
        update={
            "tool_calls": report.tool_calls[calls:],
            "errors": report.errors[errors:],
            "truncated": True,
        }
    )
    measured = _report_size(result)
    if measured != size:
        raise RuntimeError(f"retained report measures {measured} bytes, not the computed {size}")
    return result


def _record_projection(record: ToolCallRecord | ExecutionError) -> dict[str, Any]:
    """The JSON projection of one record, as the complete report renders it."""

    return record.model_dump(mode="json", by_alias=True, exclude_none=True)


def _report_size(report: ExecutionReport) -> int:
    return _json_size(report.model_dump(mode="json", by_alias=True, exclude_none=True))


def _list_size(sizes: Sequence[int]) -> int:
    """JSON size of a list's elements and separators; the envelope holds the brackets."""

    return sum(sizes) + max(0, len(sizes) - 1)


def _removal_size(sizes: Sequence[int], index: int) -> int:
    """What removing ``sizes[index]``, the first remaining element, saves."""

    return sizes[index] + (1 if index + 1 < len(sizes) else 0)


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
        and argument_number_in_range(value)
    ):
        return value, False
    if key is not None and _is_sensitive_metric_key(key):
        if _is_redaction_placeholder(value):
            # Redacting twice changes nothing, so the Server's second pass
            # keeps a record the Runtime redacted and its size (OBS-R27).
            return {"redacted": True, "size": value["size"]}, False
        return {"redacted": True, "size": _value_size(value)}, False
    if depth >= MAX_METRIC_DEPTH:
        return "[TRUNCATED]", True
    if isinstance(value, bytes | bytearray | memoryview):
        return {"redacted": True, "size": len(value)}, False
    if isinstance(value, str):
        bounded = _bounded_text(value, secrets)
        return bounded, bounded != value
    if value is None or isinstance(value, bool):
        return value, False
    if isinstance(value, int | float):
        # The Server rejects a number outside LIM-OBS-16, NaN and infinities.
        if argument_number_in_range(value):
            return value, False
        return ARGUMENT_NUMBER_MARKER, True
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
    size = _json_size(value)
    if size <= MAX_ARGUMENT_SUMMARY_BYTES:
        return value, False
    return {"summary": "[TRUNCATED]", "originalSizeBytes": size}, True


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
    return _truncate_utf8(redact_private_values(value, secrets), MAX_METRIC_TEXT_BYTES)


def _metric_error_code(error: Exception, secrets: tuple[str, ...]) -> str:
    code = safe_error_code(error)
    # A code is never partially redacted. One exposing a long private value,
    # which no Runtime code does, is replaced whole.
    if any(secret in code for secret in substring_values(secrets)):
        return "tool_call_failed"
    return code


def _bounded_error(value: str, secrets: tuple[str, ...]) -> str:
    return _truncate_utf8(_bounded_text(value, secrets), MAX_ERROR_MESSAGE_BYTES)


def _truncate_utf8(value: str, limit: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return encoded.decode("utf-8")
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


def _is_redaction_placeholder(value: Any) -> bool:
    if not isinstance(value, Mapping) or len(value) != 2 or value.get("redacted") is not True:
        return False
    size = value.get("size", False)
    return size is None or (type(size) is int and size >= 0 and argument_number_in_range(size))


def _value_size(value: Any) -> int | None:
    if isinstance(value, str | bytes | bytearray | memoryview | list | tuple | Mapping):
        return len(value)
    return None


def _metric_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized[:64] or "unknown"


def _is_sensitive_metric_key(value: str) -> bool:
    # The whole name is classified; a metric identifier would cut it at 64
    # characters and miss a sensitive word after them.
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
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
