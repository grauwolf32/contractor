"""Private Runtime protocol reports models and validation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from contractor_runtime.contracts.base import (
    _RUNTIME_ADAPTER_ERROR_CODES,
    MAX_UINT64,
    WORKER_FAILURE_CODE_PATTERN,
    RuntimeAdapterRef,
    VersionedWireModel,
    WireModel,
    _known_completion,
    _require_aware_datetime,
    _require_text,
)


class ExecutionError(WireModel):
    code: str
    message: str = Field(max_length=4096)
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


class ToolMetrics(WireModel):
    calls: int | None = Field(default=None, ge=0)
    succeeded: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)


class WorkerBudgetMetrics(WireModel):
    max_model_calls: int = Field(gt=0, le=1000)
    max_tool_calls: int = Field(gt=0, le=10_000)
    max_total_tokens: int = Field(gt=0, le=100_000_000)
    observed_model_calls: int = Field(ge=0)
    observed_tool_calls: int = Field(ge=0)
    observed_total_tokens: int = Field(ge=0)
    token_usage_unavailable: int = Field(ge=0)
    exhausted: Literal["model_calls", "tool_calls", "total_tokens"] | None = None

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if self.observed_model_calls > self.max_model_calls:
            raise ValueError("observedModelCalls exceeds maxModelCalls")
        if self.observed_tool_calls > self.max_tool_calls:
            raise ValueError("observedToolCalls exceeds maxToolCalls")
        if self.exhausted == "model_calls" and self.observed_model_calls != self.max_model_calls:
            raise ValueError("model-call exhaustion is inconsistent")
        if self.exhausted == "tool_calls" and self.observed_tool_calls != self.max_tool_calls:
            raise ValueError("tool-call exhaustion is inconsistent")
        if self.exhausted == "total_tokens" and self.observed_total_tokens < self.max_total_tokens:
            raise ValueError("token exhaustion is inconsistent")
        return self


class WorkerSummarizerMetrics(WireModel):
    attempts: int = Field(gt=0, le=MAX_UINT64)
    succeeded: int = Field(ge=0, le=MAX_UINT64)
    failed: int = Field(ge=0, le=MAX_UINT64)
    model_calls: int = Field(ge=0, le=MAX_UINT64)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    failure_codes: dict[str, int] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if self.succeeded + self.failed != self.attempts:
            raise ValueError("Worker summarizer terminal counts are inconsistent")
        if self.model_calls > self.attempts:
            raise ValueError("Worker summarizer model calls exceed attempts")
        if self.token_usage_unavailable > self.model_calls:
            raise ValueError("Worker summarizer missing-usage count exceeds model calls")
        failure_total = 0
        for code, count in self.failure_codes.items():
            if WORKER_FAILURE_CODE_PATTERN.fullmatch(code) is None or count <= 0:
                raise ValueError("Worker summarizer failure aggregate is invalid")
            failure_total += count
        if failure_total != self.failed:
            raise ValueError("Worker summarizer failure aggregate is inconsistent")
        return self


class ToolCallOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkerCompletionDiagnostics(WireModel):
    """Latest invocation facts; publication is not Audit evidence acceptance."""

    kind: Literal["audit-check-results@1"]
    phase: Literal["collecting", "sealed", "publishing", "published", "failed"]
    accepted_count: int = Field(ge=0, le=64)
    total_count: int = Field(ge=1, le=64)
    reminder_count: int = Field(ge=0, le=2)
    failure_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,63}$")

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if self.accepted_count > self.total_count or (
            self.phase in {"sealed", "publishing", "published"}
            and self.accepted_count != self.total_count
        ):
            raise ValueError("inconsistent completion counts")
        if (self.phase == "failed") != (self.failure_code is not None):
            raise ValueError("inconsistent completion failure")
        return self


ResourceReason = Literal[
    "unsupported_platform", "read_failed", "sampling_gap", "counter_reset", "invalid_report"
]
ResourceNumber = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ResourceInteger = Annotated[int, Field(ge=0, le=2**53 - 1)]


class PerformanceMetricsRequest(WireModel):
    version: Literal[1]
    interval_seconds: Literal[15]

    @model_validator(mode="before")
    @classmethod
    def reject_coerced_literals(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("version", "intervalSeconds", "interval_seconds"):
                if key in value and type(value[key]) is not int:
                    raise ValueError("performance request fields must be integers")
        return value


class DroppedSpanCounts(WireModel):
    """Locally discarded spans by reason; delivery may be ambiguous on I/O failure."""

    queue_overflow: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    encoding_failed: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    non_retryable: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    retry_exhausted: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    collector_rejected: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    deadline_exceeded: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    cancelled: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    shutdown: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)


class RuntimeAdapterMetricsV2(WireModel):
    operations: int = Field(ge=0, le=2**64 - 1)
    failed_operations: int = Field(ge=0, le=2**64 - 1)
    flush_attempted: bool | None = None
    flush_succeeded: bool | None = None
    last_error_code: str | None = None
    dropped_spans: DroppedSpanCounts | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.failed_operations > self.operations:
            raise ValueError("Runtime adapter failures exceed operations")
        if (self.flush_attempted is None) != (self.flush_succeeded is None):
            raise ValueError("Runtime adapter flush fields must be present together")
        if self.flush_attempted is False and self.flush_succeeded:
            raise ValueError("Runtime adapter flush cannot succeed when not attempted")
        if (
            self.last_error_code is not None
            and self.last_error_code not in _RUNTIME_ADAPTER_ERROR_CODES
        ):
            raise ValueError("Runtime adapter lastErrorCode is invalid")
        return self


class ExecutionMetrics(WireModel):
    duration_ms: int | None = Field(default=None, ge=0)
    model_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    tools: dict[str, ToolMetrics] = Field(default_factory=dict)
    worker_budget: WorkerBudgetMetrics | None = None
    summarizer: WorkerSummarizerMetrics | None = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: dict[str, ToolMetrics]) -> dict[str, ToolMetrics]:
        for name in value:
            _require_text("tool name", name)
        return value


class ToolCallRecord(WireModel):
    call_id: str
    tool: str
    arguments: dict[str, Any] | None = None
    arguments_truncated: bool = False
    outcome: ToolCallOutcome
    duration_ms: int | None = Field(default=None, ge=0)
    result_size_bytes: int | None = Field(default=None, ge=0)
    error: ExecutionError | None = None

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        _require_text("callId", self.call_id)
        _require_text("tool", self.tool)
        if self.arguments is None and self.arguments_truncated:
            raise ValueError("absent arguments cannot be marked truncated")
        if (self.outcome is ToolCallOutcome.SUCCEEDED) == (self.error is not None):
            raise ValueError("tool outcome and error are inconsistent")
        return self


class RuntimeResources(WireModel):
    version: Literal[1]
    scope: Literal["runtime_process"]
    status: Literal["complete", "partial", "unavailable"]
    reason: ResourceReason | None = None
    duration_seconds: ResourceNumber | None = None
    cpu_user_seconds: ResourceNumber | None = None
    cpu_system_seconds: ResourceNumber | None = None
    rss_start_bytes: ResourceInteger | None = None
    rss_end_bytes: ResourceInteger | None = None
    rss_peak_observed_bytes: ResourceInteger | None = None
    rss_sample_count: ResourceInteger | None = None
    max_sample_gap_seconds: ResourceNumber | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_and_coerced_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if any(item is None for item in value.values()):
                raise ValueError("unknown resource fields must be omitted, not null")
            if "version" in value and type(value["version"]) is not int:
                raise ValueError("resource version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_resources(self) -> Self:
        if (
            self.max_sample_gap_seconds is not None
            and self.duration_seconds is not None
            and self.max_sample_gap_seconds > self.duration_seconds
        ):
            raise ValueError("resource sampling gap exceeds duration")
        has_samples = self.rss_sample_count is not None and self.rss_sample_count > 0
        if has_samples != (self.rss_peak_observed_bytes is not None):
            raise ValueError("resource peak and sample count are inconsistent")
        boundaries = [v for v in (self.rss_start_bytes, self.rss_end_bytes) if v is not None]
        if any(
            self.rss_peak_observed_bytes is None or value > self.rss_peak_observed_bytes
            for value in boundaries
        ):
            raise ValueError("resource boundary exceeds observed peak")
        if boundaries and (self.rss_sample_count or 0) < len(boundaries):
            raise ValueError("resource sample count omits boundaries")
        if self.status == "complete" and (
            self.reason is not None
            or self.duration_seconds is None
            or self.cpu_user_seconds is None
            or self.cpu_system_seconds is None
            or len(boundaries) != 2
            or self.max_sample_gap_seconds is None
            or self.max_sample_gap_seconds > 30
        ):
            raise ValueError(
                "complete resources require successful boundaries and bounded coverage"
            )
        return self


class ExecutionReport(WireModel):
    report_id: str
    complete: bool
    metrics: ExecutionMetrics
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    truncated: bool = False
    completion: WorkerCompletionDiagnostics | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("completion", mode="before")
    @classmethod
    def known_completion(cls, value):
        return _known_completion(value)

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, value: str) -> str:
        return _require_text("reportId", value)


class RuntimeReport(WireModel):
    complete: bool
    duration_ms: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    adapters: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2] = Field(default_factory=dict)
    resources: RuntimeResources | None = None
    _resources_error: ResourceReason | None = PrivateAttr(default=None)

    @property
    def resources_error(self) -> ResourceReason | None:
        return self._resources_error

    @model_validator(mode="wrap")
    @classmethod
    def isolate_resources(cls, value: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        try:
            result = handler(value)
        except ValidationError as error:
            if not isinstance(value, dict) or "resources" not in value:
                raise
            if not all(item["loc"] and item["loc"][0] == "resources" for item in error.errors()):
                raise
            result = handler({key: item for key, item in value.items() if key != "resources"})
            result._resources_error = "invalid_report"
        if isinstance(value, dict) and "resources" in value and value["resources"] is None:
            result._resources_error = "invalid_report"
        return result

    @field_validator("stop_reason")
    @classmethod
    def validate_stop_reason(cls, value: str | None) -> str | None:
        if value is not None:
            _require_text("stopReason", value)
        return value


class AllocationFinalReport(WireModel):
    report_id: str
    allocation_id: str
    started_at: datetime
    finished_at: datetime
    worker: ExecutionReport
    runtime: RuntimeReport

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_text("reportId", self.report_id)
        _require_text("allocationId", self.allocation_id)
        _require_aware_datetime("startedAt", self.started_at)
        _require_aware_datetime("finishedAt", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finishedAt must not precede startedAt")
        if (
            len(self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > 1024 * 1024
        ):
            raise ValueError("allocation final report exceeds 1 MiB")
        return self


class RuntimeReportV2(RuntimeReport):
    adapters: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]

    @field_validator("adapters")
    @classmethod
    def validate_adapters(
        cls, value: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]
    ) -> dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]:
        if len(value) > 64:
            raise ValueError("Runtime adapter metrics exceed 64 entries")
        return value


class AllocationFinalResponse(VersionedWireModel):
    report: AllocationFinalReport
