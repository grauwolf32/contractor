"""Private Runtime protocol worker models and validation."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from contractor_runtime.contracts.artifacts import ArtifactRef
from contractor_runtime.contracts.base import (
    _STATE_METRIC_NAME_PATTERN,
    DIGEST_PATTERN,
    ID_PATTERN,
    MAX_AGENT_STATE_SNAPSHOT_BYTES,
    MAX_STATE_WORKSPACE_PATH_BYTES,
    MAX_STATE_WORKSPACE_PATHS,
    MAX_UINT64,
    MAX_WORKER_COMPLETION_BYTES,
    MAX_WORKER_FAILURE_MESSAGE_BYTES,
    MAX_WORKER_OBSERVATION_TOOLS,
    MAX_WORKER_RESULT_ARTIFACTS,
    WORKER_FAILURE_CODE_PATTERN,
    WORKER_SUBTASK_ID_PATTERN,
    TerminationError,
    VersionedWireModel,
    WireModel,
    _encoded_state_path_list_size,
    _known_completion,
    _require_state_workspace_path,
    _require_text,
    _require_worker_result_text,
    _require_worker_subtask_id,
)
from contractor_runtime.contracts.reports import ToolCallOutcome, WorkerCompletionDiagnostics
from contractor_runtime.contracts.workspace import WorkspaceObservationSummary


class StageContentRequest(VersionedWireModel):
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    objective: str
    instructions: str
    parameters: dict[str, str]
    artifacts: dict[str, ArtifactRef]
    result_artifacts: dict[str, ArtifactRef] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_text("objective", self.objective)
        _require_text("instructions", self.instructions)
        for key in self.parameters:
            _require_text("parameter name", key)
        for key, artifact in self.artifacts.items():
            _require_text("artifact context name", key)
            artifact.require_exact()
        for key, artifact in self.result_artifacts.items():
            _require_text("result artifact slot", key)
            if artifact.revision is not None:
                raise ValueError("result artifact binding must be versionless")
        return self


def _reserved_worker_result_binding(artifact: ArtifactRef) -> bool:
    return artifact.namespace in {"inputs", "outputs", "skills"} or artifact.name.startswith(
        "memory."
    )


class WorkerModelResult(WireModel):
    """Strict model-only result; it never crosses the Runtime A2A boundary."""

    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    result: str

    @model_validator(mode="after")
    def validate_model_result(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_worker_result_text(self.result)
        return self


class ToolObservationCount(WireModel):
    calls: int = Field(ge=0, le=MAX_UINT64)
    failures: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.failures > self.calls:
            raise ValueError("Worker observation failures exceed calls")
        return self


class WorkerFailure(WireModel):
    code: str = Field(pattern=WORKER_FAILURE_CODE_PATTERN.pattern)
    message: str
    retryable: bool

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if WORKER_FAILURE_CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("Worker failure code is invalid")
        if (
            not self.message.strip()
            or len(self.message.encode("utf-8")) > MAX_WORKER_FAILURE_MESSAGE_BYTES
        ):
            raise ValueError("Worker failure message must contain 1..4096 UTF-8 bytes")
        return self


class WorkerStateInvocationToolMetrics(WireModel):
    calls: int = Field(ge=0, le=MAX_UINT64)
    failures: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.failures > self.calls:
            raise ValueError("Worker State tool failures exceed calls")
        return self


class WorkerStateExecutionError(WireModel):
    code: str
    message: str = Field(max_length=4096)
    retryable: bool | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


class WorkerStateBudget(WireModel):
    max_model_calls: int = Field(gt=0, le=1000)
    max_tool_calls: int = Field(gt=0, le=10_000)
    max_total_tokens: int = Field(gt=0, le=100_000_000)
    observed_model_calls: int = Field(ge=0, le=MAX_UINT64)
    observed_tool_calls: int = Field(ge=0, le=MAX_UINT64)
    observed_total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    exhausted: Literal["model_calls", "tool_calls", "total_tokens"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

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


class WorkerStateSummarizer(WireModel):
    phase: Literal["disabled", "not_requested", "requested", "succeeded", "failed"]
    request_state_revision: int | None = Field(
        default=None, gt=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    model_calls: int = Field(ge=0, le=1)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=1)
    failure_code: str | None = Field(
        default=None,
        pattern=WORKER_FAILURE_CODE_PATTERN.pattern,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        requested = self.phase in {"requested", "succeeded", "failed"}
        if requested != (self.request_state_revision is not None):
            raise ValueError("Worker summarizer request revision is inconsistent")
        if self.phase in {"disabled", "not_requested", "requested"} and any(
            (
                self.model_calls,
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.token_usage_unavailable,
            )
        ):
            raise ValueError("inactive Worker summarizer has usage")
        if self.phase == "succeeded" and self.model_calls != 1:
            raise ValueError("successful Worker summarizer requires one model call")
        if self.token_usage_unavailable > self.model_calls:
            raise ValueError("Worker summarizer missing usage exceeds model calls")
        if (self.phase == "failed") != (self.failure_code is not None):
            raise ValueError("Worker summarizer failure code is inconsistent")
        return self


class WorkerStateWorkspaceInteraction(WireModel):
    path: str
    first_ordinal: int = Field(gt=0, le=MAX_UINT64)
    last_ordinal: int = Field(gt=0, le=MAX_UINT64)
    discovery_calls: int = Field(ge=0, le=MAX_UINT64)
    read_calls: int = Field(ge=0, le=MAX_UINT64)
    match_calls: int = Field(ge=0, le=MAX_UINT64)
    mutation_calls: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        _require_state_workspace_path(self.path)
        if self.last_ordinal < self.first_ordinal or not any(
            (
                self.discovery_calls,
                self.read_calls,
                self.match_calls,
                self.mutation_calls,
            )
        ):
            raise ValueError("Worker State workspace interaction is inconsistent")
        return self


class StageOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WorkerObservations(WireModel):
    profile: Literal["lean@1"]
    tools: dict[str, ToolObservationCount]
    workspace: WorkspaceObservationSummary | None = None
    truncated: bool

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if len(self.tools) > MAX_WORKER_OBSERVATION_TOOLS:
            raise ValueError("Worker observation tools exceed their bound")
        for name in self.tools:
            if len(name) > 128 or ID_PATTERN.fullmatch(name) is None:
                raise ValueError("Worker observation tool name is invalid")
        if (
            self.workspace is not None
            and (
                not self.workspace.scope_complete
                or not self.workspace.detail_complete
                or self.workspace.files_read_truncated
            )
            and not self.truncated
        ):
            raise ValueError("Worker observation truncation is inconsistent")
        return self


class WorkerStateInvocationMetrics(WireModel):
    model_calls: int = Field(ge=0, le=MAX_UINT64)
    model_errors: int = Field(ge=0, le=MAX_UINT64)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    cached_input_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    latest_prompt_tokens: int | None = Field(ge=0, le=MAX_UINT64)
    tool_calls: int = Field(ge=0, le=MAX_UINT64)
    tool_errors: int = Field(ge=0, le=MAX_UINT64)
    tools: dict[str, WorkerStateInvocationToolMetrics] = Field(max_length=256)
    truncated: bool

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.model_errors > self.model_calls or self.tool_errors > self.tool_calls:
            raise ValueError("Worker State invocation error counts exceed calls")
        for name in self.tools:
            if ID_PATTERN.fullmatch(name) is None or len(name) > 64:
                raise ValueError("Worker State invocation tool name is invalid")
        detailed_calls = sum(item.calls for item in self.tools.values())
        detailed_failures = sum(item.failures for item in self.tools.values())
        if detailed_calls > self.tool_calls or detailed_failures > self.tool_errors:
            raise ValueError("Worker State invocation tool detail exceeds aggregate")
        if not self.truncated and (
            detailed_calls != self.tool_calls or detailed_failures != self.tool_errors
        ):
            raise ValueError("complete Worker State invocation tool detail is inconsistent")
        return self


class WorkerStateToolCall(WireModel):
    """Strict live-State form; unlike final-report records every flag is explicit."""

    call_id: str
    tool: str
    arguments: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    arguments_truncated: bool
    outcome: ToolCallOutcome
    duration_ms: int | None = Field(
        default=None, ge=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    result_size_bytes: int | None = Field(
        default=None, ge=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    error: WorkerStateExecutionError | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        _require_text("callId", self.call_id)
        _require_text("tool", self.tool)
        if self.arguments is None and self.arguments_truncated:
            raise ValueError("absent Worker State arguments cannot be marked truncated")
        if (self.outcome is ToolCallOutcome.SUCCEEDED) == (self.error is not None):
            raise ValueError("Worker State tool outcome and error are inconsistent")
        return self


class WorkerStateWorkspaceObservation(WireModel):
    workspace_digest: str = Field(pattern=DIGEST_PATTERN.pattern)
    scope_paths: list[str] = Field(max_length=MAX_STATE_WORKSPACE_PATHS)
    scope_complete: bool
    interactions: list[WorkerStateWorkspaceInteraction] = Field(
        max_length=MAX_STATE_WORKSPACE_PATHS
    )
    detail_complete: bool

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        scope = [_require_state_workspace_path(path) for path in self.scope_paths]
        if scope != sorted(set(scope)) or _encoded_state_path_list_size(scope) > (
            MAX_STATE_WORKSPACE_PATH_BYTES
        ):
            raise ValueError("Worker State workspace scope is invalid")
        interaction_order = [(item.first_ordinal, item.path) for item in self.interactions]
        if interaction_order != sorted(interaction_order) or len(
            {item.path for item in self.interactions}
        ) != len(self.interactions):
            raise ValueError("Worker State workspace interactions are invalid")
        if (
            _encoded_state_path_list_size([item.path for item in self.interactions])
            > MAX_STATE_WORKSPACE_PATH_BYTES
        ):
            raise ValueError("Worker State workspace interaction paths exceed their bound")
        return self


class StageContentResult(VersionedWireModel):
    outcome: StageOutcome
    summary: str
    artifacts: dict[str, ArtifactRef]
    error: TerminationError | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_text("summary", self.summary)
        if self.outcome is StageOutcome.SUCCEEDED and self.error is not None:
            raise ValueError("successful result must not contain error")
        if self.outcome is StageOutcome.FAILED and self.error is None:
            raise ValueError("failed result requires error")
        for key, artifact in self.artifacts.items():
            _require_text("result artifact slot", key)
            artifact.require_exact()
        return self


class WorkerResult(WireModel):
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    result: str
    observations: WorkerObservations
    artifacts: dict[str, ArtifactRef]
    summarized: bool

    @model_validator(mode="after")
    def validate_worker_result(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_worker_result_text(self.result)
        if len(self.artifacts) > MAX_WORKER_RESULT_ARTIFACTS:
            raise ValueError("Worker result artifacts exceed their bound")
        for slot, artifact in self.artifacts.items():
            _require_text("Worker result artifact slot", slot)
            artifact.require_exact()
            if _reserved_worker_result_binding(artifact):
                raise ValueError("Worker result artifact identifies a reserved binding")
        return self


class WorkerAllocationMetricsState(WireModel):
    counters: dict[str, int] = Field(max_length=10_000)
    tool_calls: list[WorkerStateToolCall] = Field(max_length=1000)
    errors: list[WorkerStateExecutionError] = Field(max_length=100)
    final_outcome: str | None
    truncated: bool
    completion: WorkerCompletionDiagnostics | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("completion", mode="before")
    @classmethod
    def known_completion(cls, value):
        return _known_completion(value)

    worker_budget: WorkerStateBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        for name, count in self.counters.items():
            if (
                not 1 <= len(name) <= 128
                or _STATE_METRIC_NAME_PATTERN.fullmatch(name) is None
                or type(count) is not int
                or not 0 <= count <= MAX_UINT64
            ):
                raise ValueError("Worker State allocation counter is invalid")
        if self.final_outcome is not None and (
            len(self.final_outcome) > 64 or ID_PATTERN.fullmatch(self.final_outcome) is None
        ):
            raise ValueError("Worker State final outcome is invalid")
        for call in self.tool_calls:
            if call.arguments is not None:
                try:
                    encoded = json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                except (TypeError, ValueError) as error:
                    raise ValueError("Worker State tool arguments are invalid") from error
                if len(encoded) > 4096:
                    raise ValueError("Worker State tool arguments exceed their bound")
            if call.error is not None and len(call.error.message.encode("utf-8")) > 4096:
                raise ValueError("Worker State tool error exceeds its bound")
        for error in self.errors:
            if len(error.message.encode("utf-8")) > 4096:
                raise ValueError("Worker State error exceeds its bound")
        return self


class WorkerInvocationState(WireModel):
    invocation_id: str
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    phase: Literal["running", "succeeded", "failed", "cancelled"]
    metrics: WorkerStateInvocationMetrics
    summarizer: WorkerStateSummarizer
    workspace: WorkerStateWorkspaceObservation | None

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        if (
            self.invocation_id != self.invocation_id.strip()
            or not 1 <= len(self.invocation_id.encode("utf-8")) <= 128
            or any(character in self.invocation_id for character in "\r\n\t\x00")
        ):
            raise ValueError("Worker State invocationId is invalid")
        _require_worker_subtask_id(self.subtask_id)
        if self.phase != "running" and self.summarizer.phase == "requested":
            raise ValueError("terminal Worker invocation has pending summarization")
        return self


class WorkerCompletion(VersionedWireModel):
    result: WorkerResult | None = None
    failure: WorkerFailure | None = None
    invocation_id: str
    state_revision: int = Field(gt=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("WorkerCompletion requires exactly one result or failure")
        if (
            self.invocation_id != self.invocation_id.strip()
            or not 1 <= len(self.invocation_id.encode("utf-8")) <= 128
            or any(character in self.invocation_id for character in "\r\n\t\x00")
        ):
            raise ValueError("Worker invocationId is invalid")
        if (
            len(self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > MAX_WORKER_COMPLETION_BYTES
        ):
            raise ValueError("WorkerCompletion exceeds its bounded contract")
        return self


class ContractorWorkerState(WireModel):
    schema_version: Literal[2]
    state_revision: int = Field(gt=0, le=MAX_UINT64)
    metrics: WorkerAllocationMetricsState
    current_invocation: WorkerInvocationState | None
    last_completed_invocation: WorkerInvocationState | None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.current_invocation is not None and self.current_invocation.phase != "running":
            raise ValueError("current Worker invocation must be running")
        if self.last_completed_invocation is not None and (
            self.last_completed_invocation.phase == "running"
        ):
            raise ValueError("last completed Worker invocation must be terminal")
        return self


class AgentStateSnapshot(VersionedWireModel):
    state: ContractorWorkerState

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if len(self.model_dump_json(by_alias=True).encode("utf-8")) > (
            MAX_AGENT_STATE_SNAPSHOT_BYTES
        ):
            raise ValueError("AgentStateSnapshot exceeds its bounded contract")
        return self
