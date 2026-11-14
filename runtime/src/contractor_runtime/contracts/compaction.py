"""Content-free Worker compaction observations shared by State and reports."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from contractor_runtime.contracts.base import (
    MAX_EXACT_INTEGER,
    WORKER_FAILURE_CODE_PATTERN,
    WORKER_SUBTASK_ID_PATTERN,
    WireModel,
)
from contractor_runtime.contracts.settings import MAX_WORKER_COMPACTIONS

Counter = Annotated[int, Field(ge=0, le=MAX_EXACT_INTEGER)]
Generation = Annotated[int, Field(ge=0, le=MAX_WORKER_COMPACTIONS)]


class WorkerModelUsage(WireModel):
    model_calls: Counter
    input_tokens: Counter
    output_tokens: Counter
    total_tokens: Counter
    token_usage_unavailable: Counter
    duration_ms: Counter | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.token_usage_unavailable > self.model_calls:
            raise ValueError("model phase counts exceed model calls")
        return self


class WorkerModelPhases(WireModel):
    main: WorkerModelUsage
    compaction: WorkerModelUsage
    finalization: WorkerModelUsage
    summarization: WorkerModelUsage


class CompactionPromptMeasurement(WireModel):
    generation: Generation
    # Absence is unavailable. Byte counts never supply this value.
    prompt_tokens: Counter | None = Field(default=None, exclude_if=lambda v: v is None)


class WorkerCompactionProgress(WireModel):
    invocation_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    generation: Generation
    phase: Literal["idle", "requested", "running", "applied", "failed"]
    attempts: int = Field(ge=0, le=MAX_WORKER_COMPACTIONS + 1)
    failures: int = Field(ge=0, le=MAX_WORKER_COMPACTIONS + 1)
    archive_reads: Counter
    input_bytes: Counter
    summary_bytes: Counter
    reasoning_available: Counter
    reasoning_included: Counter
    reasoning_truncated: Counter
    last_cause: str | None = Field(
        default=None, pattern=WORKER_FAILURE_CODE_PATTERN.pattern, exclude_if=lambda v: v is None
    )
    before_prompt: CompactionPromptMeasurement | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    after_prompt: CompactionPromptMeasurement | None = Field(
        default=None, exclude_if=lambda v: v is None
    )

    @model_validator(mode="after")
    def validate_progress(self) -> Self:
        pending = self.phase in {"requested", "running"}
        if self.attempts != self.generation + self.failures + pending:
            raise ValueError("compaction generation or outcomes are inconsistent")
        if self.phase == "idle" and self.attempts:
            raise ValueError("idle compaction has attempts")
        if self.phase != "idle" and self.before_prompt is None:
            raise ValueError("compaction requires a before measurement")
        if (self.phase == "failed") != (self.last_cause is not None):
            raise ValueError("compaction failure cause is inconsistent")
        if self.phase == "failed" and not self.failures:
            raise ValueError("failed compaction has no failure")
        if self.phase == "applied":
            if not self.summary_bytes or not self.input_bytes:
                raise ValueError("applied compaction has no model call or byte measurements")
            if not self.generation or self.before_prompt is None or self.after_prompt is None:
                raise ValueError("applied compaction requires generation measurements")
            if (
                self.before_prompt.generation != self.generation - 1
                or self.after_prompt.generation != self.generation
            ):
                raise ValueError("compaction measurements name a different generation")
        else:
            if self.after_prompt is not None:
                raise ValueError("unapplied compaction has an after measurement")
            if self.before_prompt is not None and self.before_prompt.generation != self.generation:
                raise ValueError("compaction before measurement names a different generation")
        if max(self.reasoning_included, self.reasoning_truncated) > self.reasoning_available:
            raise ValueError("compaction reasoning counts exceed available groups")
        return self
