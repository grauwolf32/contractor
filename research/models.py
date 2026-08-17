"""Strict, versioned models for research records."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from math import isclose
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HypothesisId = Annotated[str, Field(pattern=r"^[A-Z]{1,3}[0-9]+$")]
DirectionId = Annotated[str, Field(pattern=r"^[A-Z]{1,3}$")]
ExperimentId = Annotated[str, Field(pattern=r"^EXP-[0-9]{4}-[0-9]{3,}$")]
DecisionId = Annotated[str, Field(pattern=r"^DEC-[0-9]{4}-[0-9]{3,}$")]
ArmId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")]
MetricName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]*$")]


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    TRIAGED = "triaged"
    READY = "ready"
    RUNNING = "running"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    REPLICATING = "replicating"
    PROMOTED = "promoted"
    MONITORED = "monitored"
    RETIRED = "retired"


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    FROZEN = "frozen"
    RUNNING = "running"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class DecisionOutcome(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    PROMOTED = "promoted"
    RETIRED = "retired"


class Scenario(StrEnum):
    AGENT = "agent"
    TASK = "task"
    PIPELINE = "pipeline"
    OFFLINE = "offline"


class DesignKind(StrEnum):
    PAIRED_AB = "paired_ab"
    FACTORIAL = "factorial"
    NON_INFERIORITY = "non_inferiority"
    OBSERVATIONAL = "observational"
    FAULT_INJECTION = "fault_injection"
    SCALING_SWEEP = "scaling_sweep"


class MetricDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NON_INFERIOR = "non_inferior"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class IdentifiedRecord(StrictModel):
    """Common structural contract for top-level registry records."""

    id: str


class MetricSpec(StrictModel):
    name: MetricName
    direction: MetricDirection
    minimum_effect: float | None = None
    non_inferiority_margin: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_margin(self) -> MetricSpec:
        if self.direction == MetricDirection.NON_INFERIOR and self.non_inferiority_margin is None:
            raise ValueError("non_inferior metrics require non_inferiority_margin")
        if self.direction != MetricDirection.NON_INFERIOR and self.non_inferiority_margin is not None:
            raise ValueError("non_inferiority_margin is only valid for non_inferior metrics")
        return self


class Hypothesis(IdentifiedRecord):
    schema_version: Literal["hypothesis/v1"] = "hypothesis/v1"
    id: HypothesisId
    direction: DirectionId
    title: str = Field(min_length=5)
    claim: str = Field(min_length=10)
    mechanism: str = Field(min_length=10)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    scenario: Scenario
    primary_metric: MetricSpec
    guardrails: list[MetricSpec] = Field(default_factory=list)
    dependencies: list[HypothesisId] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("guardrails")
    @classmethod
    def guardrails_are_unique(cls, value: list[MetricSpec]) -> list[MetricSpec]:
        _require_unique_metric_names(value, "guardrails")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Hypothesis:
        id_direction = self.id.rstrip("0123456789")
        if id_direction != self.direction:
            raise ValueError(f"id {self.id!r} must belong to direction {self.direction!r}")
        if self.id in self.dependencies:
            raise ValueError("a hypothesis cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("dependencies must be unique")
        if self.primary_metric.name in {item.name for item in self.guardrails}:
            raise ValueError("primary metric cannot also be a guardrail")
        return self


class Arm(StrictModel):
    id: ArmId
    description: str = Field(min_length=3)
    model: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    task_versions: dict[str, str] = Field(default_factory=dict)
    feature_flags: dict[str, bool | int | float | str] = Field(default_factory=dict)
    runtime: dict[str, bool | int | float | str] = Field(default_factory=dict)


class FixtureSelection(StrictModel):
    include: list[str] = Field(min_length=1)
    exclude: list[str] = Field(default_factory=list)
    slice: str | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> FixtureSelection:
        if len(set(self.include)) != len(self.include):
            raise ValueError("included fixtures must be unique")
        if len(set(self.exclude)) != len(self.exclude):
            raise ValueError("excluded fixtures must be unique")
        overlap = set(self.include) & set(self.exclude)
        if overlap:
            raise ValueError(f"fixtures cannot be both included and excluded: {sorted(overlap)}")
        return self


class SamplePlan(StrictModel):
    attempts_per_case: int = Field(ge=1)
    requested_seeds: list[int] = Field(min_length=1)
    sequential_looks: list[float] = Field(default_factory=lambda: [1.0])
    alpha: float = Field(default=0.05, gt=0, lt=1)
    fdr_family: str | None = None
    max_gpu_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_looks(self) -> SamplePlan:
        if len(set(self.requested_seeds)) != len(self.requested_seeds):
            raise ValueError("requested_seeds must be unique")
        if len(self.requested_seeds) != self.attempts_per_case:
            raise ValueError("requested_seeds must contain one seed per attempt")
        if self.sequential_looks != sorted(set(self.sequential_looks)):
            raise ValueError("sequential_looks must be sorted and unique")
        if not self.sequential_looks or self.sequential_looks[-1] != 1.0:
            raise ValueError("sequential_looks must end at 1.0")
        if any(look <= 0 or look > 1 for look in self.sequential_looks):
            raise ValueError("sequential_looks must be in (0, 1]")
        return self


class Experiment(IdentifiedRecord):
    schema_version: Literal["experiment/v1"] = "experiment/v1"
    id: ExperimentId
    hypothesis_id: HypothesisId
    title: str = Field(min_length=5)
    design: DesignKind
    scenario: Scenario
    status: ExperimentStatus = ExperimentStatus.DRAFT
    control: Arm
    treatments: list[Arm] = Field(min_length=1)
    fixtures: FixtureSelection
    primary_metric: MetricSpec
    guardrails: list[MetricSpec] = Field(default_factory=list)
    sample_plan: SamplePlan
    scorer: str = Field(min_length=1)
    scorer_parameters: dict[str, Any] = Field(default_factory=dict)
    exclusions: list[str] = Field(default_factory=list)
    frozen_at: datetime | None = None
    owner: str | None = None

    @field_validator("guardrails")
    @classmethod
    def guardrails_are_unique(cls, value: list[MetricSpec]) -> list[MetricSpec]:
        _require_unique_metric_names(value, "guardrails")
        return value

    @model_validator(mode="after")
    def validate_experiment(self) -> Experiment:
        arm_ids = [self.control.id, *(arm.id for arm in self.treatments)]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("arm ids must be unique within an experiment")
        if self.primary_metric.name in {item.name for item in self.guardrails}:
            raise ValueError("primary metric cannot also be a guardrail")
        if self.status in {
            ExperimentStatus.FROZEN,
            ExperimentStatus.RUNNING,
            ExperimentStatus.COMPLETE,
        } and self.frozen_at is None:
            raise ValueError(f"status {self.status!r} requires frozen_at")
        if self.status == ExperimentStatus.DRAFT and self.frozen_at is not None:
            raise ValueError("draft experiments cannot have frozen_at")
        if self.frozen_at is not None and self.frozen_at.utcoffset() is None:
            raise ValueError("frozen_at must include a timezone offset")
        return self


class MetricResult(StrictModel):
    name: MetricName
    control: float
    treatment: float
    effect: float
    interval_low: float | None = None
    interval_high: float | None = None
    adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> MetricResult:
        if not isclose(self.effect, self.treatment - self.control, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("effect must equal treatment - control")
        bounds = (self.interval_low, self.interval_high)
        if (bounds[0] is None) != (bounds[1] is None):
            raise ValueError("interval_low and interval_high must be supplied together")
        if bounds[0] is not None and bounds[1] is not None:
            if bounds[0] > bounds[1]:
                raise ValueError("interval_low cannot exceed interval_high")
            if not bounds[0] <= self.effect <= bounds[1]:
                raise ValueError("effect must lie inside its interval")
        return self


class Decision(IdentifiedRecord):
    schema_version: Literal["decision/v1"] = "decision/v1"
    id: DecisionId
    experiment_id: ExperimentId
    hypothesis_id: HypothesisId
    treatment_arm_id: ArmId
    outcome: DecisionOutcome
    decided_on: date
    primary_result: MetricResult
    guardrail_results: list[MetricResult] = Field(default_factory=list)
    actual_cases: int = Field(ge=1)
    actual_attempts: int = Field(ge=1)
    deviations: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=10)
    reviewer: str
    implementation_action: str | None = None
    follow_up: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> Decision:
        guardrails_pass = all(item.passed for item in self.guardrail_results)
        if self.outcome in {
            DecisionOutcome.SUPPORTED,
            DecisionOutcome.PROMOTED,
        } and (not self.primary_result.passed or not guardrails_pass):
            raise ValueError("supported/promoted decisions require primary and guardrails to pass")
        if self.outcome == DecisionOutcome.REFUTED and self.primary_result.passed:
            raise ValueError("refuted decisions require the primary result to fail")
        if self.actual_attempts < self.actual_cases:
            raise ValueError("actual_attempts cannot be less than actual_cases")
        return self


def _require_unique_metric_names(metrics: list[MetricSpec], field_name: str) -> None:
    names = [item.name for item in metrics]
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} metric names must be unique")
