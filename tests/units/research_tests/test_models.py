from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from research.models import (
    Decision,
    Experiment,
    Hypothesis,
    MetricResult,
    MetricSpec,
    SamplePlan,
)


def _metric_result(*, passed: bool = True) -> MetricResult:
    return MetricResult(
        name="recall",
        control=0.4,
        treatment=0.5,
        effect=0.1,
        interval_low=0.02,
        interval_high=0.18,
        adjusted_p_value=0.04,
        passed=passed,
    )


def test_models_are_immutable() -> None:
    metric = MetricSpec(name="recall", direction="increase", minimum_effect=0.05)
    with pytest.raises(ValidationError, match="frozen"):
        metric.name = "precision"  # type: ignore[misc]


def test_hypothesis_direction_must_match_exact_prefix() -> None:
    with pytest.raises(ValidationError, match="must belong to direction"):
        Hypothesis(
            id="AA1",
            direction="A",
            title="Exact direction identity",
            claim="A sufficiently descriptive research claim.",
            mechanism="A sufficiently descriptive causal mechanism.",
            scenario="offline",
            primary_metric=MetricSpec(name="recall", direction="increase"),
        )


def test_primary_metric_cannot_be_repeated_as_guardrail() -> None:
    metric = MetricSpec(name="recall", direction="increase")
    with pytest.raises(ValidationError, match="primary metric cannot also be a guardrail"):
        Hypothesis(
            id="A1",
            direction="A",
            title="No duplicate metric roles",
            claim="A sufficiently descriptive research claim.",
            mechanism="A sufficiently descriptive causal mechanism.",
            scenario="offline",
            primary_metric=metric,
            guardrails=[metric],
        )


def test_sample_plan_requires_one_unique_seed_per_attempt() -> None:
    with pytest.raises(ValidationError, match="one seed per attempt"):
        SamplePlan(attempts_per_case=2, requested_seeds=[17])
    with pytest.raises(ValidationError, match="must be unique"):
        SamplePlan(attempts_per_case=2, requested_seeds=[17, 17])


def test_frozen_experiment_requires_timezone_aware_timestamp() -> None:
    raw = {
        "id": "EXP-2026-999",
        "hypothesis_id": "A1",
        "title": "Timezone-safe frozen experiment",
        "design": "paired_ab",
        "scenario": "offline",
        "status": "frozen",
        "control": {"id": "off", "description": "control arm"},
        "treatments": [{"id": "on", "description": "treatment arm"}],
        "fixtures": {"include": ["fixture"]},
        "primary_metric": {"name": "recall", "direction": "increase"},
        "sample_plan": {"attempts_per_case": 1, "requested_seeds": [17]},
        "scorer": "scorer-v1",
        "frozen_at": datetime(2026, 8, 5),
    }
    with pytest.raises(ValidationError, match="timezone offset"):
        Experiment.model_validate(raw)
    raw["frozen_at"] = datetime(2026, 8, 5, tzinfo=UTC)
    assert Experiment.model_validate(raw).frozen_at is not None


def test_metric_result_is_internally_consistent() -> None:
    with pytest.raises(ValidationError, match="treatment - control"):
        MetricResult(name="recall", control=0.4, treatment=0.5, effect=0.2, passed=True)
    with pytest.raises(ValidationError, match="supplied together"):
        MetricResult(
            name="recall",
            control=0.4,
            treatment=0.5,
            effect=0.1,
            interval_low=0.0,
            passed=True,
        )


def test_supported_decision_requires_passing_primary_and_guardrails() -> None:
    with pytest.raises(ValidationError, match="require primary and guardrails to pass"):
        Decision(
            id="DEC-2026-999",
            experiment_id="EXP-2026-999",
            hypothesis_id="A1",
            treatment_arm_id="on",
            outcome="supported",
            decided_on=date(2026, 8, 5),
            primary_result=_metric_result(passed=False),
            actual_cases=1,
            actual_attempts=1,
            rationale="The primary decision threshold was not reached.",
            reviewer="reviewer",
        )
