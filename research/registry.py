"""Load and validate the tracked research registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from research.models import (
    Decision,
    Experiment,
    ExperimentStatus,
    Hypothesis,
    IdentifiedRecord,
    MetricSpec,
)

T = TypeVar("T", bound=IdentifiedRecord)
DEFAULT_ROOT = Path(__file__).resolve().parent


class RegistryError(ValueError):
    """A record is invalid in the context of the complete registry."""


@dataclass(frozen=True)
class ResearchRegistry:
    hypotheses: dict[str, Hypothesis]
    experiments: dict[str, Experiment]
    decisions: dict[str, Decision]

    def validate(self) -> None:
        for hypothesis in self.hypotheses.values():
            missing = set(hypothesis.dependencies) - self.hypotheses.keys()
            if missing:
                raise RegistryError(f"{hypothesis.id}: unknown dependencies {sorted(missing)}")
        _validate_dependency_cycles(self.hypotheses)

        for experiment in self.experiments.values():
            linked_hypothesis = self.hypotheses.get(experiment.hypothesis_id)
            if linked_hypothesis is None:
                raise RegistryError(
                    f"{experiment.id}: unknown hypothesis {experiment.hypothesis_id!r}"
                )
            if experiment.scenario != linked_hypothesis.scenario:
                raise RegistryError(
                    f"{experiment.id}: scenario {experiment.scenario!r} does not match "
                    f"{linked_hypothesis.id} scenario {linked_hypothesis.scenario!r}"
                )
            if not _same_metric_contract(
                experiment.primary_metric, linked_hypothesis.primary_metric
            ):
                raise RegistryError(
                    f"{experiment.id}: primary metric does not match {linked_hypothesis.id}"
                )
            if not _same_guardrail_contract(
                experiment.guardrails, linked_hypothesis.guardrails
            ):
                raise RegistryError(
                    f"{experiment.id}: guardrails do not match {linked_hypothesis.id}"
                )

        for decision in self.decisions.values():
            linked_experiment = self.experiments.get(decision.experiment_id)
            if linked_experiment is None:
                raise RegistryError(
                    f"{decision.id}: unknown experiment {decision.experiment_id!r}"
                )
            if decision.hypothesis_id != linked_experiment.hypothesis_id:
                raise RegistryError(
                    f"{decision.id}: hypothesis does not match {linked_experiment.id}"
                )
            if linked_experiment.status != ExperimentStatus.COMPLETE:
                raise RegistryError(
                    f"{decision.id}: experiment {linked_experiment.id} must be complete"
                )
            treatment_ids = {arm.id for arm in linked_experiment.treatments}
            if decision.treatment_arm_id not in treatment_ids:
                raise RegistryError(
                    f"{decision.id}: unknown treatment arm {decision.treatment_arm_id!r}"
                )
            if decision.primary_result.name != linked_experiment.primary_metric.name:
                raise RegistryError(
                    f"{decision.id}: primary result does not match {linked_experiment.id}"
                )
            expected_guardrails = {metric.name for metric in linked_experiment.guardrails}
            actual_guardrails = {result.name for result in decision.guardrail_results}
            if actual_guardrails != expected_guardrails:
                raise RegistryError(
                    f"{decision.id}: guardrail results do not match {linked_experiment.id}; "
                    f"expected {sorted(expected_guardrails)}, got {sorted(actual_guardrails)}"
                )

    def records_for(self, hypothesis_id: str) -> tuple[list[Experiment], list[Decision]]:
        experiments = [
            item for item in self.experiments.values() if item.hypothesis_id == hypothesis_id
        ]
        experiment_ids = {item.id for item in experiments}
        decisions = [
            item for item in self.decisions.values() if item.experiment_id in experiment_ids
        ]
        return experiments, decisions


def load_registry(root: Path = DEFAULT_ROOT) -> ResearchRegistry:
    registry = ResearchRegistry(
        hypotheses=_load_directory(root / "hypotheses", Hypothesis),
        experiments=_load_directory(root / "experiments", Experiment),
        decisions=_load_directory(root / "decisions", Decision),
    )
    registry.validate()
    return registry


def _load_directory(directory: Path, model: type[T]) -> dict[str, T]:
    records: dict[str, T] = {}
    if not directory.exists():
        return records
    for path in sorted((*directory.glob("*.yml"), *directory.glob("*.yaml"))):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RegistryError(f"{path}: expected a YAML mapping")
        try:
            record = model.model_validate(raw)
        except ValidationError as exc:
            raise RegistryError(f"{path}:\n{exc}") from exc
        record_id = str(record.id)
        if record_id in records:
            raise RegistryError(f"duplicate {model.__name__} id {record_id!r}: {path}")
        records[record_id] = record
    return records


def _validate_dependency_cycles(hypotheses: dict[str, Hypothesis]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(hypothesis_id: str, trail: Iterable[str]) -> None:
        if hypothesis_id in visiting:
            cycle = [*trail, hypothesis_id]
            raise RegistryError(f"hypothesis dependency cycle: {' -> '.join(cycle)}")
        if hypothesis_id in visited:
            return
        visiting.add(hypothesis_id)
        for dependency in hypotheses[hypothesis_id].dependencies:
            visit(dependency, [*trail, hypothesis_id])
        visiting.remove(hypothesis_id)
        visited.add(hypothesis_id)

    for hypothesis_id in hypotheses:
        visit(hypothesis_id, [])


def _same_metric_contract(left: MetricSpec, right: MetricSpec) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _same_guardrail_contract(left: list[MetricSpec], right: list[MetricSpec]) -> bool:
    left_by_name = {item.name: item for item in left}
    right_by_name = {item.name: item for item in right}
    if left_by_name.keys() != right_by_name.keys():
        return False
    return all(_same_metric_contract(left_by_name[name], right_by_name[name]) for name in left_by_name)
