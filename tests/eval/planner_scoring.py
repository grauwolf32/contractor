"""Planner-specific scoring for evaluating planning agent behavior.

Extracts structural metrics from ``TaskAgentRun`` to evaluate how the
planner decomposes tasks, manages its subtask budget, handles failures,
and produces outcomes — independently of the *content* quality that
output-focused evals (``test_project_information_task_eval``, etc.)
already measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from tests.eval.scoring import Score, _score_sets
from tests.eval.task_harness import TaskAgentRun, TaskMetrics

PLANNER_TOOL_NAMES = frozenset(
    {
        "add_subtask",
        "get_current_subtask",
        "list_subtasks",
        "get_records",
        "execute_current_subtask",
        "decompose_subtask",
        "skip",
        "finish",
    }
)


# ── Data models ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SubtaskRecord:
    """A subtask extracted from ``TaskResult.records``."""

    task_id: str
    title: str
    description: str
    status: str  # done | incomplete | malformed | skipped | decomposed
    output: str = ""
    summary: str = ""

    @property
    def depth(self) -> int:
        return self.task_id.count(".")

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "skipped")

    @property
    def is_child(self) -> bool:
        return "." in self.task_id


@dataclass(frozen=True)
class PlannerToolCounts:
    """Counts of planner-specific tool calls for a single task."""

    add_subtask: int = 0
    execute_current_subtask: int = 0
    decompose_subtask: int = 0
    skip: int = 0
    finish: int = 0
    list_subtasks: int = 0
    get_records: int = 0
    get_current_subtask: int = 0


@dataclass(frozen=True)
class PlanStructure:
    """Structural analysis of the planner's subtask tree for one task."""

    subtasks: list[SubtaskRecord]
    tool_counts: PlannerToolCounts
    task_completed: bool

    @property
    def total(self) -> int:
        return len(self.subtasks)

    @property
    def root_subtasks(self) -> list[SubtaskRecord]:
        return [s for s in self.subtasks if not s.is_child]

    @property
    def child_subtasks(self) -> list[SubtaskRecord]:
        return [s for s in self.subtasks if s.is_child]

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "done")

    @property
    def skipped_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "skipped")

    @property
    def decomposed_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "decomposed")

    @property
    def malformed_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "malformed")

    @property
    def incomplete_count(self) -> int:
        return sum(1 for s in self.subtasks if s.status == "incomplete")

    @property
    def max_depth(self) -> int:
        if not self.subtasks:
            return 0
        return max(s.depth for s in self.subtasks)

    @property
    def skip_rate(self) -> float:
        terminal = [s for s in self.subtasks if s.is_terminal]
        if not terminal:
            return 0.0
        return self.skipped_count / len(terminal)

    @property
    def success_rate(self) -> float:
        terminal = [s for s in self.subtasks if s.is_terminal]
        if not terminal:
            return 0.0
        return self.done_count / len(terminal)

    @property
    def decomposition_rate(self) -> float:
        roots = self.root_subtasks
        if not roots:
            return 0.0
        return sum(1 for s in roots if s.status == "decomposed") / len(roots)


# ── Extraction helpers ─────────────────────────────────────────────────


def extract_subtask_records(result: dict[str, Any]) -> list[SubtaskRecord]:
    """Parse ``SubtaskRecord`` list from a ``TaskResult``'s records field."""
    raw_records = result.get("records", [])
    out: list[SubtaskRecord] = []
    for rec in raw_records:
        if not isinstance(rec, dict) or "task_id" not in rec:
            continue
        out.append(
            SubtaskRecord(
                task_id=str(rec["task_id"]),
                title=str(rec.get("title", "")),
                description=str(rec.get("description", "")),
                status=str(rec.get("status", "unknown")),
                output=str(rec.get("output", "")),
                summary=str(rec.get("summary", "")),
            )
        )
    return out


def extract_planner_tool_counts(metrics: TaskMetrics) -> PlannerToolCounts:
    tc = metrics.tool_counts
    return PlannerToolCounts(
        add_subtask=tc.get("add_subtask", 0),
        execute_current_subtask=tc.get("execute_current_subtask", 0),
        decompose_subtask=tc.get("decompose_subtask", 0),
        skip=tc.get("skip", 0),
        finish=tc.get("finish", 0),
        list_subtasks=tc.get("list_subtasks", 0),
        get_records=tc.get("get_records", 0),
        get_current_subtask=tc.get("get_current_subtask", 0),
    )


def extract_plan_structure(
    run: TaskAgentRun,
    task_key: str,
) -> PlanStructure:
    """Build a ``PlanStructure`` for *task_key* from a ``TaskAgentRun``."""
    result: Optional[dict[str, Any]] = None
    for r in run.results:
        if r["template_key"] == task_key:
            result = r
            break

    if result is None:
        return PlanStructure(
            subtasks=[], tool_counts=PlannerToolCounts(), task_completed=False
        )

    subtasks = extract_subtask_records(result)

    metrics: Optional[TaskMetrics] = None
    for ref, m in run.metrics.items():
        if task_key in ref:
            metrics = m
            break

    tool_counts = (
        extract_planner_tool_counts(metrics) if metrics else PlannerToolCounts()
    )
    task_completed = result.get("status") == "done"

    return PlanStructure(
        subtasks=subtasks,
        tool_counts=tool_counts,
        task_completed=task_completed,
    )


# ── Topic coverage ─────────────────────────────────────────────────────


def score_topic_coverage(
    subtasks: list[SubtaskRecord],
    expected_topics: list[dict[str, Any]],
) -> Score:
    """Score how many expected topic areas the planner covered.

    Each topic entry has ``"label"`` and ``"keywords"`` (list of strings).
    A topic is covered if any subtask's title+description contains at least
    one keyword (case-insensitive).
    """
    covered: set[str] = set()
    expected: set[str] = set()

    for topic in expected_topics:
        label = topic["label"]
        keywords = topic["keywords"]
        expected.add(label)

        for sub in subtasks:
            text = f"{sub.title} {sub.description}".lower()
            if any(kw.lower() in text for kw in keywords):
                covered.add(label)
                break

    return _score_sets(covered, expected)


# ── Composite scorer ───────────────────────────────────────────────────


@dataclass(frozen=True)
class PlannerScore:
    """Composite planner evaluation result."""

    structure: PlanStructure
    topic_coverage: Optional[Score] = None

    subtask_count_ok: bool = True
    depth_ok: bool = True
    skip_rate_ok: bool = True
    budget_ok: bool = True
    completion_ok: bool = True

    def passes(self, *, min_topic_recall: float = 0.0) -> bool:
        checks = [
            self.subtask_count_ok,
            self.depth_ok,
            self.skip_rate_ok,
            self.budget_ok,
            self.completion_ok,
        ]
        if self.topic_coverage is not None and min_topic_recall > 0:
            checks.append(self.topic_coverage.recall >= min_topic_recall)
        return all(checks)

    def explain(self) -> str:
        s = self.structure
        lines = [
            f"planner: completed={s.task_completed} subtasks={s.total} "
            f"(done={s.done_count} skipped={s.skipped_count} "
            f"decomposed={s.decomposed_count} malformed={s.malformed_count})",
            f"  max_depth={s.max_depth} skip_rate={s.skip_rate:.2f} "
            f"success_rate={s.success_rate:.2f}",
            f"  tools: add={s.tool_counts.add_subtask} "
            f"exec={s.tool_counts.execute_current_subtask} "
            f"decompose={s.tool_counts.decompose_subtask} "
            f"skip={s.tool_counts.skip} finish={s.tool_counts.finish}",
        ]

        failures: list[str] = []
        if not self.subtask_count_ok:
            failures.append("subtask_count")
        if not self.depth_ok:
            failures.append("depth")
        if not self.skip_rate_ok:
            failures.append("skip_rate")
        if not self.budget_ok:
            failures.append("budget")
        if not self.completion_ok:
            failures.append("completion")
        if failures:
            lines.append(f"  FAILED checks: {failures}")

        if self.topic_coverage is not None:
            lines.append(f"  {self.topic_coverage.explain('topics')}")

        return "\n".join(lines)


def score_planner(
    run: TaskAgentRun,
    task_key: str,
    *,
    min_subtasks: int = 1,
    max_subtasks: Optional[int] = None,
    max_depth: int = 2,
    max_skip_rate: float = 1.0,
    max_budget_utilization: Optional[float] = None,
    budget: Optional[int] = None,
    must_complete: bool = True,
    expected_topics: Optional[list[dict[str, Any]]] = None,
) -> PlannerScore:
    """Score planner behavior for a specific task within a pipeline run."""
    structure = extract_plan_structure(run, task_key)

    subtask_count_ok = structure.total >= min_subtasks
    if max_subtasks is not None:
        subtask_count_ok = subtask_count_ok and structure.total <= max_subtasks

    depth_ok = structure.max_depth <= max_depth

    skip_rate_ok = structure.skip_rate <= max_skip_rate

    budget_ok = True
    if budget is not None and max_budget_utilization is not None:
        utilization = structure.total / budget if budget > 0 else 0.0
        budget_ok = utilization <= max_budget_utilization

    completion_ok = not must_complete or structure.task_completed

    topic_coverage = None
    if expected_topics:
        topic_coverage = score_topic_coverage(structure.subtasks, expected_topics)

    return PlannerScore(
        structure=structure,
        topic_coverage=topic_coverage,
        subtask_count_ok=subtask_count_ok,
        depth_ok=depth_ok,
        skip_rate_ok=skip_rate_ok,
        budget_ok=budget_ok,
        completion_ok=completion_ok,
    )
