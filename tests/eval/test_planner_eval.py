"""End-to-end eval for planner agent decomposition and execution behavior.

Complements the output-quality evals (``test_project_information_task_eval``,
``test_likec4_task_eval``) by scoring the *planning process* itself:

- Did the planner create a reasonable number of subtasks?
- Did the subtask titles/descriptions cover the expected topic areas?
- Did the planner stay within depth / budget / skip-rate constraints?
- Did the task complete successfully?

Each ``(fixture, case)`` pair is its own pytest item via the
``planner_case`` fixture (parametrized in ``conftest.pytest_generate_tests``).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.swe_agent.agent import build_swe_agent
from tests.eval.planner_scoring import score_planner
from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_task
from tests.eval.task_harness import render_metrics_table, run_task_pipeline


def _queue_project_information(
    runner,
    *,
    fs,
    model: LiteLlm,
    project_path: str,
    max_steps: int = 20,
) -> None:
    """Queue dependency_information + project_information (the common pair)."""
    swe_builder = partial(
        build_swe_agent,
        name="swe_agent",
        fs=fs,
        model=model,
        max_tokens=100_000,
    )
    runner.add_variable(name="project_path", value=project_path)
    runner.add_task(
        name="dependency_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=2,
        max_steps=max_steps,
        namespace="dependency_information",
        model=model,
    )
    runner.add_task(
        name="project_information",
        worker_builder=swe_builder,
        iterations=1,
        max_attempts=2,
        max_steps=max_steps,
        artifacts=["dependency_information/result"],
        namespace="project_information",
        model=model,
    )


_TASK_QUEUE_REGISTRY: dict[str, dict[str, Any]] = {
    "project_information": {
        "queue_fn": _queue_project_information,
        "artifact_keys": [
            "dependency_information/result",
            "project_information/result",
        ],
    },
    "dependency_information": {
        "queue_fn": _queue_project_information,
        "artifact_keys": [
            "dependency_information/result",
            "project_information/result",
        ],
    },
}


@pytest.mark.eval
@pytest.mark.asyncio
async def test_planner_behavior(planner_case, eval_model: LiteLlm, eval_sink):
    """Evaluate planner decomposition quality for a single case."""
    fixture, case = planner_case

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))

    task_key: str = case["task"]
    registry_entry = _TASK_QUEUE_REGISTRY.get(task_key)
    if registry_entry is None:
        pytest.skip(f"no queue helper registered for task '{task_key}'")

    max_steps = int(case.get("max_steps", 20))

    def queue(runner, *, _entry=registry_entry, _ms=max_steps) -> None:
        _entry["queue_fn"](
            runner,
            fs=fs,
            model=eval_model,
            project_path=str(fixture.source_root),
            max_steps=_ms,
        )

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=registry_entry["artifact_keys"],
        namespace=f"planner-eval-{fixture.slug}-{case['id']}",
        timeout_s=float(case.get("timeout_s", 1800)),
        runner_name=f"planner-{fixture.slug}-{case['id']}",
        artifact_dir=case_artifact_dir("planning_agent", fixture.slug, case["id"]),
    )

    planner_score = score_planner(
        run,
        task_key,
        min_subtasks=int(case.get("min_subtasks", 1)),
        max_subtasks=case.get("max_subtasks"),
        max_depth=int(case.get("max_depth", 2)),
        max_skip_rate=float(case.get("max_skip_rate", 1.0)),
        max_budget_utilization=case.get("max_budget_utilization"),
        budget=case.get("budget"),
        must_complete=bool(case.get("must_complete", True)),
        expected_topics=case.get("expected_topics"),
    )

    min_topic_recall = float(case.get("min_topic_recall", 0.0))
    passed = planner_score.passes(min_topic_recall=min_topic_recall)

    _tc = planner_score.topic_coverage
    eval_sink.record(
        scenario="task", unit="planning_agent", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(
            id=case["id"], passed=passed, pass_count=int(passed), attempts=1,
            metrics=metrics_from_task(run.metrics),
            detail=({"precision": round(_tc.precision, 3), "recall": round(_tc.recall, 3),
                     "f1": round(_tc.f1, 3)} if _tc is not None else {})),
        artifacts=run.artifacts,
    )
    assert passed, (
        f"planner eval failed for fixture={fixture.slug} "
        f"case={case['id']}\n"
        f"{planner_score.explain()}\n\n"
        f"metrics:\n{render_metrics_table(run.metrics)}"
    )
