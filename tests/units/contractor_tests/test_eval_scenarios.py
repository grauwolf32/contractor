"""Unit tests for the scenario harness wrappers (``tests/eval/scenarios.py``).

Uses a fake ``attempt`` coroutine (no live model) to verify the wrapper groups
cases by fixture, applies pass@X, and tags the envelope by scenario.
"""
from __future__ import annotations

import asyncio

from tests.eval.results import AttemptOutcome
from tests.eval.scenarios import (
    run_agent_eval,
    run_eval,
    run_pipeline_eval,
    run_task_eval,
)


def test_groups_cases_by_fixture_and_tags_scenario():
    async def attempt(payload, i):
        return AttemptOutcome(passed=payload["ok"], metrics={"total_tokens": 1},
                              detail={"v": payload["ok"]})

    cases = [
        ("fxA", "a1", {"ok": True}),
        ("fxA", "a2", {"ok": False}),
        ("fxB", "b1", {"ok": True}),
    ]
    run = asyncio.run(run_task_eval(unit="t", metric_kind="generic", cases=cases,
                                    attempt=attempt, pass_at=1))
    assert run.scenario == "task"
    assert [f.slug for f in run.fixtures] == ["fxA", "fxB"]   # first-seen order
    assert run.fixtures[0].cases_total == 2 and run.fixtures[0].cases_passed == 1
    assert run.fixtures[1].cases_passed == 1


def test_pass_at_repeats_and_passes_if_any():
    # Attempt passes only on its 3rd try (idx 2).
    async def attempt(payload, i):
        return AttemptOutcome(passed=(i == 2), detail={"idx": i})

    run = asyncio.run(run_agent_eval(unit="a", metric_kind="verdict",
                                     cases=[("fx", "c", {})], attempt=attempt, pass_at=3))
    assert run.scenario == "agent" and run.pass_at == 3
    case = run.fixtures[0].cases[0]
    assert case.passed is True and case.pass_count == 1 and case.attempts == 3
    assert case.runs is not None and len(case.runs) == 3


def test_scenario_aliases_set_scenario():
    async def attempt(payload, i):
        return AttemptOutcome(passed=True)

    for fn, expected in ((run_agent_eval, "agent"), (run_task_eval, "task"),
                         (run_pipeline_eval, "pipeline")):
        run = asyncio.run(fn(unit="u", metric_kind="generic",
                             cases=[("fx", "c", {})], attempt=attempt))
        assert run.scenario == expected


def test_envelope_carries_model_and_meta():
    async def attempt(payload, i):
        return AttemptOutcome(passed=True)

    run = asyncio.run(run_eval(scenario="pipeline", unit="trace", metric_kind="diff",
                               cases=[("fx", "c", {})], attempt=attempt,
                               model="lm-studio-qwen3.6", meta={"axes": ["v5", "v7"]}))
    env = run.to_envelope()
    assert env["model"] == "lm-studio-qwen3.6"
    assert env["meta"]["axes"] == ["v5", "v7"]
    assert env["scenario"] == "pipeline" and env["metric_kind"] == "diff"
