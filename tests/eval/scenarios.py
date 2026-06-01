"""The three eval scenarios as thin, uniform harness wrappers.

Every eval is one of three scenarios, differing only in *what an attempt runs*:

* :func:`run_agent_eval`    — an attempt drives one ``LlmAgent`` (``run_agent``).
* :func:`run_task_eval`     — an attempt drives one ``TaskRunner`` task
                              (``run_task_pipeline``).
* :func:`run_pipeline_eval` — an attempt drives a whole workflow / CLI run.

All three are aliases of :func:`run_eval`, which owns the shared structure:
group cases by fixture, run each case ``pass_at`` times via
:func:`tests.eval.results.pass_at`, and fold everything into the ``eval/v1``
:class:`~tests.eval.results.EvalRun` envelope.

The caller supplies an ``attempt(case, attempt_idx) -> AttemptOutcome``
coroutine that does the scenario-specific work (build → run → score). Keeping
the bespoke scoring inside ``attempt`` is what lets one wrapper serve every
eval; the wrapper only standardizes repetition + the result shape.

Typical use::

    async def attempt(case, i):
        agent = build_my_agent(...)
        run = await run_agent(agent, user_message=case["prompt"], ...)
        score = score_my_run(run, case)
        return AttemptOutcome(
            passed=score.passed,
            metrics=metrics_from_events(run.metrics_events),
            detail={"actual": ..., "expected": ...},
        )

    eval_run = await run_agent_eval(
        unit="trace_agent", metric_kind="diff", pass_at=3,
        cases=[(c["fixture"], c["id"], c) for c in CASES],
        attempt=attempt, model=str(model),
    )
    write_eval_results(eval_run, "trace-agent-run")
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Awaitable, Callable, Optional, Sequence

from tests.eval.results import (AttemptOutcome, EvalRun, FixtureResult,
                                MetricKind, Scenario, pass_at)

# A case to evaluate: (fixture_slug, case_id, case_payload).
Case = tuple[str, str, Any]
# attempt(case_payload, attempt_idx) -> AttemptOutcome
AttemptFn = Callable[[Any, int], Awaitable[AttemptOutcome]]


async def run_eval(
    *,
    scenario: Scenario,
    unit: str,
    metric_kind: MetricKind,
    cases: Sequence[Case],
    attempt: AttemptFn,
    pass_at: int = 1,
    model: Optional[str] = None,
    prompt_version: Optional[str] = None,
    timestamp: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
) -> EvalRun:
    """Run every case ``pass_at`` times and assemble the ``eval/v1`` envelope.

    Cases are grouped into fixtures by their ``fixture_slug`` (preserving first-
    seen order). Each case is scored via the shared pass@X repeater, so the
    case passes iff any of its ``pass_at`` attempts passes.
    """
    by_fixture: "OrderedDict[str, list]" = OrderedDict()
    for slug, case_id, payload in cases:
        case_result = await _pass_at(case_id, payload, attempt, pass_at)
        by_fixture.setdefault(slug, []).append(case_result)

    return EvalRun(
        scenario=scenario,
        unit=unit,
        pass_at=pass_at,
        metric_kind=metric_kind,
        model=model,
        prompt_version=prompt_version,
        timestamp=timestamp,
        fixtures=[FixtureResult(slug=s, cases=cs) for s, cs in by_fixture.items()],
        meta=meta or {},
    )


async def _pass_at(case_id: str, payload: Any, attempt: AttemptFn, n: int):
    return await pass_at(case_id, lambda i: attempt(payload, i), n)


async def run_agent_eval(**kwargs: Any) -> EvalRun:
    """Agent-level eval (pass@N): each attempt drives one ``LlmAgent``."""
    return await run_eval(scenario="agent", **kwargs)


async def run_task_eval(**kwargs: Any) -> EvalRun:
    """Task-level eval (pass@M): each attempt drives one ``TaskRunner`` task."""
    return await run_eval(scenario="task", **kwargs)


async def run_pipeline_eval(**kwargs: Any) -> EvalRun:
    """Pipeline-level eval (pass@K): each attempt drives a whole workflow."""
    return await run_eval(scenario="pipeline", **kwargs)
