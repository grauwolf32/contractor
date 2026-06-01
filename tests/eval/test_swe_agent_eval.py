"""End-to-end eval for ``swe_agent``.

The SWE agent is a generic exploration/Q&A worker.  Each ``(fixture, case)``
pair is its own pytest item via the ``swe_case`` fixture (parametrized in
``conftest.pytest_generate_tests``).  Scoring dimensions:

- tool trajectory: required tools were used (and ordered, if specified)
- phrase recall: the final response cites the expected files/symbols
"""

from __future__ import annotations

import pytest

from contractor.agents.swe_agent.agent import build_swe_agent
from tests.eval.harness import run_agent
from tests.eval.results import CaseResult, metrics_from_events
from tests.eval.scorers import score_swe_run

NAMESPACE = "swe-eval"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_swe_agent(swe_case, eval_model, eval_sink):
    fixture, case = swe_case

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))
    agent = build_swe_agent(
        name="swe_agent",
        fs=fs,
        namespace=NAMESPACE,
        model=eval_model,
        max_tokens=80_000,
    )

    run = await run_agent(
        agent,
        user_message=case["prompt"],
        timeout_s=600.0,
    )

    result = score_swe_run(run, case)
    eval_sink.record(
        scenario="agent", unit="swe_agent", metric_kind="generic",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(id=case["id"], passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_events(run.metrics_events), detail={}),
    )
    assert result.passed, f"swe_agent eval failed: case={case['id']}\n{result.explain()}"
