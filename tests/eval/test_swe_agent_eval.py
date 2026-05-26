"""End-to-end eval for `swe_agent`.

The SWE agent is a generic exploration/Q&A worker. We score each case on:
  - tool trajectory: required tools were used (and ordered, if specified)
  - phrase recall: the final response cites the expected files/symbols
"""

from __future__ import annotations

import pytest

from contractor.agents.swe_agent.agent import build_swe_agent

from tests.eval.adk_evals import score_tool_trajectory
from tests.eval.harness import run_agent
from tests.eval.scoring import score_phrases

NAMESPACE = "swe-eval"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_swe_agent_cases(fixture, fixture_fs, eval_model):
    agent = build_swe_agent(
        name="swe_agent",
        fs=fixture_fs,
        namespace=NAMESPACE,
        model=eval_model,
        max_tokens=80_000,
    )

    failures: list[str] = []

    for case in fixture.swe_cases:
        run = await run_agent(
            agent,
            user_message=case["prompt"],
            timeout_s=600.0,
        )

        names = set(run.tool_names())
        required_all = set(case.get("expected_tools_all") or [])
        required_any = set(case.get("expected_tools_any") or [])

        missing_all = required_all - names
        any_satisfied = (not required_any) or bool(required_any & names)

        phrase_score = score_phrases(run.final_text, case.get("expected_phrases", []))
        min_phrase_recall = float(case.get("min_phrase_recall", 0.66))

        trajectory_failed = False
        trajectory_info = ""
        expected_trajectory = case.get("expected_tool_trajectory")
        if expected_trajectory:
            ordered = bool(case.get("trajectory_ordered", True))
            traj = score_tool_trajectory(run, expected_trajectory, ordered=ordered)
            trajectory_failed = not traj.matched
            trajectory_info = f"  {traj.explain()}\n"

        case_failed = (
            missing_all
            or not any_satisfied
            or phrase_score.recall < min_phrase_recall
            or trajectory_failed
        )
        if case_failed:
            failures.append(
                f"case={case['id']}\n"
                f"  tools_used={sorted(names)}\n"
                f"  missing_required_tools={sorted(missing_all)}\n"
                f"  any_of_satisfied={any_satisfied} "
                f"(needed any of {sorted(required_any)})\n"
                f"  {phrase_score.explain('phrases')}\n"
                f"{trajectory_info}"
                f"  final_text_preview={run.final_text[:300]!r}"
            )

    assert not failures, "swe_agent eval failures:\n\n" + "\n\n".join(failures)
