"""End-to-end eval for the ``project_information`` task.

This goes through ``TaskRunner`` (planner + worker), not just the bare
``swe_agent``. ``project_information`` consumes
``dependency_information/result`` as an upstream artifact, so we queue
both tasks the same way ``LikeC4BuildingPipeline`` does and then score
the Markdown the second task publishes.

Scoring is structural: how many of the nine numbered category headings
the document actually contains, and whether the framework keywords the
fixture expects (e.g. ``fastapi``) appear anywhere in the body. We're
measuring "did the planner/worker chain produce the documented
output_format?", not the semantic quality of individual table rows.
"""

from __future__ import annotations

from functools import partial

import pytest
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.swe_agent.agent import build_swe_agent

from tests.eval.scoring import score_markdown_sections, score_phrases
from tests.eval.task_harness import render_metrics_table, run_task_pipeline


def _case_for(fixture, task: str) -> dict | None:
    for case in fixture.task_cases:
        if case.get("task") == task:
            return case
    return None


@pytest.mark.eval
@pytest.mark.asyncio
async def test_project_information_task(fixture, eval_model: LiteLlm):
    case = _case_for(fixture, "project_information")
    if case is None:
        pytest.skip(f"no project_information case for fixture {fixture.slug}")

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))

    def queue(runner) -> None:
        swe_builder = partial(
            build_swe_agent,
            name="swe_agent",
            fs=fs,
            model=eval_model,
            max_tokens=100_000,
        )
        runner.add_variable(name="project_path", value=str(fixture.source_root))
        runner.add_task(
            name="dependency_information",
            worker_builder=swe_builder,
            iterations=1,
            max_attempts=2,
            max_steps=20,
            namespace="dependency_information",
            model=eval_model,
        )
        runner.add_task(
            name="project_information",
            worker_builder=swe_builder,
            iterations=1,
            max_attempts=2,
            max_steps=20,
            artifacts=["dependency_information/result"],
            namespace="project_information",
            model=eval_model,
        )

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=[
            "dependency_information/result",
            "project_information/result",
        ],
        namespace=f"task-eval-{fixture.slug}-{case['id']}",
        timeout_s=float(case.get("timeout_s", 1800.0)),
        runner_name=f"project_information-{fixture.slug}",
    )

    result_text = run.result_text("project_information")
    assert result_text, (
        "project_information published no result artifact\n"
        + render_metrics_table(run.metrics)
    )

    section_score = score_markdown_sections(
        result_text, case["expected_sections"]
    )
    min_section_recall = float(case.get("min_section_recall", 0.7))

    expected_phrases = case.get("expected_phrases_any", [])
    phrase_score = score_phrases(result_text, expected_phrases)
    min_phrase_recall = float(case.get("min_phrase_recall", 0.5))

    if (
        section_score.recall < min_section_recall
        or phrase_score.recall < min_phrase_recall
    ):
        pytest.fail(
            "project_information eval failed for "
            f"fixture={fixture.slug} case={case['id']}\n"
            f"  {section_score.explain('sections')}\n"
            f"  {phrase_score.explain('phrases')}\n"
            f"  result_chars={len(result_text)}\n\n"
            f"metrics:\n{render_metrics_table(run.metrics)}"
        )
