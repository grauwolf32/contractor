"""End-to-end eval for the ``oas_update`` task (OpenAPI build).

Runs the oas_update task through TaskRunner (optionally preceded by
dependency_information + project_information when precomputed artifacts
are available) and scores the produced OpenAPI schema against ground
truth endpoints and components.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial

import pytest
import yaml
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.utils.prompt import load_prompt_with_version
from tests.eval.conftest import FIXTURES_ROOT
from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_task
from tests.eval.scorers import diff_detail, score_oas_schema
from tests.eval.task_harness import render_metrics_table, run_task_pipeline


def _load_precomputed(slug: str) -> dict[str, str] | None:
    art_dir = FIXTURES_ROOT / slug / "artifacts"
    mapping: dict[str, str] = {}
    for task_key in ("dependency_information", "project_information"):
        path = art_dir / f"{task_key}_result.txt"
        if not path.is_file():
            return None
        mapping[f"{task_key}/result"] = path.read_text(encoding="utf-8")
    return mapping


@pytest.mark.eval
@pytest.mark.asyncio
async def test_oas_build_task(fixture, fixture_fs, eval_model: LiteLlm, eval_sink):
    if not fixture.expected_oas:
        pytest.skip(f"no oas.expected.yaml for fixture {fixture.slug}")

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))
    precomputed = _load_precomputed(fixture.slug)

    _, prompt_version = load_prompt_with_version("oas_builder_agent")
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        FIXTURES_ROOT / fixture.slug / "runs"
        / f"oas_build_{prompt_version}_{ts}"
    )

    def queue(runner) -> None:
        swe_builder = partial(
            build_swe_agent,
            name="swe_agent",
            fs=fs,
            model=eval_model,
            max_tokens=100_000,
        )
        oas_builder = partial(
            build_oas_builder_agent,
            name="oas_builder",
            fs=fs,
            model=eval_model,
            max_tokens=100_000,
        )

        runner.add_variable(name="project_path", value=str(fixture.source_root))

        if precomputed is None:
            runner.add_task(
                name="dependency_information",
                worker_builder=swe_builder,
                iterations=1, max_attempts=2, max_steps=20,
                namespace="dependency_information", model=eval_model,
            )
            runner.add_task(
                name="project_information",
                worker_builder=swe_builder,
                iterations=1, max_attempts=2, max_steps=20,
                artifacts=["dependency_information/result"],
                namespace="project_information", model=eval_model,
            )

        runner.add_task(
            name="oas_update",
            worker_builder=oas_builder,
            iterations=2, max_attempts=4, max_steps=20,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="openapi-building", model=eval_model,
        )

    oas_artifact_key = "user:oas-openapi-building"

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=["oas_update/result", oas_artifact_key],
        namespace=f"task-eval-{fixture.slug}-oas_build",
        timeout_s=2400.0,
        runner_name=f"oas-build-{fixture.slug}",
        preloaded_artifacts=precomputed,
        output_dir=run_dir,
        artifact_dir=case_artifact_dir("oas_build", fixture.slug, fixture.slug),
    )

    result_text = run.artifacts.get(oas_artifact_key, "")
    if not result_text:
        result_text = run.result_text("oas_update")
    assert result_text, (
        "oas_update produced no result artifact\n"
        + render_metrics_table(run.metrics)
    )

    actual_schema = yaml.safe_load(result_text) or {}
    if not isinstance(actual_schema, dict):
        actual_schema = {}

    result = score_oas_schema(
        actual_schema, fixture.expected_oas,
        min_endpoint_precision=0.5, min_endpoint_recall=0.6,
        min_schema_recall=0.3,
    )

    summary = (
        f"fixture={fixture.slug}\n"
        f"{result.explain()}\n"
        f"  precomputed={'yes' if precomputed else 'no'}\n\n"
        f"metrics:\n{render_metrics_table(run.metrics)}"
    )
    print(f"\n{'='*60}\n{summary}\n{'='*60}")

    eval_sink.record(
        scenario="task", unit="oas_build", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(id=fixture.slug, passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_task(run.metrics), detail=diff_detail(result)),
        artifacts=run.artifacts,
    )
    assert result.passed, f"oas_build eval failed\n{summary}"
