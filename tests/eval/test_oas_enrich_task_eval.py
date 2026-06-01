"""End-to-end eval for the ``oas_enrich`` task (OpenAPI enrichment).

Runs the oas_enrich task through TaskRunner using precomputed
dependency/project artifacts. The enrichment task assumes a seed OAS
schema already exists (from a prior build pass); for eval purposes we
bootstrap the seed from the fixture's oas.expected.yaml stripped to
just paths (no request/response bodies) so the enricher has something
to refine rather than building from scratch.

Scoring compares the enriched schema against the full ground truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial

import pytest
import yaml
from google.adk.models.lite_llm import LiteLlm

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.utils.prompt import load_prompt_with_version
from tests.eval.conftest import FIXTURES_ROOT
from tests.eval.results import CaseResult, metrics_from_task
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


def _build_skeleton_seed(expected_oas: dict) -> str:
    """Build a minimal OAS seed with paths but no request/response detail."""
    seed: dict = {
        "openapi": "3.0.3",
        "info": {
            "title": expected_oas.get("info", {}).get("title", "API"),
            "version": "0.1.0",
        },
        "paths": {},
    }
    for path, methods in (expected_oas.get("paths") or {}).items():
        seed["paths"][path] = {}
        for method, spec in (methods or {}).items():
            if method.lower() in (
                "get", "post", "put", "patch", "delete", "head", "options"
            ):
                seed["paths"][path][method] = {
                    "summary": spec.get("summary", ""),
                    "responses": {"200": {"description": "OK"}},
                }
    return yaml.dump(seed, default_flow_style=False, sort_keys=False)


@pytest.mark.eval
@pytest.mark.asyncio
async def test_oas_enrich_task(fixture, fixture_fs, eval_model: LiteLlm, eval_sink):
    if not fixture.expected_oas:
        pytest.skip(f"no oas.expected.yaml for fixture {fixture.slug}")

    from cli.fs import RootedLocalFileSystem

    fs = RootedLocalFileSystem(str(fixture.source_root))
    precomputed = _load_precomputed(fixture.slug)
    if precomputed is None:
        pytest.skip(
            f"no precomputed artifacts for {fixture.slug} — "
            "run scripts/precompute_task_artifacts.py first"
        )

    _, prompt_version = load_prompt_with_version("oas_builder_agent")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        FIXTURES_ROOT / fixture.slug / "runs"
        / f"oas_enrich_{prompt_version}_{ts}"
    )

    def queue(runner) -> None:
        oas_builder = partial(
            build_oas_builder_agent,
            name="oas_builder",
            fs=fs,
            model=eval_model,
            max_tokens=120_000,
        )

        runner.add_variable(name="project_path", value=str(fixture.source_root))

        runner.add_task(
            name="oas_enrich",
            worker_builder=oas_builder,
            iterations=3, max_attempts=6, max_steps=30,
            artifacts=[
                "dependency_information/result",
                "project_information/result",
            ],
            namespace="openapi-building", model=eval_model,
        )

    run = await run_task_pipeline(
        queue_fn=queue,
        artifact_keys=["oas_enrich/result"],
        namespace=f"task-eval-{fixture.slug}-oas_enrich",
        timeout_s=2400.0,
        runner_name=f"oas-enrich-{fixture.slug}",
        preloaded_artifacts=precomputed,
        output_dir=run_dir,
    )

    result_text = run.result_text("oas_enrich")
    assert result_text, (
        "oas_enrich produced no result artifact\n"
        + render_metrics_table(run.metrics)
    )

    actual_schema = yaml.safe_load(result_text) or {}
    if not isinstance(actual_schema, dict):
        actual_schema = {}

    result = score_oas_schema(
        actual_schema, fixture.expected_oas,
        min_endpoint_precision=0.5, min_endpoint_recall=0.7,
        min_schema_recall=0.4,
    )

    summary = (
        f"fixture={fixture.slug}\n"
        f"{result.explain()}\n"
        f"  precomputed={'yes' if precomputed else 'no'}\n\n"
        f"metrics:\n{render_metrics_table(run.metrics)}"
    )
    print(f"\n{'='*60}\n{summary}\n{'='*60}")

    eval_sink.record(
        scenario="task", unit="oas_enrich", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(id=fixture.slug, passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_task(run.metrics), detail=diff_detail(result)),
    )
    assert result.passed, f"oas_enrich eval failed\n{summary}"
