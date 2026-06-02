"""End-to-end eval for `oas_builder_agent`.

The agent walks the fixture's source tree and accumulates an OpenAPI schema
into a session-scoped artifact (`user:oas-<namespace>`). We compare the
produced schema's endpoint and component sets against ground truth.
"""

from __future__ import annotations

import pytest
import yaml

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from tests.eval.harness import run_agent
from tests.eval.results import CaseResult, case_artifact_dir, metrics_from_events
from tests.eval.scorers import diff_detail, score_oas_schema

NAMESPACE = "openapi-eval"
ARTIFACT_KEY = f"user:oas-{NAMESPACE}"


def _build_user_message(source_root_label: str) -> str:
    return (
        f"Build the OpenAPI 3.0 schema for the project located at /. "
        f"(Project label: {source_root_label}.) "
        "Discover every HTTP endpoint, request body, and response by reading the "
        "source code with the available tools. Use upsert_path for each "
        "endpoint and upsert_component for each reusable schema. "
        "Stop only when list_paths reflects the full set of endpoints found in code."
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_oas_builder_endpoint_coverage(fixture, fixture_fs, eval_model, eval_sink):
    agent = build_oas_builder_agent(
        name="oas_builder",
        fs=fixture_fs,
        namespace=NAMESPACE,
        model=eval_model,
        max_tokens=80_000,
    )

    run = await run_agent(
        agent,
        user_message=_build_user_message(fixture.slug),
        timeout_s=900.0,
        artifact_dir=case_artifact_dir("oas_builder", fixture.slug, fixture.slug),
    )

    schema_text = run.artifacts.get(ARTIFACT_KEY, "")
    assert schema_text, (
        f"oas_builder produced no schema artifact ({ARTIFACT_KEY}). "
        f"Tool calls: {run.tool_names()}"
    )

    actual_schema = yaml.safe_load(schema_text) or {}
    result = score_oas_schema(
        actual_schema, fixture.expected_oas,
        min_endpoint_precision=0.7, min_endpoint_recall=0.8,
        min_schema_recall=0.5,
    )
    eval_sink.record(
        scenario="agent", unit="oas_builder", metric_kind="diff",
        fixture=fixture.slug, model=str(eval_model.model),
        case=CaseResult(id=fixture.slug, passed=result.passed,
                        pass_count=int(result.passed), attempts=1,
                        metrics=metrics_from_events(run.metrics_events), detail=diff_detail(result)),
        artifacts=run.artifacts,
    )
    assert result.passed, f"oas_builder eval failed: fixture={fixture.slug}\n{result.explain()}"
