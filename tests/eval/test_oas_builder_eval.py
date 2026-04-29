"""End-to-end eval for `oas_builder_agent`.

The agent walks the fixture's source tree and accumulates an OpenAPI schema
into a session-scoped artifact (`user:oas-<namespace>`). We compare the
produced schema's endpoint and component sets against ground truth.
"""

from __future__ import annotations

import yaml
import pytest

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent

from tests.eval.harness import run_agent
from tests.eval.scoring import score_components, score_endpoints

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
async def test_oas_builder_endpoint_coverage(fixture, fixture_fs, eval_model):
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
    )

    schema_text = run.artifacts.get(ARTIFACT_KEY, "")
    assert schema_text, (
        f"oas_builder produced no schema artifact ({ARTIFACT_KEY}). "
        f"Tool calls: {run.tool_names()}"
    )

    actual_schema = yaml.safe_load(schema_text) or {}

    endpoint_score = score_endpoints(actual_schema, fixture.expected_oas)
    schemas_score = score_components(actual_schema, fixture.expected_oas, "schemas")

    assert endpoint_score.passes(min_precision=0.7, min_recall=0.8), (
        endpoint_score.explain("endpoints")
    )
    assert schemas_score.recall >= 0.5, schemas_score.explain("schemas (recall>=0.5)")
