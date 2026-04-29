"""End-to-end eval for `oas_analyzer.report_generator`.

The analyzer takes an OpenAPI schema as input and accumulates findings into
session.state under the key `oas_analyzer::vulnerabilities`. We feed it the
fixture's ground-truth schema (so the builder is not on the critical path)
and score the findings against an expected vulnerability list.
"""

from __future__ import annotations

import json

import pytest
import yaml

from contractor.agents.oas_analyzer.agent import root_agent

from tests.eval.harness import run_agent
from tests.eval.scoring import score_vulnerabilities

VULN_STATE_KEY = "oas_analyzer::vulnerabilities"
SERVICE_INFO_KEY = "oas_analyzer::service_information"


def _user_message(schema: dict) -> str:
    return (
        "Analyze the following OpenAPI 3.0 schema. Identify endpoints relevant "
        "to your assigned security focus and call save_vulnerability for each "
        "high-confidence finding. Begin with the service review.\n\n"
        "```yaml\n"
        + yaml.safe_dump(schema, sort_keys=False)
        + "\n```"
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_oas_analyzer_finds_expected_classes(fixture, eval_model):
    # Inject the model into every LlmAgent in the analyzer tree. The analyzer
    # currently captures DEFAULT_MODEL at import time, so this monkeys the
    # configured model onto each constructed sub-agent for the duration of
    # the test.
    from google.adk.agents import LlmAgent

    def _all_llm_agents(node) -> list:
        out: list = []
        if isinstance(node, LlmAgent):
            out.append(node)
        for sub in getattr(node, "sub_agents", []) or []:
            out.extend(_all_llm_agents(sub))
        review = getattr(node, "review_agent", None)
        if review is not None:
            out.extend(_all_llm_agents(review))
        return out

    for llm_agent in _all_llm_agents(root_agent):
        llm_agent.model = eval_model

    run = await run_agent(
        root_agent,
        user_message=_user_message(fixture.expected_oas),
        timeout_s=1500.0,
    )

    vulnerabilities = run.state.get(VULN_STATE_KEY) or []
    assert isinstance(vulnerabilities, list), (
        f"expected list under {VULN_STATE_KEY}, got {type(vulnerabilities).__name__}"
    )

    assert run.state.get(SERVICE_INFO_KEY), (
        "review sub-agent did not populate oas_analyzer::service_information"
    )

    score = score_vulnerabilities(vulnerabilities, fixture.expected_vulnerabilities)

    debug_msg = score.explain("vulnerabilities") + "\nactual:\n" + json.dumps(
        vulnerabilities, indent=2, default=str
    )
    assert score.passes(min_precision=0.4, min_recall=0.5), debug_msg
