"""The Runtime reproduces every canonical digest in the cases Go also checks."""

import json
from pathlib import Path

import pytest

from contractor_runtime.contracts import (
    ResolvedAgentTemplate,
    ResolvedModelPolicy,
)
from contractor_runtime.digests import (
    verify_model_policy_digest,
    verify_template_digests,
)

CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/v1alpha1/digest-cases.json").read_text(
        encoding="utf-8"
    )
)
# Case groups only the Go Server checks, each with its reason. Every other group
# must be exercised below.
GO_ONLY_GROUPS: dict[str, str] = {
    "llmGatewayConfigs": (
        "the resolved Gateway body and its digest stay on the Go Server; the Runtime "
        "receives only the digest-bearing LLMGatewayConfigRef"
    ),
}
RUNTIME_GROUPS = {"modelPolicies", "agentTemplates"}


def test_every_case_group_is_checked_or_go_only() -> None:
    assert CASES.keys() == RUNTIME_GROUPS | GO_ONLY_GROUPS.keys()
    assert RUNTIME_GROUPS.isdisjoint(GO_ONLY_GROUPS)
    assert all(reason.strip() for reason in GO_ONLY_GROUPS.values())


@pytest.mark.parametrize(
    "value", CASES["modelPolicies"], ids=lambda value: value["ref"]["policyId"]
)
def test_model_policy_digest_case(value: dict) -> None:
    verify_model_policy_digest(ResolvedModelPolicy.model_validate(value))


@pytest.mark.parametrize(
    "value", CASES["agentTemplates"], ids=lambda value: value["ref"]["templateId"]
)
def test_agent_template_digest_case(value: dict) -> None:
    verify_template_digests(ResolvedAgentTemplate.model_validate(value))
