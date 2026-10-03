"""The Runtime reproduces every canonical digest in the cases Go also checks."""

import json
from pathlib import Path

import pytest

from contractor_runtime.contracts import (
    ResolvedAgentTemplate,
    ResolvedLLMGatewayConfig,
    ResolvedModelPolicy,
)
from contractor_runtime.digests import (
    verify_gateway_config_digest,
    verify_model_policy_digest,
    verify_template_digests,
)

CASES = json.loads(
    (Path(__file__).parents[2] / "api/testdata/v1alpha1/digest-cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(
    "value", CASES["modelPolicies"], ids=lambda value: value["ref"]["policyId"]
)
def test_model_policy_digest_case(value: dict) -> None:
    verify_model_policy_digest(ResolvedModelPolicy.model_validate(value))


@pytest.mark.parametrize(
    "value", CASES["llmGatewayConfigs"], ids=lambda value: value["ref"]["gatewayId"]
)
def test_llm_gateway_config_digest_case(value: dict) -> None:
    verify_gateway_config_digest(ResolvedLLMGatewayConfig.model_validate(value))


@pytest.mark.parametrize(
    "value", CASES["agentTemplates"], ids=lambda value: value["ref"]["templateId"]
)
def test_agent_template_digest_case(value: dict) -> None:
    verify_template_digests(ResolvedAgentTemplate.model_validate(value))
