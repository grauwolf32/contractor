from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes.spec import allocation_spec
from pydantic import ValidationError

from contractor_runtime.contracts import AllocationSpec, AllocationSpecV2
from contractor_runtime.digests import TemplateDigestMismatch, verify_template_digests

FIXTURES = Path(__file__).parents[2] / "api" / "testdata" / "v1alpha1" / "valid"


def test_optional_summarizer_round_trips_on_both_allocation_contracts() -> None:
    raw = (FIXTURES / "allocation-spec-summarizer.json").read_text(encoding="utf-8")
    legacy = AllocationSpec.model_validate_json(raw)
    assert legacy.agent_template.summarizer is not None
    assert legacy.agent_template.summarizer.soft_total_tokens == 20_000
    assert legacy.agent_template.summarizer.soft_prompt_tokens == 12_000
    verify_template_digests(legacy.agent_template)

    current = allocation_spec(summarizer=True)
    encoded = current.model_dump_json(by_alias=True, exclude_none=True)
    decoded = AllocationSpecV2.model_validate_json(encoded)
    assert decoded == current
    assert decoded.agent_template.summarizer is not None
    assert decoded.agent_template.summarizer.model_policy.model == "worker-summarizer-model"
    verify_template_digests(decoded.agent_template)


def test_omitted_summarizer_is_absent_and_behaviorally_disabled() -> None:
    allocation = allocation_spec()
    encoded = allocation.model_dump(by_alias=True, exclude_none=True)
    assert "summarizer" not in encoded["agentTemplate"]
    verify_template_digests(allocation.agent_template)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: (
            value.pop("softTotalTokens"),
            value.pop("softPromptTokens"),
        ),
        lambda value: value.__setitem__("softTotalTokens", 0),
        lambda value: value.__setitem__("softPromptTokens", -1),
        lambda value: value.__setitem__("softTotalTokens", 32_768),
        lambda value: value["modelPolicy"].pop("maxOutputTokens"),
        lambda value: value["modelPolicy"].pop("maxTotalTokens"),
        lambda value: value["modelPolicy"].__setitem__("maxModelCalls", 2),
        lambda value: value["modelPolicy"].__setitem__("maxToolCalls", 1),
        lambda value: value["modelPolicy"].__setitem__("maxWorkerCalls", 1),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_invalid_summarizer_wire_shape_is_rejected(mutation: Any) -> None:
    allocation = json.loads(
        (FIXTURES / "allocation-spec-summarizer.json").read_text(encoding="utf-8")
    )
    mutation(allocation["agentTemplate"]["summarizer"])
    with pytest.raises(ValidationError):
        AllocationSpec.model_validate(allocation)


def test_effective_worker_budget_must_remain_above_soft_total_threshold() -> None:
    allocation = allocation_spec(summarizer=True)
    candidate = allocation.model_dump(by_alias=True, exclude_none=True)
    candidate["modelPolicy"]["maxTotalTokens"] = 20_000
    with pytest.raises(ValidationError, match="effective Worker maxTotalTokens"):
        AllocationSpecV2.model_validate(candidate)


def test_summarizer_policy_and_thresholds_are_covered_by_template_digests() -> None:
    allocation = allocation_spec(summarizer=True)
    template = allocation.agent_template
    verify_template_digests(template)

    changed_threshold = template.model_copy(deep=True)
    assert changed_threshold.summarizer is not None
    changed_threshold.summarizer.soft_prompt_tokens += 1
    with pytest.raises(TemplateDigestMismatch, match="AgentTemplate"):
        verify_template_digests(changed_threshold)

    changed_policy = template.model_copy(deep=True)
    assert changed_policy.summarizer is not None
    changed_policy.summarizer.model_policy.model = "different-summary-model"
    with pytest.raises(TemplateDigestMismatch, match="ModelPolicy"):
        verify_template_digests(changed_policy)
