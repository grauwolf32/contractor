from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes.spec import allocation_spec
from pydantic import ValidationError

from contractor_runtime.contracts import AllocationSpec
from contractor_runtime.digests import TemplateDigestMismatch, verify_template_digests

FIXTURES = Path(__file__).parents[2] / "api" / "testdata" / "v1alpha1" / "valid"


def test_optional_summarizer_round_trips_on_both_allocation_contracts() -> None:
    raw = (FIXTURES / "allocation-spec-summarizer.json").read_text(encoding="utf-8")
    legacy = AllocationSpec.model_validate_json(raw)
    assert legacy.agent_template.summarizer is not None
    assert legacy.agent_template.summarizer.cumulative_budget == 20_000
    assert legacy.agent_template.summarizer.context_window_ratio == 0.9
    assert legacy.agent_template.model_policy.context_window_tokens == 131_072
    assert legacy.agent_template.summarizer.model_policy.context_window_tokens == 131_072
    verify_template_digests(legacy.agent_template)

    current = allocation_spec(summarizer=True)
    encoded = current.model_dump_json(by_alias=True, exclude_none=True)
    decoded = AllocationSpec.model_validate_json(encoded)
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
        lambda value: value.pop("contextWindowRatio"),
        lambda value: value.__setitem__("cumulativeBudget", 0),
        lambda value: value.__setitem__("contextWindowRatio", 0),
        lambda value: value.__setitem__("contextWindowRatio", 1),
        lambda value: value.__setitem__("cumulativeBudget", 32_768),
        lambda value: value["modelPolicy"].pop("maxOutputTokens"),
        lambda value: value["modelPolicy"].pop("contextWindowTokens"),
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
        AllocationSpec.model_validate(candidate)


def test_summarized_effective_worker_requires_context_window_metadata() -> None:
    allocation = allocation_spec(summarizer=True)
    candidate = allocation.model_dump(by_alias=True, exclude_none=True)
    candidate["modelPolicy"].pop("contextWindowTokens")
    with pytest.raises(ValidationError, match="effective Worker modelPolicy"):
        AllocationSpec.model_validate(candidate)


def test_summarizer_policy_and_thresholds_are_covered_by_template_digests() -> None:
    allocation = allocation_spec(summarizer=True)
    template = allocation.agent_template
    verify_template_digests(template)

    changed_threshold = template.model_copy(deep=True)
    assert changed_threshold.summarizer is not None
    changed_threshold.summarizer.context_window_ratio = 0.85
    with pytest.raises(TemplateDigestMismatch, match="AgentTemplate"):
        verify_template_digests(changed_threshold)

    changed_policy = template.model_copy(deep=True)
    assert changed_policy.summarizer is not None
    changed_policy.summarizer.model_policy.model = "different-summary-model"
    with pytest.raises(TemplateDigestMismatch, match="ModelPolicy"):
        verify_template_digests(changed_policy)


def test_summary_instruction_golden_and_digest_tampering() -> None:
    from contractor_runtime.digests import _agent_template_digest

    raw = (FIXTURES / "allocation-spec-summarizer-instructions.json").read_text(encoding="utf-8")
    template = AllocationSpec.model_validate_json(raw).agent_template
    verify_template_digests(template)
    assert template.summarizer is not None
    assert template.summarizer.instructions is not None
    template.summarizer.instructions.ref = "instructions/other.md"
    with pytest.raises(TemplateDigestMismatch, match="AgentTemplate"):
        verify_template_digests(template)
    template.ref.digest = _agent_template_digest(template)
    verify_template_digests(template)
    template.summarizer.instructions.text += "Changed."
    with pytest.raises(TemplateDigestMismatch, match="instruction digest"):
        verify_template_digests(template)


@pytest.mark.parametrize(
    "text,valid", [("界" * 8000, True), ("界" * 8001, False), (" \n\t", False)]
)
def test_summary_instruction_wire_character_limit(text: str, valid: bool) -> None:
    import hashlib

    allocation = allocation_spec(summarizer=True).model_dump(by_alias=True, exclude_none=True)
    allocation["agentTemplate"]["summarizer"]["instructions"] = {
        "ref": "instructions/summary.md",
        "digest": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }
    if valid:
        decoded = AllocationSpec.model_validate(allocation)
        assert decoded.agent_template.summarizer.instructions.text == text
    else:
        with pytest.raises(ValidationError):
            AllocationSpec.model_validate(allocation)
