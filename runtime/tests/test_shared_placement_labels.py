from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contractor_runtime.contracts import AgentRegistration, decode_private
from contractor_runtime.contracts.placement import validate_placement_labels

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"


def test_shared_placement_labels() -> None:
    fixture = json.loads((FIXTURES / "placement-label-cases.json").read_text())
    assert fixture["cases"]
    assert fixture["maxLabels"] == 32
    assert fixture["maxLabelBytes"] == 63
    registration = json.loads((FIXTURES / "valid" / "agent-registration.json").read_text())
    for case in fixture["cases"]:
        registration["placementLabels"] = case["labels"]
        if case["valid"]:
            validate_placement_labels(case["labels"])
            model = decode_private(AgentRegistration, json.dumps(registration))
            assert model.placement_labels == case["labels"], case["name"]
        else:
            with pytest.raises(ValueError, match="placement labels"):
                validate_placement_labels(case["labels"])
            with pytest.raises(ValidationError):
                AgentRegistration.model_validate_json(json.dumps(registration))


def test_shared_placement_labels_reject_non_arrays() -> None:
    registration = json.loads((FIXTURES / "valid" / "agent-registration.json").read_text())
    for value in (None, "cloud/aws", 42, {}):
        registration["placementLabels"] = value
        with pytest.raises(ValidationError):
            AgentRegistration.model_validate_json(json.dumps(registration))
