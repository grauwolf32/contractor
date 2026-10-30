from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from contractor_runtime.contracts import AgentRegistration, AllocationSpec, decode_private
from contractor_runtime.contracts.base import (
    MAX_RUNTIME_LABEL_BYTES,
    MAX_RUNTIME_LABELS,
    _require_runtime_labels,
)

FIXTURES = Path(__file__).resolve().parents[2] / "api" / "testdata" / "v1alpha1"


def _cases() -> list[dict[str, Any]]:
    fixture = json.loads((FIXTURES / "runtime-label-set-cases.json").read_text())
    assert fixture["maxLabels"] == MAX_RUNTIME_LABELS
    assert fixture["maxLabelBytes"] == MAX_RUNTIME_LABEL_BYTES
    assert fixture["cases"]
    return fixture["cases"]


def _stored(case: dict[str, Any]) -> bool:
    """The Runtime reads only stored sets, which are already canonical."""

    return bool(case["valid"]) and case["labels"] == case.get("canonical")


@pytest.mark.parametrize("case", _cases(), ids=lambda case: str(case["name"]))
def test_shared_runtime_label_set_cases(case: dict[str, Any]) -> None:
    labels = list(case["labels"])
    registration = json.loads((FIXTURES / "valid" / "agent-registration.json").read_text())
    registration["initialLabels"] = labels
    allocation = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    provenance = allocation["resolvedRuntimeConfigProvenance"]
    template = provenance["runLabels"][0]
    provenance["runLabels"] = [{**template, "label": label} for label in labels]
    if _stored(case):
        _require_runtime_labels("labels", labels)
        model = decode_private(AgentRegistration, json.dumps(registration))
        assert model.initial_labels == labels
        decode_private(AllocationSpec, json.dumps(allocation))
    else:
        with pytest.raises(ValueError, match=r"Runtime label|sorted and unique"):
            _require_runtime_labels("labels", labels)
        with pytest.raises(ValidationError):
            AgentRegistration.model_validate_json(json.dumps(registration))
        with pytest.raises(ValidationError):
            AllocationSpec.model_validate_json(json.dumps(allocation))
