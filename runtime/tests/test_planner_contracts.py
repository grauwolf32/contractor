from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contractor_runtime.contracts import API_VERSION
from contractor_runtime.contracts.base import json_byte_size
from contractor_runtime.contracts.planner import (
    MAX_PLANNER_INVOCATION_BYTES,
    MAX_PLANNER_OPERATION_BYTES,
    MAX_PLANNER_OPERATIONS,
    MAX_PLANNER_TOOLS,
    PlannerCompaction,
    PlannerOperation,
    PlannerOperationResult,
)


def test_shared_planner_execution_bounds_and_utf8_envelope():
    bounds = json.loads(
        (
            Path(__file__).parents[2] / "api/testdata/v1alpha1/planner-execution-limits.json"
        ).read_text()
    )
    assert (
        bounds["operationBytes"],
        bounds["invocationBytes"],
        bounds["tools"],
        bounds["operations"],
    ) == (
        MAX_PLANNER_OPERATION_BYTES,
        MAX_PLANNER_INVOCATION_BYTES,
        MAX_PLANNER_TOOLS,
        MAX_PLANNER_OPERATIONS,
    )
    schema = PlannerCompaction.model_json_schema(by_alias=True)["properties"]
    assert [schema[name]["maximum"] for name in ("keepRecentGroups", "maxCompactions")] == [
        bounds["recentGroups"],
        bounds["compactions"],
    ]
    assert schema["instructions"]["maxLength"] == bounds["instructions"]
    for model in (PlannerOperation, PlannerOperationResult):
        payload = {"apiVersion": API_VERSION}
        member = "arguments" if model is PlannerOperation else "result"
        payload.update(
            {"sequence": 1, "name": "before_model"}
            if model is PlannerOperation
            else {"stop": False, "escalate": False}
        )
        payload[member] = {"text": ""}
        available = bounds["operationBytes"] - json_byte_size(payload)
        payload[member]["text"] = "é" * (available // 2) + "x" * (available % 2)
        assert model.model_validate(payload)
        payload[member]["text"] += "x"
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_shared_planner_allocation_null_settings():
    from private_schemas import private_validator

    from contractor_runtime.contracts import AllocationSpec, decode_private

    root = Path(__file__).parents[2] / "api/testdata/v1alpha1"
    value = json.loads((root / "valid/allocation-spec-planner.json").read_text())
    patch = json.loads((root / "planner-allocation-settings.json").read_text())
    value["runtimeSettings"].update(patch)
    private_validator("allocation.schema.json").validate(value)
    allocation = decode_private(AllocationSpec, json.dumps(value))
    settings = allocation.runtime_settings.model_dump(by_alias=True)
    assert all(settings[name] is None for name in patch)
