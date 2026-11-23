"""Strict private completion contract shared with the Server."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from contractor_runtime.contracts import (
    AgentRegistration,
    AllocationSpec,
    RuntimeCompletionCapabilities,
    WorkerCompletionContract,
    decode_private,
)

ROOT = Path(__file__).parents[2]
CASES = json.loads((ROOT / "api/testdata/v1alpha1/audit-completion-cases.json").read_bytes())
MODELS = {
    "contract": WorkerCompletionContract,
    "capabilities": RuntimeCompletionCapabilities,
    "allocation": AllocationSpec,
    "registration": AgentRegistration,
}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_go_completion_wire_fixtures(case):
    raw = json.dumps(case["value"]).encode()
    if case["valid"]:
        value = decode_private(MODELS[case["model"]], raw)
        assert value.model_dump(mode="json", by_alias=True, exclude_none=True) == case["value"]
    else:
        with pytest.raises(ValueError):
            decode_private(MODELS[case["model"]], raw)


def test_shared_private_schemas_accept_complete_contracts():
    schemas = {
        path.name: json.loads(path.read_bytes())
        for path in (ROOT / "api/v1alpha1").glob("*.schema.json")
    }
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()
    )
    private_alloc = json.loads((ROOT / "api/v1alpha1/allocation.schema.json").read_bytes())
    private_registration = json.loads(
        (ROOT / "api/v1alpha1/agent-registration.schema.json").read_bytes()
    )
    for case in CASES:
        if case["valid"] and case["model"] in {
            "allocation",
            "registration",
            "contract",
        }:
            schema = {
                "allocation": private_alloc,
                "registration": private_registration,
                "contract": schemas["worker-completion-contract.schema.json"],
            }[case["model"]]
            Draft202012Validator(schema, registry=registry).validate(case["value"])
    for case in CASES:
        if not case["valid"] and case["model"] == "contract":
            assert list(
                Draft202012Validator(
                    schemas["worker-completion-contract.schema.json"], registry=registry
                ).iter_errors(case["value"])
            )


def test_shared_optional_completion_fields_are_omitted_without_opt_in(tmp_path):
    from contractor_runtime.factories import built_in_factories

    assert "audit-results@2" in built_in_factories(tmp_path).toolsets
    raw = json.loads((ROOT / "api/testdata/v1alpha1/valid/agent-registration.json").read_bytes())
    registration = AgentRegistration.model_validate_json(json.dumps(raw))
    assert registration.capabilities is None
    assert "capabilities" not in registration.model_dump(by_alias=True, exclude_none=True)
    from fakes.spec import allocation_spec

    spec = allocation_spec()
    assert spec.completion_contract is None
    assert "completionContract" not in spec.model_dump(by_alias=True, exclude_none=True)
