from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError
from referencing import Registry, Resource

from contractor_runtime.contracts import (
    AbortAllocationRequest,
    AgentHeartbeat,
    AgentRegistration,
    AgentRegistrationResponse,
    AllocationSpec,
    ArtifactReadResult,
    FinalizeAllocationRequest,
    HeartbeatResponse,
    ReleaseAllocationRequest,
    RuntimeSettings,
    StageContentRequest,
    StageContentResult,
)

FIXTURES = Path(__file__).parents[2] / "api" / "testdata" / "v1alpha1"

VALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration.json": AgentRegistration,
    "agent-registration-response.json": AgentRegistrationResponse,
    "agent-heartbeat.json": AgentHeartbeat,
    "heartbeat-response.json": HeartbeatResponse,
    "allocation-spec.json": AllocationSpec,
    "finalize-allocation.json": FinalizeAllocationRequest,
    "abort-allocation.json": AbortAllocationRequest,
    "release-allocation.json": ReleaseAllocationRequest,
    "artifact-read-result.json": ArtifactReadResult,
    "stage-content-request.json": StageContentRequest,
    "stage-content-result-success.json": StageContentResult,
    "stage-content-result-failure.json": StageContentResult,
}

INVALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration-idle-with-allocation.json": AgentRegistration,
    "agent-heartbeat-missing-allocation.json": AgentHeartbeat,
    "heartbeat-response-unknown-action.json": HeartbeatResponse,
    "allocation-spec-bad-api-version.json": AllocationSpec,
    "stage-content-request-unknown-field.json": StageContentRequest,
    "stage-content-result-unversioned-artifact.json": StageContentResult,
    "stage-content-result-success-with-error.json": StageContentResult,
    "artifact-read-result-unversioned.json": ArtifactReadResult,
}

FIXTURE_SCHEMAS = {
    "agent-registration": "agent-registration.schema.json",
    "agent-registration-response": "agent-registration-response.schema.json",
    "agent-heartbeat": "agent-heartbeat.schema.json",
    "heartbeat-response": "agent-heartbeat.schema.json",
    "allocation-spec": "allocation.schema.json",
    "finalize-allocation": "allocation.schema.json",
    "abort-allocation": "allocation.schema.json",
    "release-allocation": "allocation.schema.json",
    "artifact-read-result": "artifact.schema.json",
    "stage-content-request": "stage-content.schema.json",
    "stage-content-result-success": "stage-content.schema.json",
    "stage-content-result-failure": "stage-content.schema.json",
    "agent-registration-idle-with-allocation": "agent-registration.schema.json",
    "agent-heartbeat-missing-allocation": "agent-heartbeat.schema.json",
    "heartbeat-response-unknown-action": "agent-heartbeat.schema.json",
    "allocation-spec-bad-api-version": "allocation.schema.json",
    "stage-content-request-unknown-field": "stage-content.schema.json",
    "stage-content-result-unversioned-artifact": "stage-content.schema.json",
    "stage-content-result-success-with-error": "stage-content.schema.json",
    "artifact-read-result-unversioned": "artifact.schema.json",
}


@pytest.mark.parametrize(("filename", "model"), VALID_MODELS.items())
def test_valid_golden_fixture_round_trip(filename: str, model: type[BaseModel]) -> None:
    raw = (FIXTURES / "valid" / filename).read_text(encoding="utf-8")
    value = model.model_validate_json(raw)
    encoded = value.model_dump_json(by_alias=True, exclude_none=True)
    assert json.loads(encoded) == json.loads(raw)


@pytest.mark.parametrize(("filename", "model"), INVALID_MODELS.items())
def test_invalid_golden_fixture_is_rejected(filename: str, model: type[BaseModel]) -> None:
    raw = (FIXTURES / "invalid" / filename).read_text(encoding="utf-8")
    with pytest.raises(ValidationError):
        model.model_validate_json(raw)


def test_runtime_settings_repr_is_redacted_but_json_is_wire_usable() -> None:
    token = "recognizable-secret-token"
    settings = RuntimeSettings.model_validate(
        {
            "llmGatewayUrl": "https://gateway.example/v1",
            "llmGatewayToken": token,
            "artifactApiUrl": "https://server.example/private/v1",
            "requestTimeoutSeconds": 30,
        }
    )
    assert token not in repr(settings)
    assert token not in str(settings)
    assert json.loads(settings.model_dump_json(by_alias=True))["llmGatewayToken"] == token


def test_all_golden_files_have_an_assigned_model() -> None:
    valid_files = {path.name for path in (FIXTURES / "valid").glob("*.json")}
    invalid_files = {path.name for path in (FIXTURES / "invalid").glob("*.json")}
    assert valid_files == VALID_MODELS.keys()
    assert invalid_files == INVALID_MODELS.keys()


def test_schema_files_are_json_objects() -> None:
    schema_root = Path(__file__).parents[2] / "api" / "v1alpha1"
    schemas: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8")) for path in schema_root.glob("*.schema.json")
    ]
    assert len(schemas) >= 5
    expected_draft = "https://json-schema.org/draft/2020-12/schema"
    assert all(schema.get("$schema") == expected_draft for schema in schemas)


def test_json_schemas_accept_and_reject_the_golden_fixtures() -> None:
    schema_root = Path(__file__).parents[2] / "api" / "v1alpha1"
    schemas = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in schema_root.glob("*.schema.json")
    }
    resources = [(schema["$id"], Resource.from_contents(schema)) for schema in schemas.values()]
    registry = Registry().with_resources(resources)

    for kind, expected_valid in (("valid", True), ("invalid", False)):
        for path in (FIXTURES / kind).glob("*.json"):
            schema_name = FIXTURE_SCHEMAS[path.stem]
            validator = Draft202012Validator(schemas[schema_name], registry=registry)
            errors = list(validator.iter_errors(json.loads(path.read_text(encoding="utf-8"))))
            assert bool(errors) is not expected_valid, (path.name, errors)
