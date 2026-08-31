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
    AgentRegistrationResponseV2,
    AgentRegistrationV2,
    AllocationFinalResponse,
    AllocationSpec,
    AllocationSpecV2,
    ArtifactListResult,
    ArtifactReadResult,
    FinalizeAllocationRequest,
    HeartbeatResponse,
    PrivateProtocolDecodeError,
    ReleaseAllocationRequest,
    ResolvedLLMGatewayConfig,
    ResolvedRuntimeConfigProvenanceV2,
    RuntimeReportV2,
    RuntimeSettings,
    RuntimeSettingsV2,
    StageContentRequest,
    StageContentResult,
    decode_private_v2,
    encode_private_v2,
)
from contractor_runtime.digests import GatewayDigestMismatch, verify_gateway_config_digest

FIXTURES = Path(__file__).parents[2] / "api" / "testdata" / "v1alpha1"
PRIVATE_V2_FIXTURES = Path(__file__).parents[2] / "testdata" / "contracts" / "private-v2"

PRIVATE_V2_VALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration.json": AgentRegistrationV2,
    "agent-registration-response.json": AgentRegistrationResponseV2,
    "runtime-settings-empty.json": RuntimeSettingsV2,
    "runtime-settings-telemetry.json": RuntimeSettingsV2,
    "runtime-settings-proxy.json": RuntimeSettingsV2,
    "runtime-settings-combined.json": RuntimeSettingsV2,
    "runtime-provenance.json": ResolvedRuntimeConfigProvenanceV2,
    "runtime-report.json": RuntimeReportV2,
}

PRIVATE_V2_INVALID_MODELS: dict[str, tuple[type[BaseModel], str]] = {
    "registration-versionless.json": (AgentRegistrationV2, "version"),
    "registration-v1.json": (AgentRegistrationV2, "version"),
    "registration-unsorted-labels.json": (AgentRegistrationV2, "invariant"),
    "registration-unsorted-adapters.json": (AgentRegistrationV2, "invariant"),
    "runtime-settings-duplicate-key.json": (RuntimeSettingsV2, "duplicate_key"),
    "runtime-settings-unknown-adapter.json": (RuntimeSettingsV2, "invariant"),
    "runtime-settings-two-proxy-auth.json": (RuntimeSettingsV2, "invariant"),
    "runtime-settings-secret-error.json": (RuntimeSettingsV2, "invariant"),
    "runtime-provenance-secret-field.json": (ResolvedRuntimeConfigProvenanceV2, "schema"),
}

VALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration.json": AgentRegistration,
    "agent-registration-response.json": AgentRegistrationResponse,
    "agent-heartbeat.json": AgentHeartbeat,
    "heartbeat-response.json": HeartbeatResponse,
    "llm-gateway-config.json": ResolvedLLMGatewayConfig,
    "allocation-spec.json": AllocationSpec,
    "allocation-final-response.json": AllocationFinalResponse,
    "finalize-allocation.json": FinalizeAllocationRequest,
    "abort-allocation.json": AbortAllocationRequest,
    "release-allocation.json": ReleaseAllocationRequest,
    "artifact-read-result.json": ArtifactReadResult,
    "artifact-list-result.json": ArtifactListResult,
    "stage-content-request.json": StageContentRequest,
    "stage-content-result-success.json": StageContentResult,
    "stage-content-result-failure.json": StageContentResult,
}

INVALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration-idle-with-allocation.json": AgentRegistration,
    "agent-registration-oversized-software-version.json": AgentRegistration,
    "agent-heartbeat-missing-allocation.json": AgentHeartbeat,
    "heartbeat-response-unknown-action.json": HeartbeatResponse,
    "llm-gateway-config-secret-field.json": ResolvedLLMGatewayConfig,
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
    "llm-gateway-config": "llm-gateway-config.schema.json",
    "allocation-spec": "allocation.schema.json",
    "allocation-final-response": "allocation.schema.json",
    "finalize-allocation": "allocation.schema.json",
    "abort-allocation": "allocation.schema.json",
    "release-allocation": "allocation.schema.json",
    "artifact-read-result": "artifact.schema.json",
    "artifact-list-result": "artifact.schema.json",
    "stage-content-request": "stage-content.schema.json",
    "stage-content-result-success": "stage-content.schema.json",
    "stage-content-result-failure": "stage-content.schema.json",
    "agent-registration-idle-with-allocation": "agent-registration.schema.json",
    "agent-registration-oversized-software-version": "agent-registration.schema.json",
    "agent-heartbeat-missing-allocation": "agent-heartbeat.schema.json",
    "heartbeat-response-unknown-action": "agent-heartbeat.schema.json",
    "llm-gateway-config-secret-field": "llm-gateway-config.schema.json",
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


@pytest.mark.parametrize(("filename", "model"), PRIVATE_V2_VALID_MODELS.items())
def test_private_v2_valid_fixture_is_shared_canonical_json(
    filename: str, model: type[BaseModel]
) -> None:
    raw = (PRIVATE_V2_FIXTURES / "valid" / filename).read_bytes().strip()
    value = decode_private_v2(model, raw)
    assert encode_private_v2(value) == raw


@pytest.mark.parametrize(("filename", "case"), PRIVATE_V2_INVALID_MODELS.items())
def test_private_v2_invalid_fixture_has_safe_reason(
    filename: str, case: tuple[type[BaseModel], str]
) -> None:
    model, reason = case
    raw = (PRIVATE_V2_FIXTURES / "invalid" / filename).read_bytes()
    with pytest.raises(PrivateProtocolDecodeError) as failure:
        decode_private_v2(model, raw)
    assert failure.value.reason == reason
    rendered = f"{failure.value!s} {failure.value!r}"
    for canary in (
        "recognizable-secret-canary",
        "recognizable-provenance-secret",
        "proxy-password-canary",
        "proxy-bearer-canary",
        "unknown-secret-adapter",
    ):
        assert canary not in rendered


def test_private_v2_allocation_composes_and_redacts_settings() -> None:
    value = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    value["runtimeSettings"] = json.loads(
        (PRIVATE_V2_FIXTURES / "valid" / "runtime-settings-combined.json").read_text()
    )
    value["resolvedRuntimeConfigProvenance"] = json.loads(
        (PRIVATE_V2_FIXTURES / "valid" / "runtime-provenance.json").read_text()
    )
    allocation = decode_private_v2(AllocationSpecV2, json.dumps(value))
    canonical = encode_private_v2(allocation)
    decode_private_v2(AllocationSpecV2, canonical)
    rendered = f"{allocation.runtime_settings!s} {allocation.runtime_settings!r}"
    for secret in ("gateway-secret", "telemetry-secret", "proxy-bearer-secret"):
        assert secret not in rendered
        assert secret.encode() in canonical


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


def test_runtime_settings_accepts_explicit_unauthenticated_gateway() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "llmGatewayUrl": "http://127.0.0.1:4000/v1",
            "llmGatewayToken": "",
            "artifactApiUrl": "https://server.example/private/v1",
            "requestTimeoutSeconds": 30,
        }
    )
    assert settings.llm_gateway_token.get_secret_value() == ""


def test_resolved_gateway_digest_matches_go_fixture() -> None:
    raw = (FIXTURES / "valid" / "llm-gateway-config.json").read_text(encoding="utf-8")
    gateway = ResolvedLLMGatewayConfig.model_validate_json(raw)
    verify_gateway_config_digest(gateway)
    assert "token" not in gateway.model_dump_json(by_alias=True).lower()
    changed = gateway.model_copy(update={"url": "http://127.0.0.1:4001/v1"})
    with pytest.raises(GatewayDigestMismatch):
        verify_gateway_config_digest(changed)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("maxOutputTokens", 0),
        ("maxModelCalls", 0),
        ("maxModelCalls", 1001),
        ("maxToolCalls", -1),
        ("maxToolCalls", 10001),
        ("maxTotalTokens", 0),
        ("maxTotalTokens", 100000001),
    ],
)
def test_worker_budget_wire_fields_are_required_and_bounded(field: str, invalid: int) -> None:
    raw = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    for policy_path in (("agentTemplate", "modelPolicy"), ("modelPolicy",)):
        missing = json.loads(json.dumps(raw))
        policy: dict[str, Any] = missing
        for component in policy_path:
            policy = policy[component]
        del policy[field]
        with pytest.raises(ValidationError):
            AllocationSpec.model_validate_json(json.dumps(missing))

        invalid_value = json.loads(json.dumps(raw))
        policy = invalid_value
        for component in policy_path:
            policy = policy[component]
        policy[field] = invalid
        with pytest.raises(ValidationError):
            AllocationSpec.model_validate_json(json.dumps(invalid_value))


def test_worker_policy_rejects_planner_only_limit() -> None:
    raw = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    raw["modelPolicy"]["maxWorkerCalls"] = 1
    with pytest.raises(ValidationError, match="incompatible with adk@1 Worker"):
        AllocationSpec.model_validate_json(json.dumps(raw))


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
