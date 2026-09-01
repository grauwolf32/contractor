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
    AllocationWorkspaceSpecV2,
    ArtifactListResult,
    ArtifactReadResult,
    FinalizeAllocationRequest,
    HeartbeatResponse,
    PrivateProtocolDecodeError,
    ReleaseAllocationRequest,
    ResolvedAgentTemplate,
    ResolvedLLMGatewayConfig,
    ResolvedRuntimeConfigProvenanceV2,
    RuntimeReportV2,
    RuntimeSettings,
    RuntimeSettingsV2,
    StageContentRequest,
    StageContentResult,
    WorkspaceCapabilitiesV2,
    decode_private_v2,
    encode_private_v2,
)
from contractor_runtime.digests import (
    GatewayDigestMismatch,
    _agent_template_digest,
    verify_gateway_config_digest,
)

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
    "workspace-capabilities.json": WorkspaceCapabilitiesV2,
    "allocation-workspace-overlay.json": AllocationWorkspaceSpecV2,
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
    "runtime-settings-caido-secret-error.json": (RuntimeSettingsV2, "invariant"),
    "runtime-provenance-secret-field.json": (ResolvedRuntimeConfigProvenanceV2, "schema"),
    "workspace-capabilities-unsorted-modes.json": (
        WorkspaceCapabilitiesV2,
        "invariant",
    ),
    "allocation-workspace-versionless-source.json": (
        AllocationWorkspaceSpecV2,
        "invariant",
    ),
}

VALID_MODELS: dict[str, type[BaseModel]] = {
    "agent-registration.json": AgentRegistration,
    "agent-registration-response.json": AgentRegistrationResponse,
    "agent-heartbeat.json": AgentHeartbeat,
    "heartbeat-response.json": HeartbeatResponse,
    "llm-gateway-config.json": ResolvedLLMGatewayConfig,
    "allocation-spec.json": AllocationSpec,
    "allocation-spec-skills.json": AllocationSpec,
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
    "allocation-spec-resolved-skill-versionless.json": AllocationSpec,
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
    "allocation-spec-skills": "allocation.schema.json",
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
    "allocation-spec-resolved-skill-versionless": "allocation.schema.json",
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
        "caido-invalid-secret-canary",
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
    for secret in (
        "gateway-secret",
        "telemetry-secret",
        "proxy-bearer-secret",
        "caido-bearer-secret",
    ):
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


def test_agent_template_skill_refs_are_strict_and_empty_wire_is_compatible() -> None:
    allocation = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    raw = allocation["agentTemplate"]

    omitted = ResolvedAgentTemplate.model_validate(raw)
    explicit_empty = ResolvedAgentTemplate.model_validate({**raw, "skills": []})
    assert "skills" not in omitted.model_dump(by_alias=True)
    assert "skills" not in explicit_empty.model_dump(by_alias=True)

    valid = ResolvedAgentTemplate.model_validate(
        {
            **raw,
            "skills": [
                {"namespace": "skills", "name": "analysis2"},
                {"namespace": "skills", "name": "review"},
            ],
        }
    )
    assert [skill.name for skill in valid.skills] == ["analysis2", "review"]

    for skills in (
        [{"namespace": "other", "name": "review"}],
        [{"namespace": "skills", "name": "Review"}],
        [{"namespace": "skills", "name": "review", "revision": "rev-1"}],
        [
            {"namespace": "skills", "name": "review"},
            {"namespace": "skills", "name": "review"},
        ],
        [
            {"namespace": "skills", "name": "review"},
            {"namespace": "skills", "name": "analysis2"},
        ],
    ):
        with pytest.raises(ValidationError):
            ResolvedAgentTemplate.model_validate({**raw, "skills": skills})


def test_nonempty_agent_template_skill_digest_matches_go() -> None:
    allocation = AllocationSpec.model_validate_json(
        (FIXTURES / "valid" / "allocation-spec-skills.json").read_text()
    )
    assert (
        _agent_template_digest(allocation.agent_template)
        == "sha256:f088d3a6c4b2ecd9e430da5f503db36d023cdb8791efec29468f91d7120293d6"
    )


def test_agent_template_skills_reserve_native_names_and_require_tool_budget() -> None:
    allocation = json.loads((FIXTURES / "valid" / "allocation-spec.json").read_text())
    raw = allocation["agentTemplate"]
    skill = [{"namespace": "skills", "name": "review"}]

    collision = json.loads(json.dumps(raw))
    collision["skills"] = skill
    collision["toolsets"][0]["tools"][0] = "load_skill"
    with pytest.raises(ValidationError, match="reserved by Agent Skills"):
        ResolvedAgentTemplate.model_validate(collision)

    unbudgeted = json.loads(json.dumps(raw))
    unbudgeted["skills"] = skill
    unbudgeted["toolsets"] = []
    del unbudgeted["modelPolicy"]["maxToolCalls"]
    with pytest.raises(ValidationError, match="incompatible with adk@1 Worker"):
        ResolvedAgentTemplate.model_validate(unbudgeted)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("resolvedSkills"),
        lambda value: value["resolvedSkills"].append(value["resolvedSkills"][1].copy()),
        lambda value: value["resolvedSkills"].__setitem__(1, value["resolvedSkills"][0].copy()),
        lambda value: value["resolvedSkills"].reverse(),
        lambda value: value["resolvedSkills"][0]["artifact"].pop("revision"),
        lambda value: value["resolvedSkills"][0]["artifact"].__setitem__("namespace", "other"),
        lambda value: value["resolvedSkills"][0]["artifact"].__setitem__("name", "review"),
        lambda value: value["resolvedSkills"][0].__setitem__("packageDigest", "sha256:ABC"),
    ],
)
def test_allocation_resolved_skills_rejects_every_manifest_mismatch(mutation: Any) -> None:
    value = json.loads((FIXTURES / "valid" / "allocation-spec-skills.json").read_text())
    mutation(value)
    with pytest.raises(ValidationError):
        AllocationSpec.model_validate(value)


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
