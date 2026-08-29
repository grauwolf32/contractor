"""Strict contractor/v1alpha1 DTOs shared with the Go Control Plane."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

API_VERSION = "contractor/v1alpha1"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


def _require_text(field: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _require_selector(field: str, value: str) -> str:
    if value.count("@") != 1:
        raise ValueError(f"{field} must use exact <id>@<version> syntax")
    identifier, version = value.split("@")
    if ID_PATTERN.fullmatch(identifier) is None or VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"{field} has an invalid exact selector")
    return value


def _require_digest(field: str, value: str) -> str:
    if DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be sha256 followed by 64 lowercase hex characters")
    return value


def _require_url(field: str, value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username is not None:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL without user information")
    return value


def _require_aware_datetime(field: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include an offset")
    return value


class WireModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
        strict=True,
    )


class VersionedWireModel(WireModel):
    api_version: Literal[API_VERSION]


class AgentObservedState(StrEnum):
    IDLE = "idle"
    ALLOCATED = "allocated"
    DRAINING = "draining"
    FENCED = "fenced"


class ReconciliationAction(StrEnum):
    CONTINUE = "continue"
    DRAIN = "drain"
    RELEASE = "release"
    REREGISTER = "reregister"


class ToolsetCapability(WireModel):
    ref: str
    tools: list[str]

    @model_validator(mode="after")
    def validate_capability(self) -> Self:
        _require_selector("ref", self.ref)
        if not self.tools or len(self.tools) != len(set(self.tools)):
            raise ValueError("tools must be non-empty and unique")
        for tool in self.tools:
            _require_text("tool", tool)
        return self


class AgentRegistration(VersionedWireModel):
    instance_id: str
    started_at: datetime
    control_url: str
    a2a_url: str
    supported_runtimes: list[str]
    supported_toolsets: list[ToolsetCapability]
    supported_sandbox_profiles: list[str]
    observed_state: AgentObservedState
    allocation_id: str | None = None

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        _require_text("instanceId", self.instance_id)
        _require_aware_datetime("startedAt", self.started_at)
        _require_url("controlUrl", self.control_url)
        _require_url("a2aUrl", self.a2a_url)
        _validate_capability_refs(
            self.supported_runtimes,
            self.supported_toolsets,
            self.supported_sandbox_profiles,
        )
        _validate_observed_allocation(self.observed_state, self.allocation_id)
        return self


class AgentRegistrationResponse(VersionedWireModel):
    heartbeat_interval_seconds: int = Field(gt=0)
    confirmed_lease_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.confirmed_lease_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("confirmed lease must exceed heartbeat interval")
        return self


class AgentHeartbeat(VersionedWireModel):
    instance_id: str
    heartbeat_seq: int = Field(gt=0)
    echoed_ack_seq: int = Field(ge=0)
    observed_state: AgentObservedState
    allocation_id: str | None = None

    @model_validator(mode="after")
    def validate_heartbeat(self) -> Self:
        _require_text("instanceId", self.instance_id)
        _validate_observed_allocation(self.observed_state, self.allocation_id)
        return self


class HeartbeatResponse(VersionedWireModel):
    ack_seq: int = Field(gt=0)
    action: ReconciliationAction
    allocation_id: str | None = None

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        if self.action in {ReconciliationAction.DRAIN, ReconciliationAction.RELEASE}:
            _require_text("allocationId", self.allocation_id or "")
        elif self.allocation_id is not None:
            _require_text("allocationId", self.allocation_id)
        return self


class AgentTemplateRef(WireModel):
    template_id: str
    version: str
    digest: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("agentTemplateRef", f"{self.template_id}@{self.version}")
        _require_digest("agentTemplateRef.digest", self.digest)
        return self


class WorkerRuntimeRef(WireModel):
    runtime_id: str
    version: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("workerRuntimeRef", f"{self.runtime_id}@{self.version}")
        return self


class ModelPolicyRef(WireModel):
    policy_id: str
    version: str
    digest: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("modelPolicyRef", f"{self.policy_id}@{self.version}")
        _require_digest("modelPolicyRef.digest", self.digest)
        return self


class ToolsetRef(WireModel):
    toolset_id: str
    version: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("toolsetRef", f"{self.toolset_id}@{self.version}")
        return self


class SandboxProfileRef(WireModel):
    sandbox_profile_id: str
    version: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("sandboxProfileRef", f"{self.sandbox_profile_id}@{self.version}")
        return self


class ResolvedInstructions(WireModel):
    ref: str
    digest: str
    text: str

    @model_validator(mode="after")
    def validate_instructions(self) -> Self:
        _require_text("instructions.ref", self.ref)
        _require_digest("instructions.digest", self.digest)
        _require_text("instructions.text", self.text)
        return self


class ResolvedModelPolicy(WireModel):
    ref: ModelPolicyRef
    model: str
    max_output_tokens: int = Field(gt=0)
    temperature: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_text("model", self.model)
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        return self


class ToolsetSelection(WireModel):
    ref: ToolsetRef
    tools: list[str]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.tools or len(self.tools) != len(set(self.tools)):
            raise ValueError("selected tools must be non-empty and unique")
        for tool in self.tools:
            _require_text("selected tool", tool)
        return self


class ResolvedAgentTemplate(WireModel):
    ref: AgentTemplateRef
    description: str
    runtime: WorkerRuntimeRef
    instructions: ResolvedInstructions
    model_policy: ResolvedModelPolicy
    toolsets: list[ToolsetSelection]
    sandbox_profile: SandboxProfileRef

    @model_validator(mode="after")
    def validate_template(self) -> Self:
        _require_text("description", self.description)
        refs = [f"{item.ref.toolset_id}@{item.ref.version}" for item in self.toolsets]
        if len(refs) != len(set(refs)):
            raise ValueError("toolset refs must be unique")
        visible = [tool for selection in self.toolsets for tool in selection.tools]
        if len(visible) != len(set(visible)):
            raise ValueError("model-visible tool names must be unique across Toolsets")
        return self


class RuntimeSettings(WireModel):
    llm_gateway_url: str
    llm_gateway_token: SecretStr
    artifact_api_url: str
    request_timeout_seconds: int = Field(gt=0)

    @field_validator("llm_gateway_url", "artifact_api_url")
    @classmethod
    def validate_url(cls, value: str, info: Any) -> str:
        return _require_url(info.field_name, value)

    @field_validator("llm_gateway_token")
    @classmethod
    def validate_token(cls, value: SecretStr) -> SecretStr:
        _require_text("llmGatewayToken", value.get_secret_value())
        return value

    @field_serializer("llm_gateway_token", when_used="json")
    def serialize_token(self, value: SecretStr) -> str:
        return value.get_secret_value()


class AllocationSpec(VersionedWireModel):
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    lease_expires_at: datetime
    agent_template: ResolvedAgentTemplate
    runtime_settings: RuntimeSettings

    @model_validator(mode="after")
    def validate_spec(self) -> Self:
        for field, value in (
            ("allocationId", self.allocation_id),
            ("runId", self.run_id),
            ("stageExecutionId", self.stage_execution_id),
            ("logicalAgentName", self.logical_agent_name),
            ("namespace", self.namespace),
        ):
            _require_text(field, value)
        if "/" in self.namespace:
            raise ValueError("namespace must not contain slash")
        _require_aware_datetime("leaseExpiresAt", self.lease_expires_at)
        return self


class PrepareAllocationRequest(VersionedWireModel):
    spec: AllocationSpec


class WorkerHandle(WireModel):
    allocation_id: str
    agent_template_ref: AgentTemplateRef
    worker_runtime_ref: WorkerRuntimeRef
    agent_card: dict[str, Any]
    lease_expires_at: datetime

    @model_validator(mode="after")
    def validate_handle(self) -> Self:
        _require_text("allocationId", self.allocation_id)
        if not self.agent_card:
            raise ValueError("agentCard must not be empty")
        _require_aware_datetime("leaseExpiresAt", self.lease_expires_at)
        return self


class PrepareAllocationResponse(VersionedWireModel):
    worker_handle: WorkerHandle


class FinalizeAllocationRequest(VersionedWireModel):
    allocation_id: str
    finalization_id: str
    deadline: datetime

    @model_validator(mode="after")
    def validate_finalize(self) -> Self:
        _validate_lifecycle(
            self.allocation_id, "finalizationId", self.finalization_id, self.deadline
        )
        return self


class TerminationError(WireModel):
    code: str
    message: str
    retryable: bool

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


class AbortAllocationRequest(VersionedWireModel):
    allocation_id: str
    abort_id: str
    reason: TerminationError
    deadline: datetime

    @model_validator(mode="after")
    def validate_abort(self) -> Self:
        _validate_lifecycle(self.allocation_id, "abortId", self.abort_id, self.deadline)
        return self


class ReleaseAllocationRequest(VersionedWireModel):
    allocation_id: str

    @field_validator("allocation_id")
    @classmethod
    def validate_allocation_id(cls, value: str) -> str:
        return _require_text("allocationId", value)


class ExecutionReport(WireModel):
    allocation_id: str
    started_at: datetime
    finished_at: datetime
    complete: bool
    counters: dict[str, int]
    errors: list[TerminationError]
    truncated: bool

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_text("allocationId", self.allocation_id)
        _require_aware_datetime("startedAt", self.started_at)
        _require_aware_datetime("finishedAt", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finishedAt must not precede startedAt")
        if any(not key or value < 0 for key, value in self.counters.items()):
            raise ValueError("counters require non-empty keys and non-negative values")
        return self


class AllocationFinalResponse(VersionedWireModel):
    report: ExecutionReport


class ArtifactRef(WireModel):
    namespace: str
    name: str
    revision: str | None = None

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        if not self.namespace.strip() or "/" in self.namespace:
            raise ValueError("namespace must be non-empty and contain no slash")
        if not self.name.strip() or "/" in self.name:
            raise ValueError("name must be non-empty and contain no slash")
        if self.revision is not None:
            _require_text("revision", self.revision)
        return self

    def require_exact(self) -> Self:
        if self.revision is None:
            raise ValueError("artifact revision is required")
        return self


class ArtifactReadResult(VersionedWireModel):
    artifact: ArtifactRef
    media_type: str
    size: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        self.artifact.require_exact()
        _validate_media_type(self.media_type)
        return self


class ArtifactWriteResult(ArtifactReadResult):
    pass


class ArtifactListResult(VersionedWireModel):
    artifacts: list[ArtifactRef]

    @model_validator(mode="after")
    def validate_artifacts(self) -> Self:
        if any(artifact.revision is not None for artifact in self.artifacts):
            raise ValueError("listed artifact refs must be versionless")
        return self


class StageContentRequest(VersionedWireModel):
    objective: str
    instructions: str
    parameters: dict[str, str]
    artifacts: dict[str, ArtifactRef]

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        _require_text("objective", self.objective)
        _require_text("instructions", self.instructions)
        for key in self.parameters:
            _require_text("parameter name", key)
        for key, artifact in self.artifacts.items():
            _require_text("artifact context name", key)
            artifact.require_exact()
        return self


class StageOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StageContentResult(VersionedWireModel):
    outcome: StageOutcome
    summary: str
    artifacts: dict[str, ArtifactRef]
    error: TerminationError | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_text("summary", self.summary)
        if self.outcome is StageOutcome.SUCCEEDED and self.error is not None:
            raise ValueError("successful result must not contain error")
        if self.outcome is StageOutcome.FAILED and self.error is None:
            raise ValueError("failed result requires error")
        for key, artifact in self.artifacts.items():
            _require_text("result artifact slot", key)
            artifact.require_exact()
        return self


def _validate_capability_refs(
    runtimes: list[str], toolsets: list[ToolsetCapability], sandboxes: list[str]
) -> None:
    if not runtimes or not sandboxes:
        raise ValueError("runtime and sandbox capability lists must not be empty")
    if len(runtimes) != len(set(runtimes)) or len(sandboxes) != len(set(sandboxes)):
        raise ValueError("capability refs must be unique")
    for value in runtimes:
        _require_selector("supportedRuntimes", value)
    for value in sandboxes:
        _require_selector("supportedSandboxProfiles", value)
    refs = [item.ref for item in toolsets]
    if len(refs) != len(set(refs)):
        raise ValueError("toolset capability refs must be unique")


def _validate_observed_allocation(state: AgentObservedState, allocation_id: str | None) -> None:
    if state is AgentObservedState.IDLE:
        if allocation_id is not None:
            raise ValueError("idle agent must not report allocationId")
        return
    _require_text("allocationId", allocation_id or "")


def _validate_lifecycle(
    allocation_id: str, id_field: str, id_value: str, deadline: datetime
) -> None:
    _require_text("allocationId", allocation_id)
    _require_text(id_field, id_value)
    _require_aware_datetime("deadline", deadline)


def _validate_media_type(value: str) -> None:
    parts = value.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or value != value.lower()
        or any(char in value for char in "; ")
    ):
        raise ValueError("mediaType must be lowercase type/subtype without parameters")
