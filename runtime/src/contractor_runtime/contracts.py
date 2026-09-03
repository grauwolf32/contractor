"""Strict contractor/v1alpha1 DTOs shared with the Go Control Plane."""

from __future__ import annotations

import ipaddress
import json
import math
import re
import ssl
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

import jcs
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

API_VERSION = "contractor/v1alpha1"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NATIVE_SKILL_TOOL_NAMES = frozenset({"list_skills", "load_skill", "load_skill_resource"})
WORKER_SUBTASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
WORKER_FAILURE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RUN_METADATA_LABEL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
MAX_RUN_METADATA_LABELS = 32
MAX_RUN_METADATA_LABEL_KEY_BYTES = 63
MAX_RUN_METADATA_LABEL_VALUE_BYTES = 256
MAX_WORKER_RESULT_BYTES = 64 * 1024
MAX_WORKER_FAILURE_MESSAGE_BYTES = 4 * 1024
MAX_WORKER_RESULT_ARTIFACTS = 128
MAX_WORKER_OBSERVATION_TOOLS = 256
MAX_WORKER_FILES_READ = 25
MAX_WORKER_COMPLETION_BYTES = 256 * 1024
MAX_AGENT_STATE_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_STATE_WORKSPACE_PATHS = 10_000
MAX_STATE_WORKSPACE_PATH_BYTES = 2 * 1024 * 1024
MAX_UINT64 = 2**64 - 1
_STATE_METRIC_NAME_PATTERN = re.compile(r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$")


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


def _require_inference_gateway_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError(
            "llmGatewayConfig.url must be an absolute HTTP(S) URL with an explicit "
            "path and no userinfo, query, or fragment"
        )
    return value


def _require_management_gateway_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or "?" in value
        or "#" in value
        or parsed.query
        or parsed.fragment
        or parsed.path
    ):
        raise ValueError("managementUrl must be a canonical HTTP(S) origin")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("HTTP managementUrl is allowed only for a loopback IP origin")
    return value


def _require_aware_datetime(field: str, value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include an offset")
    return value


def normalize_run_metadata_labels(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("runMetadataLabels must be an object")
    if len(value) > MAX_RUN_METADATA_LABELS:
        raise ValueError("runMetadataLabels exceed 32 entries")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("runMetadataLabels contain an invalid key")
    result: dict[str, str] = {}
    for key in sorted(value):
        label_value = value[key]
        if (
            not key
            or len(key.encode("utf-8")) > MAX_RUN_METADATA_LABEL_KEY_BYTES
            or RUN_METADATA_LABEL_KEY_PATTERN.fullmatch(key) is None
            or key.startswith("contractor.")
        ):
            raise ValueError("runMetadataLabels contain an invalid key")
        if not isinstance(label_value, str):
            raise ValueError("runMetadataLabels contain an invalid value")
        encoded = label_value.encode("utf-8")
        if not encoded or len(encoded) > MAX_RUN_METADATA_LABEL_VALUE_BYTES or "\0" in label_value:
            raise ValueError("runMetadataLabels contain an invalid value")
        result[key] = label_value
    return result


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
    software_version: str = Field(min_length=1, max_length=128, pattern=VERSION_PATTERN.pattern)
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
        if self.action is ReconciliationAction.DRAIN:
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


class LLMGatewayConfigRef(WireModel):
    gateway_id: str
    version: str
    digest: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        _require_selector("llmGatewayConfigRef", f"{self.gateway_id}@{self.version}")
        _require_digest("llmGatewayConfigRef.digest", self.digest)
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
    context_window_tokens: int | None = Field(default=None, gt=0, le=100_000_000)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_model_calls: int | None = Field(default=None, gt=0, le=1000)
    max_tool_calls: int | None = Field(default=None, gt=0, le=10_000)
    max_worker_calls: int | None = Field(default=None, gt=0, le=10_000)
    max_total_tokens: int | None = Field(default=None, gt=0, le=100_000_000)
    temperature: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_text("model", self.model)
        if self.temperature is not None and not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if (
            self.context_window_tokens is not None
            and self.max_output_tokens is not None
            and self.max_output_tokens >= self.context_window_tokens
        ):
            raise ValueError("maxOutputTokens must be below contextWindowTokens")
        return self


def _require_worker_policy(policy: ResolvedModelPolicy, *, has_tools: bool) -> None:
    if (
        policy.max_output_tokens is None
        or policy.max_model_calls is None
        or policy.max_total_tokens is None
        or (has_tools and policy.max_tool_calls is None)
        or policy.max_worker_calls is not None
    ):
        raise ValueError("modelPolicy is incompatible with adk@1 Worker")


def _require_worker_summarizer_policy(policy: ResolvedModelPolicy) -> None:
    if (
        policy.context_window_tokens is None
        or policy.max_output_tokens is None
        or policy.max_model_calls != 1
        or policy.max_tool_calls is not None
        or policy.max_worker_calls is not None
    ):
        raise ValueError("modelPolicy is incompatible with the Worker terminal summarizer")


class LLMGatewayCredentialManager(WireModel):
    implementation: Literal["litellm-virtual-keys@1"]
    management_url: str

    @field_validator("management_url")
    @classmethod
    def validate_management_url(cls, value: str) -> str:
        return _require_management_gateway_origin(value)


class ResolvedLLMGatewayConfig(WireModel):
    ref: LLMGatewayConfigRef
    protocol: Literal["openai-compatible@1"]
    url: str
    credential_manager: LLMGatewayCredentialManager | None = None

    @field_validator("url")
    @classmethod
    def validate_inference_url(cls, value: str) -> str:
        return _require_inference_gateway_url(value)


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


class WorkerSummarizerConfig(WireModel):
    model_policy: ResolvedModelPolicy
    cumulative_budget: int | None = Field(default=None, gt=0, le=100_000_000)
    context_window_ratio: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_summarizer(self) -> Self:
        _require_worker_summarizer_policy(self.model_policy)
        if not math.isfinite(self.context_window_ratio):
            raise ValueError("Worker summarizer contextWindowRatio must be finite")
        return self


class ResolvedAgentTemplate(WireModel):
    ref: AgentTemplateRef
    description: str
    runtime: WorkerRuntimeRef
    instructions: ResolvedInstructions
    model_policy: ResolvedModelPolicy
    summarizer: WorkerSummarizerConfig | None = None
    toolsets: list[ToolsetSelection]
    skills: list[ArtifactRef] = Field(
        default_factory=list, max_length=32, exclude_if=lambda value: not value
    )
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
        skill_names: list[str] = []
        for skill in self.skills:
            if (
                skill.namespace != "skills"
                or skill.revision is not None
                or SKILL_NAME_PATTERN.fullmatch(skill.name) is None
                or len(skill.name) > 64
            ):
                raise ValueError("skills must be sorted versionless skills/<portable-name> refs")
            skill_names.append(skill.name)
        if skill_names != sorted(set(skill_names)):
            raise ValueError("skills must be sorted and unique")
        if self.skills and NATIVE_SKILL_TOOL_NAMES.intersection(visible):
            raise ValueError("model-visible tool name is reserved by Agent Skills")
        _require_worker_policy(self.model_policy, has_tools=bool(visible or self.skills))
        if self.summarizer is not None:
            _require_worker_summarizer_policy(self.summarizer.model_policy)
            if self.model_policy.context_window_tokens is None:
                raise ValueError("summarized Worker modelPolicy requires contextWindowTokens")
            if (
                self.summarizer.cumulative_budget is not None
                and self.model_policy.max_total_tokens is not None
                and self.summarizer.cumulative_budget >= self.model_policy.max_total_tokens
            ):
                raise ValueError(
                    "Worker summarizer cumulativeBudget must be below Worker maxTotalTokens"
                )
        return self


class ResolvedSkill(WireModel):
    name: str
    artifact: ArtifactRef
    package_digest: str

    @model_validator(mode="after")
    def validate_skill(self) -> Self:
        if SKILL_NAME_PATTERN.fullmatch(self.name) is None or len(self.name) > 64:
            raise ValueError("resolved Skill name is invalid")
        if self.artifact.namespace != "skills" or self.artifact.name != self.name:
            raise ValueError("resolved Skill artifact must identify skills/<name>")
        self.artifact.require_exact()
        _require_digest("resolved Skill packageDigest", self.package_digest)
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

    @field_serializer("llm_gateway_token", when_used="json")
    def serialize_token(self, value: SecretStr) -> str:
        return value.get_secret_value()


class AllocationSpec(VersionedWireModel):
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    run_metadata_labels: dict[str, str]
    lease_expires_at: datetime
    agent_template: ResolvedAgentTemplate
    resolved_skills: list[ResolvedSkill] = Field(max_length=32)
    model_policy: ResolvedModelPolicy
    runtime_settings: RuntimeSettings

    @field_validator("run_metadata_labels")
    @classmethod
    def validate_run_metadata_labels(cls, value: dict[str, str]) -> dict[str, str]:
        return normalize_run_metadata_labels(value)

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
        names = [skill.name for skill in self.resolved_skills]
        if names != sorted(set(names)):
            raise ValueError("resolvedSkills must be sorted and unique")
        if names != [skill.name for skill in self.agent_template.skills]:
            raise ValueError("resolvedSkills must exactly match AgentTemplate skills")
        _require_worker_policy(
            self.model_policy,
            has_tools=bool(
                self.agent_template.skills
                or any(selection.tools for selection in self.agent_template.toolsets)
            ),
        )
        if (
            self.agent_template.summarizer is not None
            and self.agent_template.summarizer.cumulative_budget is not None
            and self.model_policy.max_total_tokens is not None
            and self.agent_template.summarizer.cumulative_budget
            >= self.model_policy.max_total_tokens
        ):
            raise ValueError(
                "Worker summarizer cumulativeBudget must be below effective Worker maxTotalTokens"
            )
        if (
            self.agent_template.summarizer is not None
            and self.model_policy.context_window_tokens is None
        ):
            raise ValueError("summarized effective Worker modelPolicy requires contextWindowTokens")
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


class ExecutionError(WireModel):
    code: str
    message: str = Field(max_length=4096)
    retryable: bool | None = None

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


class ToolMetrics(WireModel):
    calls: int | None = Field(default=None, ge=0)
    succeeded: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)


class WorkerBudgetMetrics(WireModel):
    max_model_calls: int = Field(gt=0, le=1000)
    max_tool_calls: int = Field(gt=0, le=10_000)
    max_total_tokens: int = Field(gt=0, le=100_000_000)
    observed_model_calls: int = Field(ge=0)
    observed_tool_calls: int = Field(ge=0)
    observed_total_tokens: int = Field(ge=0)
    token_usage_unavailable: int = Field(ge=0)
    exhausted: Literal["model_calls", "tool_calls", "total_tokens"] | None = None

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if self.observed_model_calls > self.max_model_calls:
            raise ValueError("observedModelCalls exceeds maxModelCalls")
        if self.observed_tool_calls > self.max_tool_calls:
            raise ValueError("observedToolCalls exceeds maxToolCalls")
        if self.exhausted == "model_calls" and self.observed_model_calls != self.max_model_calls:
            raise ValueError("model-call exhaustion is inconsistent")
        if self.exhausted == "tool_calls" and self.observed_tool_calls != self.max_tool_calls:
            raise ValueError("tool-call exhaustion is inconsistent")
        if self.exhausted == "total_tokens" and self.observed_total_tokens < self.max_total_tokens:
            raise ValueError("token exhaustion is inconsistent")
        return self


class WorkerSummarizerMetrics(WireModel):
    attempts: int = Field(gt=0, le=MAX_UINT64)
    succeeded: int = Field(ge=0, le=MAX_UINT64)
    failed: int = Field(ge=0, le=MAX_UINT64)
    model_calls: int = Field(ge=0, le=MAX_UINT64)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    failure_codes: dict[str, int] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if self.succeeded + self.failed != self.attempts:
            raise ValueError("Worker summarizer terminal counts are inconsistent")
        if self.model_calls > self.attempts:
            raise ValueError("Worker summarizer model calls exceed attempts")
        if self.token_usage_unavailable > self.model_calls:
            raise ValueError("Worker summarizer missing-usage count exceeds model calls")
        failure_total = 0
        for code, count in self.failure_codes.items():
            if WORKER_FAILURE_CODE_PATTERN.fullmatch(code) is None or count <= 0:
                raise ValueError("Worker summarizer failure aggregate is invalid")
            failure_total += count
        if failure_total != self.failed:
            raise ValueError("Worker summarizer failure aggregate is inconsistent")
        return self


class ExecutionMetrics(WireModel):
    duration_ms: int | None = Field(default=None, ge=0)
    model_calls: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    tools: dict[str, ToolMetrics] = Field(default_factory=dict)
    worker_budget: WorkerBudgetMetrics | None = None
    summarizer: WorkerSummarizerMetrics | None = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: dict[str, ToolMetrics]) -> dict[str, ToolMetrics]:
        for name in value:
            _require_text("tool name", name)
        return value


class ToolCallOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ToolCallRecord(WireModel):
    call_id: str
    tool: str
    arguments: dict[str, Any] | None = None
    arguments_truncated: bool = False
    outcome: ToolCallOutcome
    duration_ms: int | None = Field(default=None, ge=0)
    result_size_bytes: int | None = Field(default=None, ge=0)
    error: ExecutionError | None = None

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        _require_text("callId", self.call_id)
        _require_text("tool", self.tool)
        if self.arguments is None and self.arguments_truncated:
            raise ValueError("absent arguments cannot be marked truncated")
        if (self.outcome is ToolCallOutcome.SUCCEEDED) == (self.error is not None):
            raise ValueError("tool outcome and error are inconsistent")
        return self


class ExecutionReport(WireModel):
    report_id: str
    complete: bool
    metrics: ExecutionMetrics
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    truncated: bool = False

    @field_validator("report_id")
    @classmethod
    def validate_report_id(cls, value: str) -> str:
        return _require_text("reportId", value)


class RuntimeReport(WireModel):
    complete: bool
    duration_ms: int | None = Field(default=None, ge=0)
    stop_reason: str | None = None
    adapters: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2] = Field(default_factory=dict)

    @field_validator("stop_reason")
    @classmethod
    def validate_stop_reason(cls, value: str | None) -> str | None:
        if value is not None:
            _require_text("stopReason", value)
        return value


class AllocationFinalReport(WireModel):
    report_id: str
    allocation_id: str
    started_at: datetime
    finished_at: datetime
    worker: ExecutionReport
    runtime: RuntimeReport

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        _require_text("reportId", self.report_id)
        _require_text("allocationId", self.allocation_id)
        _require_aware_datetime("startedAt", self.started_at)
        _require_aware_datetime("finishedAt", self.finished_at)
        if self.finished_at < self.started_at:
            raise ValueError("finishedAt must not precede startedAt")
        if (
            len(self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > 1024 * 1024
        ):
            raise ValueError("allocation final report exceeds 1 MiB")
        return self


class AllocationFinalResponse(VersionedWireModel):
    report: AllocationFinalReport


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
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    objective: str
    instructions: str
    parameters: dict[str, str]
    artifacts: dict[str, ArtifactRef]
    result_artifacts: dict[str, ArtifactRef] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_text("objective", self.objective)
        _require_text("instructions", self.instructions)
        for key in self.parameters:
            _require_text("parameter name", key)
        for key, artifact in self.artifacts.items():
            _require_text("artifact context name", key)
            artifact.require_exact()
        for key, artifact in self.result_artifacts.items():
            _require_text("result artifact slot", key)
            if artifact.revision is not None:
                raise ValueError("result artifact binding must be versionless")
        return self


def _require_worker_subtask_id(value: str) -> str:
    if (
        len(value.encode("ascii", errors="ignore")) != len(value)
        or not 1 <= len(value) <= 128
        or WORKER_SUBTASK_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("subtaskId is invalid")
    return value


def _require_worker_result_text(value: str) -> str:
    if not value.strip() or len(value.encode("utf-8")) > MAX_WORKER_RESULT_BYTES:
        raise ValueError("Worker result must contain 1..65536 UTF-8 bytes")
    return value


def _reserved_worker_result_binding(artifact: ArtifactRef) -> bool:
    return artifact.namespace in {"inputs", "outputs", "skills"} or artifact.name.startswith(
        "memory."
    )


class WorkerModelResult(WireModel):
    """Strict model-only result; it never crosses the Runtime A2A boundary."""

    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    result: str

    @model_validator(mode="after")
    def validate_model_result(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_worker_result_text(self.result)
        return self


class ToolObservationCount(WireModel):
    calls: int = Field(ge=0, le=MAX_UINT64)
    failures: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.failures > self.calls:
            raise ValueError("Worker observation failures exceed calls")
        return self


class WorkspaceObservationSummary(WireModel):
    scoped_files: int = Field(ge=0, le=MAX_UINT64)
    scope_complete: bool
    discovered_files: int = Field(ge=0, le=MAX_UINT64)
    read_files: int = Field(ge=0, le=MAX_UINT64)
    matched_files: int = Field(ge=0, le=MAX_UINT64)
    modified_files: int = Field(ge=0, le=MAX_UINT64)
    detail_complete: bool
    unread_files: int | None = Field(default=None, ge=0, le=MAX_UINT64)
    files_read: list[str] = Field(max_length=MAX_WORKER_FILES_READ)
    files_read_truncated: bool

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        seen: set[str] = set()
        for path in self.files_read:
            if not path:
                raise ValueError("Worker observed workspace path must not be empty")
            _require_workspace_target(path)
            if path in seen:
                raise ValueError("Worker filesRead paths must be unique")
            seen.add(path)
        if len(self.files_read) > self.read_files:
            raise ValueError("Worker filesRead detail exceeds readFiles")
        if self.files_read_truncated != (len(self.files_read) < self.read_files):
            raise ValueError("Worker filesRead truncation is inconsistent")
        coverage_complete = self.scope_complete and self.detail_complete
        if coverage_complete != (self.unread_files is not None):
            raise ValueError("Worker unreadFiles completeness is inconsistent")
        if self.unread_files is not None and self.unread_files > self.scoped_files:
            raise ValueError("Worker unreadFiles exceeds scopedFiles")
        return self


class WorkerObservations(WireModel):
    profile: Literal["lean@1"]
    tools: dict[str, ToolObservationCount]
    workspace: WorkspaceObservationSummary | None = None
    truncated: bool

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if len(self.tools) > MAX_WORKER_OBSERVATION_TOOLS:
            raise ValueError("Worker observation tools exceed their bound")
        for name in self.tools:
            if len(name) > 128 or ID_PATTERN.fullmatch(name) is None:
                raise ValueError("Worker observation tool name is invalid")
        if (
            self.workspace is not None
            and (
                not self.workspace.scope_complete
                or not self.workspace.detail_complete
                or self.workspace.files_read_truncated
            )
            and not self.truncated
        ):
            raise ValueError("Worker observation truncation is inconsistent")
        return self


class WorkerResult(WireModel):
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    result: str
    observations: WorkerObservations
    artifacts: dict[str, ArtifactRef]
    summarized: bool

    @model_validator(mode="after")
    def validate_worker_result(self) -> Self:
        _require_worker_subtask_id(self.subtask_id)
        _require_worker_result_text(self.result)
        if len(self.artifacts) > MAX_WORKER_RESULT_ARTIFACTS:
            raise ValueError("Worker result artifacts exceed their bound")
        for slot, artifact in self.artifacts.items():
            _require_text("Worker result artifact slot", slot)
            artifact.require_exact()
            if _reserved_worker_result_binding(artifact):
                raise ValueError("Worker result artifact identifies a reserved binding")
        return self


class WorkerFailure(WireModel):
    code: str = Field(pattern=WORKER_FAILURE_CODE_PATTERN.pattern)
    message: str
    retryable: bool

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        if WORKER_FAILURE_CODE_PATTERN.fullmatch(self.code) is None:
            raise ValueError("Worker failure code is invalid")
        if (
            not self.message.strip()
            or len(self.message.encode("utf-8")) > MAX_WORKER_FAILURE_MESSAGE_BYTES
        ):
            raise ValueError("Worker failure message must contain 1..4096 UTF-8 bytes")
        return self


class WorkerCompletion(VersionedWireModel):
    result: WorkerResult | None = None
    failure: WorkerFailure | None = None
    invocation_id: str
    state_revision: int = Field(gt=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        if (self.result is None) == (self.failure is None):
            raise ValueError("WorkerCompletion requires exactly one result or failure")
        if (
            self.invocation_id != self.invocation_id.strip()
            or not 1 <= len(self.invocation_id.encode("utf-8")) <= 128
            or any(character in self.invocation_id for character in "\r\n\t\x00")
        ):
            raise ValueError("Worker invocationId is invalid")
        if (
            len(self.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8"))
            > MAX_WORKER_COMPLETION_BYTES
        ):
            raise ValueError("WorkerCompletion exceeds its bounded contract")
        return self


class WorkerStateInvocationToolMetrics(WireModel):
    calls: int = Field(ge=0, le=MAX_UINT64)
    failures: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.failures > self.calls:
            raise ValueError("Worker State tool failures exceed calls")
        return self


class WorkerStateExecutionError(WireModel):
    code: str
    message: str = Field(max_length=4096)
    retryable: bool | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_error(self) -> Self:
        _require_text("code", self.code)
        _require_text("message", self.message)
        return self


class WorkerStateToolCall(WireModel):
    """Strict live-State form; unlike final-report records every flag is explicit."""

    call_id: str
    tool: str
    arguments: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    arguments_truncated: bool
    outcome: ToolCallOutcome
    duration_ms: int | None = Field(
        default=None, ge=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    result_size_bytes: int | None = Field(
        default=None, ge=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    error: WorkerStateExecutionError | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_call(self) -> Self:
        _require_text("callId", self.call_id)
        _require_text("tool", self.tool)
        if self.arguments is None and self.arguments_truncated:
            raise ValueError("absent Worker State arguments cannot be marked truncated")
        if (self.outcome is ToolCallOutcome.SUCCEEDED) == (self.error is not None):
            raise ValueError("Worker State tool outcome and error are inconsistent")
        return self


class WorkerStateBudget(WireModel):
    max_model_calls: int = Field(gt=0, le=1000)
    max_tool_calls: int = Field(gt=0, le=10_000)
    max_total_tokens: int = Field(gt=0, le=100_000_000)
    observed_model_calls: int = Field(ge=0, le=MAX_UINT64)
    observed_tool_calls: int = Field(ge=0, le=MAX_UINT64)
    observed_total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    exhausted: Literal["model_calls", "tool_calls", "total_tokens"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_observations(self) -> Self:
        if self.observed_model_calls > self.max_model_calls:
            raise ValueError("observedModelCalls exceeds maxModelCalls")
        if self.observed_tool_calls > self.max_tool_calls:
            raise ValueError("observedToolCalls exceeds maxToolCalls")
        if self.exhausted == "model_calls" and self.observed_model_calls != self.max_model_calls:
            raise ValueError("model-call exhaustion is inconsistent")
        if self.exhausted == "tool_calls" and self.observed_tool_calls != self.max_tool_calls:
            raise ValueError("tool-call exhaustion is inconsistent")
        if self.exhausted == "total_tokens" and self.observed_total_tokens < self.max_total_tokens:
            raise ValueError("token exhaustion is inconsistent")
        return self


class WorkerStateInvocationMetrics(WireModel):
    model_calls: int = Field(ge=0, le=MAX_UINT64)
    model_errors: int = Field(ge=0, le=MAX_UINT64)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    cached_input_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=MAX_UINT64)
    latest_prompt_tokens: int | None = Field(ge=0, le=MAX_UINT64)
    tool_calls: int = Field(ge=0, le=MAX_UINT64)
    tool_errors: int = Field(ge=0, le=MAX_UINT64)
    tools: dict[str, WorkerStateInvocationToolMetrics] = Field(max_length=256)
    truncated: bool

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.model_errors > self.model_calls or self.tool_errors > self.tool_calls:
            raise ValueError("Worker State invocation error counts exceed calls")
        for name in self.tools:
            if ID_PATTERN.fullmatch(name) is None or len(name) > 64:
                raise ValueError("Worker State invocation tool name is invalid")
        detailed_calls = sum(item.calls for item in self.tools.values())
        detailed_failures = sum(item.failures for item in self.tools.values())
        if detailed_calls > self.tool_calls or detailed_failures > self.tool_errors:
            raise ValueError("Worker State invocation tool detail exceeds aggregate")
        if not self.truncated and (
            detailed_calls != self.tool_calls or detailed_failures != self.tool_errors
        ):
            raise ValueError("complete Worker State invocation tool detail is inconsistent")
        return self


class WorkerStateSummarizer(WireModel):
    phase: Literal["disabled", "not_requested", "requested", "succeeded", "failed"]
    request_state_revision: int | None = Field(
        default=None, gt=0, le=MAX_UINT64, exclude_if=lambda value: value is None
    )
    model_calls: int = Field(ge=0, le=1)
    input_tokens: int = Field(ge=0, le=MAX_UINT64)
    output_tokens: int = Field(ge=0, le=MAX_UINT64)
    total_tokens: int = Field(ge=0, le=MAX_UINT64)
    token_usage_unavailable: int = Field(ge=0, le=1)
    failure_code: str | None = Field(
        default=None,
        pattern=WORKER_FAILURE_CODE_PATTERN.pattern,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        requested = self.phase in {"requested", "succeeded", "failed"}
        if requested != (self.request_state_revision is not None):
            raise ValueError("Worker summarizer request revision is inconsistent")
        if self.phase in {"disabled", "not_requested", "requested"} and any(
            (
                self.model_calls,
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.token_usage_unavailable,
            )
        ):
            raise ValueError("inactive Worker summarizer has usage")
        if self.phase == "succeeded" and self.model_calls != 1:
            raise ValueError("successful Worker summarizer requires one model call")
        if self.token_usage_unavailable > self.model_calls:
            raise ValueError("Worker summarizer missing usage exceeds model calls")
        if (self.phase == "failed") != (self.failure_code is not None):
            raise ValueError("Worker summarizer failure code is inconsistent")
        return self


def _require_state_workspace_path(value: str) -> str:
    if (
        not value
        or value != unicodedata.normalize("NFC", value)
        or len(value.encode("utf-8")) > 4096
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or "://" in value
    ):
        raise ValueError("Worker State workspace path is invalid")
    parts = value.split("/")
    if len(parts) > 128:
        raise ValueError("Worker State workspace path is invalid")
    for part in parts:
        if part in {"", ".", ".."} or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in part
        ):
            raise ValueError("Worker State workspace path is invalid")
    if (
        len(parts[0]) >= 2
        and parts[0][0].isascii()
        and parts[0][0].isalpha()
        and parts[0][1] == ":"
    ):
        raise ValueError("Worker State workspace path is invalid")
    return value


def _encoded_state_path_list_size(paths: list[str]) -> int:
    return len(json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class WorkerStateWorkspaceInteraction(WireModel):
    path: str
    first_ordinal: int = Field(gt=0, le=MAX_UINT64)
    last_ordinal: int = Field(gt=0, le=MAX_UINT64)
    discovery_calls: int = Field(ge=0, le=MAX_UINT64)
    read_calls: int = Field(ge=0, le=MAX_UINT64)
    match_calls: int = Field(ge=0, le=MAX_UINT64)
    mutation_calls: int = Field(ge=0, le=MAX_UINT64)

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        _require_state_workspace_path(self.path)
        if self.last_ordinal < self.first_ordinal or not any(
            (
                self.discovery_calls,
                self.read_calls,
                self.match_calls,
                self.mutation_calls,
            )
        ):
            raise ValueError("Worker State workspace interaction is inconsistent")
        return self


class WorkerStateWorkspaceObservation(WireModel):
    workspace_digest: str = Field(pattern=DIGEST_PATTERN.pattern)
    scope_paths: list[str] = Field(max_length=MAX_STATE_WORKSPACE_PATHS)
    scope_complete: bool
    interactions: list[WorkerStateWorkspaceInteraction] = Field(
        max_length=MAX_STATE_WORKSPACE_PATHS
    )
    detail_complete: bool

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        scope = [_require_state_workspace_path(path) for path in self.scope_paths]
        if scope != sorted(set(scope)) or _encoded_state_path_list_size(scope) > (
            MAX_STATE_WORKSPACE_PATH_BYTES
        ):
            raise ValueError("Worker State workspace scope is invalid")
        interaction_order = [(item.first_ordinal, item.path) for item in self.interactions]
        if interaction_order != sorted(interaction_order) or len(
            {item.path for item in self.interactions}
        ) != len(self.interactions):
            raise ValueError("Worker State workspace interactions are invalid")
        if (
            _encoded_state_path_list_size([item.path for item in self.interactions])
            > MAX_STATE_WORKSPACE_PATH_BYTES
        ):
            raise ValueError("Worker State workspace interaction paths exceed their bound")
        return self


class WorkerInvocationState(WireModel):
    invocation_id: str
    subtask_id: str = Field(pattern=WORKER_SUBTASK_ID_PATTERN.pattern)
    phase: Literal["running", "succeeded", "failed", "cancelled"]
    metrics: WorkerStateInvocationMetrics
    summarizer: WorkerStateSummarizer
    workspace: WorkerStateWorkspaceObservation | None

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        if (
            self.invocation_id != self.invocation_id.strip()
            or not 1 <= len(self.invocation_id.encode("utf-8")) <= 128
            or any(character in self.invocation_id for character in "\r\n\t\x00")
        ):
            raise ValueError("Worker State invocationId is invalid")
        _require_worker_subtask_id(self.subtask_id)
        if self.phase != "running" and self.summarizer.phase == "requested":
            raise ValueError("terminal Worker invocation has pending summarization")
        return self


class WorkerAllocationMetricsState(WireModel):
    counters: dict[str, int] = Field(max_length=10_000)
    tool_calls: list[WorkerStateToolCall] = Field(max_length=1000)
    errors: list[WorkerStateExecutionError] = Field(max_length=100)
    final_outcome: str | None
    truncated: bool
    worker_budget: WorkerStateBudget | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        for name, count in self.counters.items():
            if (
                not 1 <= len(name) <= 128
                or _STATE_METRIC_NAME_PATTERN.fullmatch(name) is None
                or type(count) is not int
                or not 0 <= count <= MAX_UINT64
            ):
                raise ValueError("Worker State allocation counter is invalid")
        if self.final_outcome is not None and (
            len(self.final_outcome) > 64 or ID_PATTERN.fullmatch(self.final_outcome) is None
        ):
            raise ValueError("Worker State final outcome is invalid")
        for call in self.tool_calls:
            if call.arguments is not None:
                try:
                    encoded = json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                except (TypeError, ValueError) as error:
                    raise ValueError("Worker State tool arguments are invalid") from error
                if len(encoded) > 4096:
                    raise ValueError("Worker State tool arguments exceed their bound")
            if call.error is not None and len(call.error.message.encode("utf-8")) > 4096:
                raise ValueError("Worker State tool error exceeds its bound")
        for error in self.errors:
            if len(error.message.encode("utf-8")) > 4096:
                raise ValueError("Worker State error exceeds its bound")
        return self


class ContractorWorkerState(WireModel):
    schema_version: Literal[2]
    state_revision: int = Field(gt=0, le=MAX_UINT64)
    metrics: WorkerAllocationMetricsState
    current_invocation: WorkerInvocationState | None
    last_completed_invocation: WorkerInvocationState | None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if self.current_invocation is not None and self.current_invocation.phase != "running":
            raise ValueError("current Worker invocation must be running")
        if self.last_completed_invocation is not None and (
            self.last_completed_invocation.phase == "running"
        ):
            raise ValueError("last completed Worker invocation must be terminal")
        return self


class AgentStateSnapshot(VersionedWireModel):
    state: ContractorWorkerState

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if len(self.model_dump_json(by_alias=True).encode("utf-8")) > (
            MAX_AGENT_STATE_SNAPSHOT_BYTES
        ):
            raise ValueError("AgentStateSnapshot exceeds its bounded contract")
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
    if state is AgentObservedState.FENCED and allocation_id is None:
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


# Private Runtime protocol v2 is intentionally separate from the active v1
# DTOs above. V8-004 switches both peers atomically after durable principal
# state exists; importing these models alone cannot activate v2 behavior.
PRIVATE_PROTOCOL_VERSION_V2 = 2
RUNTIME_ADAPTER_REFS = frozenset({"caido-graphql@1", "http-proxy@1", "otlp-http@1"})
RUNTIME_CREDENTIAL_KINDS = frozenset(
    {"caido-bearer@1", "http-proxy-basic@1", "http-proxy-bearer@1", "otlp-headers@1"}
)
PROXY_TARGETS = frozenset({"llm-gateway", "tool-http", "tool-subprocess"})
_RUNTIME_AGENT_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_RUNTIME_ADAPTER_ERROR_CODES = frozenset(
    {
        "close_failed",
        "delivery_failed",
        "flush_failed",
        "flush_timeout",
        "queue_overflow",
        "request_failed",
    }
)
_CERTIFICATE_PATTERN = re.compile(
    r"-----BEGIN CERTIFICATE-----\s+.+?\s+-----END CERTIFICATE-----", re.DOTALL
)
_FORBIDDEN_RUNTIME_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class PrivateProtocolDecodeError(ValueError):
    """Bounded private-wire failure whose rendering never includes input."""

    def __init__(self, reason: Literal["version", "duplicate_key", "schema", "invariant"]):
        self.reason = reason
        super().__init__(f"private protocol v2 {reason} error")


def _require_runtime_adapter_ref(value: str) -> str:
    if value not in RUNTIME_ADAPTER_REFS:
        raise ValueError("unknown RuntimeAdapter ref")
    return value


RuntimeAdapterRef = Annotated[str, AfterValidator(_require_runtime_adapter_ref)]
RuntimeCredentialKind = Literal[
    "caido-bearer@1",
    "http-proxy-basic@1",
    "http-proxy-bearer@1",
    "otlp-headers@1",
]
HTTPProxyTarget = Literal["llm-gateway", "tool-http", "tool-subprocess"]
WorkspaceModeV2 = Literal["direct", "overlay"]
WorkspaceStorageV2 = Literal["local", "memory"]


def _require_sorted_unique(field: str, values: list[str], *, maximum: int) -> None:
    if len(values) > maximum or values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted, unique, and contain at most {maximum} items")


def _require_runtime_label(field: str, value: str) -> str:
    if (
        len(value.encode("ascii", errors="ignore")) != len(value)
        or not 1 <= len(value) <= 63
        or ID_PATTERN.fullmatch(value) is None
        or value == "default"
    ):
        raise ValueError(f"{field} contains an invalid Runtime label")
    return value


def _require_runtime_endpoint(field: str, value: str) -> str:
    parsed = urlsplit(value)
    if (
        value != value.strip()
        or not 1 <= len(value.encode("utf-8")) <= 2048
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
    ):
        raise ValueError(
            f"{field} must be a bounded absolute HTTP(S) URL without userinfo, query, or fragment"
        )
    return value


def _require_workspace_target(value: str) -> str:
    if value == "":
        return value
    if (
        value != unicodedata.normalize("NFC", value)
        or len(value.encode("utf-8")) > 1024
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
        or "://" in value
    ):
        raise ValueError("workspace target is invalid")
    parts = value.split("/")
    if len(parts) > 32:
        raise ValueError("workspace target is invalid")
    for part in parts:
        if part in {"", ".", ".."} or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in part
        ):
            raise ValueError("workspace target is invalid")
    if len(parts[0]) >= 2 and parts[0][0].isalpha() and parts[0][1] == ":":
        raise ValueError("workspace target is invalid")
    return value


class WorkspaceLimitsV2(WireModel):
    max_files: int = Field(gt=0)
    max_expanded_bytes: int = Field(gt=0)
    max_managed_text_bytes: int = Field(gt=0)
    max_file_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if (
            self.max_file_bytes > self.max_expanded_bytes
            or self.max_managed_text_bytes > self.max_expanded_bytes
        ):
            raise ValueError("workspace limits are invalid")
        return self


class WorkspaceCapabilitiesV2(WireModel):
    storage: WorkspaceStorageV2
    modes: list[WorkspaceModeV2] = Field(min_length=1, max_length=2)
    limits: WorkspaceLimitsV2

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        _require_sorted_unique("workspace capability modes", self.modes, maximum=2)
        return self


class AllocationWorkspaceSourceV2(WireModel):
    artifact: ArtifactRef
    target: str

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        self.artifact.require_exact()
        _require_workspace_target(self.target)
        return self


class AllocationWorkspaceStateV2(WireModel):
    artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        self.artifact.require_exact()
        return self


class AllocationWorkspaceExportV2(WireModel):
    state: str = Field(pattern=ID_PATTERN.pattern)
    diff: str = Field(pattern=ID_PATTERN.pattern)

    @model_validator(mode="after")
    def validate_export(self) -> Self:
        if self.state == self.diff:
            raise ValueError("workspace export slots must be distinct")
        return self


class AllocationWorkspaceSpecV2(WireModel):
    mode: WorkspaceModeV2
    sources: list[AllocationWorkspaceSourceV2] = Field(min_length=1, max_length=32)
    state: AllocationWorkspaceStateV2 | None = None
    export: AllocationWorkspaceExportV2 | None = None

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        targets = [source.target for source in self.sources]
        for index, target in enumerate(targets):
            for other_index, other in enumerate(targets):
                if index == other_index:
                    continue
                if target == other or target == "" or other.startswith(f"{target}/"):
                    raise ValueError("workspace source targets must be unique and non-overlapping")
        if self.export is not None and self.mode != "overlay":
            raise ValueError("workspace export requires overlay mode")
        return self


class AgentRegistrationV2(AgentRegistration):
    private_protocol_version: Literal[PRIVATE_PROTOCOL_VERSION_V2]
    initial_labels: list[str] = Field(max_length=32)
    supported_runtime_adapters: list[RuntimeAdapterRef] = Field(max_length=64)
    workspace_capabilities: WorkspaceCapabilitiesV2 | None = None

    @model_validator(mode="after")
    def validate_v2_registration(self) -> Self:
        if len(self.supported_runtimes) > 128 or len(self.supported_toolsets) > 128:
            raise ValueError("Runtime capability collection exceeds its bound")
        if len(self.supported_sandbox_profiles) > 128:
            raise ValueError("Runtime capability collection exceeds its bound")
        for label in self.initial_labels:
            _require_runtime_label("initialLabels", label)
        _require_sorted_unique("initialLabels", self.initial_labels, maximum=32)
        _require_sorted_unique(
            "supportedRuntimeAdapters", self.supported_runtime_adapters, maximum=64
        )
        return self


class AgentRegistrationResponseV2(AgentRegistrationResponse):
    private_protocol_version: Literal[PRIVATE_PROTOCOL_VERSION_V2]
    runtime_agent_id: str = Field(pattern=_RUNTIME_AGENT_ID_PATTERN.pattern)
    labels: list[str] = Field(max_length=32)
    label_revision: int = Field(gt=0, le=2**64 - 1)

    @model_validator(mode="after")
    def validate_v2_response(self) -> Self:
        for label in self.labels:
            _require_runtime_label("labels", label)
        _require_sorted_unique("labels", self.labels, maximum=32)
        return self


class TelemetrySettingsV2(WireModel):
    adapter: RuntimeAdapterRef
    endpoint: str
    headers: dict[str, SecretStr]
    capture_content: bool
    flush_timeout_seconds: int = Field(ge=1, le=10)

    @model_validator(mode="after")
    def validate_telemetry(self) -> Self:
        if self.adapter != "otlp-http@1":
            raise ValueError("telemetry adapter must be otlp-http@1")
        _require_runtime_endpoint("telemetry.endpoint", self.endpoint)
        if len(self.headers) > 32:
            raise ValueError("telemetry headers exceed 32 entries")
        total = 0
        for name, wrapped in self.headers.items():
            value = wrapped.get_secret_value()
            if (
                not 1 <= len(name) <= 64
                or _HEADER_NAME_PATTERN.fullmatch(name) is None
                or name.lower() in _FORBIDDEN_RUNTIME_HEADERS
                or "\r" in name
                or "\n" in name
            ):
                raise ValueError("telemetry header name is invalid")
            if not 1 <= len(value.encode("utf-8")) <= 4096 or "\r" in value or "\n" in value:
                raise ValueError("telemetry header value is invalid")
            total += len(value.encode("utf-8"))
        if total > 16 * 1024:
            raise ValueError("telemetry header values exceed 16 KiB")
        if self.capture_content:
            raise ValueError("telemetry captureContent must be false")
        return self

    @field_serializer("headers", when_used="json")
    def serialize_headers(self, value: dict[str, SecretStr]) -> dict[str, str]:
        return {name: secret.get_secret_value() for name, secret in value.items()}


class HTTPProxyBasicAuthV2(WireModel):
    username: SecretStr
    password: SecretStr

    @model_validator(mode="after")
    def validate_auth(self) -> Self:
        username = self.username.get_secret_value()
        password = self.password.get_secret_value()
        if not 1 <= len(username.encode("utf-8")) <= 256:
            raise ValueError("HTTP proxy username is outside its size bound")
        if not 1 <= len(password.encode("utf-8")) <= 8192:
            raise ValueError("HTTP proxy password is outside its size bound")
        return self

    @field_serializer("username", "password", when_used="json")
    def serialize_secret(self, value: SecretStr) -> str:
        return value.get_secret_value()


class HTTPProxySettingsV2(WireModel):
    adapter: RuntimeAdapterRef
    proxy_url: str
    basic_auth: HTTPProxyBasicAuthV2 | None = None
    bearer_token: SecretStr | None = None
    ca_bundle_pem: str | None = None
    targets: list[HTTPProxyTarget] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_proxy(self) -> Self:
        if self.adapter != "http-proxy@1":
            raise ValueError("HTTP proxy adapter must be http-proxy@1")
        _require_runtime_endpoint("httpProxy.proxyUrl", self.proxy_url)
        if self.basic_auth is not None and self.bearer_token is not None:
            raise ValueError("HTTP proxy basicAuth and bearerToken are mutually exclusive")
        if self.bearer_token is not None:
            token = self.bearer_token.get_secret_value()
            if not 1 <= len(token.encode("utf-8")) <= 8192:
                raise ValueError("HTTP proxy bearerToken is outside its size bound")
        if self.ca_bundle_pem is not None:
            _validate_ca_bundle("HTTP proxy", self.ca_bundle_pem)
        _require_sorted_unique("httpProxy.targets", self.targets, maximum=3)
        return self

    @field_serializer("bearer_token", when_used="json")
    def serialize_bearer(self, value: SecretStr | None) -> str | None:
        return None if value is None else value.get_secret_value()


class CaidoSettingsV2(WireModel):
    adapter: RuntimeAdapterRef
    endpoint: str
    bearer_token: SecretStr | None = None
    ca_bundle_pem: str | None = None
    request_timeout_seconds: int

    @model_validator(mode="after")
    def validate_caido(self) -> Self:
        if self.adapter != "caido-graphql@1":
            raise ValueError("Caido adapter must be caido-graphql@1")
        _require_runtime_endpoint("caido.endpoint", self.endpoint)
        if self.bearer_token is not None:
            token = self.bearer_token.get_secret_value()
            if not 1 <= len(token.encode("utf-8")) <= 8192:
                raise ValueError("Caido bearerToken is outside its size bound")
        if self.ca_bundle_pem is not None:
            _validate_ca_bundle("Caido", self.ca_bundle_pem)
        if not 1 <= self.request_timeout_seconds <= 120:
            raise ValueError("Caido requestTimeoutSeconds must be from 1 through 120")
        return self

    @field_serializer("bearer_token", when_used="json")
    def serialize_bearer(self, value: SecretStr | None) -> str | None:
        return None if value is None else value.get_secret_value()


class RuntimeSettingsV2(WireModel):
    llm_gateway_url: str
    llm_gateway_token: SecretStr | None = None
    artifact_api_url: str
    telemetry: TelemetrySettingsV2 | None = None
    http_proxy: HTTPProxySettingsV2 | None = None
    caido: CaidoSettingsV2 | None = None
    request_timeout_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        _require_runtime_endpoint("runtimeSettings.llmGatewayUrl", self.llm_gateway_url)
        if len(self.artifact_api_url.encode("utf-8")) > 2048:
            raise ValueError("runtimeSettings.artifactApiUrl exceeds 2048 bytes")
        _require_url("runtimeSettings.artifactApiUrl", self.artifact_api_url)
        return self

    @field_serializer("llm_gateway_token", when_used="json")
    def serialize_token(self, value: SecretStr | None) -> str | None:
        return None if value is None else value.get_secret_value()


class RuntimeConfigRefV2(WireModel):
    name: str
    version: str
    digest: str

    @model_validator(mode="after")
    def validate_ref(self) -> Self:
        if not 1 <= len(self.name) <= 63:
            raise ValueError("RuntimeConfig ref name is invalid")
        _require_selector("RuntimeConfig ref", f"{self.name}@{self.version}")
        if len(self.version) > 128:
            raise ValueError("RuntimeConfig ref version is invalid")
        _require_digest("RuntimeConfig ref digest", self.digest)
        return self


class RuntimeLabelBindingProvenanceV2(WireModel):
    label: str
    binding_revision: int = Field(gt=0, le=2**64 - 1)
    config: RuntimeConfigRefV2


class RuntimeCredentialRefV2(WireModel):
    credential_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN.pattern)
    kind: RuntimeCredentialKind


class LLMCredentialRefV2(WireModel):
    credential_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN.pattern)


class ResolvedRuntimeConfigProvenanceV2(WireModel):
    default: RuntimeLabelBindingProvenanceV2
    run_labels: list[RuntimeLabelBindingProvenanceV2] = Field(max_length=32)
    agent_labels: list[RuntimeLabelBindingProvenanceV2] = Field(max_length=32)
    runtime_adapters: list[RuntimeAdapterRef] = Field(max_length=64)
    llm_gateway_config: LLMGatewayConfigRef | None = None
    llm_credential: LLMCredentialRefV2 | None = None
    runtime_credential_refs: list[RuntimeCredentialRefV2] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if self.default.label != "default":
            raise ValueError("provenance default binding is invalid")
        for field, values in (
            ("runLabels", self.run_labels),
            ("agentLabels", self.agent_labels),
        ):
            for value in values:
                _require_runtime_label(field, value.label)
            _require_sorted_unique(field, [value.label for value in values], maximum=32)
        _require_sorted_unique("runtimeAdapters", self.runtime_adapters, maximum=64)
        if self.llm_credential is not None and self.llm_gateway_config is None:
            raise ValueError("LLM credential provenance requires a Gateway config ref")
        credential_keys = [
            f"{value.kind}\0{value.credential_id}" for value in self.runtime_credential_refs
        ]
        _require_sorted_unique("runtimeCredentialRefs", credential_keys, maximum=64)
        return self


class AllocationSpecV2(AllocationSpec):
    runtime_settings: RuntimeSettingsV2
    resolved_runtime_config_provenance: ResolvedRuntimeConfigProvenanceV2
    workspace: AllocationWorkspaceSpecV2 | None = None


class PrepareAllocationRequestV2(VersionedWireModel):
    spec: AllocationSpecV2


class RuntimeAdapterMetricsV2(WireModel):
    operations: int = Field(ge=0, le=2**64 - 1)
    failed_operations: int = Field(ge=0, le=2**64 - 1)
    flush_attempted: bool | None = None
    flush_succeeded: bool | None = None
    last_error_code: str | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.failed_operations > self.operations:
            raise ValueError("Runtime adapter failures exceed operations")
        if (self.flush_attempted is None) != (self.flush_succeeded is None):
            raise ValueError("Runtime adapter flush fields must be present together")
        if self.flush_attempted is False and self.flush_succeeded:
            raise ValueError("Runtime adapter flush cannot succeed when not attempted")
        if (
            self.last_error_code is not None
            and self.last_error_code not in _RUNTIME_ADAPTER_ERROR_CODES
        ):
            raise ValueError("Runtime adapter lastErrorCode is invalid")
        return self


class RuntimeReportV2(RuntimeReport):
    adapters: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]

    @field_validator("adapters")
    @classmethod
    def validate_adapters(
        cls, value: dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]
    ) -> dict[RuntimeAdapterRef, RuntimeAdapterMetricsV2]:
        if len(value) > 64:
            raise ValueError("Runtime adapter metrics exceed 64 entries")
        return value


def decode_private_v2[PrivateModelT: WireModel](
    model: type[PrivateModelT], raw: str | bytes
) -> PrivateModelT:
    """Decode one secret-bearing v2 document without reflecting input in errors."""

    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except _DuplicateJSONKey:
        raise PrivateProtocolDecodeError("duplicate_key") from None
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        raise PrivateProtocolDecodeError("schema") from None
    if not isinstance(value, dict):
        raise PrivateProtocolDecodeError("schema")
    if (
        issubclass(model, (AgentRegistrationV2, AgentRegistrationResponseV2))
        and value.get("privateProtocolVersion") != PRIVATE_PROTOCOL_VERSION_V2
    ):
        raise PrivateProtocolDecodeError("version")
    try:
        # Keep Pydantic's strict JSON conversions (notably RFC 3339 strings to
        # aware datetimes) after the duplicate-key pre-scan above.
        return model.model_validate_json(raw)
    except ValidationError as error:
        reason: Literal["schema", "invariant"] = "invariant"
        if any(item["type"] != "value_error" for item in error.errors(include_input=False)):
            reason = "schema"
        raise PrivateProtocolDecodeError(reason) from None


def encode_private_v2(value: WireModel) -> bytes:
    """Return RFC 8785 canonical private JSON; callers must not log it."""

    dumped = value.model_dump(mode="json", by_alias=True, exclude_none=True)
    return jcs.canonicalize(dumped)


class _DuplicateJSONKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey
        result[key] = value
    return result


def _validate_ca_bundle(owner: str, value: str) -> None:
    if not 1 <= len(value.encode("utf-8")) <= 64 * 1024 or "PRIVATE KEY" in value:
        raise ValueError(f"{owner} CA bundle is invalid")
    certificates = _CERTIFICATE_PATTERN.findall(value)
    remainder = _CERTIFICATE_PATTERN.sub("", value)
    if not 1 <= len(certificates) <= 8 or remainder.strip():
        raise ValueError(f"{owner} CA bundle is invalid")
    try:
        for certificate in certificates:
            ssl.PEM_cert_to_DER_cert(certificate)
    except ValueError:
        raise ValueError(f"{owner} CA bundle is invalid") from None
