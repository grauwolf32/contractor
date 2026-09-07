"""Private Runtime protocol settings models and validation."""

from __future__ import annotations

import math
import ssl
from typing import Any, Literal, Self

from pydantic import (
    Field,
    SecretStr,
    field_serializer,
    field_validator,
    model_validator,
)

from contractor_runtime.contracts.artifacts import ArtifactRef
from contractor_runtime.contracts.base import (
    _CERTIFICATE_PATTERN,
    _FORBIDDEN_RUNTIME_HEADERS,
    _HEADER_NAME_PATTERN,
    ID_PATTERN,
    NATIVE_SKILL_TOOL_NAMES,
    SKILL_NAME_PATTERN,
    HTTPProxyTarget,
    RuntimeAdapterRef,
    RuntimeCredentialKind,
    WireModel,
    _require_digest,
    _require_inference_gateway_url,
    _require_management_gateway_origin,
    _require_runtime_endpoint,
    _require_runtime_label,
    _require_selector,
    _require_sorted_unique,
    _require_text,
    _require_url,
)


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


class LLMGatewayCredentialManager(WireModel):
    implementation: Literal["litellm-virtual-keys@1"]
    management_url: str

    @field_validator("management_url")
    @classmethod
    def validate_management_url(cls, value: str) -> str:
        return _require_management_gateway_origin(value)


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


class TelemetryExportSettings(WireModel):
    batch_size_bytes: int = Field(strict=True, ge=1024 * 1024, le=64 * 1024 * 1024)
    max_attempts: int = Field(strict=True, ge=1, le=10)
    max_pending_spans: int = Field(strict=True, ge=1, le=2048)
    max_pending_bytes: int = Field(strict=True, ge=1024 * 1024, le=64 * 1024 * 1024)

    @classmethod
    def defaults(cls) -> TelemetryExportSettings:
        return cls(
            batchSizeBytes=8 * 1024 * 1024,
            maxAttempts=2,
            maxPendingSpans=2048,
            maxPendingBytes=64 * 1024 * 1024,
        )

    @model_validator(mode="after")
    def validate_queue_capacity(self) -> Self:
        if self.max_pending_bytes < self.batch_size_bytes:
            raise ValueError("telemetry export maxPendingBytes must cover batchSizeBytes")
        return self


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


class RuntimeCredentialRefV2(WireModel):
    credential_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN.pattern)
    kind: RuntimeCredentialKind


class LLMCredentialRefV2(WireModel):
    credential_id: str = Field(min_length=1, max_length=128, pattern=ID_PATTERN.pattern)


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


class ResolvedLLMGatewayConfig(WireModel):
    ref: LLMGatewayConfigRef
    protocol: Literal["openai-compatible@1"]
    url: str
    credential_manager: LLMGatewayCredentialManager | None = None

    @field_validator("url")
    @classmethod
    def validate_inference_url(cls, value: str) -> str:
        return _require_inference_gateway_url(value)


class TelemetrySettingsV2(WireModel):
    adapter: RuntimeAdapterRef
    endpoint: str
    headers: dict[str, SecretStr]
    capture_content: bool
    flush_timeout_seconds: int = Field(ge=1, le=10)
    export: TelemetryExportSettings | None = None

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
        return self

    @field_serializer("headers", when_used="json")
    def serialize_headers(self, value: dict[str, SecretStr]) -> dict[str, str]:
        return {name: secret.get_secret_value() for name, secret in value.items()}


class HTTPOriginTargetSettingsV2(WireModel):
    url: str
    basic_auth: HTTPProxyBasicAuthV2 | None = None
    bearer_token: SecretStr | None = None

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        _require_runtime_endpoint("httpOriginTarget.url", self.url)
        if self.basic_auth is not None and self.bearer_token is not None:
            raise ValueError("HTTP origin target basicAuth and bearerToken are mutually exclusive")
        if self.basic_auth is not None and ":" in self.basic_auth.username.get_secret_value():
            raise ValueError("HTTP origin target username cannot contain a colon")
        if self.bearer_token is not None:
            token = self.bearer_token.get_secret_value()
            if not 1 <= len(token.encode("utf-8")) <= 8192:
                raise ValueError("HTTP origin target bearerToken is outside its size bound")
        return self

    @field_serializer("bearer_token", when_used="json")
    def serialize_bearer(self, value: SecretStr | None) -> str | None:
        return None if value is None else value.get_secret_value()


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


class RuntimeLabelBindingProvenanceV2(WireModel):
    label: str
    binding_revision: int = Field(gt=0, le=2**64 - 1)
    config: RuntimeConfigRefV2


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


class RuntimeSettingsV2(WireModel):
    llm_gateway_url: str
    llm_gateway_token: SecretStr | None = None
    artifact_api_url: str
    telemetry: TelemetrySettingsV2 | None = None
    http_proxy: HTTPProxySettingsV2 | None = None
    caido: CaidoSettingsV2 | None = None
    http_origin_target: HTTPOriginTargetSettingsV2 | None = None
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


class WorkerSummarizerConfig(WireModel):
    instructions: ResolvedInstructions | None = None
    model_policy: ResolvedModelPolicy
    cumulative_budget: int | None = Field(default=None, gt=0, le=100_000_000)
    context_window_ratio: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_summarizer(self) -> Self:
        if self.instructions is not None and len(self.instructions.text) > 8000:
            raise ValueError(
                "Worker summarizer instructions must contain at most 8000 Unicode characters"
            )
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
