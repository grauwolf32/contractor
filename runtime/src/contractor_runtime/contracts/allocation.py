"""Private Runtime protocol allocation models and validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from contractor_runtime.contracts.artifacts import ArtifactRef
from contractor_runtime.contracts.base import (
    ARTIFACT_NAME_PATTERN,
    NAME_PATTERN,
    TerminationError,
    VersionedWireModel,
    WireModel,
    WorkerSessionMode,
    _require_aware_datetime,
    _require_resource_id,
    _require_text,
    normalize_run_metadata_labels,
)
from contractor_runtime.contracts.engagement_allocation import EngagementAllocationProjection
from contractor_runtime.contracts.reports import PerformanceMetricsRequest
from contractor_runtime.contracts.settings import (
    AgentTemplateRef,
    CompactWorkerContextManagement,
    ResolvedAgentTemplate,
    ResolvedModelPolicy,
    ResolvedRuntimeConfigProvenance,
    ResolvedSkill,
    RuntimeSettings,
    WorkerContextManagement,
    WorkerRuntimeRef,
    _require_worker_compactor_policy,
    _require_worker_policy,
    _require_worker_summarizer_policy,
)
from contractor_runtime.contracts.workspace import AllocationWorkspaceSpec

# audit-check-results@1 collects through exactly these audit-results@2 tools.
AUDIT_COMPLETION_TOOLS = frozenset(
    {"read_audit_task", "submit_result", "submit_results", "evidence"}
)


class WorkerCompletionContract(WireModel):
    kind: Literal["audit-check-results@1"]
    context: ArtifactRef
    result_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_completion_refs(self) -> Self:
        self.context.require_exact()
        if (
            self.context.namespace != "inputs"
            or self.result_artifact.revision is not None
            or self.result_artifact.namespace == "inputs"
        ):
            raise ValueError("completion needs an exact Run context and a versionless output")
        return self

    def validate_allocation(
        self,
        namespace: str,
        template: ResolvedAgentTemplate,
        context_management: WorkerContextManagement,
    ) -> None:
        if self.result_artifact.namespace != namespace:
            raise ValueError("completion output is outside Worker namespace")
        selections = [
            selection
            for selection in template.toolsets
            if selection.ref.toolset_id == "audit-results"
        ]
        if (
            context_management.strategy == "terminal@1"
            or (template.summarizer is not None and context_management.strategy != "compact@1")
            or len(selections) != 1
            or selections[0].ref.version != "2"
            or sorted(selections[0].tools) != sorted(AUDIT_COMPLETION_TOOLS)
        ):
            raise ValueError(
                "Audit completion needs audit-results@2 with both tools "
                "and no terminal summarization"
            )


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


def _validate_lifecycle(
    allocation_id: str, id_field: str, id_value: str, deadline: datetime
) -> None:
    _require_text("allocationId", allocation_id)
    _require_text(id_field, id_value)
    _require_aware_datetime("deadline", deadline)


class ReleaseAllocationRequest(VersionedWireModel):
    allocation_id: str

    @field_validator("allocation_id")
    @classmethod
    def validate_allocation_id(cls, value: str) -> str:
        return _require_text("allocationId", value)


class AllocationSpec(VersionedWireModel):
    nullable_fields = frozenset({"workspace", "performance_metrics"})

    # Pydantic 2.13 silently ignores a disabled optional field-name alias.
    # Keeping this authority field's exact wire spelling makes other names unknown.
    engagement: EngagementAllocationProjection | None = None
    completion_contract: WorkerCompletionContract | None = None
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    worker_session_mode: WorkerSessionMode
    context_management: WorkerContextManagement
    run_metadata_labels: dict[str, str]
    project_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    lease_expires_at: datetime
    agent_template: ResolvedAgentTemplate
    resolved_skills: list[ResolvedSkill] = Field(max_length=32)
    model_policy: ResolvedModelPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    summarizer_model_policy: ResolvedModelPolicy | None = None
    compactor_model_policy: ResolvedModelPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    runtime_settings: RuntimeSettings
    resolved_runtime_config_provenance: ResolvedRuntimeConfigProvenance
    workspace: AllocationWorkspaceSpec | None = None
    performance_metrics: PerformanceMetricsRequest | None = None

    @field_validator("compactor_model_policy", mode="before")
    @classmethod
    def require_compactor_policy_when_present(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("compactorModelPolicy must not be null")
        return value

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("projectId must not be null")
        return _require_resource_id("projectId", value)

    @field_validator("run_metadata_labels")
    @classmethod
    def validate_run_metadata_labels(cls, value: dict[str, str]) -> dict[str, str]:
        return normalize_run_metadata_labels(value)

    @model_validator(mode="after")
    def validate_spec(self) -> Self:
        assignment = self.engagement.http if self.engagement else None
        materials = self.runtime_settings.http_identities or {}
        identities = assignment.identities if assignment else []
        expected = {identity.id for identity in identities if identity.credential is not None}
        if set(materials) != expected:
            raise ValueError("engagement private material differs from assignment")
        for identity in identities:
            if identity.credential is None:
                continue
            material = materials[identity.id]
            if material.credential != identity.credential:
                raise ValueError("engagement private credential differs from assignment")
        if self.completion_contract is not None:
            self.completion_contract.validate_allocation(
                self.namespace, self.agent_template, self.context_management
            )
        elif any(
            selection.ref.toolset_id == "audit-results" and selection.ref.version == "2"
            for selection in self.agent_template.toolsets
        ):
            raise ValueError("audit-results@2 requires a trusted completion contract")
        for field, value in (
            ("allocationId", self.allocation_id),
            ("runId", self.run_id),
            ("stageExecutionId", self.stage_execution_id),
            ("namespace", self.namespace),
        ):
            _require_text(field, value)
        if NAME_PATTERN.fullmatch(self.logical_agent_name) is None:
            raise ValueError("logicalAgentName must be a name of at most 128 characters")
        if ARTIFACT_NAME_PATTERN.fullmatch(self.namespace) is None:
            raise ValueError(
                "namespace must be a portable ASCII Artifact name of 1 through 128 characters"
            )
        _require_aware_datetime("leaseExpiresAt", self.lease_expires_at)
        names = [skill.name for skill in self.resolved_skills]
        if names != sorted(set(names)):
            raise ValueError("resolvedSkills must be sorted and unique")
        if names != [skill.name for skill in self.agent_template.skills]:
            raise ValueError("resolvedSkills must exactly match AgentTemplate skills")
        self._validate_context_management()
        if self.agent_template.is_tool_worker:
            if (
                "model_policy" in self.model_fields_set
                or "summarizer_model_policy" in self.model_fields_set
                or "compactor_model_policy" in self.model_fields_set
                or "completion_contract" in self.model_fields_set
                or "llm_gateway_url" in self.runtime_settings.model_fields_set
                or self.runtime_settings.llm_gateway_url is not None
                or self.runtime_settings.llm_gateway_token is not None
                or self.resolved_runtime_config_provenance.llm_gateway_config is not None
                or self.resolved_runtime_config_provenance.llm_credential is not None
                or self.workspace is not None
                or self.completion_contract is not None
            ):
                raise ValueError(
                    "tool@1 allocation forbids model access, workspace and completion contracts"
                )
            return self
        if self.model_policy is None or self.runtime_settings.llm_gateway_url is None:
            raise ValueError("modeled allocation requires modelPolicy and llmGatewayUrl")
        if (
            self.agent_template.runtime.runtime_id == "planner"
            and self.agent_template.runtime.version == "1"
        ):
            if (
                not self.model_policy.max_output_tokens
                or not self.model_policy.max_model_calls
                or not self.model_policy.max_worker_calls
                or not self.model_policy.max_total_tokens
                or self.model_policy.max_tool_calls is not None
                or self.agent_template.toolsets
                or self.agent_template.skills
                or self.workspace is not None
                or self.summarizer_model_policy is not None
                or self.compactor_model_policy is not None
                or self.completion_contract is not None
                or self.engagement is not None
                or self.runtime_settings.telemetry is not None
                or self.runtime_settings.http_proxy is not None
                or self.runtime_settings.caido is not None
                or self.runtime_settings.interactsh is not None
                or self.runtime_settings.http_origin_target is not None
                or self.runtime_settings.http_identities
                or self.resolved_runtime_config_provenance.agent_labels
                or self.resolved_runtime_config_provenance.runtime_adapters
                or self.resolved_runtime_config_provenance.runtime_credential_refs
            ):
                raise ValueError("Planner allocation configuration is invalid")
        else:
            _require_worker_policy(
                self.model_policy,
                has_tools=bool(
                    self.agent_template.skills
                    or any(selection.tools for selection in self.agent_template.toolsets)
                ),
            )
        if self.summarizer_model_policy is not None:
            _require_worker_summarizer_policy(self.summarizer_model_policy)
        if isinstance(self.context_management, CompactWorkerContextManagement):
            if (
                self.model_policy.context_window_tokens is None
                or self.model_policy.max_output_tokens is None
            ):
                raise ValueError(
                    "compact@1 Worker modelPolicy requires contextWindowTokens and maxOutputTokens"
                )
            return self
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

    def _validate_context_management(self) -> None:
        """Checks the one effective context branch (WRK-R184)."""
        strategy = self.context_management.strategy
        summarizer = self.agent_template.summarizer
        if strategy == "none" and summarizer is not None:
            raise ValueError("contextManagement none requires a template without summarizer")
        if strategy == "terminal@1" and (
            summarizer is None
            or self.agent_template.is_tool_worker
            or self.summarizer_model_policy is None
        ):
            raise ValueError(
                "contextManagement terminal@1 requires template summarizer behavior "
                "and summarizerModelPolicy"
            )
        if self.summarizer_model_policy is not None and strategy != "terminal@1":
            raise ValueError("summarizerModelPolicy requires terminal@1 context management")
        if strategy != "compact@1":
            if self.compactor_model_policy is not None:
                raise ValueError("compactorModelPolicy requires compact@1 context management")
            return
        if self.compactor_model_policy is None:
            raise ValueError("compact@1 requires compactorModelPolicy")
        _require_worker_compactor_policy(self.compactor_model_policy)
        if self.agent_template.is_tool_worker:
            raise ValueError("compact@1 requires a modeled ADK Worker")
        if self.worker_session_mode != "isolated":
            raise ValueError("compact@1 supports only isolated Worker sessions")


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


class AbortAllocationRequest(VersionedWireModel):
    allocation_id: str
    abort_id: str
    reason: TerminationError
    deadline: datetime

    @model_validator(mode="after")
    def validate_abort(self) -> Self:
        _validate_lifecycle(self.allocation_id, "abortId", self.abort_id, self.deadline)
        return self


class PrepareAllocationRequest(VersionedWireModel):
    spec: AllocationSpec
