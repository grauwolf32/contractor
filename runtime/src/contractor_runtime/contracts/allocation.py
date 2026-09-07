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
    TerminationError,
    VersionedWireModel,
    WireModel,
    WorkerSessionMode,
    _require_aware_datetime,
    _require_text,
    normalize_run_metadata_labels,
)
from contractor_runtime.contracts.reports import PerformanceMetricsRequest
from contractor_runtime.contracts.settings import (
    AgentTemplateRef,
    ResolvedAgentTemplate,
    ResolvedModelPolicy,
    ResolvedRuntimeConfigProvenanceV2,
    ResolvedSkill,
    RuntimeSettings,
    RuntimeSettingsV2,
    WorkerRuntimeRef,
    _require_worker_policy,
)
from contractor_runtime.contracts.workspace import AllocationWorkspaceSpecV2


class WorkerCompletionContract(WireModel):
    kind: Literal["audit-check-results@1"]
    task: ArtifactRef
    execution_manifest: ArtifactRef
    result_artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_completion_refs(self) -> Self:
        self.task.require_exact()
        self.execution_manifest.require_exact()
        if (
            self.task.namespace != "inputs"
            or self.execution_manifest.namespace != "inputs"
            or self.task.name == self.execution_manifest.name
            or self.result_artifact.revision is not None
            or self.result_artifact.namespace == "inputs"
        ):
            raise ValueError("completion needs distinct exact Run inputs and a versionless output")
        return self

    def validate_allocation(self, namespace: str, template: ResolvedAgentTemplate) -> None:
        if self.result_artifact.namespace != namespace:
            raise ValueError("completion output is outside Worker namespace")
        selections = [
            selection
            for selection in template.toolsets
            if selection.ref.toolset_id == "audit-results"
        ]
        if (
            template.summarizer is not None
            or len(selections) != 1
            or selections[0].ref.version != "2"
            or sorted(selections[0].tools) != ["read_audit_task", "submit_check_result"]
        ):
            raise ValueError(
                "Audit completion needs audit-results@2 with both tools and no summarizer"
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
    completion_contract: WorkerCompletionContract | None = None
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    worker_session_mode: WorkerSessionMode
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
        if self.completion_contract is not None:
            self.completion_contract.validate_allocation(self.namespace, self.agent_template)
        for field, value in (
            ("allocationId", self.allocation_id),
            ("runId", self.run_id),
            ("stageExecutionId", self.stage_execution_id),
            ("logicalAgentName", self.logical_agent_name),
            ("namespace", self.namespace),
        ):
            _require_text(field, value)
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


class AllocationSpecV2(AllocationSpec):
    runtime_settings: RuntimeSettingsV2
    resolved_runtime_config_provenance: ResolvedRuntimeConfigProvenanceV2
    workspace: AllocationWorkspaceSpecV2 | None = None
    performance_metrics: PerformanceMetricsRequest | None = None


class PrepareAllocationRequestV2(VersionedWireModel):
    spec: AllocationSpecV2
