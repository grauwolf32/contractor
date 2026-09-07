"""Private Runtime protocol registration models and validation."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    model_validator,
)

from contractor_runtime.contracts.base import (
    _RUNTIME_AGENT_ID_PATTERN,
    VERSION_PATTERN,
    AgentObservedState,
    ReconciliationAction,
    RuntimeAdapterRef,
    VersionedWireModel,
    WireModel,
    _require_aware_datetime,
    _require_runtime_label,
    _require_selector,
    _require_sorted_unique,
    _require_text,
    _require_url,
)
from contractor_runtime.contracts.workspace import WorkspaceCapabilities


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


def _validate_observed_allocation(state: AgentObservedState, allocation_id: str | None) -> None:
    if state is AgentObservedState.IDLE:
        if allocation_id is not None:
            raise ValueError("idle agent must not report allocationId")
        return
    if state is AgentObservedState.FENCED and allocation_id is None:
        return
    _require_text("allocationId", allocation_id or "")


class AgentRegistrationResponse(VersionedWireModel):
    heartbeat_interval_seconds: int = Field(gt=0)
    confirmed_lease_seconds: int = Field(gt=0)
    runtime_agent_id: str = Field(pattern=_RUNTIME_AGENT_ID_PATTERN.pattern)
    labels: list[str] = Field(max_length=32)
    label_revision: int = Field(gt=0, le=2**64 - 1)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.confirmed_lease_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("confirmed lease must exceed heartbeat interval")
        return self

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        for label in self.labels:
            _require_runtime_label("labels", label)
        _require_sorted_unique("labels", self.labels, maximum=32)
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


class RuntimeCompletionCapabilities(WireModel):
    completion_contracts: list[Literal["audit-check-results@1"]] = Field(max_length=1)


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
    capabilities: RuntimeCompletionCapabilities | None = None
    initial_labels: list[str] = Field(max_length=32)
    supported_runtime_adapters: list[RuntimeAdapterRef] = Field(max_length=64)
    workspace_capabilities: WorkspaceCapabilities | None = None
    supported_performance_metrics_versions: list[Annotated[int, Field(strict=True, ge=1, le=1)]] = (
        Field(default_factory=list, max_length=1, exclude_if=lambda value: not value)
    )

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

    @model_validator(mode="after")
    def validate_runtime_capabilities(self) -> Self:
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
