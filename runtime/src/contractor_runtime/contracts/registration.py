"""Private Runtime protocol registration models and validation."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    model_validator,
)

from contractor_runtime.contracts.base import (
    MAX_RUNTIME_CAPABILITY_REFS,
    MAX_TOOLSET_CAPABILITY_TOOLS,
    NAME_PATTERN,
    VERSION_PATTERN,
    AgentObservedState,
    ReconciliationAction,
    RuntimeAdapterRef,
    VersionedWireModel,
    WireModel,
    _require_aware_datetime,
    _require_capability_ref,
    _require_resource_id,
    _require_sorted_unique,
    _require_url,
)
from contractor_runtime.contracts.placement import validate_placement_labels
from contractor_runtime.contracts.workspace import WorkspaceCapabilities


class ToolsetCapability(WireModel):
    ref: str
    tools: list[str]

    @model_validator(mode="after")
    def validate_capability(self) -> Self:
        _require_capability_ref("ref", self.ref)
        if (
            not self.tools
            or len(self.tools) > MAX_TOOLSET_CAPABILITY_TOOLS
            or len(self.tools) != len(set(self.tools))
        ):
            raise ValueError(f"tools must be 1 to {MAX_TOOLSET_CAPABILITY_TOOLS} unique tool names")
        for tool in self.tools:
            if NAME_PATTERN.fullmatch(tool) is None:
                raise ValueError("tools must use the shared tool name grammar")
        return self


MAX_ALLOCATION_SLOTS = 4


class ObservedAllocation(WireModel):
    allocation_id: str
    observed_state: Literal[
        AgentObservedState.ALLOCATED, AgentObservedState.DRAINING, AgentObservedState.FENCED
    ]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_resource_id("allocationId", self.allocation_id)
        return self


def _validate_observed_allocations(state, allocations):
    ids = [item.allocation_id for item in allocations]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate allocation observation")
    if state is AgentObservedState.IDLE and allocations:
        raise ValueError("idle Runtime must not report allocations")
    if state in {AgentObservedState.ALLOCATED, AgentObservedState.DRAINING} and not allocations:
        raise ValueError("allocated Runtime must report allocations")


class AllocationAction(WireModel):
    allocation_id: str
    action: Literal[ReconciliationAction.DRAIN, ReconciliationAction.RELEASE]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        _require_resource_id("allocationId", self.allocation_id)
        return self


class AgentRegistrationResponse(VersionedWireModel):
    heartbeat_interval_seconds: int = Field(gt=0)
    confirmed_lease_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.confirmed_lease_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("confirmed lease must exceed heartbeat interval")
        return self


class HeartbeatResponse(VersionedWireModel):
    ack_seq: int = Field(gt=0)
    action: Literal[ReconciliationAction.CONTINUE, ReconciliationAction.REREGISTER]
    allocation_actions: list[AllocationAction] = Field(max_length=2 * MAX_ALLOCATION_SLOTS)

    @model_validator(mode="after")
    def validate_response(self) -> Self:
        ids = [item.allocation_id for item in self.allocation_actions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate allocation action")
        if self.action is ReconciliationAction.REREGISTER and self.allocation_actions:
            raise ValueError("reregister must not include allocation actions")
        return self


class RuntimeCompletionCapabilities(WireModel):
    completion_contracts: list[Literal["audit-check-results@1"]] = Field(max_length=1)


def _validate_capability_refs(
    runtimes: list[str], toolsets: list[ToolsetCapability], sandboxes: list[str]
) -> None:
    if not runtimes or not sandboxes:
        raise ValueError("runtime and sandbox capability lists must not be empty")
    if max(len(runtimes), len(toolsets), len(sandboxes)) > MAX_RUNTIME_CAPABILITY_REFS:
        raise ValueError("Runtime capability collection exceeds its bound")
    if len(runtimes) != len(set(runtimes)) or len(sandboxes) != len(set(sandboxes)):
        raise ValueError("capability refs must be unique")
    for value in runtimes:
        _require_capability_ref("supportedRuntimes", value)
    for value in sandboxes:
        _require_capability_ref("supportedSandboxProfiles", value)
    refs = [item.ref for item in toolsets]
    if len(refs) != len(set(refs)):
        raise ValueError("toolset capability refs must be unique")


class AgentHeartbeat(VersionedWireModel):
    instance_id: str
    heartbeat_seq: int = Field(gt=0)
    echoed_ack_seq: int = Field(ge=0)
    observed_state: AgentObservedState
    allocations: list[ObservedAllocation] = Field(max_length=MAX_ALLOCATION_SLOTS)

    @model_validator(mode="after")
    def validate_heartbeat(self) -> Self:
        _require_resource_id("instanceId", self.instance_id)
        _validate_observed_allocations(self.observed_state, self.allocations)
        return self


class AgentRegistration(VersionedWireModel):
    nullable_fields = frozenset({"capabilities", "workspace_capabilities"})

    allocation_slots: int = Field(ge=1, le=MAX_ALLOCATION_SLOTS)
    instance_id: str
    software_version: str = Field(min_length=1, max_length=128, pattern=VERSION_PATTERN.pattern)
    started_at: datetime
    control_url: str
    a2a_url: str
    supported_runtimes: list[str]
    supported_toolsets: list[ToolsetCapability]
    supported_sandbox_profiles: list[str]
    observed_state: AgentObservedState
    allocations: list[ObservedAllocation] = Field(max_length=MAX_ALLOCATION_SLOTS)
    capabilities: RuntimeCompletionCapabilities | None = None
    placement_labels: list[str] = Field(
        default_factory=list, max_length=32, exclude_if=lambda value: not value
    )
    supported_runtime_adapters: list[RuntimeAdapterRef] = Field(max_length=64)
    workspace_capabilities: WorkspaceCapabilities | None = None
    supported_performance_metrics_versions: list[Annotated[int, Field(strict=True, ge=1, le=1)]] = (
        Field(default_factory=list, max_length=1, exclude_if=lambda value: not value)
    )

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        _require_resource_id("instanceId", self.instance_id)
        _require_aware_datetime("startedAt", self.started_at)
        _require_url("controlUrl", self.control_url)
        _require_url("a2aUrl", self.a2a_url)
        _validate_capability_refs(
            self.supported_runtimes,
            self.supported_toolsets,
            self.supported_sandbox_profiles,
        )
        _validate_observed_allocations(self.observed_state, self.allocations)
        if len(self.allocations) > self.allocation_slots:
            raise ValueError("observations exceed Runtime capacity")
        return self

    @model_validator(mode="after")
    def validate_runtime_capabilities(self) -> Self:
        validate_placement_labels(self.placement_labels)
        _require_sorted_unique(
            "supportedRuntimeAdapters", self.supported_runtime_adapters, maximum=64
        )
        return self
