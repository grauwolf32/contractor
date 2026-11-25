"""Concurrency-safe process and Runtime Agent allocation state."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from contractor_runtime import __version__
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AgentHeartbeat,
    AgentObservedState,
    AgentRegistration,
    RuntimeCompletionCapabilities,
)
from contractor_runtime.contracts.registration import MAX_ALLOCATION_SLOTS, ObservedAllocation
from contractor_runtime.settings import Settings


class ProcessState(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    ALLOCATED = "allocated"
    DRAINING = "draining"
    FENCED = "fenced"
    STOPPING = "stopping"


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    instance_id: str
    started_at: datetime
    process_state: ProcessState
    allocation_id: str | None
    route_dispatches: int
    allocations: tuple[str, ...]


class RuntimeState:
    """Owns one process identity, admission fence and bounded allocation membership."""

    def __init__(
        self,
        *,
        now: datetime | None = None,
        instance_id: str | None = None,
        capabilities: CapabilitySnapshot | None = None,
        allocation_slots: int = MAX_ALLOCATION_SLOTS,
    ) -> None:
        if type(allocation_slots) is not int or not 1 <= allocation_slots <= MAX_ALLOCATION_SLOTS:
            raise ValueError("invalid allocation slot count")
        self.allocation_slots = allocation_slots
        self._lock = asyncio.Lock()
        self._instance_id = instance_id or f"runtime-{uuid.uuid4()}"
        self._started_at = now or datetime.now(UTC)
        self._process_state = ProcessState.STARTING
        self._allocations: dict[str, ProcessState] = {}
        self._route_dispatches = 0
        self._capabilities = capabilities

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def started_at(self) -> datetime:
        return self._started_at

    @property
    def capabilities(self) -> CapabilitySnapshot:
        snapshot = self._capabilities
        if snapshot is None:
            raise RuntimeError("Runtime Agent capabilities have not been discovered")
        return snapshot

    async def install_capabilities(self, capabilities: CapabilitySnapshot) -> None:
        """Freeze the one startup snapshot before the first registration."""

        async with self._lock:
            if self._process_state is not ProcessState.STARTING:
                raise RuntimeError("capabilities can be installed only while starting")
            if self._capabilities is not None and self._capabilities != capabilities:
                raise RuntimeError("capabilities are immutable for the process instance")
            self._capabilities = capabilities

    async def snapshot(self, allocation_id: str | None = None) -> StateSnapshot:
        async with self._lock:
            phase = self._process_state
            if phase is ProcessState.IDLE:
                phase = (
                    self._allocations.get(allocation_id, ProcessState.IDLE)
                    if allocation_id is not None
                    else ProcessState.ALLOCATED
                    if self._allocations
                    else ProcessState.IDLE
                )
            return StateSnapshot(
                instance_id=self._instance_id,
                started_at=self._started_at,
                process_state=phase,
                allocation_id=allocation_id if allocation_id in self._allocations else None,
                route_dispatches=self._route_dispatches,
                allocations=tuple(sorted(self._allocations)),
            )

    async def reserve_allocation(self, allocation_id: str) -> None:
        async with self._lock:
            if (
                self._process_state is not ProcessState.IDLE
                or len(self._allocations) >= self.allocation_slots
            ):
                raise RuntimeError("Runtime allocation capacity is unavailable")
            if allocation_id in self._allocations:
                raise RuntimeError("allocation already reserved")
            self._allocations[allocation_id] = ProcessState.IDLE

    async def discard_preparation(self, allocation_id: str) -> None:
        async with self._lock:
            if self._allocations.get(allocation_id) not in {ProcessState.IDLE, ProcessState.FENCED}:
                raise RuntimeError("cannot discard a committed allocation")
            del self._allocations[allocation_id]

    async def registration(self, settings: Settings) -> AgentRegistration:
        """Build a registration without publishing an internal idle transition.

        The wire contract has no ``starting`` value. The request advertises the
        slot state it will enter only if this registration is acknowledged;
        internal state remains ``starting`` until that response validates.
        """

        async with self._lock:
            capabilities = self._capabilities
            if capabilities is None:
                raise RuntimeError("Runtime Agent capabilities have not been discovered")
            observed_state, allocations = self._wire_state(prospective_idle=True)
            return AgentRegistration(
                apiVersion=API_VERSION,
                instanceId=self._instance_id,
                allocationSlots=self.allocation_slots,
                softwareVersion=__version__,
                startedAt=self._started_at,
                controlUrl=settings.advertised_control_url,
                a2aUrl=settings.advertised_a2a_url,
                placementLabels=list(settings.placement_labels),
                supportedRuntimes=list(capabilities.runtimes),
                supportedToolsets=capabilities.wire_toolsets(),
                capabilities=(
                    RuntimeCompletionCapabilities(
                        completionContracts=list(capabilities.completion_contracts)
                    )
                    if capabilities.completion_contracts
                    else None
                ),
                supportedSandboxProfiles=list(capabilities.sandbox_profiles),
                supportedRuntimeAdapters=list(capabilities.runtime_adapters),
                supportedPerformanceMetricsVersions=list(capabilities.performance_metrics_versions),
                workspaceCapabilities=(
                    capabilities.workspace.wire() if capabilities.workspace is not None else None
                ),
                observedState=observed_state,
                allocations=allocations,
            )

    async def mark_registered(self) -> None:
        async with self._lock:
            if self._process_state is not ProcessState.STARTING:
                raise RuntimeError("registration acknowledgement requires starting state")
            self._process_state = ProcessState.IDLE

    async def heartbeat(self, sequence: int, echoed_ack: int) -> AgentHeartbeat:
        async with self._lock:
            observed_state, allocations = self._wire_state(prospective_idle=False)
            return AgentHeartbeat(
                apiVersion=API_VERSION,
                instanceId=self._instance_id,
                heartbeatSeq=sequence,
                echoedAckSeq=echoed_ack,
                observedState=observed_state,
                allocations=allocations,
            )

    async def begin_stopping(self) -> None:
        async with self._lock:
            self._process_state = ProcessState.STOPPING

    async def commit_allocation(self, allocation_id: str) -> None:
        async with self._lock:
            if (
                self._process_state is not ProcessState.IDLE
                or self._allocations.get(allocation_id) is not ProcessState.IDLE
            ):
                raise RuntimeError("allocation commit requires reserved capacity")
            self._allocations[allocation_id] = ProcessState.ALLOCATED

    async def begin_draining(self, allocation_id: str) -> None:
        async with self._lock:
            if self._allocations.get(allocation_id) not in {
                ProcessState.ALLOCATED,
                ProcessState.DRAINING,
            }:
                raise RuntimeError("draining requires the active allocation")
            self._allocations[allocation_id] = ProcessState.DRAINING

    async def fence_allocation(self, allocation_id: str) -> None:
        async with self._lock:
            if allocation_id not in self._allocations:
                raise RuntimeError("cannot fence an unowned allocation")
            self._allocations[allocation_id] = ProcessState.FENCED

    async def fence_control_lease(self) -> None:
        async with self._lock:
            if self._process_state in {ProcessState.STARTING, ProcessState.STOPPING}:
                raise RuntimeError("cannot fence the current process state")
            self._process_state = ProcessState.FENCED

    async def confirm_release(
        self, allocation_id: str | None, *, keep_fenced: bool = False
    ) -> None:
        async with self._lock:
            if allocation_id is None:
                if self._allocations or self._process_state is not ProcessState.FENCED:
                    raise RuntimeError("lease recovery requires an empty fenced Runtime")
                self._process_state = ProcessState.FENCED if keep_fenced else ProcessState.IDLE
                return
            if self._allocations.get(allocation_id) is not ProcessState.FENCED:
                raise RuntimeError("release confirmation requires the matching fenced allocation")
            del self._allocations[allocation_id]
            if keep_fenced:
                self._process_state = ProcessState.FENCED

    async def record_route_dispatch(self) -> None:
        async with self._lock:
            self._route_dispatches += 1

    def _wire_state(self, *, prospective_idle: bool):
        allocations = [
            ObservedAllocation(allocationId=identity, observedState=phase.value)
            for identity, phase in sorted(self._allocations.items())
            if phase is not ProcessState.IDLE
        ]
        if self._process_state is ProcessState.STARTING and prospective_idle:
            return AgentObservedState.IDLE, []
        if self._process_state is ProcessState.FENCED:
            return AgentObservedState.FENCED, allocations
        if self._process_state is not ProcessState.IDLE:
            raise RuntimeError(f"state {self._process_state} cannot emit a heartbeat")
        return AgentObservedState.ALLOCATED if allocations else AgentObservedState.IDLE, allocations
