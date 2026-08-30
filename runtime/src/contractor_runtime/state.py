"""Concurrency-safe process and single-slot Runtime Agent state."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from contractor_runtime.contracts import (
    API_VERSION,
    AgentHeartbeat,
    AgentObservedState,
    AgentRegistration,
    ToolsetCapability,
)
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


class RuntimeState:
    """Owns the process identity and one exclusive allocation slot."""

    def __init__(self, *, now: datetime | None = None, instance_id: str | None = None) -> None:
        self._lock = asyncio.Lock()
        self._instance_id = instance_id or f"runtime-{uuid.uuid4()}"
        self._started_at = now or datetime.now(UTC)
        self._process_state = ProcessState.STARTING
        self._allocation_id: str | None = None
        self._route_dispatches = 0

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def started_at(self) -> datetime:
        return self._started_at

    async def snapshot(self) -> StateSnapshot:
        async with self._lock:
            return StateSnapshot(
                instance_id=self._instance_id,
                started_at=self._started_at,
                process_state=self._process_state,
                allocation_id=self._allocation_id,
                route_dispatches=self._route_dispatches,
            )

    async def registration(self, settings: Settings) -> AgentRegistration:
        """Build a registration without publishing an internal idle transition.

        The wire contract has no ``starting`` value. The request advertises the
        slot state it will enter only if this registration is acknowledged;
        internal state remains ``starting`` until that response validates.
        """

        async with self._lock:
            observed_state, allocation_id = self._wire_state(prospective_idle=True)
            return AgentRegistration(
                apiVersion=API_VERSION,
                instanceId=self._instance_id,
                startedAt=self._started_at,
                controlUrl=settings.advertised_control_url,
                a2aUrl=settings.advertised_a2a_url,
                supportedRuntimes=["adk@1"],
                supportedToolsets=[
                    ToolsetCapability(
                        ref="openapi@1",
                        tools=[
                            "get_openapi_component",
                            "get_openapi_info",
                            "get_openapi_path",
                            "initialize_openapi",
                            "list_openapi_components",
                            "list_openapi_paths",
                            "list_openapi_servers",
                            "load_openapi",
                            "read_openapi_document",
                            "remove_openapi_component",
                            "remove_openapi_path",
                            "set_openapi_info",
                            "set_openapi_servers",
                            "upsert_openapi_component",
                            "upsert_openapi_path",
                            "validate_openapi",
                        ],
                    ),
                    ToolsetCapability(
                        ref="run-artifacts@1",
                        tools=["list_artifacts", "read_artifact", "write_artifact"],
                    ),
                    ToolsetCapability(
                        ref="source-analysis@1",
                        tools=[
                            "list_source_files",
                            "open_source_archive",
                            "read_source",
                            "search_source",
                        ],
                    ),
                    ToolsetCapability(
                        ref="text-artifacts@1",
                        tools=["read_text_artifact", "write_text_artifact"],
                    ),
                ],
                supportedSandboxProfiles=["local-workdir@1"],
                observedState=observed_state,
                allocationId=allocation_id,
            )

    async def mark_registered(self) -> None:
        async with self._lock:
            if self._process_state is not ProcessState.STARTING:
                raise RuntimeError("registration acknowledgement requires starting state")
            self._process_state = ProcessState.IDLE

    async def heartbeat(self, sequence: int, echoed_ack: int) -> AgentHeartbeat:
        async with self._lock:
            observed_state, allocation_id = self._wire_state(prospective_idle=False)
            return AgentHeartbeat(
                apiVersion=API_VERSION,
                instanceId=self._instance_id,
                heartbeatSeq=sequence,
                echoedAckSeq=echoed_ack,
                observedState=observed_state,
                allocationId=allocation_id,
            )

    async def begin_stopping(self) -> None:
        async with self._lock:
            self._process_state = ProcessState.STOPPING

    async def commit_allocation(self, allocation_id: str) -> None:
        async with self._lock:
            if self._process_state is not ProcessState.IDLE or self._allocation_id is not None:
                raise RuntimeError("allocation commit requires an idle slot")
            self._allocation_id = allocation_id
            self._process_state = ProcessState.ALLOCATED

    async def begin_draining(self, allocation_id: str) -> None:
        async with self._lock:
            if self._allocation_id != allocation_id or self._process_state not in {
                ProcessState.ALLOCATED,
                ProcessState.DRAINING,
            }:
                raise RuntimeError("draining requires the active allocation")
            self._process_state = ProcessState.DRAINING

    async def fence_allocation(self, allocation_id: str) -> None:
        async with self._lock:
            if self._allocation_id not in {None, allocation_id}:
                raise RuntimeError("cannot fence a different active allocation")
            if self._process_state in {ProcessState.STARTING, ProcessState.STOPPING}:
                raise RuntimeError("cannot fence the current process state")
            self._allocation_id = allocation_id
            self._process_state = ProcessState.FENCED

    async def fence_control_lease(self) -> None:
        """Fence the current slot without inventing an allocation identity."""

        async with self._lock:
            if self._process_state in {ProcessState.STARTING, ProcessState.STOPPING}:
                raise RuntimeError("cannot fence the current process state")
            self._process_state = ProcessState.FENCED

    async def confirm_release(self, allocation_id: str | None) -> None:
        async with self._lock:
            if self._process_state is ProcessState.IDLE and self._allocation_id is None:
                if allocation_id is None:
                    return
                raise RuntimeError("released allocation identity no longer matches")
            if (
                self._process_state is not ProcessState.FENCED
                or self._allocation_id != allocation_id
            ):
                raise RuntimeError("release confirmation requires the matching fenced slot")
            self._allocation_id = None
            self._process_state = ProcessState.IDLE

    async def record_route_dispatch(self) -> None:
        async with self._lock:
            self._route_dispatches += 1

    def _wire_state(self, *, prospective_idle: bool) -> tuple[AgentObservedState, str | None]:
        if self._process_state is ProcessState.STARTING and prospective_idle:
            return AgentObservedState.IDLE, None
        mapping = {
            ProcessState.IDLE: AgentObservedState.IDLE,
            ProcessState.ALLOCATED: AgentObservedState.ALLOCATED,
            ProcessState.DRAINING: AgentObservedState.DRAINING,
            ProcessState.FENCED: AgentObservedState.FENCED,
        }
        observed = mapping.get(self._process_state)
        if observed is None:
            raise RuntimeError(f"state {self._process_state} cannot emit a heartbeat")
        return observed, self._allocation_id
