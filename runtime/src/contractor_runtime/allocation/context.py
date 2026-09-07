"""Resources and state owned by one Runtime allocation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from contractor_runtime.adapters import (
    AllocationAdapterHost,
)
from contractor_runtime.contracts import (
    AllocationFinalResponse,
    PrepareAllocationResponse,
    RuntimeSettings,
)
from contractor_runtime.factories import (
    SandboxFactory,
    ToolInstance,
    WorkerRuntime,
)
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
)
from contractor_runtime.sandbox.lifecycle import PreparedExecution
from contractor_runtime.state import ProcessState
from contractor_runtime.telemetry.resources import ResourceCollector
from contractor_runtime.worker.state import WorkerStateStore
from contractor_runtime.workspace import AllocationWorkspace


@dataclass(frozen=True, slots=True)
class AllocationSnapshot:
    allocation_id: str
    stage_execution_id: str
    process_state: ProcessState
    workspace: str
    tool_names: tuple[str, ...]
    has_runtime_settings: bool
    has_worker: bool
    runtime_adapter_refs: tuple[str, ...]
    has_project_workspace: bool


@dataclass(slots=True)
class _AllocationContext:
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    fingerprint: str
    started_at: datetime
    workspace: AllocationWorkspace
    sandbox: SandboxFactory
    project_workspace: DirectWorkspaceSession | None = field(repr=False)
    adapter_host: AllocationAdapterHost = field(repr=False)
    tools: dict[str, ToolInstance]
    worker_state: WorkerStateStore | None
    runtime_settings: RuntimeSettings | None = field(repr=False)
    worker: WorkerRuntime | None = field(repr=False)
    execution: PreparedExecution | None = field(default=None, repr=False)
    resource_metrics: ResourceCollector | None = field(default=None, repr=False)
    prepare_response: PrepareAllocationResponse | None = None
    termination_kind: str | None = None
    termination_id: str | None = None
    terminal_response: AllocationFinalResponse | None = None
    release_prepared: bool = False
    release_cleanup_task: asyncio.Task[None] | None = field(default=None, repr=False)
