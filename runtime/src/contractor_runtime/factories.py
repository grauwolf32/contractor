"""Descriptor-backed factories for allocation-local Runtime Agent resources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from contractor_runtime.contracts import ResolvedAgentTemplate, RuntimeSettings
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory


class WorkerRuntime(Protocol):
    @property
    def agent_card(self) -> Mapping[str, Any]: ...

    async def finalize(self, deadline: datetime) -> None: ...

    async def abort(self, deadline: datetime) -> None: ...


class ToolInstance(Protocol):
    @property
    def name(self) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerBuildContext:
    allocation_id: str
    run_id: str
    stage_execution_id: str
    logical_agent_name: str
    namespace: str
    agent_template: ResolvedAgentTemplate
    workspace: AllocationWorkspace
    tools: Mapping[str, ToolInstance]
    state: Any
    a2a_base_url: str
    runtime_settings: RuntimeSettings = field(repr=False)


class WorkerRuntimeFactory(Protocol):
    ref: str

    async def create(self, context: WorkerBuildContext) -> WorkerRuntime: ...


class ToolsetFactory(Protocol):
    ref: str
    exported_tools: frozenset[str]

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
    ) -> Mapping[str, ToolInstance]: ...


class SandboxFactory(Protocol):
    ref: str

    async def prepare(self) -> AllocationWorkspace: ...

    async def cleanup(self, workspace: AllocationWorkspace) -> None: ...


@dataclass(frozen=True, slots=True)
class FactoryRegistry:
    worker_runtimes: Mapping[str, WorkerRuntimeFactory]
    toolsets: Mapping[str, ToolsetFactory]
    sandbox_profiles: Mapping[str, SandboxFactory]

    def __post_init__(self) -> None:
        _validate_registry("WorkerRuntime", self.worker_runtimes)
        _validate_registry("Toolset", self.toolsets)
        _validate_registry("SandboxProfile", self.sandbox_profiles)


def built_in_factories(work_root: Path) -> FactoryRegistry:
    runtime = StubADKWorkerRuntimeFactory()
    toolset = RunArtifactsToolsetFactory()
    sandbox = LocalWorkdirFactory(work_root)
    return FactoryRegistry(
        worker_runtimes={runtime.ref: runtime},
        toolsets={toolset.ref: toolset},
        sandbox_profiles={sandbox.ref: sandbox},
    )


class StubADKWorkerRuntimeFactory:
    """Lifecycle-complete stand-in replaced by the real ADK adapter in MVP-012."""

    ref = "adk@1"

    async def create(self, context: WorkerBuildContext) -> WorkerRuntime:
        return StubWorkerRuntime(context)


class StubWorkerRuntime:
    def __init__(self, context: WorkerBuildContext) -> None:
        endpoint = (
            f"{context.a2a_base_url.rstrip('/')}/private/v1/allocations/{context.allocation_id}/a2a"
        )
        self._agent_card: dict[str, Any] = {
            "name": context.logical_agent_name,
            "description": context.agent_template.description,
            "url": endpoint,
            "protocolVersion": "1.0",
            "version": context.agent_template.ref.version,
            "capabilities": {},
            "defaultInputModes": ["application/json"],
            "defaultOutputModes": ["application/json"],
            "skills": [],
        }
        self.stopped = False

    @property
    def agent_card(self) -> Mapping[str, Any]:
        return dict(self._agent_card)

    async def finalize(self, deadline: datetime) -> None:
        self.stopped = True

    async def abort(self, deadline: datetime) -> None:
        self.stopped = True


class RunArtifactsToolsetFactory:
    ref = "run-artifacts@1"
    exported_tools = frozenset({"list_artifacts", "read_artifact", "write_artifact"})

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
    ) -> Mapping[str, ToolInstance]:
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        return {name: StubTool(name=name, toolset_ref=self.ref) for name in selected}


@dataclass(frozen=True, slots=True)
class StubTool:
    name: str
    toolset_ref: str

    async def close(self) -> None:
        return None


def _validate_registry(kind: str, entries: Mapping[str, Any]) -> None:
    if not entries:
        raise ValueError(f"{kind} registry must not be empty")
    for ref, factory in entries.items():
        if ref != factory.ref:
            raise ValueError(f"{kind} registry key {ref!r} does not match descriptor ref")
