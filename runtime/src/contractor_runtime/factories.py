"""Descriptor-backed factories for allocation-local Runtime Agent resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from contractor_runtime.adapters import (
    AdapterHandles,
    RuntimeAdapterFactory,
)
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adk_runtime import AdkWorkerRuntimeFactory, ModelFactory
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ResolvedAgentTemplate, ResolvedModelPolicy, RuntimeSettings
from contractor_runtime.toolsets.likec4 import LikeC4ToolsetFactory
from contractor_runtime.toolsets.openapi import OpenAPIToolsetFactory
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.toolsets.source_analysis import SourceAnalysisToolsetFactory
from contractor_runtime.toolsets.text_artifacts import TextArtifactsToolsetFactory
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
    model_policy: ResolvedModelPolicy
    workspace: AllocationWorkspace
    tools: Mapping[str, ToolInstance]
    state: Any
    a2a_base_url: str
    runtime_settings: RuntimeSettings = field(repr=False)
    adapter_handles: AdapterHandles = field(
        default=EMPTY_ADAPTER_HANDLES,
        repr=False,
    )


class WorkerRuntimeFactory(Protocol):
    ref: str

    async def probe(self) -> bool: ...

    async def create(self, context: WorkerBuildContext) -> WorkerRuntime: ...


class ToolsetFactory(Protocol):
    ref: str
    exported_tools: frozenset[str]

    async def probe(self) -> frozenset[str]: ...

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
    ) -> Mapping[str, ToolInstance]: ...


class SandboxFactory(Protocol):
    ref: str

    async def probe(self) -> bool: ...

    async def prepare(self) -> AllocationWorkspace: ...

    async def cleanup(self, workspace: AllocationWorkspace) -> None: ...


@dataclass(frozen=True, slots=True)
class FactoryRegistry:
    worker_runtimes: Mapping[str, WorkerRuntimeFactory]
    toolsets: Mapping[str, ToolsetFactory]
    sandbox_profiles: Mapping[str, SandboxFactory]
    runtime_adapters: Mapping[str, RuntimeAdapterFactory] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_registry("WorkerRuntime", self.worker_runtimes)
        _validate_registry("Toolset", self.toolsets)
        _validate_registry("SandboxProfile", self.sandbox_profiles)
        _validate_registry("RuntimeAdapter", self.runtime_adapters, allow_empty=True)


def built_in_factories(
    work_root: Path,
    artifact_client_factory: Callable[[str, RuntimeSettings], ArtifactClient] | None = None,
    model_factory: ModelFactory | None = None,
) -> FactoryRegistry:
    runtime = AdkWorkerRuntimeFactory(model_factory)
    artifact_toolset = RunArtifactsToolsetFactory(artifact_client_factory)
    likec4_toolset = LikeC4ToolsetFactory(artifact_client_factory)
    openapi_toolset = OpenAPIToolsetFactory(artifact_client_factory)
    source_toolset = SourceAnalysisToolsetFactory(artifact_client_factory)
    text_toolset = TextArtifactsToolsetFactory(artifact_client_factory)
    sandbox = LocalWorkdirFactory(work_root)
    return FactoryRegistry(
        worker_runtimes={runtime.ref: runtime},
        toolsets={
            artifact_toolset.ref: artifact_toolset,
            likec4_toolset.ref: likec4_toolset,
            openapi_toolset.ref: openapi_toolset,
            source_toolset.ref: source_toolset,
            text_toolset.ref: text_toolset,
        },
        sandbox_profiles={sandbox.ref: sandbox},
        runtime_adapters={},
    )


class StubADKWorkerRuntimeFactory:
    """Lifecycle-complete stand-in replaced by the real ADK adapter in MVP-012."""

    ref = "adk@1"

    async def probe(self) -> bool:
        return True

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


def _validate_registry(kind: str, entries: Mapping[str, Any], *, allow_empty: bool = False) -> None:
    if not entries and not allow_empty:
        raise ValueError(f"{kind} registry must not be empty")
    for ref, factory in entries.items():
        if ref != factory.ref:
            raise ValueError(f"{kind} registry key {ref!r} does not match descriptor ref")
