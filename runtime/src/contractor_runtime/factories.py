"""Descriptor-backed factories for allocation-local Runtime Agent resources."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from contractor_runtime.adapters import (
    AdapterHandles,
    RuntimeAdapterFactory,
)
from contractor_runtime.adapters.caido_graphql import CaidoGraphQLAdapterFactory
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.adapters.http_proxy import HTTPProxyAdapterFactory
from contractor_runtime.adapters.otlp_http import OTLPHTTPAdapterFactory
from contractor_runtime.adk_runtime import AdkWorkerRuntimeFactory, ModelFactory
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import (
    AllocationWorkspaceExportV2,
    ResolvedAgentTemplate,
    ResolvedModelPolicy,
    ResolvedSkill,
    RuntimeSettings,
)
from contractor_runtime.projectfs import WorkspaceProvider, build_workspace_provider
from contractor_runtime.settings import WorkspaceSettings
from contractor_runtime.toolsets.caido import CaidoToolsetFactory
from contractor_runtime.toolsets.edit_files import EditFilesToolsetFactory
from contractor_runtime.toolsets.filesystem import FilesystemToolsetFactory
from contractor_runtime.toolsets.http_tools import HTTPToolsetFactory
from contractor_runtime.toolsets.likec4 import LikeC4ToolsetFactory
from contractor_runtime.toolsets.memory import MemoryToolsetFactory
from contractor_runtime.toolsets.openapi import OpenAPIToolsetFactory
from contractor_runtime.toolsets.run_artifacts import RunArtifactsToolsetFactory
from contractor_runtime.toolsets.source_analysis import SourceAnalysisToolsetFactory
from contractor_runtime.toolsets.text_artifacts import TextArtifactsToolsetFactory
from contractor_runtime.toolsets.workspace_changes import WorkspaceChangesToolsetFactory
from contractor_runtime.workspace import AllocationWorkspace, LocalWorkdirFactory

if TYPE_CHECKING:
    from contractor_runtime.agent_skills.runtime import PreparedAgentSkills
    from contractor_runtime.projectfs import DirectWorkspaceSession
    from contractor_runtime.projectfs.storage import (
        WorkspaceChanges,
        WorkspaceReader,
        WorkspaceWriter,
    )


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
    resolved_skills: tuple[ResolvedSkill, ...] = ()
    agent_skills: PreparedAgentSkills | None = field(default=None, repr=False)
    project_workspace: DirectWorkspaceSession | None = field(default=None, repr=False)
    workspace_export: AllocationWorkspaceExportV2 | None = None


class WorkerRuntimeFactory(Protocol):
    ref: str
    supports_agent_skills: bool

    async def probe(self) -> bool: ...

    async def create(self, context: WorkerBuildContext) -> WorkerRuntime: ...


class ToolsetFactory(Protocol):
    ref: str
    exported_tools: frozenset[str]
    infrastructure_channels: Mapping[str, frozenset[InfrastructureChannel]]

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
        project_workspace: WorkspaceReader | WorkspaceWriter | WorkspaceChanges | None = None,
    ) -> Mapping[str, ToolInstance]: ...


class SandboxFactory(Protocol):
    ref: str

    async def probe(self) -> bool: ...

    async def prepare(self) -> AllocationWorkspace: ...

    async def cleanup(self, workspace: AllocationWorkspace) -> None: ...


InfrastructureChannel = Literal[
    "caido-graphql-client", "runtime-http-client", "runtime-subprocess-launcher"
]
INFRASTRUCTURE_CHANNELS = frozenset(
    {"caido-graphql-client", "runtime-http-client", "runtime-subprocess-launcher"}
)


@dataclass(frozen=True, slots=True)
class FactoryRegistry:
    worker_runtimes: Mapping[str, WorkerRuntimeFactory]
    toolsets: Mapping[str, ToolsetFactory]
    sandbox_profiles: Mapping[str, SandboxFactory]
    runtime_adapters: Mapping[str, RuntimeAdapterFactory] = field(default_factory=dict)
    workspace_provider: WorkspaceProvider | None = field(default=None, repr=False)
    artifact_client_factory: Callable[[str, RuntimeSettings], ArtifactClient] | None = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        _validate_registry("WorkerRuntime", self.worker_runtimes)
        _validate_registry("Toolset", self.toolsets)
        for ref, factory in self.toolsets.items():
            _validate_toolset_channels(ref, factory)
        _validate_registry("SandboxProfile", self.sandbox_profiles)
        _validate_registry("RuntimeAdapter", self.runtime_adapters, allow_empty=True)


def built_in_factories(
    work_root: Path,
    artifact_client_factory: Callable[[str, RuntimeSettings], ArtifactClient] | None = None,
    model_factory: ModelFactory | None = None,
    enabled_runtime_adapters: Sequence[str] | None = None,
    workspace_settings: WorkspaceSettings | None = None,
) -> FactoryRegistry:
    runtime = AdkWorkerRuntimeFactory(model_factory, artifact_client_factory)
    filesystem_toolset = FilesystemToolsetFactory()
    http_toolset = HTTPToolsetFactory(artifact_client_factory)
    caido_toolset = CaidoToolsetFactory(artifact_client_factory)
    edit_files_toolset = EditFilesToolsetFactory()
    workspace_changes_toolset = WorkspaceChangesToolsetFactory()
    artifact_toolset = RunArtifactsToolsetFactory(artifact_client_factory)
    likec4_toolset = LikeC4ToolsetFactory(artifact_client_factory)
    memory_toolset = MemoryToolsetFactory(artifact_client_factory)
    openapi_toolset = OpenAPIToolsetFactory(artifact_client_factory)
    source_toolset = SourceAnalysisToolsetFactory(artifact_client_factory)
    text_toolset = TextArtifactsToolsetFactory(artifact_client_factory)
    sandbox = LocalWorkdirFactory(work_root)
    telemetry = OTLPHTTPAdapterFactory()
    proxy = HTTPProxyAdapterFactory()
    caido = CaidoGraphQLAdapterFactory()
    runtime_adapters = {caido.ref: caido, proxy.ref: proxy, telemetry.ref: telemetry}
    if enabled_runtime_adapters is not None:
        enabled = frozenset(enabled_runtime_adapters)
        unknown = enabled - runtime_adapters.keys()
        if unknown:
            raise ValueError("enabled Runtime adapter is not built in")
        runtime_adapters = {
            ref: factory for ref, factory in runtime_adapters.items() if ref in enabled
        }
    return FactoryRegistry(
        worker_runtimes={runtime.ref: runtime},
        toolsets={
            artifact_toolset.ref: artifact_toolset,
            caido_toolset.ref: caido_toolset,
            edit_files_toolset.ref: edit_files_toolset,
            filesystem_toolset.ref: filesystem_toolset,
            http_toolset.ref: http_toolset,
            likec4_toolset.ref: likec4_toolset,
            memory_toolset.ref: memory_toolset,
            openapi_toolset.ref: openapi_toolset,
            source_toolset.ref: source_toolset,
            text_toolset.ref: text_toolset,
            workspace_changes_toolset.ref: workspace_changes_toolset,
        },
        sandbox_profiles={sandbox.ref: sandbox},
        runtime_adapters=runtime_adapters,
        workspace_provider=build_workspace_provider(workspace_settings),
        artifact_client_factory=artifact_client_factory,
    )


class StubADKWorkerRuntimeFactory:
    """Lifecycle-complete stand-in replaced by the real ADK adapter in MVP-012."""

    ref = "adk@1"
    supports_agent_skills = False

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


def _validate_toolset_channels(ref: str, factory: ToolsetFactory) -> None:
    channels = getattr(factory, "infrastructure_channels", None)
    if not isinstance(channels, Mapping):
        raise ValueError(f"Toolset {ref!r} has no infrastructure-channel descriptor")
    if not set(channels) <= factory.exported_tools:
        raise ValueError(f"Toolset {ref!r} describes a channel for an unknown tool")
    for tool, selected in channels.items():
        if (
            not isinstance(selected, frozenset)
            or not selected
            or not selected <= INFRASTRUCTURE_CHANNELS
        ):
            raise ValueError(f"Toolset {ref!r} has invalid channels for {tool!r}")
