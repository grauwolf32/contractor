"""Allocation preparation and tool composition without slot-state authority."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from functools import partial
from typing import Any

from contractor_runtime.adapters import (
    AdapterHandles,
    AllocationAdapterHost,
)
from contractor_runtime.adapters.http_transport import HTTPTransport
from contractor_runtime.allocation.context import _AllocationContext
from contractor_runtime.allocation.errors import AllocationError
from contractor_runtime.allocation.redaction import _untrusted_agent_card_text
from contractor_runtime.contracts import (
    API_VERSION,
    AllocationSpec,
    CompactWorkerContextManagement,
    PrepareAllocationResponse,
    WorkerHandle,
    WorkerSummarizerConfig,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    SandboxFactory,
    ToolInstance,
    WorkerBuildContext,
    WorkerRuntime,
    WorkerRuntimeFactory,
)
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.redaction import (
    contains_private_value,
    runtime_setting_values,
    substring_values,
)
from contractor_runtime.sandbox.contracts import PODMAN_PROFILES
from contractor_runtime.sandbox.lifecycle import PreparedExecution
from contractor_runtime.sitemap import SitemapClient
from contractor_runtime.sitemap_collection import SitemapCollector
from contractor_runtime.telemetry.resources import ResourceCollector
from contractor_runtime.toolsets.common.report_observations import REPORT_FORMATS, prepare_report
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.worker.state import WorkerStateStore
from contractor_runtime.workspace import AllocationWorkspace


def _effective_summarizer(spec: AllocationSpec) -> WorkerSummarizerConfig | None:
    """Returns the terminal summarizer only when it is the effective branch."""
    summarizer = spec.agent_template.summarizer
    if spec.context_management.strategy != "terminal@1" or summarizer is None:
        return None
    if spec.summarizer_model_policy is None:
        raise RuntimeError("terminal allocation is missing summarizerModelPolicy")
    return WorkerSummarizerConfig(
        **summarizer.model_dump(by_alias=True, exclude_none=True),
        modelPolicy=spec.summarizer_model_policy,
    )


class _SourceObservationSink:
    def __init__(self, collector: SitemapCollector, kind: str) -> None:
        self._collector = collector
        self._kind = kind

    async def receive(self, observations: Sequence[dict], *, gaps: Sequence[dict] = ()) -> None:
        await self._collector.receive_source(self._kind, observations, gaps=gaps)


class AllocationPreparation:
    """Acquire allocation resources; expose partial ownership to service rollback.

    This owner cannot change slot state or acknowledge commands. Only a successful
    build returns a complete context for AllocationService to commit.
    """

    def __init__(
        self,
        factories: FactoryRegistry,
        *,
        now: Callable[[], datetime],
        a2a_base_url: str,
        private_bypass_hosts: tuple[str, ...],
        confirmed_control_deadline: Callable[[], float | None],
    ) -> None:
        self._factories = factories
        self._now = now
        self._a2a_base_url = a2a_base_url
        self._private_bypass_hosts = private_bypass_hosts
        self._confirmed_control_deadline = confirmed_control_deadline
        self.adapter_host: AllocationAdapterHost | None = None
        self.workspace: AllocationWorkspace | None = None
        self.project_workspace: DirectWorkspaceSession | None = None
        self.tools: dict[str, ToolInstance] = {}
        self.resources: ToolResources | None = None
        self.worker_state: WorkerStateStore | None = None
        self.collector: SitemapCollector | None = None
        self.worker: WorkerRuntime | None = None
        self.execution: PreparedExecution | None = None
        self.http_transport: HTTPTransport | None = None

    async def build(
        self,
        spec: AllocationSpec,
        sandbox: SandboxFactory,
        runtime_factory: WorkerRuntimeFactory,
        *,
        fingerprint: str,
        resource_metrics: ResourceCollector | None,
    ) -> _AllocationContext:
        adapter_deadline = min(
            spec.lease_expires_at,
            self._now() + timedelta(seconds=spec.runtime_settings.request_timeout_seconds),
        )
        lifecycle = self._factories.execution_lifecycle
        if lifecycle is not None:
            # Even embedded preparation cannot hydrate/delete orphan
            # storage until this service's predecessor is removed.
            await lifecycle.recover(
                deadline=time.monotonic()
                + max(0.0, (adapter_deadline - self._now()).total_seconds())
            )
        self.adapter_host = await AllocationAdapterHost.create(
            spec,
            self._factories.runtime_adapters,
            deadline=adapter_deadline,
            now=self._now,
            private_bypass_hosts=self._private_bypass_hosts,
        )
        self.workspace = await sandbox.prepare()
        if spec.workspace is not None:
            if not (self._factories.workspace_provider is not None):
                raise RuntimeError("Expected self._factories.workspace_provider is not None")
            if not (self._factories.artifact_client_factory is not None):
                raise RuntimeError("Expected self._factories.artifact_client_factory is not None")
            artifact_reader = self._factories.artifact_client_factory(
                spec.allocation_id, spec.runtime_settings
            )
            remaining = min(
                spec.runtime_settings.request_timeout_seconds,
                max(0.0, (spec.lease_expires_at - self._now()).total_seconds()),
            )
            self.project_workspace = await hydrate_workspace(
                provider=self._factories.workspace_provider,
                spec=spec.workspace,
                artifact_reader=artifact_reader,
                allocation_id=spec.allocation_id,
                timeout_seconds=remaining,
            )
        if sandbox.ref in PODMAN_PROFILES:
            if not (lifecycle is not None and self.project_workspace is not None):
                raise RuntimeError(
                    "Expected lifecycle is not None and project_workspace is not None"
                )
            self.execution = lifecycle.allocate(
                spec.allocation_id, self.project_workspace, sandbox_profile=sandbox.ref
            )
            await self.execution.prepare(deadline=adapter_deadline)
        self.worker_state = WorkerStateStore()
        self.resources = ToolResources(metrics=self.worker_state.metrics)
        if (
            spec.agent_template.sitemap is not None
            and spec.agent_template.sitemap.collect_observations
        ):
            if self._factories.artifact_client_factory is None:
                raise RuntimeError(
                    "automatic collection requires the allocation Artifact transport"
                )
            collection_artifacts = self._factories.artifact_client_factory(
                spec.allocation_id, spec.runtime_settings
            )
            self.collector = SitemapCollector(
                SitemapClient(collection_artifacts),
                report_projector=partial(prepare_report, collection_artifacts),
                report_formats=REPORT_FORMATS,
            )
            self.resources.http_observation_sink = _SourceObservationSink(self.collector, "http")
            self.resources.caido_observation_sink = _SourceObservationSink(self.collector, "caido")
            self.resources.report_publication_sink = self.collector
            self.resources.sitemap_read_flush = self.collector.checkpoint
            self.resources.before_completion = self.collector.checkpoint
        if self.execution is not None:
            self.execution.bind_failure(self.resources.execution.fail)
        if (
            spec.engagement is not None
            and spec.engagement.http is not None
            and self._factories.http_transport_factory is not None
        ):
            self.http_transport = self._factories.http_transport_factory.create(spec)
            self.http_transport.configure(authority_deadline=self._confirmed_control_deadline)
        async with asyncio.timeout(max(0.0, (adapter_deadline - self._now()).total_seconds())):
            await self._create_tools(
                spec,
                self.workspace,
                self.project_workspace,
                self.resources,
                self.adapter_host.handles,
                self.execution,
                self.tools,
                adapter_deadline,
                self.http_transport,
            )
        # Target identity credentials belong only to controlled HTTP
        # owner. The ordinary Worker/summarizer must not copy them
        # into its own long-lived redaction or model state.
        worker_settings = spec.runtime_settings.model_copy(update={"http_identities": None})
        self.worker = await runtime_factory.create(
            WorkerBuildContext(
                allocation_id=spec.allocation_id,
                run_id=spec.run_id,
                stage_execution_id=spec.stage_execution_id,
                logical_agent_name=spec.logical_agent_name,
                namespace=spec.namespace,
                worker_session_mode=spec.worker_session_mode,
                description=spec.agent_template.description,
                instruction=(
                    spec.agent_template.instructions.text
                    if spec.agent_template.instructions is not None
                    else ""
                ),
                execution=spec.agent_template.execution,
                template_ref=spec.agent_template.ref,
                card_version=spec.agent_template.ref.version,
                model_policy=spec.model_policy,
                workspace=self.workspace,
                tools=self.tools,
                state=self.worker_state,
                resources=self.resources,
                a2a_base_url=self._a2a_base_url,
                runtime_settings=worker_settings,
                summarizer=_effective_summarizer(spec),
                compaction=(
                    spec.context_management
                    if isinstance(spec.context_management, CompactWorkerContextManagement)
                    else None
                ),
                compactor_model_policy=spec.compactor_model_policy,
                completion_contract=spec.completion_contract,
                response_recovery=spec.agent_template.response_recovery,
                adapter_handles=self.adapter_host.handles.for_worker(),
                resolved_skills=tuple(spec.resolved_skills),
                project_workspace=self.project_workspace,
                workspace_export=(spec.workspace.export if spec.workspace is not None else None),
            )
        )
        handle = WorkerHandle(
            allocationId=spec.allocation_id,
            agentTemplateRef=spec.agent_template.ref,
            workerRuntimeRef=spec.agent_template.runtime,
            agentCard=dict(self.worker.agent_card),
            leaseExpiresAt=spec.lease_expires_at,
        )
        self._assert_safe_handle(handle, spec, self.workspace, self._a2a_base_url)
        response = PrepareAllocationResponse(
            apiVersion=API_VERSION,
            workerHandle=handle,
        )
        context = _AllocationContext(
            allocation_id=spec.allocation_id,
            run_id=spec.run_id,
            stage_execution_id=spec.stage_execution_id,
            logical_agent_name=spec.logical_agent_name,
            namespace=spec.namespace,
            fingerprint=fingerprint,
            started_at=self._now(),
            workspace=self.workspace,
            sandbox=sandbox,
            project_workspace=self.project_workspace,
            adapter_host=self.adapter_host,
            tools=self.tools,
            worker_state=self.worker_state,
            resources=self.resources,
            runtime_settings=spec.runtime_settings,
            worker=self.worker,
            execution=self.execution,
            resource_metrics=resource_metrics,
            collector=self.collector,
            prepare_response=response,
            http_transport=self.http_transport,
        )
        return context

    async def _create_tools(
        self,
        spec: AllocationSpec,
        workspace: AllocationWorkspace,
        project_workspace: DirectWorkspaceSession | None,
        resources: ToolResources,
        adapter_handles: AdapterHandles,
        execution: PreparedExecution | None,
        result: dict[str, ToolInstance],
        preparation_deadline: datetime,
        http_transport: HTTPTransport | None = None,
    ) -> None:
        for selection in spec.agent_template.toolsets:
            ref = f"{selection.ref.toolset_id}@{selection.ref.version}"
            factory = self._factories.toolsets.get(ref)
            if factory is None:
                raise AllocationError(
                    "unsupported_toolset",
                    "AgentTemplate selects an unsupported Toolset",
                    retryable=False,
                    status_code=422,
                )
            if not set(selection.tools) <= factory.exported_tools:
                raise AllocationError(
                    "unsupported_tool",
                    "AgentTemplate selects a tool not exported by its Toolset",
                    retryable=False,
                    status_code=422,
                )
            channels = frozenset(
                channel
                for name in selection.tools
                for channel in factory.infrastructure_channels.get(name, frozenset())
            )
            created = await factory.create_selected(
                **(
                    {
                        "controlled_http": http_transport,
                        "deployment_marker": (
                            spec.agent_template.sitemap.deployment
                            if spec.agent_template.sitemap is not None
                            else None
                        ),
                        "preparation_deadline": asyncio.get_running_loop().time()
                        + max(0.0, (preparation_deadline - self._now()).total_seconds()),
                    }
                    if ref == "http-tools@1"
                    else {}
                ),
                **({"controlled_http": http_transport} if ref == "scan@1" else {}),
                **(
                    {"completion_contract": spec.completion_contract}
                    if ref == "audit-results@2"
                    else {}
                ),
                **(
                    {"sandbox_executor": execution.executor if execution else None}
                    if ref == "code-execution@1"
                    else {}
                ),
                selected=selection.tools,
                allocation_id=spec.allocation_id,
                run_id=spec.run_id,
                namespace=spec.namespace,
                runtime_settings=spec.runtime_settings,
                workspace=workspace,
                resources=resources,
                adapter_handles=adapter_handles.for_tool_channels(channels),
                project_workspace=self._workspace_tool_view(factory, project_workspace),
            )
            collision = set(result) & set(created)
            # Transfer every returned resource to prepare's rollback owner
            # before validation or the next await. Invalid duplicate names
            # retain separate cleanup entries and never reach a Worker.
            for name, tool in created.items():
                cleanup_name = name
                while cleanup_name in result:
                    cleanup_name = f"invalid:{cleanup_name}"
                result[cleanup_name] = tool
            if set(created) != set(selection.tools):
                raise AllocationError(
                    "invalid_toolset_factory",
                    "Toolset factory returned a different visible tool set",
                    retryable=False,
                    status_code=500,
                )
            if any(tool.name != name for name, tool in created.items()):
                raise AllocationError(
                    "invalid_toolset_factory",
                    "Toolset factory returned a tool with a mismatched visible name",
                    retryable=False,
                    status_code=500,
                )
            # The Server decides from the same declaration whether the Worker's
            # Planner gets workspace projections (TLS-R38), so a tool may report
            # workspace observations exactly when its Toolset declares it.
            observing = getattr(factory, "workspace_observation_tools", frozenset())
            if any(
                (name in observing) != callable(getattr(tool, "contractor_observation", None))
                for name, tool in created.items()
            ):
                raise AllocationError(
                    "invalid_toolset_factory",
                    "Toolset factory returned a tool whose workspace observation is undeclared",
                    retryable=False,
                    status_code=500,
                )
            if collision:
                raise AllocationError(
                    "duplicate_visible_tool",
                    "Toolset factories produced a duplicate model-visible tool",
                    retryable=False,
                    status_code=422,
                )

    @staticmethod
    def _workspace_tool_view(factory: Any, workspace: DirectWorkspaceSession | None) -> Any | None:
        if workspace is None:
            return None
        access = getattr(factory, "workspace_access", None)
        if access == "source":
            try:
                return workspace.source_view()
            except WorkspaceStorageError:
                raise AllocationError(
                    "workspace_mode_unsupported",
                    "Trace findings require overlay workspace mode",
                    retryable=False,
                    status_code=422,
                ) from None
        if access == "read":
            return workspace.reader_view()
        if access == "write":
            return workspace.writer_view()
        if access == "changes":
            try:
                return workspace.changes_view()
            except WorkspaceStorageError:
                raise AllocationError(
                    "workspace_mode_unsupported",
                    "selected Toolset requires overlay workspace mode",
                    retryable=False,
                    status_code=422,
                ) from None
        if access is not None:
            raise AllocationError(
                "invalid_toolset_factory",
                "Toolset declares an invalid project workspace access mode",
                retryable=False,
                status_code=422,
            )
        return None

    @staticmethod
    def _assert_safe_handle(
        handle: WorkerHandle,
        spec: AllocationSpec,
        workspace: AllocationWorkspace,
        a2a_base_url: str,
    ) -> None:
        wire = handle.model_dump(mode="json", by_alias=True)
        encoded = json.dumps(wire, ensure_ascii=False)
        endpoint = f"{a2a_base_url.rstrip('/')}/private/v1/allocations/{spec.allocation_id}/a2a"
        card_values, card_keys = _untrusted_agent_card_text(
            handle.agent_card,
            allocation_id=spec.allocation_id,
            logical_agent_name=spec.logical_agent_name,
            description=spec.agent_template.description,
            version=spec.agent_template.ref.version,
            endpoint=endpoint,
        )
        private_values = runtime_setting_values(spec.runtime_settings)
        long_values = substring_values(private_values)
        leaked_value = any(contains_private_value(value, private_values) for value in card_values)
        leaked_key = any(secret in key for key in card_keys for secret in long_values)
        if leaked_value or leaked_key or str(workspace.path) in encoded:
            raise AllocationError(
                "unsafe_worker_handle",
                "Worker runtime exposed private allocation data in its Agent Card",
                retryable=False,
                status_code=500,
            )
