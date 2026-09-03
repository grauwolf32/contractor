"""Idempotent lifecycle for the Runtime Agent's one allocation slot."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from contractor_runtime.adapters import (
    AdapterHandles,
    AdapterPreparationError,
    AllocationAdapterHost,
)
from contractor_runtime.agent_skills.runtime import AgentSkillPreparationError
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    AgentStateSnapshot,
    AllocationFinalReport,
    AllocationFinalResponse,
    AllocationSpec,
    AllocationSpecV2,
    FinalizeAllocationRequest,
    PrepareAllocationResponse,
    ReleaseAllocationRequest,
    RuntimeReport,
    RuntimeSettings,
    RuntimeSettingsV2,
    TerminationError,
    WorkerHandle,
)
from contractor_runtime.digests import (
    TemplateDigestMismatch,
    verify_model_policy_digest,
    verify_template_digests,
)
from contractor_runtime.factories import (
    FactoryRegistry,
    SandboxFactory,
    ToolInstance,
    WorkerBuildContext,
    WorkerRuntime,
    WorkerRuntimeFactory,
)
from contractor_runtime.metrics import MetricsState
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    WorkspacePreparationError,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.worker_state import WorkerStateStore
from contractor_runtime.workspace import AllocationWorkspace

RESERVED_NAMESPACES = frozenset({"inputs", "outputs", "skills"})


class AllocationError(Exception):
    """Stable private-API error which never includes allocation secrets."""

    def __init__(self, code: str, message: str, *, retryable: bool, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code

    def payload(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


# Backward-compatible import surface for toolset tests and embedders. The
# concrete allocation object is now the revisioned State store.
WorkerState = WorkerStateStore


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
    prepare_response: PrepareAllocationResponse | None = None
    termination_kind: str | None = None
    termination_id: str | None = None
    terminal_response: AllocationFinalResponse | None = None
    release_prepared: bool = False
    release_cleanup_task: asyncio.Task[None] | None = field(default=None, repr=False)


class AllocationService:
    """Constructs, drains, and erases exactly one in-process Worker context."""

    def __init__(
        self,
        state: RuntimeState,
        factories: FactoryRegistry,
        capabilities: CapabilitySnapshot | None = None,
        *,
        a2a_base_url: str,
        private_bypass_urls: Sequence[str] = (),
        now: Callable[[], datetime] | None = None,
        force_exit: Callable[[int], Any] = os._exit,
    ) -> None:
        self._state = state
        self._factories = factories
        # Tests and embedded callers may inject an already frozen snapshot.
        # The process entry point resolves it from RuntimeState after the
        # private listener is ready and startup discovery has completed.
        self._capabilities = capabilities
        self._a2a_base_url = a2a_base_url.rstrip("/")
        self._private_bypass_hosts = _url_hosts((*private_bypass_urls, a2a_base_url))
        self._now = now or (lambda: datetime.now(UTC))
        self._force_exit = force_exit
        self._lock = asyncio.Lock()
        self._context: _AllocationContext | None = None
        self._released_allocation_id: str | None = None
        self._fingerprint_key = os.urandom(32)

    async def snapshot(self) -> AllocationSnapshot | None:
        async with self._lock:
            context = self._context
            if context is None:
                return None
            state = await self._state.snapshot()
            return AllocationSnapshot(
                allocation_id=context.allocation_id,
                stage_execution_id=context.stage_execution_id,
                process_state=state.process_state,
                workspace=str(context.workspace.path),
                tool_names=tuple(sorted(context.tools)),
                has_runtime_settings=context.runtime_settings is not None,
                has_worker=context.worker is not None,
                runtime_adapter_refs=context.adapter_host.refs,
                has_project_workspace=context.project_workspace is not None,
            )

    async def active_a2a_application(self, allocation_id: str) -> Any | None:
        """Resolve A2A only while this exact allocation remains active."""

        async with self._lock:
            context = self._context
            if context is None or context.allocation_id != allocation_id or context.worker is None:
                return None
            state = await self._state.snapshot()
            if (
                state.process_state is not ProcessState.ALLOCATED
                or state.allocation_id != allocation_id
            ):
                return None
            return getattr(context.worker, "a2a_application", None)

    async def agent_state_snapshot(self, allocation_id: str) -> AgentStateSnapshot:
        """Return only the exact active Worker's Contractor-owned State."""

        async with self._lock:
            context = self._require_context(allocation_id)
            state = await self._state.snapshot()
            worker = context.worker
            if (
                worker is None
                or state.process_state is not ProcessState.ALLOCATED
                or state.allocation_id != allocation_id
            ):
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=True,
                    status_code=409,
                )
            try:
                return await worker.agent_state_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception:
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=True,
                    status_code=503,
                ) from None

    async def prepare(self, spec: AllocationSpec) -> PrepareAllocationResponse:
        async with self._lock:
            fingerprint = _spec_fingerprint(spec, self._fingerprint_key)
            if self._context is not None:
                context = self._context
                if (
                    context.allocation_id == spec.allocation_id
                    and context.stage_execution_id == spec.stage_execution_id
                ):
                    if context.fingerprint != fingerprint or context.prepare_response is None:
                        raise _conflict("prepare request differs from the active allocation")
                    return context.prepare_response
                raise _conflict("Runtime Agent already owns another allocation")

            state = await self._state.snapshot()
            if state.process_state is not ProcessState.IDLE:
                raise _conflict("Runtime Agent slot is not idle")
            self._validate_spec(spec)

            sandbox = self._sandbox_factory(spec)
            runtime_factory = self._runtime_factory(spec)
            adapter_host: AllocationAdapterHost | None = None
            workspace: AllocationWorkspace | None = None
            project_workspace: DirectWorkspaceSession | None = None
            tools: dict[str, ToolInstance] = {}
            worker_state: WorkerStateStore | None = None
            worker: WorkerRuntime | None = None
            try:
                adapter_deadline = min(
                    spec.lease_expires_at,
                    self._now() + timedelta(seconds=spec.runtime_settings.request_timeout_seconds),
                )
                if isinstance(spec, AllocationSpecV2):
                    adapter_host = await AllocationAdapterHost.create(
                        spec,
                        self._factories.runtime_adapters,
                        deadline=adapter_deadline,
                        now=self._now,
                        private_bypass_hosts=self._private_bypass_hosts,
                    )
                else:
                    adapter_host = AllocationAdapterHost.empty()
                workspace = await sandbox.prepare()
                if isinstance(spec, AllocationSpecV2) and spec.workspace is not None:
                    assert self._factories.workspace_provider is not None
                    assert self._factories.artifact_client_factory is not None
                    artifact_reader = self._factories.artifact_client_factory(
                        spec.allocation_id, spec.runtime_settings
                    )
                    remaining = min(
                        spec.runtime_settings.request_timeout_seconds,
                        max(0.0, (spec.lease_expires_at - self._now()).total_seconds()),
                    )
                    project_workspace = await hydrate_workspace(
                        provider=self._factories.workspace_provider,
                        spec=spec.workspace,
                        artifact_reader=artifact_reader,
                        allocation_id=spec.allocation_id,
                        timeout_seconds=remaining,
                    )
                worker_state = WorkerStateStore()
                tools = await self._create_tools(
                    spec,
                    workspace,
                    project_workspace,
                    worker_state,
                    adapter_host.handles,
                )
                worker = await runtime_factory.create(
                    WorkerBuildContext(
                        allocation_id=spec.allocation_id,
                        run_id=spec.run_id,
                        stage_execution_id=spec.stage_execution_id,
                        logical_agent_name=spec.logical_agent_name,
                        namespace=spec.namespace,
                        description=spec.agent_template.description,
                        instruction=spec.agent_template.instructions.text,
                        card_version=spec.agent_template.ref.version,
                        model_policy=spec.model_policy,
                        workspace=workspace,
                        tools=tools,
                        state=worker_state,
                        a2a_base_url=self._a2a_base_url,
                        runtime_settings=spec.runtime_settings,
                        summarizer=spec.agent_template.summarizer,
                        adapter_handles=adapter_host.handles.for_worker(),
                        resolved_skills=tuple(spec.resolved_skills),
                        project_workspace=project_workspace,
                        workspace_export=(
                            spec.workspace.export
                            if isinstance(spec, AllocationSpecV2) and spec.workspace is not None
                            else None
                        ),
                    )
                )
                handle = WorkerHandle(
                    allocationId=spec.allocation_id,
                    agentTemplateRef=spec.agent_template.ref,
                    workerRuntimeRef=spec.agent_template.runtime,
                    agentCard=dict(worker.agent_card),
                    leaseExpiresAt=spec.lease_expires_at,
                )
                self._assert_safe_handle(handle, spec.runtime_settings, workspace)
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
                    workspace=workspace,
                    sandbox=sandbox,
                    project_workspace=project_workspace,
                    adapter_host=adapter_host,
                    tools=tools,
                    worker_state=worker_state,
                    runtime_settings=spec.runtime_settings,
                    worker=worker,
                    prepare_response=response,
                )
                await self._state.commit_allocation(spec.allocation_id)
                self._context = context
                self._released_allocation_id = None
                return response
            except AdapterPreparationError as error:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                if not error.cleanup_confirmed:
                    await self._state.fence_allocation(spec.allocation_id)
                    self._force_exit(70)
                raise AllocationError(
                    "runtime_adapter_prepare_failed",
                    "allocation Runtime adapter preparation failed",
                    retryable=error.retryable,
                    status_code=503,
                ) from None
            except AgentSkillPreparationError as error:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                raise AllocationError(
                    error.code,
                    f"allocation Agent Skill preparation failed ({error.code})",
                    retryable=error.retryable,
                    status_code=error.status_code,
                ) from None
            except WorkspacePreparationError as error:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                if not error.cleanup_confirmed:
                    await self._state.fence_allocation(spec.allocation_id)
                    self._force_exit(70)
                raise AllocationError(
                    error.code,
                    f"allocation workspace preparation failed ({error.code})",
                    retryable=error.retryable,
                    status_code=error.status_code,
                ) from None
            except AllocationError:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                raise
            except asyncio.CancelledError:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                raise
            except Exception as error:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    workspace,
                    project_workspace,
                    tools,
                    worker,
                    adapter_host,
                )
                raise AllocationError(
                    "allocation_preparation_failed",
                    f"allocation resource preparation failed ({type(error).__name__})",
                    retryable=True,
                    status_code=503,
                ) from None

    async def finalize(self, request: FinalizeAllocationRequest) -> AllocationFinalResponse:
        return await self._terminate(
            allocation_id=request.allocation_id,
            kind="finalize",
            operation_id=request.finalization_id,
            deadline=request.deadline,
            reason=None,
        )

    async def abort(self, request: AbortAllocationRequest) -> AllocationFinalResponse:
        return await self._terminate(
            allocation_id=request.allocation_id,
            kind="abort",
            operation_id=request.abort_id,
            deadline=request.deadline,
            reason=request.reason,
        )

    async def release(self, request: ReleaseAllocationRequest) -> None:
        async with self._lock:
            if self._context is None:
                # Release is an idempotent cleanup command.  A restarted
                # process cannot distinguish an allocation it already cleaned
                # from one that belonged to its predecessor, but in either
                # case there is no local allocation state left to remove.
                return
            context = self._require_context(request.allocation_id)
            state = await self._state.snapshot()
            if state.process_state not in {ProcessState.DRAINING, ProcessState.FENCED}:
                raise _conflict("allocation must be draining or fenced before release")
            if context.worker is not None:
                raise _conflict("Worker must be stopped before release")
            if context.release_prepared:
                return
            await self._prepare_release_cleanup(context)
            await self._state.fence_allocation(context.allocation_id)

    async def confirm_release(self, allocation_id: str | None) -> None:
        """Apply a heartbeat-confirmed authoritative release.

        The private release endpoint only prepares local cleanup. Keeping this
        second edge separate makes a lost HTTP response safe: the slot remains
        fenced and the same release can be retried.
        """

        async with self._lock:
            context = self._context
            if context is None:
                if allocation_id is not None and self._released_allocation_id == allocation_id:
                    return
                await self._state.confirm_release(allocation_id)
                return
            if allocation_id != context.allocation_id:
                raise _conflict("release action identifies another allocation")
            if context.worker is not None:
                raise _conflict("active Worker must be drained before release")
            if not context.release_prepared:
                await self._prepare_release_cleanup(context)
            await self._state.confirm_release(context.allocation_id)
            self._released_allocation_id = context.allocation_id
            context.terminal_response = None
            self._context = None

    async def _prepare_release_cleanup(self, context: _AllocationContext) -> None:
        timeout_seconds = (
            context.runtime_settings.request_timeout_seconds
            if context.runtime_settings is not None
            else 5
        )
        cleanup_task = context.release_cleanup_task
        if cleanup_task is not None and cleanup_task.done():
            try:
                cleanup_task.result()
            except (Exception, asyncio.CancelledError):
                # The completed attempt may have made partial, idempotent
                # progress. A retry gets a fresh outer deadline and resumes
                # from the resources which remain in the context.
                context.release_cleanup_task = None
                cleanup_task = None
            else:
                return

        if cleanup_task is None:
            deadline = self._now() + timedelta(seconds=timeout_seconds)
            cleanup_task = asyncio.create_task(
                self._run_release_cleanup(context, deadline),
                name=f"allocation-release-cleanup-{context.allocation_id}",
            )
            cleanup_task.add_done_callback(_consume_background_task)
            context.release_cleanup_task = cleanup_task

        try:
            await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=float(timeout_seconds))
        except asyncio.CancelledError:
            await self._state.fence_allocation(context.allocation_id)
            current = asyncio.current_task()
            if cleanup_task.cancelled() and (current is None or not current.cancelling()):
                context.release_cleanup_task = None
                raise AllocationError(
                    "allocation_cleanup_failed",
                    "allocation cleanup failed (CancelledError)",
                    retryable=True,
                    status_code=503,
                ) from None
            raise
        except Exception as error:
            if cleanup_task.done():
                context.release_cleanup_task = None
            await self._state.fence_allocation(context.allocation_id)
            raise AllocationError(
                "allocation_cleanup_failed",
                f"allocation cleanup failed ({type(error).__name__})",
                retryable=True,
                status_code=503,
            ) from None

    async def _run_release_cleanup(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        await _close_tools(context.tools)
        if context.project_workspace is not None:
            await self._cleanup_project_workspace(context.project_workspace)
            context.project_workspace = None
        await context.sandbox.cleanup(context.workspace)
        await context.adapter_host.rollback(deadline=deadline, now=self._now)

        context.tools.clear()
        context.worker_state = None
        context.runtime_settings = None
        context.prepare_response = None
        context.release_prepared = True

    async def expire_control_lease(self, shutdown_grace_seconds: float) -> None:
        async with self._lock:
            context = self._context
            if context is None:
                await self._state.fence_control_lease()
                return
            await self._stop_worker_for_fence(
                context,
                shutdown_grace_seconds,
                TerminationError(
                    code="control_lease_expired",
                    message="Runtime Agent confirmed control lease expired",
                    retryable=True,
                ),
            )

    async def reconcile_drain(self, allocation_id: str, shutdown_grace_seconds: float) -> None:
        async with self._lock:
            context = self._context
            if context is None or context.allocation_id != allocation_id:
                return
            await self._stop_worker_for_fence(
                context,
                shutdown_grace_seconds,
                TerminationError(
                    code="control_plane_reconciliation",
                    message="Control Plane requested allocation reconciliation",
                    retryable=True,
                ),
            )

    def _validate_spec(self, spec: AllocationSpec) -> None:
        capabilities = self._capabilities or self._state.capabilities
        if spec.namespace in RESERVED_NAMESPACES:
            raise AllocationError(
                "invalid_agent_namespace",
                "agent allocation cannot use a Run-reserved namespace",
                retryable=False,
                status_code=422,
            )
        if spec.lease_expires_at <= self._now():
            raise AllocationError(
                "allocation_lease_expired",
                "allocation lease has already expired",
                retryable=True,
                status_code=409,
            )
        try:
            verify_template_digests(spec.agent_template)
            verify_model_policy_digest(spec.model_policy)
        except TemplateDigestMismatch:
            raise AllocationError(
                "template_digest_mismatch",
                "resolved AgentTemplate integrity verification failed",
                retryable=False,
                status_code=422,
            ) from None

        runtime = spec.agent_template.runtime
        runtime_ref = f"{runtime.runtime_id}@{runtime.version}"
        if not capabilities.supports_runtime(runtime_ref):
            raise AllocationError(
                "unsupported_worker_runtime",
                "AgentTemplate selects an unavailable WorkerRuntime",
                retryable=False,
                status_code=422,
            )
        runtime_factory = self._factories.worker_runtimes.get(runtime_ref)
        if spec.resolved_skills and not bool(
            getattr(runtime_factory, "supports_agent_skills", False)
        ):
            raise AllocationError(
                "skill_runtime_unsupported",
                "selected WorkerRuntime does not support Agent Skills",
                retryable=False,
                status_code=422,
            )
        sandbox = spec.agent_template.sandbox_profile
        sandbox_ref = f"{sandbox.sandbox_profile_id}@{sandbox.version}"
        if not capabilities.supports_sandbox(sandbox_ref):
            raise AllocationError(
                "unsupported_sandbox_profile",
                "AgentTemplate selects an unavailable SandboxProfile",
                retryable=False,
                status_code=422,
            )
        required_infrastructure_channels: set[str] = set()
        for selection in spec.agent_template.toolsets:
            toolset_ref = f"{selection.ref.toolset_id}@{selection.ref.version}"
            if not capabilities.has_toolset(toolset_ref):
                raise AllocationError(
                    "unsupported_toolset",
                    "AgentTemplate selects an unavailable Toolset",
                    retryable=False,
                    status_code=422,
                )
            if not capabilities.supports_tools(toolset_ref, selection.tools):
                raise AllocationError(
                    "unsupported_tool",
                    "AgentTemplate selects an unavailable tool",
                    retryable=False,
                    status_code=422,
                )
            factory = self._factories.toolsets.get(toolset_ref)
            if factory is not None:
                required_infrastructure_channels.update(
                    channel
                    for tool in selection.tools
                    for channel in factory.infrastructure_channels.get(tool, frozenset())
                )
            if getattr(factory, "requires_workspace", False) and (
                not isinstance(spec, AllocationSpecV2) or spec.workspace is None
            ):
                raise AllocationError(
                    "workspace_required",
                    "selected Toolset requires an allocation project workspace",
                    retryable=False,
                    status_code=422,
                )
            if (
                toolset_ref == "workspace-changes@1"
                and isinstance(spec, AllocationSpecV2)
                and spec.workspace is not None
                and spec.workspace.mode != "overlay"
            ):
                raise AllocationError(
                    "workspace_mode_unsupported",
                    "selected Toolset requires overlay workspace mode",
                    retryable=False,
                    status_code=422,
                )

        if isinstance(spec, AllocationSpecV2):
            if (
                "caido-graphql-client" in required_infrastructure_channels
                and spec.runtime_settings.caido is None
            ):
                raise AllocationError(
                    "caido_not_configured",
                    "selected Caido tools require resolved Runtime configuration",
                    retryable=False,
                    status_code=422,
                )
            if spec.workspace is not None:
                provider = self._factories.workspace_provider
                if (
                    provider is None
                    or capabilities.workspace is None
                    or capabilities.workspace != provider.capability
                    or not capabilities.supports_workspace_mode(spec.workspace.mode)
                ):
                    raise AllocationError(
                        "workspace_mode_unsupported",
                        "allocation workspace mode is not available on this Runtime Agent",
                        retryable=False,
                        status_code=422,
                    )
                if self._factories.artifact_client_factory is None:
                    raise AllocationError(
                        "workspace_source_unavailable",
                        "allocation workspace Artifact client is unavailable",
                        retryable=True,
                        status_code=503,
                    )
            required = set(spec.resolved_runtime_config_provenance.runtime_adapters)
            configured: set[str] = set()
            if spec.runtime_settings.telemetry is not None:
                configured.add(spec.runtime_settings.telemetry.adapter)
            if spec.runtime_settings.http_proxy is not None:
                configured.add(spec.runtime_settings.http_proxy.adapter)
            if spec.runtime_settings.caido is not None:
                configured.add(spec.runtime_settings.caido.adapter)
            if (
                required != configured
                or not capabilities.supports_runtime_adapters(required)
                or not required <= set(self._factories.runtime_adapters)
            ):
                raise AllocationError(
                    "unsupported_runtime_adapter",
                    "allocation Runtime adapter settings do not match frozen capabilities",
                    retryable=False,
                    status_code=422,
                )

    def _runtime_factory(self, spec: AllocationSpec) -> WorkerRuntimeFactory:
        ref = f"{spec.agent_template.runtime.runtime_id}@{spec.agent_template.runtime.version}"
        factory = self._factories.worker_runtimes.get(ref)
        if factory is None:
            raise AllocationError(
                "unsupported_worker_runtime",
                "AgentTemplate selects an unsupported WorkerRuntime",
                retryable=False,
                status_code=422,
            )
        return factory

    def _sandbox_factory(self, spec: AllocationSpec) -> SandboxFactory:
        selected = spec.agent_template.sandbox_profile
        ref = f"{selected.sandbox_profile_id}@{selected.version}"
        factory = self._factories.sandbox_profiles.get(ref)
        if factory is None:
            raise AllocationError(
                "unsupported_sandbox_profile",
                "AgentTemplate selects an unsupported SandboxProfile",
                retryable=False,
                status_code=422,
            )
        return factory

    async def _create_tools(
        self,
        spec: AllocationSpec,
        workspace: AllocationWorkspace,
        project_workspace: DirectWorkspaceSession | None,
        worker_state: WorkerStateStore,
        adapter_handles: AdapterHandles,
    ) -> dict[str, ToolInstance]:
        result: dict[str, ToolInstance] = {}
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
                selected=selection.tools,
                allocation_id=spec.allocation_id,
                run_id=spec.run_id,
                namespace=spec.namespace,
                runtime_settings=spec.runtime_settings,
                workspace=workspace,
                state=worker_state,
                adapter_handles=adapter_handles.for_tool_channels(channels),
                project_workspace=self._workspace_tool_view(factory, project_workspace),
            )
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
            collision = set(result) & set(created)
            if collision:
                raise AllocationError(
                    "duplicate_visible_tool",
                    "Toolset factories produced a duplicate model-visible tool",
                    retryable=False,
                    status_code=422,
                )
            result.update(created)
        return result

    @staticmethod
    def _workspace_tool_view(factory: Any, workspace: DirectWorkspaceSession | None) -> Any | None:
        if workspace is None:
            return None
        access = getattr(factory, "workspace_access", None)
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

    async def _terminate(
        self,
        *,
        allocation_id: str,
        kind: str,
        operation_id: str,
        deadline: datetime,
        reason: TerminationError | None,
    ) -> AllocationFinalResponse:
        async with self._lock:
            context = self._require_context(allocation_id)
            if (
                context.termination_kind == "lease"
                and context.worker is None
                and context.terminal_response is not None
            ):
                # A local confirmed-lease loss may stop the Worker before the
                # Scheduler observes the matching Control Plane loss. Bind the
                # cached report to the first authoritative terminal operation
                # instead of trying to stop an already absent Worker.
                context.termination_kind = kind
                context.termination_id = operation_id
                return context.terminal_response
            if context.termination_kind is not None:
                if context.termination_kind == kind and context.termination_id == operation_id:
                    if context.terminal_response is None:
                        raise _conflict("allocation termination is still incomplete")
                    return context.terminal_response
                raise _conflict("allocation already has a different terminal operation")

            context.termination_kind = kind
            context.termination_id = operation_id
            await self._state.begin_draining(allocation_id)
            worker = context.worker
            if worker is None:
                raise _conflict("allocation Worker is already absent")
            remaining = (deadline - self._now()).total_seconds()
            stop_task: asyncio.Task[None] | None = None
            try:
                if remaining <= 0:
                    raise TimeoutError
                operation = (
                    worker.finalize(deadline) if kind == "finalize" else worker.abort(deadline)
                )
                stop_task = asyncio.create_task(operation, name=f"worker-{kind}-{allocation_id}")
                done, _ = await asyncio.wait({stop_task}, timeout=remaining)
                if not done:
                    stop_task.cancel()
                    stop_task.add_done_callback(_consume_background_task)
                    raise TimeoutError
                await stop_task
            except asyncio.CancelledError:
                if stop_task is not None and not stop_task.done():
                    stop_task.cancel()
                    stop_task.add_done_callback(_consume_background_task)
                await self._state.fence_allocation(allocation_id)
                self._force_exit(70)
                raise
            except Exception as error:
                await self._state.fence_allocation(allocation_id)
                self._force_exit(70)
                raise AllocationError(
                    "worker_stop_unconfirmed",
                    f"in-process Worker stop could not be guaranteed ({type(error).__name__})",
                    retryable=False,
                    status_code=503,
                ) from None

            context.worker = None
            await self._stop_tools_or_exit(context, deadline)
            await self._stop_adapters_or_exit(context, deadline)
            if kind == "abort":
                await self._discard_project_workspace_or_exit(context, deadline)
            response = AllocationFinalResponse(
                apiVersion=API_VERSION,
                report=_build_report(context, self._now(), reason),
            )
            context.terminal_response = response
            return response

    async def _stop_worker_for_fence(
        self,
        context: _AllocationContext,
        timeout_seconds: float,
        reason: TerminationError,
    ) -> None:
        """Stop a Worker after authority is lost without making the slot idle."""

        if timeout_seconds <= 0:
            raise ValueError("Worker shutdown grace must be positive")
        if context.worker is None:
            if context.release_cleanup_task is not None:
                await self._state.fence_allocation(context.allocation_id)
                return
            deadline = self._now() + timedelta(seconds=timeout_seconds)
            await self._stop_tools_or_exit(context, deadline)
            await self._stop_adapters_or_exit(context, deadline)
            await self._discard_project_workspace_or_exit(context, deadline)
            await self._state.fence_allocation(context.allocation_id)
            return

        state = await self._state.snapshot()
        if state.process_state is ProcessState.ALLOCATED:
            await self._state.begin_draining(context.allocation_id)
        worker = context.worker
        deadline = self._now() + timedelta(seconds=timeout_seconds)
        stop_task = asyncio.create_task(
            worker.abort(deadline), name=f"worker-lease-abort-{context.allocation_id}"
        )
        try:
            done, _ = await asyncio.wait({stop_task}, timeout=timeout_seconds)
            if not done:
                stop_task.cancel()
                stop_task.add_done_callback(_consume_background_task)
                raise TimeoutError
            await stop_task
        except asyncio.CancelledError:
            if not stop_task.done():
                stop_task.cancel()
                stop_task.add_done_callback(_consume_background_task)
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise
        except Exception as error:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise AllocationError(
                "worker_stop_unconfirmed",
                f"in-process Worker stop could not be guaranteed ({type(error).__name__})",
                retryable=False,
                status_code=503,
            ) from None

        context.worker = None
        await self._stop_tools_or_exit(context, deadline)
        await self._stop_adapters_or_exit(context, deadline)
        await self._discard_project_workspace_or_exit(context, deadline)
        context.termination_kind = "lease"
        context.termination_id = None
        context.terminal_response = AllocationFinalResponse(
            apiVersion=API_VERSION,
            report=_build_report(context, self._now(), reason),
        )
        await self._state.fence_allocation(context.allocation_id)

    async def _stop_tools_or_exit(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        try:
            await _await_before_deadline(
                lambda: _close_tools(context.tools),
                deadline=deadline,
                now=self._now,
            )
        except asyncio.CancelledError:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise
        except Exception:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise AllocationError(
                "tool_cleanup_unconfirmed",
                "allocation Toolset cleanup could not be guaranteed",
                retryable=False,
                status_code=503,
            ) from None

    async def _stop_adapters_or_exit(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        try:
            await context.adapter_host.terminate(deadline=deadline, now=self._now)
        except asyncio.CancelledError:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise
        except Exception:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise AllocationError(
                "runtime_adapter_close_unconfirmed",
                "allocation Runtime adapter close could not be guaranteed",
                retryable=False,
                status_code=503,
            ) from None

    async def _discard_project_workspace_or_exit(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        project_workspace = context.project_workspace
        if project_workspace is None:
            return
        try:
            await _await_before_deadline(
                lambda: self._cleanup_project_workspace(project_workspace),
                deadline=deadline,
                now=self._now,
            )
        except asyncio.CancelledError:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise
        except Exception:
            await self._state.fence_allocation(context.allocation_id)
            self._force_exit(70)
            raise AllocationError(
                "workspace_cleanup_unconfirmed",
                "allocation project workspace cleanup could not be guaranteed",
                retryable=False,
                status_code=503,
            ) from None
        context.project_workspace = None

    async def _cleanup_project_workspace(
        self,
        project_workspace: DirectWorkspaceSession,
    ) -> None:
        await project_workspace.close()
        provider = self._factories.workspace_provider
        if provider is None:
            raise RuntimeError("project workspace provider is unavailable")
        await provider.cleanup(project_workspace.storage)

    async def _rollback_prepare(
        self,
        spec: AllocationSpec,
        sandbox: SandboxFactory,
        workspace: AllocationWorkspace | None,
        project_workspace: DirectWorkspaceSession | None,
        tools: Mapping[str, ToolInstance],
        worker: WorkerRuntime | None,
        adapter_host: AllocationAdapterHost | None,
    ) -> None:
        failed = False
        cancelled: asyncio.CancelledError | None = None
        deadline = self._now() + timedelta(seconds=spec.runtime_settings.request_timeout_seconds)

        async def attempt(operation: Callable[[], Awaitable[None]]) -> None:
            nonlocal failed, cancelled
            try:
                await _await_before_deadline(operation, deadline=deadline, now=self._now)
            except asyncio.CancelledError as error:
                failed = True
                cancelled = error
            except Exception:
                failed = True

        if worker is not None:
            await attempt(lambda: worker.abort(deadline))
        await attempt(lambda: _close_tools(tools))
        if project_workspace is not None:
            await attempt(lambda: self._cleanup_project_workspace(project_workspace))
        if workspace is not None:
            await attempt(lambda: sandbox.cleanup(workspace))
        if adapter_host is not None:
            await attempt(
                lambda: adapter_host.rollback(
                    deadline=deadline,
                    now=self._now,
                )
            )
        if failed:
            await self._state.fence_allocation(spec.allocation_id)
            self._force_exit(70)
        if cancelled is not None:
            raise cancelled

    def _require_context(self, allocation_id: str) -> _AllocationContext:
        if self._context is None or self._context.allocation_id != allocation_id:
            raise _not_found()
        return self._context

    @staticmethod
    def _assert_safe_handle(
        handle: WorkerHandle,
        settings: RuntimeSettings,
        workspace: AllocationWorkspace,
    ) -> None:
        wire = handle.model_dump(mode="json", by_alias=True)
        encoded = json.dumps(wire, ensure_ascii=False)
        handle_strings = _nested_strings(wire)
        private_values = _runtime_setting_values(settings)
        leaked = any(
            value in handle_strings or (len(value.encode("utf-8")) >= 16 and value in encoded)
            for value in private_values
        )
        if leaked or str(workspace.path) in encoded:
            raise AllocationError(
                "unsafe_worker_handle",
                "Worker runtime exposed private allocation data in its Agent Card",
                retryable=False,
                status_code=500,
            )


def _build_report(
    context: _AllocationContext,
    finished_at: datetime,
    reason: TerminationError | None,
) -> AllocationFinalReport:
    metrics = context.worker_state.metrics if context.worker_state is not None else MetricsState()
    duration_ms = max(0, int((finished_at - context.started_at).total_seconds() * 1000))
    errors = (reason,) if reason is not None else ()
    stop_reason = reason.code if reason is not None else "finalized"
    worker = metrics.build_report(
        report_id=f"worker-{context.allocation_id}",
        duration_ms=duration_ms,
        extra_errors=errors,
    )
    return AllocationFinalReport(
        reportId=f"allocation-final-{context.allocation_id}",
        allocationId=context.allocation_id,
        startedAt=context.started_at,
        finishedAt=finished_at,
        worker=worker,
        runtime=RuntimeReport(
            complete=True,
            durationMs=duration_ms,
            stopReason=stop_reason,
            adapters=context.adapter_host.report_metrics(),
        ),
    )


async def _close_tools(tools: Mapping[str, ToolInstance]) -> None:
    for name in reversed(tuple(tools)):
        await tools[name].close()
        if isinstance(tools, MutableMapping):
            del tools[name]


async def _await_before_deadline(
    operation: Callable[[], Awaitable[None]],
    *,
    deadline: datetime,
    now: Callable[[], datetime],
) -> None:
    remaining = (deadline - now()).total_seconds()
    if remaining <= 0:
        raise TimeoutError
    task = asyncio.create_task(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining)
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_background_task)
        raise
    if not done:
        task.cancel()
        task.add_done_callback(_consume_background_task)
        raise TimeoutError
    await task


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    task.exception()


def _spec_fingerprint(spec: AllocationSpec, key: bytes) -> str:
    encoded = spec.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _runtime_setting_values(settings: RuntimeSettings) -> tuple[str, ...]:
    values = [settings.llm_gateway_url, settings.artifact_api_url]
    token = settings.llm_gateway_token
    if token is not None:
        values.append(token.get_secret_value())
    if isinstance(settings, RuntimeSettingsV2):
        if settings.telemetry is not None:
            values.append(settings.telemetry.endpoint)
            values.extend(
                secret.get_secret_value() for secret in settings.telemetry.headers.values()
            )
        if settings.http_proxy is not None:
            proxy = settings.http_proxy
            values.append(proxy.proxy_url)
            if proxy.basic_auth is not None:
                values.extend(
                    (
                        proxy.basic_auth.username.get_secret_value(),
                        proxy.basic_auth.password.get_secret_value(),
                    )
                )
            if proxy.bearer_token is not None:
                values.append(proxy.bearer_token.get_secret_value())
            if proxy.ca_bundle_pem is not None:
                values.append(proxy.ca_bundle_pem)
        if settings.caido is not None:
            caido = settings.caido
            values.append(caido.endpoint)
            if caido.bearer_token is not None:
                values.append(caido.bearer_token.get_secret_value())
            if caido.ca_bundle_pem is not None:
                values.append(caido.ca_bundle_pem)
    return tuple(value for value in values if value)


def _url_hosts(urls: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    for value in urls:
        parsed = urlsplit(value)
        if parsed.hostname is not None:
            values.add(parsed.hostname)
        if parsed.netloc:
            values.add(parsed.netloc)
    return tuple(sorted(values))


def _nested_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        return {item for nested in value.values() for item in _nested_strings(nested)}
    if isinstance(value, (list, tuple)):
        return {item for nested in value for item in _nested_strings(nested)}
    return set()


def _conflict(message: str) -> AllocationError:
    return AllocationError("allocation_conflict", message, retryable=False, status_code=409)


def _not_found() -> AllocationError:
    return AllocationError(
        "allocation_not_found",
        "allocation is not active on this Runtime Agent",
        retryable=False,
        status_code=404,
    )
