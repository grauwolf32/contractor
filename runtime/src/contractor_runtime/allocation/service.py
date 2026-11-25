"""Bounded admission and independent allocation lifecycle ownership."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError

from contractor_runtime.adapters import (
    AdapterPreparationError,
    AllocationAdapterHost,
)
from contractor_runtime.adapters.http_transport import HTTPTransport
from contractor_runtime.agent_skills.runtime import AgentSkillPreparationError
from contractor_runtime.allocation.cleanup import (
    AllocationResourceCleanup,
    _await_before_deadline,
    _close_tools,
)
from contractor_runtime.allocation.context import _AllocationContext
from contractor_runtime.allocation.errors import AllocationError, _conflict, _not_found
from contractor_runtime.allocation.identity import _spec_fingerprint
from contractor_runtime.allocation.preparation import AllocationPreparation
from contractor_runtime.allocation.redaction import _url_hosts
from contractor_runtime.allocation.reports import _build_report
from contractor_runtime.allocation.validation import validate_spec
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.contracts import (
    API_VERSION,
    AbortAllocationRequest,
    AgentStateSnapshot,
    AllocationFinalResponse,
    AllocationSpec,
    FinalizeAllocationRequest,
    PrepareAllocationResponse,
    ReleaseAllocationRequest,
    TerminationError,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.factories import (
    FactoryRegistry,
    SandboxFactory,
    ToolInstance,
    WorkerRuntime,
    WorkerRuntimeFactory,
)
from contractor_runtime.log import MAX_LOG_EXTRA_TEXT
from contractor_runtime.projectfs import (
    DirectWorkspaceSession,
    WorkspacePreparationError,
)
from contractor_runtime.sandbox.lifecycle import PreparedExecution
from contractor_runtime.sitemap_collection import SitemapCollector
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.telemetry.resources import ProcessReading, ResourceCollector
from contractor_runtime.threads import consume_task_exception
from contractor_runtime.toolsets.http.errors import HTTPToolError
from contractor_runtime.toolsets.security_findings.collection import FindingsError
from contractor_runtime.worker.adk_caches import release_adk_tool_caches
from contractor_runtime.workspace import AllocationWorkspace

logger = logging.getLogger(__name__)


def _preparation_error(error: Exception) -> AllocationError:
    if isinstance(error, AdapterPreparationError):
        return AllocationError(
            "runtime_adapter_prepare_failed",
            "allocation Runtime adapter preparation failed",
            retryable=error.retryable,
            status_code=503,
        )
    if isinstance(error, AgentSkillPreparationError):
        return AllocationError(
            error.code,
            f"allocation Agent Skill preparation failed ({error.code})",
            retryable=error.retryable,
            status_code=error.status_code,
        )
    if isinstance(error, WorkspacePreparationError):
        return AllocationError(
            error.code,
            f"allocation workspace preparation failed ({error.code})",
            retryable=error.retryable,
            status_code=error.status_code,
        )
    if isinstance(error, (FindingsError, HTTPToolError)):
        return AllocationError(
            error.code,
            (
                f"allocation findings preparation failed ({error.code})"
                if isinstance(error, FindingsError)
                else f"allocation HTTP preparation failed ({error.code})"
            ),
            retryable=error.retryable,
            status_code=503 if error.retryable else 422,
        )
    return AllocationError(
        "allocation_preparation_failed",
        f"allocation resource preparation failed ({type(error).__name__})",
        retryable=True,
        status_code=503,
    )


class Allocation:
    """Constructs, drains, and erases one allocation under its own lifecycle lock."""

    def __init__(
        self,
        state: RuntimeState,
        factories: FactoryRegistry,
        capabilities: CapabilitySnapshot | None = None,
        *,
        a2a_base_url: str,
        allocation_id: str,
        private_bypass_urls: Sequence[str] = (),
        now: Callable[[], datetime] | None = None,
        confirmed_control_deadline: Callable[[], float | None] | None = None,
        force_exit: Callable[[int], Any] = os._exit,
        resource_collector_factory: Callable[[], ResourceCollector] = ResourceCollector,
    ) -> None:
        self._state = state
        self._allocation_id = allocation_id
        self._released = False
        self.exit_requested = False
        self._factories = factories
        # Tests and embedded callers may inject an already frozen snapshot.
        # The process entry point resolves it from RuntimeState after the
        # private listener is ready and startup discovery has completed.
        self._capabilities = capabilities
        self._a2a_base_url = a2a_base_url.rstrip("/")
        self._private_bypass_hosts = _url_hosts((*private_bypass_urls, a2a_base_url))
        self._now = now or (lambda: datetime.now(UTC))
        self._cleanup = AllocationResourceCleanup(factories, self._now)
        self._confirmed_control_deadline = confirmed_control_deadline or (lambda: None)
        self._exit_process = force_exit
        self._resource_collector_factory = resource_collector_factory
        self._lock = asyncio.Lock()
        self._context: _AllocationContext | None = None
        self._fingerprint_key = os.urandom(32)

    def _force_exit(self, code: int) -> Any:
        self.exit_requested = True
        # Embedders/tests may replace process exit. Even then an unconfirmed
        # shutdown must not retain a sampler or manufacture a final boundary.
        if self._context is not None and self._context.execution is not None:
            self._context.execution.reject()
        if self._context is not None and self._context.resource_metrics is not None:
            self._context.resource_metrics.close()
        logger.error("Runtime process exit requested: code=%d", code)
        return self._exit_process(code)

    def _reserve_stop_flush(self, context: _AllocationContext, deadline: datetime) -> datetime:
        """Reserve the existing flush cap inside this stop's original deadline."""

        timeout = context.adapter_host.flush_timeout_seconds
        if timeout <= 0:
            return deadline
        context.stop_deadline = min(context.stop_deadline or deadline, deadline)
        remaining = max(0.0, (context.stop_deadline - self._now()).total_seconds())
        return context.stop_deadline - timedelta(seconds=min(timeout, remaining / 2))

    async def _exit_after_stop_failure(
        self, context: _AllocationContext, *, phase: str, error: BaseException
    ) -> None:
        await self._state.fence_allocation(context.allocation_id)
        # Never format the exception message: it can contain credentials or a
        # provider response. The original Worker error remains in its spans.
        logger.error(
            "Allocation cleanup failed: phase=%s exception=%s",
            phase,
            type(error).__name__[:MAX_LOG_EXTRA_TEXT],
        )
        if context.stop_deadline is not None:
            # Reuse the bounded flush while producer handles remain valid.
            # Repeated caller cancellation cannot skip it or cancel its sender.
            task = asyncio.create_task(
                context.adapter_host.flush(deadline=context.stop_deadline, now=self._now),
                name="allocation-emergency-trace-flush",
            )
            await join_owned_task(task)
            try:
                if not task.result():
                    logger.error("Trace flush cancellation before process exit was unconfirmed")
            except BaseException:
                logger.error("Trace flush before process exit did not settle")
        self._force_exit(70)

    def _start_resources(self, spec: AllocationSpec) -> ResourceCollector | None:
        if spec.performance_metrics is None:
            return None
        try:
            collector = self._resource_collector_factory()
        except Exception:
            collector = ResourceCollector(reader=lambda _: ProcessReading(reason="read_failed"))
        collector.start()
        return collector

    async def active_a2a_application(self, allocation_id: str) -> Any | None:
        """Resolve A2A only while this exact allocation remains active."""

        async with self._lock:
            context = self._context
            if context is None or context.allocation_id != allocation_id or context.worker is None:
                return None
            if context.resources is not None and context.resources.execution.failure is not None:
                return None
            state = await self._state.snapshot(self._allocation_id)
            if (
                state.process_state is not ProcessState.ALLOCATED
                or state.allocation_id != allocation_id
            ):
                return None
            return getattr(context.worker, "a2a_application", None)

    async def agent_state_snapshot(
        self, allocation_id: str, *, if_none_match: str | None = None
    ) -> AgentStateSnapshot | None:
        """Read the active Worker's State; None means a validated ETag still matches."""

        async with self._lock:
            context, worker = await self._worker_for_state_read(allocation_id)
            if if_none_match is not None and context.worker_state is not None:
                revision = await context.worker_state.validated_state_revision()
                if revision is not None and if_none_match == (
                    f'"contractor-agent-state-v1-{revision}"'
                ):
                    return None
            context.state_reads += 1
            context.state_reads_drained.clear()
        try:
            try:
                snapshot = await worker.agent_state_snapshot()
            except asyncio.CancelledError:
                raise
            except ValidationError:
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=False,
                    status_code=503,
                ) from None
            except Exception:
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=True,
                    status_code=503,
                ) from None
            # Stop has detached this Worker. Reject before taking the lifecycle
            # lock: release may hold it while joining admitted State readers.
            if context.worker is not worker:
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=True,
                    status_code=409,
                )
        finally:
            # No Worker or State owner may remain in this frame while the
            # final lifecycle check waits behind release. The context itself
            # is cleared by release; the snapshot is an independent value.
            del worker
            context.state_reads -= 1
            if context.state_reads == 0:
                context.state_reads_drained.set()
        async with self._lock:
            current, _ = await self._worker_for_state_read(allocation_id)
            # A context binds one Worker immutably until stop detaches it.
            if current is not context:
                raise AllocationError(
                    "agent_state_unavailable",
                    "allocation Worker State is unavailable",
                    retryable=True,
                    status_code=409,
                )
            return snapshot

    async def _worker_for_state_read(
        self, allocation_id: str
    ) -> tuple[_AllocationContext, WorkerRuntime]:
        """Check live ownership while the caller holds the allocation lock."""

        context = self._require_context(allocation_id)
        state = await self._state.snapshot(self._allocation_id)
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
        return context, worker

    async def prepare(self, spec: AllocationSpec) -> PrepareAllocationResponse:
        async with self._lock:
            if self._released or spec.allocation_id != self._allocation_id:
                raise _conflict("allocation ownership has ended")
            fingerprint = _spec_fingerprint(spec, self._fingerprint_key)
            if self._context is not None:
                context = self._context
                if (
                    context.allocation_id == spec.allocation_id
                    and context.stage_execution_id == spec.stage_execution_id
                ):
                    if (
                        context.fingerprint != fingerprint
                        or context.prepare_response is None
                        or spec.lease_expires_at
                        < context.prepare_response.worker_handle.lease_expires_at
                    ):
                        raise _conflict("prepare request differs from the active allocation")
                    return context.prepare_response
                raise _conflict("Runtime Agent already owns another allocation")

            state = await self._state.snapshot(self._allocation_id)
            if state.process_state is not ProcessState.IDLE:
                raise _conflict("Runtime Agent slot is not idle")
            self._validate_spec(spec)
            sandbox = self._sandbox_factory(spec)
            runtime_factory = self._runtime_factory(spec)
            prepared = AllocationPreparation(
                self._factories,
                now=self._now,
                a2a_base_url=self._a2a_base_url,
                private_bypass_hosts=self._private_bypass_hosts,
                confirmed_control_deadline=self._confirmed_control_deadline,
            )
            # The slot lock and validation have accepted this owner. Include
            # adapter/workspace/Worker preparation in the requested interval.
            resource_metrics = self._start_resources(spec)
            try:
                context = await prepared.build(
                    spec,
                    sandbox,
                    runtime_factory,
                    fingerprint=fingerprint,
                    resource_metrics=resource_metrics,
                )
                response = context.prepare_response
                if response is None:
                    raise RuntimeError("Allocation preparation returned no response")
                await self._state.commit_allocation(spec.allocation_id)
                self._context = context
                return response
            except (Exception, asyncio.CancelledError) as error:
                await self._rollback_prepare(
                    spec,
                    sandbox,
                    prepared.workspace,
                    prepared.project_workspace,
                    prepared.tools,
                    prepared.worker,
                    prepared.adapter_host,
                    prepared.execution,
                    collector=prepared.collector,
                    http_transport=prepared.http_transport,
                )
                if isinstance(error, (AllocationError, asyncio.CancelledError)):
                    raise
                if (
                    isinstance(error, (AdapterPreparationError, WorkspacePreparationError))
                    and not error.cleanup_confirmed
                ):
                    await self._state.fence_allocation(spec.allocation_id)
                    self._force_exit(70)
                raise _preparation_error(error) from None
            finally:
                if resource_metrics is not None and (
                    self._context is None or self._context.resource_metrics is not resource_metrics
                ):
                    # Failed prepare has no final-report delivery route. Keep
                    # its existing rollback semantics, discard observations.
                    resource_metrics.close()

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
            state = await self._state.snapshot(self._allocation_id)
            if state.process_state not in {ProcessState.DRAINING, ProcessState.FENCED}:
                raise _conflict("allocation must be draining or fenced before release")
            if context.worker is not None:
                raise _conflict("Worker must be stopped before release")
            if context.release_prepared:
                return
            await self._prepare_release_cleanup(context)
            await self._state.fence_allocation(context.allocation_id)

    async def confirm_release(
        self, allocation_id: str | None, *, keep_fenced: bool = False
    ) -> None:
        """Apply a heartbeat-confirmed authoritative release.

        The private release endpoint only prepares local cleanup. Keeping this
        second edge separate makes a lost HTTP response safe: the slot remains
        fenced and the same release can be retried.
        """

        async with self._lock:
            context = self._context
            if context is None:
                return
            if allocation_id != context.allocation_id:
                raise _conflict("release action identifies another allocation")
            if context.worker is not None:
                raise _conflict("active Worker must be drained before release")
            if not context.release_prepared:
                await self._prepare_release_cleanup(context)
            await self._state.confirm_release(context.allocation_id, keep_fenced=keep_fenced)
            self._released = True
            context.terminal_response = None
            self._context = None

    async def _prepare_release_cleanup(self, context: _AllocationContext) -> None:
        if context.resource_metrics is not None:
            context.resource_metrics.close()
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
            cleanup_task.add_done_callback(consume_task_exception)
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

    async def _run_release_cleanup(self, context: _AllocationContext, deadline: datetime) -> None:
        await self._close_http_or_exit(context, deadline)
        await self._cleanup.release(context, deadline)

    async def expire_control_lease(self, shutdown_grace_seconds: float) -> None:
        if self._context is None and self._lock.locked():
            # A long prepare holds the slot lock. Fence now so the expired
            # lease cannot wait for it; its commit then fails and rolls back.
            await self._state.fence_control_lease()
        await self._drain_allocation(
            None,
            shutdown_grace_seconds,
            TerminationError(
                code="control_lease_expired",
                message="Runtime Agent confirmed control lease expired",
                retryable=True,
            ),
        )

    async def reconcile_drain(self, allocation_id: str, shutdown_grace_seconds: float) -> None:
        if self._context is None or self._context.allocation_id != allocation_id:
            # Only the lock owner installs a context. A drain for an absent
            # allocation is a no-op and must not wait behind a long prepare;
            # the Control Plane repeats it if the allocation later commits.
            return
        await self._drain_allocation(
            allocation_id,
            shutdown_grace_seconds,
            TerminationError(
                code="control_plane_reconciliation",
                message="Control Plane requested allocation reconciliation",
                retryable=True,
            ),
        )

    async def _drain_allocation(
        self,
        allocation_id: str | None,
        timeout_seconds: float,
        reason: TerminationError,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Worker shutdown grace must be positive")
        deadline = self._now() + timedelta(seconds=timeout_seconds)
        try:
            # Lock acquisition has no resource I/O or cancellation cleanup.
            # Its wait consumes the same budget as the subsequent teardown.
            async with asyncio.timeout(timeout_seconds):
                await self._lock.acquire()
        except TimeoutError:
            await self._state.fence_control_lease()
            logger.error("Allocation cleanup failed: phase=lifecycle_lock exception=TimeoutError")
            self._force_exit(70)
            raise AllocationError(
                "worker_stop_unconfirmed",
                "allocation stop could not acquire lifecycle ownership before its deadline",
                retryable=False,
                status_code=503,
            ) from None
        try:
            context = self._context
            if context is None:
                if allocation_id is None:
                    await self._state.fence_control_lease()
                return
            if allocation_id is not None and context.allocation_id != allocation_id:
                return
            await self._stop_worker_for_fence(context, deadline, reason)
        finally:
            self._lock.release()

    def _validate_spec(self, spec: AllocationSpec) -> None:
        validate_spec(
            spec,
            capabilities=self._capabilities or self._state.capabilities,
            factories=self._factories,
            now=self._now,
            runtime_factory=self._runtime_factory,
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

    async def _terminate(
        self,
        *,
        allocation_id: str,
        kind: Literal["finalize", "abort"],
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
            if context.execution is not None:
                context.execution.reject()
            await self._state.begin_draining(allocation_id)
            producer_deadline = self._reserve_stop_flush(context, deadline)
            await self._stop_producers_or_exit(context, producer_deadline, deadline, kind=kind)
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
        deadline: datetime,
        reason: TerminationError,
    ) -> None:
        """Stop a Worker after authority is lost without making the slot idle."""

        if context.execution is not None:
            context.execution.reject()
        state = await self._state.snapshot(self._allocation_id)
        if state.process_state is ProcessState.ALLOCATED:
            await self._state.begin_draining(context.allocation_id)
        producer_deadline = self._reserve_stop_flush(context, deadline)
        if context.worker is None:
            await self._close_http_or_exit(context, producer_deadline)
            if context.release_cleanup_task is not None:
                await self._state.fence_allocation(context.allocation_id)
                return
            await self._stop_tools_or_exit(context, producer_deadline)
            await self._stop_adapters_or_exit(context, deadline)
            await self._discard_project_workspace_or_exit(context, deadline)
            await self._state.fence_allocation(context.allocation_id)
            return

        await self._stop_producers_or_exit(context, producer_deadline, deadline, kind="abort")
        await self._discard_project_workspace_or_exit(context, deadline)
        context.termination_kind = "lease"
        context.termination_id = None
        context.terminal_response = AllocationFinalResponse(
            apiVersion=API_VERSION,
            report=_build_report(context, self._now(), reason),
        )
        await self._state.fence_allocation(context.allocation_id)

    async def _stop_producers_or_exit(
        self,
        context: _AllocationContext,
        producer_deadline: datetime,
        deadline: datetime,
        *,
        kind: Literal["finalize", "abort"],
    ) -> None:
        """Stop producers in order before flushing adapters under their reserved budget."""

        await self._close_http_or_exit(context, producer_deadline)
        worker = context.worker
        if worker is None:
            raise _conflict("allocation Worker is already absent")
        operation = worker.finalize if kind == "finalize" else worker.abort
        try:
            await _await_before_deadline(
                lambda: operation(producer_deadline),
                deadline=producer_deadline,
                now=self._now,
            )
        except asyncio.CancelledError as error:
            await self._exit_after_stop_failure(context, phase="worker", error=error)
            raise
        except Exception as error:
            await self._exit_after_stop_failure(context, phase="worker", error=error)
            raise AllocationError(
                "worker_stop_unconfirmed",
                f"in-process Worker stop could not be guaranteed ({type(error).__name__})",
                retryable=False,
                status_code=503,
            ) from None

        context.worker = None
        await self._stop_tools_or_exit(context, producer_deadline)
        await self._stop_execution_or_exit(context, producer_deadline)
        await self._stop_adapters_or_exit(context, deadline)

    async def _close_http_or_exit(self, context: _AllocationContext, deadline: datetime) -> None:
        if context.http_transport is None:
            return
        try:
            await _await_before_deadline(
                context.http_transport.close, deadline=deadline, now=self._now
            )
        except BaseException as error:
            await self._exit_after_stop_failure(context, phase="http", error=error)
            raise

    async def _stop_tools_or_exit(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        try:
            await _await_before_deadline(
                lambda: self._cleanup.close_producers(context),
                deadline=deadline,
                now=self._now,
            )
        except asyncio.CancelledError as error:
            await self._exit_after_stop_failure(context, phase="tool", error=error)
            raise
        except Exception as error:
            await self._exit_after_stop_failure(context, phase="tool", error=error)
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
        except asyncio.CancelledError as error:
            await self._exit_after_stop_failure(context, phase="adapter", error=error)
            raise
        except Exception as error:
            await self._exit_after_stop_failure(context, phase="adapter", error=error)
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
        await self._stop_execution_or_exit(context, deadline, remove=True)
        project_workspace = context.project_workspace
        if project_workspace is None:
            return
        try:
            await _await_before_deadline(
                lambda: self._cleanup.project_workspace(project_workspace, deadline=deadline),
                deadline=deadline,
                now=self._now,
            )
        except asyncio.CancelledError as error:
            await self._exit_after_stop_failure(context, phase="workspace", error=error)
            raise
        except Exception as error:
            await self._exit_after_stop_failure(context, phase="workspace", error=error)
            raise AllocationError(
                "workspace_cleanup_unconfirmed",
                "allocation project workspace cleanup could not be guaranteed",
                retryable=False,
                status_code=503,
            ) from None
        context.project_workspace = None

    async def _stop_execution_or_exit(
        self, context: _AllocationContext, deadline: datetime, *, remove: bool = False
    ) -> None:
        if context.execution is None:
            return
        context.execution.reject()
        try:
            operation = context.execution.remove if remove else context.execution.stop
            await _await_before_deadline(
                lambda: operation(deadline=deadline), deadline=deadline, now=self._now
            )
        except BaseException as error:
            await self._exit_after_stop_failure(context, phase="execution", error=error)
            if isinstance(error, asyncio.CancelledError):
                raise
            raise AllocationError(
                "sandbox_cleanup_unconfirmed",
                "allocation sandbox cleanup is unconfirmed",
                retryable=False,
                status_code=503,
            ) from None

    async def _rollback_prepare(
        self,
        spec: AllocationSpec,
        sandbox: SandboxFactory,
        workspace: AllocationWorkspace | None,
        project_workspace: DirectWorkspaceSession | None,
        tools: Mapping[str, ToolInstance],
        worker: WorkerRuntime | None,
        adapter_host: AllocationAdapterHost | None,
        execution: PreparedExecution | None = None,
        collector: SitemapCollector | None = None,
        http_transport: HTTPTransport | None = None,
    ) -> None:
        if execution is not None:
            # Keep a full owner even when readiness was never committed. A
            # release confirmation must not bypass an uncertain prepare.
            if not (workspace is not None and adapter_host is not None):
                raise RuntimeError("Expected workspace is not None and adapter_host is not None")
            context = _AllocationContext(
                allocation_id=spec.allocation_id,
                run_id=spec.run_id,
                stage_execution_id=spec.stage_execution_id,
                logical_agent_name=spec.logical_agent_name,
                namespace=spec.namespace,
                fingerprint=_spec_fingerprint(spec, self._fingerprint_key),
                started_at=self._now(),
                workspace=workspace,
                sandbox=sandbox,
                project_workspace=project_workspace,
                adapter_host=adapter_host,
                tools=dict(tools),
                worker_state=None,
                resources=None,
                runtime_settings=spec.runtime_settings,
                worker=worker,
                execution=execution,
                http_transport=http_transport,
                collector=collector,
            )
            self._context = context
            execution.reject()
            deadline = self._now() + timedelta(
                seconds=spec.runtime_settings.request_timeout_seconds
            )
            try:
                if worker is not None:
                    await _await_before_deadline(
                        lambda: worker.abort(deadline), deadline=deadline, now=self._now
                    )
                    context.worker = None
                await self._prepare_release_cleanup(context)
            except BaseException:
                await self._state.fence_allocation(spec.allocation_id)
                self._force_exit(70)
                raise
            self._context = None
            return
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

        if http_transport is not None:
            await attempt(http_transport.close)
        if worker is not None:
            await attempt(lambda: worker.abort(deadline))
        await attempt(lambda: _close_tools(tools))
        if collector is not None:
            await attempt(collector.close)
        # A Worker built before the failure already registered the tools in
        # ADK's process-wide caches, and this prepare never reaches release
        # (RTA-R116).
        try:
            release_adk_tool_caches()
        except Exception:
            failed = True
        if project_workspace is not None:
            await attempt(
                lambda: self._cleanup.project_workspace(project_workspace, deadline=deadline)
            )
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


class AllocationService:
    """Own bounded allocations and route lifecycle work without a process-wide I/O lock."""

    def __init__(self, state, factories, capabilities=None, **options):
        self._state = state
        self._factories = factories
        self._capabilities = capabilities
        self._options = options
        self._allocations: dict[str, Allocation] = {}
        self._lock = asyncio.Lock()
        self._exit_process = options.pop("force_exit", os._exit)
        self._exit_requested = False

    def _force_exit(self, code):
        self._exit_requested = True
        for owner in self._allocations.values():
            context = owner._context
            if context is not None:
                if context.execution is not None:
                    context.execution.reject()
                if context.resource_metrics is not None:
                    context.resource_metrics.close()
        return self._exit_process(code)

    def _require_allocation(self, allocation_id):
        owner = self._allocations.get(allocation_id)
        if owner is None:
            raise _not_found()
        return owner

    async def prepare(self, spec):
        async with self._lock:
            owner = self._allocations.get(spec.allocation_id)
            if owner is None:
                if self._exit_requested:
                    raise _conflict("Runtime is stopping")
                try:
                    await self._state.reserve_allocation(spec.allocation_id)
                except RuntimeError:
                    raise _conflict("Runtime allocation capacity is unavailable") from None
                try:
                    owner = Allocation(
                        self._state,
                        self._factories,
                        self._capabilities,
                        allocation_id=spec.allocation_id,
                        force_exit=self._force_exit,
                        **self._options,
                    )
                except BaseException:
                    await self._state.discard_preparation(spec.allocation_id)
                    raise
                self._allocations[spec.allocation_id] = owner
        try:
            return await owner.prepare(spec)
        except BaseException:
            # A confirmed rollback has no resources. Uncertain cleanup retains its
            # owner until the existing process-exit fence resolves it.
            async with self._lock:
                if (
                    owner._context is None
                    and not owner.exit_requested
                    and not owner._lock.locked()
                    and self._allocations.get(spec.allocation_id) is owner
                ):
                    self._allocations.pop(spec.allocation_id)
                    await self._state.discard_preparation(spec.allocation_id)
            raise

    async def finalize(self, request):
        return await self._require_allocation(request.allocation_id).finalize(request)

    async def abort(self, request):
        return await self._require_allocation(request.allocation_id).abort(request)

    async def release(self, request):
        owner = self._allocations.get(request.allocation_id)
        if owner is not None:
            await owner.release(request)

    async def confirm_release(self, allocation_id, *, keep_fenced=False):
        if allocation_id is None:
            await self._state.confirm_release(None, keep_fenced=keep_fenced)
            return
        owner = self._allocations.get(allocation_id)
        if owner is not None:
            await owner.confirm_release(allocation_id, keep_fenced=keep_fenced)
            async with self._lock:
                if owner._released and self._allocations.get(allocation_id) is owner:
                    self._allocations.pop(allocation_id)
        elif keep_fenced:
            await self._state.fence_control_lease()

    async def active_a2a_application(self, allocation_id):
        owner = self._allocations.get(allocation_id)
        return await owner.active_a2a_application(allocation_id) if owner is not None else None

    async def agent_state_snapshot(self, allocation_id, *, if_none_match=None):
        return await self._require_allocation(allocation_id).agent_state_snapshot(
            allocation_id,
            if_none_match=if_none_match,
        )

    async def expire_control_lease(self, shutdown_grace_seconds):
        await self._state.fence_control_lease()
        owners = tuple(self._allocations.values())
        await asyncio.gather(
            *(owner.expire_control_lease(shutdown_grace_seconds) for owner in owners)
        )

    async def close(self, shutdown_grace_seconds: float) -> None:
        """Join every allocation before process-owned transports and cache close."""
        await self._state.fence_control_lease()

        async def close_owner(owner):
            await owner.expire_control_lease(shutdown_grace_seconds)
            if owner._context is not None:
                await owner.release(
                    ReleaseAllocationRequest(
                        apiVersion=API_VERSION, allocationId=owner._allocation_id
                    )
                )
                await self.confirm_release(owner._allocation_id, keep_fenced=True)

        await asyncio.gather(*(close_owner(owner) for owner in tuple(self._allocations.values())))

    async def reconcile_drain(self, allocation_id, shutdown_grace_seconds):
        owner = self._allocations.get(allocation_id)
        if owner is not None:
            await owner.reconcile_drain(allocation_id, shutdown_grace_seconds)
