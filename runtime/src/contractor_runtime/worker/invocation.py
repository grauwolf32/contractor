"""Invocation ownership and the absolute execution deadline."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import Any

from contractor_runtime.contracts import (
    ResolvedModelPolicy,
    WorkerSummarizerConfig,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.worker_completion import PreparedWorkerCompletion
from contractor_runtime.worker.budget import _InvocationBudget
from contractor_runtime.worker.prompts import _summary_prompt_boundary
from contractor_runtime.worker.sessions import (
    WorkerSessionLifecycle,
    WorkerSessionLifecycleError,
)
from contractor_runtime.worker.state import WorkerStateStore

INVOCATION_CLEANUP_TIMEOUT_SECONDS = 5.0


class InvocationDeadline:
    """The effective Worker-call deadline on the event loop's monotonic clock.

    It is converted once, at invocation start, so continuations never reset it.
    ``scope`` cancels the work in flight once the loop runs its timer; ADK then
    joins its node task. ``check`` refuses to start the next phase when the
    deadline passed while this task ran without yielding to that timer
    (WRK-R167).
    """

    def __init__(self, deadline: datetime) -> None:
        self._loop = asyncio.get_running_loop()
        self.when = self._loop.time() + (deadline - datetime.now(UTC)).total_seconds()
        self.scope = asyncio.timeout_at(self.when)

    def passed(self) -> bool:
        return self.scope.expired() or self._loop.time() >= self.when

    def check(self) -> None:
        if self.passed():
            raise TimeoutError("Worker invocation deadline exceeded")


class InvocationLifecycle:
    """Own one admitted invocation through session acquisition and joined release."""

    def __init__(
        self,
        *,
        sessions: WorkerSessionLifecycle,
        state: WorkerStateStore,
        policy: ResolvedModelPolicy,
        summarizer: WorkerSummarizerConfig | None,
        completion: PreparedWorkerCompletion | None,
        stop_accepting: Callable[[], None],
    ) -> None:
        self._session_lifecycle = sessions
        self._worker_state = state
        self._policy = policy
        self._summarizer = summarizer
        self._metrics = state.metrics
        self._completion = completion
        self._stop_accepting = stop_accepting
        self.lock = asyncio.Lock()
        self.task: asyncio.Task[Any] | None = None
        self.budget: _InvocationBudget | None = None
        self.deadline: InvocationDeadline | None = None

    @contextlib.asynccontextmanager
    async def own(
        self,
    ) -> AsyncGenerator[tuple[str, _InvocationBudget]]:
        """Own the lock, task, session and budget across every acquisition/exit."""

        await self.lock.acquire()
        session_acquired = False
        invocation_failed = False
        completion = self._completion
        previous_diagnostics = completion.diagnostics if completion is not None else None

        async def record_completion_cancellation():
            if completion is not None and completion.diagnostics is not previous_diagnostics:
                await completion.record_phase("failed", "worker_cancelled")

        try:
            self.task = asyncio.current_task()
            try:
                session_id = await self._session_lifecycle.begin_invocation(
                    await self._worker_state.snapshot()
                )
            except asyncio.CancelledError:
                if self._session_lifecycle.failed:
                    self._stop_accepting()
                raise
            except WorkerSessionLifecycleError:
                self._stop_accepting()
                raise
            except Exception as error:
                self._stop_accepting()
                raise WorkerSessionLifecycleError("state_snapshot_failed") from error
            session_acquired = True
            policy = self._policy
            budget = _InvocationBudget(
                max_model_calls=policy.max_model_calls,
                max_tool_calls=policy.max_tool_calls,
                max_total_tokens=policy.max_total_tokens,
                metrics=self._metrics,
                cumulative_budget=(
                    self._summarizer.cumulative_budget if self._summarizer is not None else None
                ),
                summary_prompt_boundary=_summary_prompt_boundary(policy, self._summarizer),
            )
            self.budget = budget
            yield session_id, budget
        except asyncio.CancelledError:
            invocation_failed = True
            await record_completion_cancellation()
            raise
        except BaseException:
            invocation_failed = True
            raise
        finally:
            try:
                if session_acquired:
                    try:
                        await self.release_session()
                    except asyncio.CancelledError:
                        await record_completion_cancellation()
                        raise
                    except Exception:
                        self._stop_accepting()
                        if not invocation_failed:
                            raise
            finally:
                self.budget = None
                self.task = None
                self.lock.release()

    async def release_session(self) -> None:
        async def release() -> None:
            try:
                await asyncio.wait_for(
                    self._session_lifecycle.finish_invocation(),
                    timeout=INVOCATION_CLEANUP_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                raise WorkerSessionLifecycleError("cleanup_timeout") from error

        task = asyncio.create_task(release(), name="worker-invocation-session-release")
        if await join_owned_task(task) or task.cancelled():
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
            if self._session_lifecycle.failed:
                self._stop_accepting()
            raise asyncio.CancelledError
        task.result()
