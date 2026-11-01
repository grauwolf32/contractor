"""Deadline-bounded cleanup primitives; lifecycle ownership stays in the service."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from datetime import datetime

from contractor_runtime.allocation.context import _AllocationContext
from contractor_runtime.factories import (
    FactoryRegistry,
    ToolInstance,
)
from contractor_runtime.projectfs import DirectWorkspaceSession
from contractor_runtime.threads import consume_task_exception
from contractor_runtime.worker.adk_caches import release_adk_tool_caches


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
        task.add_done_callback(consume_task_exception)
        raise
    if not done:
        task.cancel()
        task.add_done_callback(consume_task_exception)
        raise TimeoutError
    await task


class AllocationResourceCleanup:
    """Close and erase resources; command state, fencing and replay stay in the service."""

    def __init__(self, factories: FactoryRegistry, now: Callable[[], datetime]) -> None:
        self._factories = factories
        self._now = now

    async def release(
        self,
        context: _AllocationContext,
        deadline: datetime,
    ) -> None:
        await _close_tools(context.tools)
        if context.collector is not None:
            await context.collector.close()
        if context.execution is not None:
            context.execution.reject()
            await context.execution.remove(deadline=deadline)
        if context.project_workspace is not None:
            await self.project_workspace(context.project_workspace, deadline=deadline)
            context.project_workspace = None
        await context.sandbox.cleanup(context.workspace)
        await context.adapter_host.rollback(deadline=deadline, now=self._now)

        context.tools.clear()
        context.resources = None
        context.worker_state = None
        context.runtime_settings = None
        context.prepare_response = None
        # After every Toolset is closed and State and secrets are dropped:
        # ADK's process-wide caches would otherwise keep the wrapped tools,
        # and through them their clients and MetricsState (RTA-R116). A
        # failure leaves release unprepared, so the retry repeats this step.
        release_adk_tool_caches()
        context.release_prepared = True

    async def project_workspace(
        self,
        project_workspace: DirectWorkspaceSession,
        *,
        deadline: datetime,
    ) -> None:
        remaining = (deadline - self._now()).total_seconds()
        await project_workspace.close(deadline=time.monotonic() + max(0.0, remaining))
        provider = self._factories.workspace_provider
        if provider is None:
            raise RuntimeError("project workspace provider is unavailable")
        await provider.cleanup(project_workspace.storage)

    @staticmethod
    async def close_producers(context: _AllocationContext) -> None:
        await _close_tools(context.tools)
        if context.collector is not None:
            await context.collector.close()
