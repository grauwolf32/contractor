# contractor/runners/trace_plugin.py
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from google.adk.events import Event
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from contractor.runners.plugins.base import (
    BaseAdkPlugin,
    PluginContext,
    resolve_tool_args,
    resolve_tool_response,
    snapshot_state,
)


class AdkTracePlugin(BaseAdkPlugin):
    """
    Runner plugin that converts ADK lifecycle hooks into ``TaskRunnerEvent``s
    carrying full tool-call / tool-result / event traces.
    """

    def __init__(
        self,
        *,
        task_name: str,
        task_id: int,
        iteration: int,
        session_id: str,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        super().__init__(
            plugin_prefix="trace",
            ctx=PluginContext(
                task_name=task_name,
                task_id=task_id,
                iteration=iteration,
                session_id=session_id,
            ),
            emit=emit,
        )

    # ── Run lifecycle ─────────────────────────────────────────────────────

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        invocation_id, _ = self._identity(invocation_context)
        await self._emit("adk_before_run", invocation_id=invocation_id)

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        invocation_id, _ = self._identity(invocation_context)
        await self._emit("adk_after_run", invocation_id=invocation_id)

    # ── Tool lifecycle ────────────────────────────────────────────────────

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        **_: Any,
    ) -> Optional[dict]:
        invocation_id, agent_name = self._identity(tool_context)
        await self._emit(
            "tool_call",
            tool_name=tool.name,
            tool_args=resolve_tool_args(tool_args, args),
            agent_name=agent_name,
            invocation_id=invocation_id,
            state=snapshot_state(tool_context.state),
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        tool_response: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        **_: Any,
    ) -> Optional[dict]:
        invocation_id, agent_name = self._identity(tool_context)
        await self._emit(
            "tool_result",
            tool_name=tool.name,
            tool_args=resolve_tool_args(tool_args, args),
            result=resolve_tool_response(tool_response, result),
            agent_name=agent_name,
            invocation_id=invocation_id,
            state=snapshot_state(tool_context.state),
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        error: Exception | str | None = None,
        **_: Any,
    ) -> Optional[dict]:
        invocation_id, agent_name = self._identity(tool_context)
        await self._emit(
            "tool_error",
            tool_name=tool.name,
            tool_args=resolve_tool_args(tool_args, args),
            error=repr(error) if error is not None else None,
            agent_name=agent_name,
            invocation_id=invocation_id,
            state=snapshot_state(tool_context.state),
        )
        return None

    # ── Event stream ──────────────────────────────────────────────────────

    async def on_event_callback(
        self,
        *,
        invocation_context: Any,
        event: Event,
    ) -> Optional[Event]:
        invocation_id, _ = self._identity(invocation_context)
        await self._emit(
            "adk_event",
            author=getattr(event, "author", None),
            event=event,
            invocation_id=invocation_id,
        )
        return None
