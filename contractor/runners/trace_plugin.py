from google.adk.plugins.base_plugin import BasePlugin
from typing import Any, Awaitable, Callable, Optional
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.events import Event


def _snapshot_state(state_obj: Any) -> dict[str, Any]:
    if state_obj is None:
        return {}
    to_dict = getattr(state_obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    if isinstance(state_obj, dict):
        return dict(state_obj)
    return {}


class AdkTracePlugin(BasePlugin):
    """
    Global Runner plugin that converts ADK lifecycle hooks into TaskRunnerEvent.
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
        super().__init__(name=f"trace_plugin_{task_name}_{task_id}_{iteration}")
        self._task_name = task_name
        self._task_id = task_id
        self._iteration = iteration
        self._session_id = session_id
        self._emit = emit

    async def before_run_callback(self, *, invocation_context) -> None:
        await self._emit(
            type="adk_before_run",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=getattr(invocation_context, "invocation_id", None),
        )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        **_: Any,
    ) -> Optional[dict]:
        actual_args = tool_args if tool_args is not None else (args or {})
        await self._emit(
            type="tool_call",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            tool_name=tool.name,
            tool_args=actual_args,
            agent_name=getattr(tool_context, "agent_name", None),
            state=_snapshot_state(tool_context.state),
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
        actual_args = tool_args if tool_args is not None else (args or {})
        actual_result = tool_response if tool_response is not None else (result or {})
        await self._emit(
            type="tool_result",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            tool_name=tool.name,
            tool_args=actual_args,
            result=actual_result,
            agent_name=getattr(tool_context, "agent_name", None),
            state=_snapshot_state(tool_context.state),
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
        actual_args = tool_args if tool_args is not None else (args or {})
        await self._emit(
            type="tool_error",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            tool_name=tool.name,
            tool_args=actual_args,
            error=repr(error) if error is not None else None,
            agent_name=getattr(tool_context, "agent_name", None),
            state=_snapshot_state(tool_context.state),
        )
        return None

    async def on_event_callback(
        self,
        *,
        invocation_context,
        event: Event,
    ) -> Optional[Event]:
        await self._emit(
            type="adk_event",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            author=getattr(event, "author", None),
            event=event,
            invocation_id=getattr(invocation_context, "invocation_id", None),
        )
        return None

    async def after_run_callback(self, *, invocation_context) -> None:
        await self._emit(
            type="adk_after_run",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=getattr(invocation_context, "invocation_id", None),
        )
