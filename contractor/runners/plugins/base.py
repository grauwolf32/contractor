# contractor/runners/_base_adk_plugin.py
from __future__ import annotations

from typing import Any, Awaitable, Callable

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


def snapshot_state(state_obj: Any) -> dict[str, Any]:
    """Best-effort conversion of an ADK state object to a plain dict."""
    if state_obj is None:
        return {}
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(state_obj, method_name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
    if isinstance(state_obj, dict):
        return dict(state_obj)
    return {}


def resolve_tool_args(
    tool_args: dict[str, Any] | None,
    args: dict[str, Any] | None,
) -> dict[str, Any]:
    """ADK passes tool arguments under varying kwarg names — normalise them."""
    return tool_args if tool_args is not None else (args or {})


def resolve_tool_response(
    tool_response: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """ADK passes tool results under varying kwarg names — normalise them."""
    return tool_response if tool_response is not None else result


class PluginContext:
    """Immutable bundle of identifiers every plugin callback needs."""

    __slots__ = ("task_name", "task_id", "iteration", "session_id")

    def __init__(
        self,
        *,
        task_name: str,
        task_id: int,
        iteration: int,
        session_id: str,
    ) -> None:
        self.task_name = task_name
        self.task_id = task_id
        self.iteration = iteration
        self.session_id = session_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "task_id": self.task_id,
            "iteration": self.iteration,
            "session_id": self.session_id,
        }


class BaseAdkPlugin(BasePlugin):
    """
    Shared base for all TaskRunner ADK plugins.

    Provides:
      - A ``PluginContext`` with the common identification fields.
      - A thin ``_emit`` wrapper that auto-injects those fields.
      - Helpers for extracting invocation / agent identity from ADK contexts.
    """

    def __init__(
        self,
        *,
        plugin_prefix: str,
        ctx: PluginContext,
        emit: Callable[..., Awaitable[None]],
    ) -> None:
        name = f"{plugin_prefix}_{ctx.task_name}_{ctx.task_id}_{ctx.iteration}"
        super().__init__(name=name)
        self._ctx = ctx
        self._raw_emit = emit

    async def _emit(self, event_type: str, **payload: Any) -> None:
        await self._raw_emit(type=event_type, **self._ctx.as_dict(), **payload)

    @staticmethod
    def _identity(context: Any) -> tuple[str | None, str | None]:
        """Extract (invocation_id, agent_name) from a tool/callback context."""
        return (
            getattr(context, "invocation_id", None),
            getattr(context, "agent_name", None),
        )