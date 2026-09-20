"""An enclosing Worker deadline inherited by its asynchronous tool calls."""

from contextvars import ContextVar

invocation_deadline: ContextVar[float | None] = ContextVar(
    "contractor_tool_invocation_deadline", default=None
)
