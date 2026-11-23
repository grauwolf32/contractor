"""Consecutive-series admission, separate from domain idempotency receipts."""

from dataclasses import dataclass
from typing import Any

from google.adk.plugins import BasePlugin

from contractor_runtime.toolsets.common.repeats import RepeatRule


@dataclass(slots=True)
class _Series:
    owner: Any
    rule: RepeatRule
    count: int = 0


class ToolRepeatPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="contractor_tool_repeats")
        self._series: _Series | None = None
        self._pending: dict[int, _Series] = {}

    def reset(self) -> None:
        self._series = None
        self._pending.clear()

    def interrupt(self) -> None:
        self._series = None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        # No awaits between examining and reserving a slot on this event loop.
        owner = getattr(tool, "func", tool)
        policy = getattr(owner, "contractor_repeat_rule", None)
        rule = policy(tool_args) if callable(policy) else None
        if rule is None:
            self.interrupt()
            return None
        if not isinstance(rule, RepeatRule) or rule.limit < 1:
            raise RuntimeError("invalid tool repeat rule")
        series = self._series
        if series is None or series.owner is not owner or series.rule != rule:
            series = self._series = _Series(owner, rule)
        if series.count >= rule.limit:
            return {
                "ok": False,
                "error": {
                    "code": "tool_repeat_limit",
                    "message": rule.message,
                    "retryable": False,
                },
            }
        series.count += 1
        self._pending[id(tool_context)] = series
        return None

    def settled(self, context: Any, failed: bool) -> None:
        # Called by the instrumentation owner also for SDK error short circuits.
        series = self._pending.pop(id(context), None)
        if series is not None and failed:
            series.count -= 1
