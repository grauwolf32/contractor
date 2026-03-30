# contractor/runners/metrics_plugin.py
from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any, Awaitable, Callable, Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from contractor.runners.plugins.base import (
    BaseAdkPlugin,
    PluginContext,
    resolve_tool_args,
    resolve_tool_response,
    snapshot_state,
)


# ─── Small helpers ────────────────────────────────────────────────────────────

_UNKNOWN_INVOCATION = "unknown_invocation"
_UNKNOWN_AGENT = "unknown_agent"

_ERROR_STATUS_VALUES = frozenset({"error", "failed", "failure"})
_ERROR_SENTINEL_KEYS = ("error", "error_message", "errors")
_EMPTY_VALUES: tuple[Any, ...] = (None, False, "", [], {})


def _jsonable(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return repr(value)


def _fingerprint(
    invocation_id: str,
    agent_name: str,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    return f"{invocation_id}||{agent_name}||{tool_name}||{_jsonable(args)}"


def _safe_int(value: Any) -> int:
    """Extract an integer from an ADK usage field, defaulting to 0."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# ─── Metric containers ───────────────────────────────────────────────────────


@unique
class _CallOutcome(StrEnum):
    SUCCESS = "success"
    EXCEPTION = "exception"
    RESULT_ERROR = "result_error"


@dataclass(slots=True)
class ToolMetrics:
    calls_total: int = 0
    success_total: int = 0
    exception_errors_total: int = 0
    result_errors_total: int = 0

    def record_outcome(self, outcome: _CallOutcome) -> None:
        match outcome:
            case _CallOutcome.SUCCESS:
                self.success_total += 1
            case _CallOutcome.EXCEPTION:
                self.exception_errors_total += 1
            case _CallOutcome.RESULT_ERROR:
                self.result_errors_total += 1

    def as_dict(self) -> dict[str, int]:
        return {
            "calls_total": self.calls_total,
            "success_total": self.success_total,
            "exception_errors_total": self.exception_errors_total,
            "result_errors_total": self.result_errors_total,
        }


@dataclass(slots=True)
class AgentMetrics:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    tools: dict[str, ToolMetrics] = field(
        default_factory=lambda: defaultdict(ToolMetrics)
    )

    def add_usage(self, usage: dict[str, int]) -> None:
        self.llm_calls += 1
        self.prompt_tokens += usage.get("prompt", 0)
        self.completion_tokens += usage.get("completion", 0)
        self.total_tokens += usage.get("total", 0)
        self.thoughts_tokens += usage.get("thoughts", 0)
        self.cached_tokens += usage.get("cached", 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "cached_tokens": self.cached_tokens,
            "tools": {name: m.as_dict() for name, m in self.tools.items()},
        }


# ─── Call tracking ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class _TrackedCall:
    call_id: int
    invocation_id: str
    agent_name: str
    tool_name: str
    args: dict[str, Any]
    exception_seen: bool = False
    result_error_seen: bool = False
    finished: bool = False

    @property
    def has_error(self) -> bool:
        return self.exception_seen or self.result_error_seen


class _CallTracker:
    """
    Correlates before_tool → (on_error / after_tool) pairs so that each
    logical call is counted exactly once regardless of callback ordering.
    """

    __slots__ = ("_seq", "_calls", "_pending_by_fp")

    def __init__(self) -> None:
        self._seq = itertools.count(1)
        self._calls: dict[int, _TrackedCall] = {}
        self._pending_by_fp: dict[str, deque[int]] = defaultdict(deque)

    def register(
        self,
        invocation_id: str,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> _TrackedCall:
        call = _TrackedCall(
            call_id=next(self._seq),
            invocation_id=invocation_id,
            agent_name=agent_name,
            tool_name=tool_name,
            args=args,
        )
        fp = _fingerprint(invocation_id, agent_name, tool_name, args)
        self._calls[call.call_id] = call
        self._pending_by_fp[fp].append(call.call_id)
        return call

    def resolve(
        self,
        invocation_id: str,
        agent_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> _TrackedCall | None:
        fp = _fingerprint(invocation_id, agent_name, tool_name, args)
        queue = self._pending_by_fp.get(fp)
        if not queue:
            return None
        for call_id in queue:
            call = self._calls.get(call_id)
            if call is not None and not call.finished:
                return call
        return None

    def finish(self, call: _TrackedCall) -> None:
        call.finished = True

    def cleanup_invocation(self, invocation_id: str) -> None:
        to_delete = [
            cid
            for cid, call in self._calls.items()
            if call.invocation_id == invocation_id
        ]
        for cid in to_delete:
            self._calls.pop(cid, None)

        empty_fps: list[str] = []
        for fp, queue in self._pending_by_fp.items():
            remaining = deque(cid for cid in queue if cid in self._calls)
            if remaining:
                self._pending_by_fp[fp] = remaining
            else:
                empty_fps.append(fp)

        for fp in empty_fps:
            del self._pending_by_fp[fp]


# ─── Result error detection ──────────────────────────────────────────────────

ResultErrorDetector = Callable[[Any], bool]


def default_result_error_detector(tool_response: Any) -> bool:
    """Heuristic: does this tool response dict look like an error?"""
    if not isinstance(tool_response, dict):
        return False

    status = str(tool_response.get("status", "")).strip().lower()
    if status in _ERROR_STATUS_VALUES:
        return True

    return any(
        tool_response.get(key) not in _EMPTY_VALUES for key in _ERROR_SENTINEL_KEYS
    )


# ─── Plugin ───────────────────────────────────────────────────────────────────


class AdkMetricsPlugin(BaseAdkPlugin):
    """
    Runner plugin that tracks per-agent, per-tool call counts, success/error
    rates, and LLM token usage, then emits a summary at invocation end.
    """

    def __init__(
        self,
        *,
        task_name: str,
        task_id: int,
        iteration: int,
        session_id: str,
        emit: Callable[..., Awaitable[None]],
        plugin_name: str | None = None,
        result_error_detector: ResultErrorDetector | None = None,
    ) -> None:
        ctx = PluginContext(
            task_name=task_name,
            task_id=task_id,
            iteration=iteration,
            session_id=session_id,
        )
        super().__init__(
            plugin_prefix=plugin_name or "metrics",
            ctx=ctx,
            emit=emit,
        )
        self._detect_result_error = result_error_detector or default_result_error_detector
        self._metrics: dict[str, dict[str, AgentMetrics]] = defaultdict(
            lambda: defaultdict(AgentMetrics)
        )
        self._tracker = _CallTracker()

    # ── Bucket helpers ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_identity(
        invocation_id: str | None, agent_name: str | None
    ) -> tuple[str, str]:
        return (
            invocation_id or _UNKNOWN_INVOCATION,
            agent_name or _UNKNOWN_AGENT,
        )

    def _agent_bucket(self, inv: str, agent: str) -> AgentMetrics:
        return self._metrics[inv][agent]

    def _tool_bucket(self, inv: str, agent: str, tool_name: str) -> ToolMetrics:
        return self._agent_bucket(inv, agent).tools[tool_name]

    # ── Usage extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_usage(llm_response: Any) -> dict[str, int]:
        usage = getattr(llm_response, "usage_metadata", None)
        if usage is None:
            return {k: 0 for k in ("prompt", "completion", "total", "thoughts", "cached")}

        data = snapshot_state(usage)
        return {
            "prompt": _safe_int(data.get("prompt_token_count")),
            "completion": _safe_int(data.get("candidates_token_count")),
            "total": _safe_int(data.get("total_token_count")),
            "thoughts": _safe_int(data.get("thoughts_token_count")),
            "cached": _safe_int(data.get("cached_content_token_count")),
        }

    # ── Tool callbacks ────────────────────────────────────────────────────

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        **_: Any,
    ) -> Optional[dict]:
        actual_args = resolve_tool_args(tool_args, args)
        raw_inv, raw_agent = self._identity(tool_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)

        self._tracker.register(inv, agent, tool.name, actual_args)
        self._tool_bucket(inv, agent, tool.name).calls_total += 1

        await self._emit(
            "metrics_tool_call",
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            tool_args=actual_args,
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
        actual_args = resolve_tool_args(tool_args, args)
        raw_inv, raw_agent = self._identity(tool_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)

        bucket = self._tool_bucket(inv, agent, tool.name)
        call = self._tracker.resolve(inv, agent, tool.name, actual_args)

        # Only count once per logical call
        if call is None or not call.exception_seen:
            bucket.record_outcome(_CallOutcome.EXCEPTION)
            if call:
                call.exception_seen = True

        await self._emit(
            "metrics_tool_exception_error",
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            tool_args=actual_args,
            error=repr(error) if error is not None else None,
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
        actual_args = resolve_tool_args(tool_args, args)
        actual_result = resolve_tool_response(tool_response, result)
        raw_inv, raw_agent = self._identity(tool_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)

        bucket = self._tool_bucket(inv, agent, tool.name)
        call = self._tracker.resolve(inv, agent, tool.name, actual_args)
        is_error = self._detect_result_error(actual_result)

        outcome = self._determine_outcome(call, is_error)
        bucket.record_outcome(outcome)

        if call:
            if is_error:
                call.result_error_seen = True
            self._tracker.finish(call)

        await self._emit(
            "metrics_tool_result",
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            tool_args=actual_args,
            result=actual_result,
            result_error=is_error,
        )
        return None

    @staticmethod
    def _determine_outcome(
        call: _TrackedCall | None, is_result_error: bool
    ) -> _CallOutcome:
        """Decide the single outcome to record for a completed tool call."""
        if is_result_error:
            if call and (call.result_error_seen or call.exception_seen):
                # Already counted — but we still need to return *something*.
                # The caller should check `call.result_error_seen` before
                # recording; here we return the category for the emit payload.
                return _CallOutcome.RESULT_ERROR
            return _CallOutcome.RESULT_ERROR

        if call and call.has_error:
            # An exception was already counted; don't double-count as success.
            return _CallOutcome.SUCCESS  # won't be recorded (caller checks)
        return _CallOutcome.SUCCESS

    # ── LLM callback ──────────────────────────────────────────────────────

    async def after_model_callback(
        self,
        *,
        callback_context: Any,
        llm_response: Any,
    ) -> None:
        raw_inv, raw_agent = self._identity(callback_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)
        usage = self._extract_usage(llm_response)

        self._agent_bucket(inv, agent).add_usage(usage)

        await self._emit(
            "metrics_llm_usage",
            invocation_id=inv,
            agent_name=agent,
            usage=usage,
            model=getattr(llm_response, "model_version", None),
        )

    # ── Run end ───────────────────────────────────────────────────────────

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        raw_inv, _ = self._identity(invocation_context)
        inv = raw_inv or _UNKNOWN_INVOCATION

        by_agent = self._metrics.pop(inv, {})

        await self._emit(
            "metrics_summary",
            invocation_id=inv,
            agents={name: m.as_dict() for name, m in by_agent.items()},
        )

        self._tracker.cleanup_invocation(inv)