# contractor/runners/metrics_plugin.py
from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique
from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from contractor.runners.agio import AgioEventType
from contractor.runners.plugins.base import (
    BaseAdkPlugin,
    PluginContext,
    resolve_tool_args,
    resolve_tool_response,
    snapshot_state,
)
from contractor.tools.fs.const import FS_COVERAGE_STATE_KEY

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


def _args_hash(args: Any) -> str:
    """Stable 16-hex digest of canonical-JSON-serialised tool args.

    Matches the algorithm scripts/analyze_metrics.py uses for retry/streak
    detection so emitter and analyzer agree byte-for-byte.
    """
    return hashlib.sha256(_jsonable(args).encode()).hexdigest()[:16]


def _fingerprint(
    invocation_id: str,
    agent_name: str,
    tool_name: str,
    args: dict[str, Any],
) -> str:
    return f"{invocation_id}||{agent_name}||{tool_name}||{_jsonable(args)}"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thoughts_tokens: int = 0
    cached_tokens: int = 0
    fs_coverage: dict[str, int] | None = None
    tools: dict[str, ToolMetrics] = field(
        default_factory=lambda: defaultdict(ToolMetrics)
    )

    def add_usage(self, usage: dict[str, int]) -> None:
        self.llm_calls += 1
        self.input_tokens += usage.get("input", 0)
        self.output_tokens += usage.get("output", 0)
        self.total_tokens += usage.get("total", 0)
        self.thoughts_tokens += usage.get("thoughts", 0)
        self.cached_tokens += usage.get("cached", 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "cached_tokens": self.cached_tokens,
            "fs_coverage": dict(self.fs_coverage) if self.fs_coverage else None,
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
    args_hash: str
    started_at: str
    started_monotonic: float
    exception_seen: bool = False
    finished: bool = False


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
            args_hash=_args_hash(args),
            started_at=_utcnow_iso(),
            started_monotonic=time.monotonic(),
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
        self._detect_result_error = (
            result_error_detector or default_result_error_detector
        )
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
            return dict.fromkeys(("input", "output", "total", "thoughts", "cached"), 0)

        data = snapshot_state(usage)
        return {
            "input": _safe_int(data.get("prompt_token_count")),
            "output": _safe_int(data.get("candidates_token_count")),
            "total": _safe_int(data.get("total_token_count")),
            "thoughts": _safe_int(data.get("thoughts_token_count")),
            "cached": _safe_int(data.get("cached_content_token_count")),
        }

    @staticmethod
    def _timing_payload(call: _TrackedCall | None) -> dict[str, Any]:
        """Build the timing/identity fields shared by tool result/exception events.

        When the before_tool callback was missed (call is None) we still emit
        args_hash via _args_hash() at the call site if needed; here we just
        skip duration since we don't know when the call started.
        """
        if call is None:
            return {}
        elapsed_ms = max(0.0, (time.monotonic() - call.started_monotonic) * 1000.0)
        return {
            "tool_call_id": f"call_{call.call_id}",
            "args_hash": call.args_hash,
            "started_at": call.started_at,
            "execution_time_ms": round(elapsed_ms, 3),
        }

    @staticmethod
    def _payload_size(value: Any) -> int:
        """Approximate JSON byte size of a tool argument/result payload."""
        if value is None:
            return 0
        try:
            return len(json.dumps(value, default=str, ensure_ascii=False).encode("utf-8"))
        except (TypeError, ValueError):
            return len(repr(value).encode("utf-8"))

    # ── Tool callbacks ────────────────────────────────────────────────────

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_context: ToolContext,
        tool_args: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict | None:
        actual_args = resolve_tool_args(tool_args, args)
        raw_inv, raw_agent = self._identity(tool_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)

        call = self._tracker.register(inv, agent, tool.name, actual_args)
        self._tool_bucket(inv, agent, tool.name).calls_total += 1

        await self._emit(
            AgioEventType.TOOL_CALL,
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            tool_call_id=f"call_{call.call_id}",
            arguments=actual_args,
            arguments_size=self._payload_size(actual_args),
            args_hash=call.args_hash,
            started_at=call.started_at,
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
    ) -> dict | None:
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

        timing = self._timing_payload(call)
        await self._emit(
            AgioEventType.TOOL_EXCEPTION,
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            arguments=actual_args,
            arguments_size=self._payload_size(actual_args),
            error=repr(error) if error is not None else None,
            **timing,
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
    ) -> dict | None:
        actual_args = resolve_tool_args(tool_args, args)
        actual_result = resolve_tool_response(tool_response, result)
        raw_inv, raw_agent = self._identity(tool_context)
        inv, agent = self._normalise_identity(raw_inv, raw_agent)

        bucket = self._tool_bucket(inv, agent, tool.name)
        call = self._tracker.resolve(inv, agent, tool.name, actual_args)
        is_error = self._detect_result_error(actual_result)

        # ADK fires after_tool after on_tool_error only when a plugin returns
        # a non-None error response. The exception was already counted in
        # on_tool_error_callback — skip recording here to avoid double-counting.
        if not (call and call.exception_seen):
            outcome = (
                _CallOutcome.RESULT_ERROR if is_error else _CallOutcome.SUCCESS
            )
            bucket.record_outcome(outcome)

        timing = self._timing_payload(call)
        if call:
            self._tracker.finish(call)

        successful = not is_error and not (call and call.exception_seen)
        await self._emit(
            AgioEventType.TOOL_RESULT,
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool.name,
            arguments=actual_args,
            arguments_size=self._payload_size(actual_args),
            result=actual_result,
            result_size=self._payload_size(actual_result),
            successful=successful,
            result_error=is_error,
            **timing,
        )

        await self._maybe_record_fs_coverage(inv, agent, tool_context)
        return None

    async def _maybe_record_fs_coverage(
        self,
        inv: str,
        agent: str,
        tool_context: ToolContext,
    ) -> None:
        state = snapshot_state(getattr(tool_context, "state", None))
        snapshot = state.get(FS_COVERAGE_STATE_KEY)
        if not isinstance(snapshot, dict):
            return

        agent_bucket = self._agent_bucket(inv, agent)
        if agent_bucket.fs_coverage == snapshot:
            return

        agent_bucket.fs_coverage = dict(snapshot)
        await self._emit(
            AgioEventType.FS_COVERAGE,
            invocation_id=inv,
            agent_name=agent,
            fs_coverage=agent_bucket.fs_coverage,
        )

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
            AgioEventType.LLM_USAGE,
            invocation_id=inv,
            agent_name=agent,
            usage=usage,
            model=getattr(llm_response, "model_version", None),
        )

    # ── Run end ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_callback_state(invocation_context: Any) -> dict[str, Any] | None:
        session = getattr(invocation_context, "session", None)
        if session is None:
            return None
        state = getattr(session, "state", None)
        if not isinstance(state, dict):
            return None
        cb_state = state.get("callbacks")
        if isinstance(cb_state, dict) and cb_state:
            return dict(cb_state)
        return None

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        raw_inv, _ = self._identity(invocation_context)
        inv = raw_inv or _UNKNOWN_INVOCATION

        by_agent = self._metrics.pop(inv, {})

        # Sanity check: every before_tool call should pair with one
        # after_tool (or on_tool_error). When ADK batches parallel tool
        # invocations the callbacks can desync — we surface the
        # imbalance here so downstream analysis can decide whether to
        # trust the counts. ``calls_total`` increments in before_tool;
        # success/exception/result-error sum in after_tool / on_error.
        imbalances: list[dict[str, Any]] = []
        for agent_name, m in by_agent.items():
            for tool_name, tm in m.tools.items():
                accounted = (
                    tm.success_total
                    + tm.exception_errors_total
                    + tm.result_errors_total
                )
                if tm.calls_total != accounted:
                    imbalances.append(
                        {
                            "agent_name": agent_name,
                            "tool_name": tool_name,
                            "calls_total": tm.calls_total,
                            "accounted": accounted,
                            "missing": tm.calls_total - accounted,
                        }
                    )

        await self._emit(
            AgioEventType.RUN_SUMMARY,
            invocation_id=inv,
            agents={name: m.as_dict() for name, m in by_agent.items()},
            callback_imbalances=imbalances,
        )

        cb_state = self._extract_callback_state(invocation_context)
        if cb_state:
            await self._emit(
                AgioEventType.CALLBACK_SUMMARY,
                invocation_id=inv,
                callbacks=cb_state,
            )

        self._tracker.cleanup_invocation(inv)
