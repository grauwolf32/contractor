from __future__ import annotations

import itertools
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


def _to_dictish(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)

    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return value
            except Exception:
                pass

    return {}


def _jsonable(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return repr(value)


def _fingerprint(
    invocation_id: str | None,
    agent_name: str | None,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str:
    return "||".join(
        [
            invocation_id or "unknown_invocation",
            agent_name or "unknown_agent",
            tool_name,
            _jsonable(args or {}),
        ]
    )


@dataclass
class ToolMetrics:
    calls_total: int = 0
    success_total: int = 0
    exception_errors_total: int = 0
    result_errors_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls_total": self.calls_total,
            "success_total": self.success_total,
            "exception_errors_total": self.exception_errors_total,
            "result_errors_total": self.result_errors_total,
        }


@dataclass
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "thoughts_tokens": self.thoughts_tokens,
            "cached_tokens": self.cached_tokens,
            "tools": {
                tool_name: metrics.as_dict()
                for tool_name, metrics in self.tools.items()
            },
        }


@dataclass
class CallState:
    call_id: int
    invocation_id: str
    agent_name: str
    tool_name: str
    args: dict[str, Any]
    exception_seen: bool = False
    result_error_seen: bool = False
    finished: bool = False


class AdkMetricsPlugin(BasePlugin):
    def __init__(
        self,
        *,
        task_name: str,
        task_id: int,
        iteration: int,
        session_id: str,
        emit: Callable[..., Awaitable[None]],
        plugin_name: str | None = None,
        result_error_detector: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        super().__init__(
            name=plugin_name or f"metrics_plugin_{task_name}_{task_id}_{iteration}"
        )
        self._task_name = task_name
        self._task_id = task_id
        self._iteration = iteration
        self._session_id = session_id
        self._emit = emit
        self._result_error_detector = (
            result_error_detector or self._default_result_error_detector
        )

        self._metrics: dict[str, dict[str, AgentMetrics]] = defaultdict(
            lambda: defaultdict(AgentMetrics)
        )

        self._call_seq = itertools.count(1)
        self._pending_by_fp: dict[str, deque[int]] = defaultdict(deque)
        self._calls: dict[int, CallState] = {}

    def _agent_bucket(
        self, invocation_id: str | None, agent_name: str | None
    ) -> AgentMetrics:
        inv = invocation_id or "unknown_invocation"
        agent = agent_name or "unknown_agent"
        return self._metrics[inv][agent]

    def _tool_bucket(
        self,
        invocation_id: str | None,
        agent_name: str | None,
        tool_name: str,
    ) -> ToolMetrics:
        return self._agent_bucket(invocation_id, agent_name).tools[tool_name]

    def _extract_usage(self, llm_response: Any) -> dict[str, int]:
        usage = getattr(llm_response, "usage_metadata", None)
        data = _to_dictish(usage)
        return {
            "prompt": int(data.get("prompt_token_count") or 0),
            "completion": int(data.get("candidates_token_count") or 0),
            "total": int(data.get("total_token_count") or 0),
            "thoughts": int(data.get("thoughts_token_count") or 0),
            "cached": int(data.get("cached_content_token_count") or 0),
        }

    def _register_call(
        self,
        invocation_id: str | None,
        agent_name: str | None,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> CallState:
        inv = invocation_id or "unknown_invocation"
        agent = agent_name or "unknown_agent"
        actual_args = args or {}

        call = CallState(
            call_id=next(self._call_seq),
            invocation_id=inv,
            agent_name=agent,
            tool_name=tool_name,
            args=actual_args,
        )

        fp = _fingerprint(inv, agent, tool_name, actual_args)
        self._calls[call.call_id] = call
        self._pending_by_fp[fp].append(call.call_id)
        return call

    def _resolve_call(
        self,
        invocation_id: str | None,
        agent_name: str | None,
        tool_name: str,
        args: dict[str, Any] | None,
    ) -> Optional[CallState]:
        inv = invocation_id or "unknown_invocation"
        agent = agent_name or "unknown_agent"
        fp = _fingerprint(inv, agent, tool_name, args or {})
        queue = self._pending_by_fp.get(fp)
        if not queue:
            return None

        for call_id in queue:
            call = self._calls.get(call_id)
            if call is not None and not call.finished:
                return call
        return None

    def _finish_call(self, call: CallState) -> None:
        call.finished = True

    def _cleanup_invocation_calls(self, invocation_id: str) -> None:
        to_delete = [
            cid
            for cid, call in self._calls.items()
            if call.invocation_id == invocation_id
        ]
        for cid in to_delete:
            self._calls.pop(cid, None)

        fp_to_delete: list[str] = []
        for fp, queue in self._pending_by_fp.items():
            rest = deque([cid for cid in queue if cid in self._calls])
            if rest:
                self._pending_by_fp[fp] = rest
            else:
                fp_to_delete.append(fp)

        for fp in fp_to_delete:
            self._pending_by_fp.pop(fp, None)

    def _default_result_error_detector(self, tool_response: Any) -> bool:
        if not isinstance(tool_response, dict):
            return False

        status = str(tool_response.get("status", "")).strip().lower()
        if status in {"error", "failed", "failure"}:
            return True

        if tool_response.get("error") not in (None, False, "", [], {}):
            return True

        if tool_response.get("error_message") not in (None, ""):
            return True

        if tool_response.get("errors") not in (None, [], {}):
            return True

        return False

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
        invocation_id = getattr(tool_context, "invocation_id", None)
        agent_name = getattr(tool_context, "agent_name", None)

        self._register_call(invocation_id, agent_name, tool.name, actual_args)
        self._tool_bucket(invocation_id, agent_name, tool.name).calls_total += 1

        await self._emit(
            type="metrics_tool_call",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=invocation_id,
            agent_name=agent_name,
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
        actual_args = tool_args if tool_args is not None else (args or {})
        invocation_id = getattr(tool_context, "invocation_id", None)
        agent_name = getattr(tool_context, "agent_name", None)

        bucket = self._tool_bucket(invocation_id, agent_name, tool.name)
        call = self._resolve_call(invocation_id, agent_name, tool.name, actual_args)

        if call and not call.exception_seen:
            call.exception_seen = True
            bucket.exception_errors_total += 1
        elif call is None:
            bucket.exception_errors_total += 1

        await self._emit(
            type="metrics_tool_exception_error",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=invocation_id,
            agent_name=agent_name,
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
        actual_args = tool_args if tool_args is not None else (args or {})
        actual_result = tool_response if tool_response is not None else result
        invocation_id = getattr(tool_context, "invocation_id", None)
        agent_name = getattr(tool_context, "agent_name", None)

        bucket = self._tool_bucket(invocation_id, agent_name, tool.name)
        call = self._resolve_call(invocation_id, agent_name, tool.name, actual_args)
        is_result_error = self._result_error_detector(actual_result)

        if is_result_error:
            if call:
                if not call.result_error_seen and not call.exception_seen:
                    call.result_error_seen = True
                    bucket.result_errors_total += 1
            else:
                bucket.result_errors_total += 1
        else:
            if call:
                if not call.exception_seen and not call.result_error_seen:
                    bucket.success_total += 1
            else:
                bucket.success_total += 1

        if call:
            self._finish_call(call)

        await self._emit(
            type="metrics_tool_result",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=invocation_id,
            agent_name=agent_name,
            tool_name=tool.name,
            tool_args=actual_args,
            result=actual_result,
            result_error=is_result_error,
        )
        return None

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        invocation_id = getattr(callback_context, "invocation_id", None)
        agent_name = getattr(callback_context, "agent_name", None)
        usage = self._extract_usage(llm_response)

        bucket = self._agent_bucket(invocation_id, agent_name)
        bucket.llm_calls += 1
        bucket.prompt_tokens += usage["prompt"]
        bucket.completion_tokens += usage["completion"]
        bucket.total_tokens += usage["total"]
        bucket.thoughts_tokens += usage["thoughts"]
        bucket.cached_tokens += usage["cached"]

        await self._emit(
            type="metrics_llm_usage",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=invocation_id,
            agent_name=agent_name,
            usage=usage,
            model=getattr(llm_response, "model_version", None),
        )
        return None

    async def after_run_callback(self, *, invocation_context) -> None:
        invocation_id = (
            getattr(invocation_context, "invocation_id", None) or "unknown_invocation"
        )
        by_agent = self._metrics.pop(invocation_id, {})

        await self._emit(
            type="metrics_summary",
            task_name=self._task_name,
            task_id=self._task_id,
            iteration=self._iteration,
            session_id=self._session_id,
            invocation_id=invocation_id,
            agents={
                agent_name: metrics.as_dict()
                for agent_name, metrics in by_agent.items()
            },
        )

        self._cleanup_invocation_calls(invocation_id)
