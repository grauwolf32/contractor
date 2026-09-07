"""Single ADK instrumentation plugin for one allocation-local Worker."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from google.adk.plugins import BasePlugin

from contractor_runtime.adapters import RuntimeInstrumentation, RuntimeSpan, TelemetryAttribute
from contractor_runtime.adapters.content import capture_span_content, model_request_content
from contractor_runtime.metrics import (
    MetricsState,
    bind_tool_metric_correlation,
    reset_tool_metric_correlation,
)
from contractor_runtime.model_response import output_limit_reached
from contractor_runtime.observations import (
    WorkspaceObservationReducer,
    WorkspaceObservationSource,
    WorkspaceToolObservation,
)
from contractor_runtime.token_usage import project_token_usage
from contractor_runtime.worker_state import InvocationPhase, WorkerStateStore

_SAFE_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_UINT64 = 2**64 - 1
_MAX_INVOCATION_TOOL_NAMES = 256


class InvocationBudget(Protocol):
    def before_model_call(self) -> None: ...

    def before_tool_call(self) -> None: ...

    def after_model_response(self, usage: Any | None) -> None: ...

    def should_summarize(self) -> bool: ...


class WorkerSummarizationRequested(RuntimeError):
    """Content-free control signal used to unwind the normal ADK Runner."""

    def __init__(self) -> None:
        super().__init__("Worker terminal summarization requested")


@dataclass(slots=True)
class InvocationMetricsReducer:
    """Content-free counters for exactly one Worker invocation."""

    model_calls: int = 0
    model_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    token_usage_unavailable: int = 0
    latest_prompt_tokens: int | None = None
    tool_calls: int = 0
    tool_errors: int = 0
    tools: dict[str, dict[str, int]] = field(default_factory=dict)
    truncated: bool = False

    def record_model_call(self) -> None:
        self.model_calls = _saturating_add(self.model_calls, 1)

    def record_model_usage(self, usage: Any | None) -> None:
        projected = project_token_usage(usage)
        self.latest_prompt_tokens = projected.prompt_tokens
        if projected.total_unavailable:
            self.token_usage_unavailable = _saturating_add(self.token_usage_unavailable, 1)
        for value, field_name in (
            (projected.prompt_tokens, "input_tokens"),
            (projected.output_tokens, "output_tokens"),
            (projected.total_tokens, "total_tokens"),
            (projected.cached_input_tokens, "cached_input_tokens"),
        ):
            if value is not None:
                setattr(self, field_name, _saturating_add(getattr(self, field_name), value))

    def record_model_error(self) -> None:
        self.model_errors = _saturating_add(self.model_errors, 1)

    def record_tool_call(self, name: str, *, failed: bool) -> None:
        self.tool_calls = _saturating_add(self.tool_calls, 1)
        if failed:
            self.tool_errors = _saturating_add(self.tool_errors, 1)
        identifier = _metric_identifier(name)
        aggregate = self.tools.get(identifier)
        if aggregate is None:
            if len(self.tools) >= _MAX_INVOCATION_TOOL_NAMES:
                self.truncated = True
                return
            aggregate = {"calls": 0, "failures": 0}
            self.tools[identifier] = aggregate
        aggregate["calls"] = _saturating_add(aggregate["calls"], 1)
        if failed:
            aggregate["failures"] = _saturating_add(aggregate["failures"], 1)

    def snapshot(self) -> dict[str, Any]:
        return {
            "modelCalls": self.model_calls,
            "modelErrors": self.model_errors,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
            "cachedInputTokens": self.cached_input_tokens,
            "tokenUsageUnavailable": self.token_usage_unavailable,
            "latestPromptTokens": self.latest_prompt_tokens,
            "toolCalls": self.tool_calls,
            "toolErrors": self.tool_errors,
            "tools": {name: dict(values) for name, values in sorted(self.tools.items())},
            "truncated": self.truncated,
        }


@dataclass(slots=True)
class _PendingTool:
    correlation_id: str
    invocation_id: str
    name: str
    started_ns: int
    span: RuntimeSpan | None
    owner: Any
    ordinal: int
    observation_cursor: int | None
    metric_token: Any
    error: Exception | None = None


class _RecordedToolFailure(RuntimeError):
    def __init__(self, code: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__("bounded Worker tool failure")


class WorkerInstrumentationPlugin(BasePlugin):
    """Own ADK callback correlation, budgets, telemetry and State publication."""

    def __init__(
        self,
        *,
        state: WorkerStateStore,
        budget: Callable[[], InvocationBudget | None],
        instrumentation: RuntimeInstrumentation | None,
        model_alias: str,
        observe_artifacts: Callable[[Any, int], None],
        workspace_observation_source: WorkspaceObservationSource | None = None,
        summarizer_enabled: bool = False,
    ) -> None:
        super().__init__(name="contractor_worker_instrumentation")
        self._state = state
        self._metrics: MetricsState = state.metrics
        self._budget = budget
        self._instrumentation = instrumentation
        self._model_alias = model_alias
        self._observe_artifacts = observe_artifacts
        self._workspace_observation_source = workspace_observation_source
        self._summarizer_enabled = summarizer_enabled
        self._lock = asyncio.Lock()
        self._prepared: tuple[str, str] | None = None
        self._continuation: str | None = None
        self._active_session_identity: tuple[Any, ...] | None = None
        self._active_invocation_id: str | None = None
        self._invocation_metrics: InvocationMetricsReducer | None = None
        self._workspace_observations: WorkspaceObservationReducer | None = None
        self._pending_models: list[RuntimeSpan | None] = []
        self._pending_auxiliary_models: dict[str, RuntimeSpan | None] = {}
        self._pending_tools: dict[int, _PendingTool] = {}
        self._next_tool_ordinal = 1
        self._projection_failed = False
        self._closed = False

    def prepare_invocation(self, *, invocation_id: str, subtask_id: str) -> None:
        if self._closed:
            raise RuntimeError("Worker instrumentation is closed")
        if self._prepared is not None or self._active_invocation_id is not None:
            raise RuntimeError("Worker instrumentation already has an invocation")
        self._projection_failed = False
        self._prepared = (invocation_id, subtask_id)

    def prepare_continuation(self, *, invocation_id: str) -> None:
        """Authorize one more Runner turn of the active logical invocation.

        The Runtime completion boundary calls this only after the prior turn has
        settled. It grants no additional model/tool budget and does not begin a
        new invocation, clear observations or reset the reducer.
        """
        if (
            self._closed
            or self._active_invocation_id != invocation_id
            or self._prepared is not None
            or self._continuation is not None
            or self._pending_models
            or self._pending_auxiliary_models
            or self._pending_tools
        ):
            raise RuntimeError("Worker instrumentation cannot continue this invocation")
        self._state.execution.check()
        self._continuation = invocation_id

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        async with self._lock:
            if self._continuation is not None:
                if (
                    self._continuation != invocation_context.invocation_id
                    or not self._is_active(invocation_context.invocation_id)
                    or self._active_session_identity != _session_identity(invocation_context)
                ):
                    raise RuntimeError("Worker instrumentation continuation is stale")
                self._state.execution.check()
                self._continuation = None
                _install_session_snapshot(invocation_context, await self._state.snapshot())
                return
        workspace_observations: WorkspaceObservationReducer | None = None
        workspace_projection_failed = False
        source = self._workspace_observation_source
        if source is not None:
            try:
                metadata = await source.observation_metadata()
                workspace_observations = WorkspaceObservationReducer.from_metadata(metadata)
            except asyncio.CancelledError:
                raise
            except Exception:
                workspace_projection_failed = True
        async with self._lock:
            prepared = self._prepared
            if prepared is None or prepared[0] != invocation_context.invocation_id:
                raise RuntimeError("Worker instrumentation invocation is not prepared")
            self._prepared = None
            self._active_invocation_id = prepared[0]
            self._active_session_identity = _session_identity(invocation_context)
            self._invocation_metrics = InvocationMetricsReducer()
            self._workspace_observations = workspace_observations
            self._projection_failed = self._projection_failed or workspace_projection_failed
            self._pending_models.clear()
            self._pending_auxiliary_models.clear()
            self._pending_tools.clear()
            self._next_tool_ordinal = 1
            snapshot = await self._state.begin_invocation(
                invocation_id=prepared[0],
                subtask_id=prepared[1],
                metrics=self._invocation_metrics.snapshot(),
                workspace=self._workspace_snapshot(),
                summarizer_enabled=self._summarizer_enabled,
            )
            _install_session_snapshot(invocation_context, snapshot)

    async def before_model_callback(self, *, callback_context: Any, llm_request: Any) -> None:
        async with self._lock:
            self._state.execution.check()
            if not self._is_active(callback_context.invocation_id):
                return
            budget = self._budget()
            if budget is not None and budget.should_summarize():
                if self._pending_tools:
                    raise RuntimeError("Worker model call overlapped unfinished tools")
                snapshot = await self._state.request_summarization(
                    invocation_id=callback_context.invocation_id
                )
                _install_session_snapshot(callback_context, snapshot)
                raise WorkerSummarizationRequested()
            if budget is not None:
                budget.before_model_call()
            reducer = self._require_reducer()
            self._metrics.record_model_call()
            reducer.record_model_call()
            self._pending_models.append(
                _start_span(
                    self._instrumentation,
                    "contractor.worker.model",
                    {
                        "operation.kind": "model",
                        "model.alias": self._model_alias,
                    },
                )
            )
            capture_span_content(
                self._pending_models[-1], input=lambda: model_request_content(llm_request)
            )
            await self._publish_locked(callback_context)

    async def after_model_callback(self, *, callback_context: Any, llm_response: Any) -> None:
        async with self._lock:
            if not self._is_active(callback_context.invocation_id) or not self._pending_models:
                return
            span = self._pending_models.pop(0)
            usage = llm_response.usage_metadata
            capture_span_content(span, output=lambda: llm_response.content)
            self._metrics.record_model_usage(usage)
            self._require_reducer().record_model_usage(usage)
            attributes = _usage_attributes(usage)
            limited = output_limit_reached(llm_response)
            if limited:
                attributes["error.type"] = "OutputTokenLimitExceeded"
            _end_span(span, outcome="failed" if limited else "succeeded", attributes=attributes)
            try:
                budget = self._budget()
                if budget is not None:
                    budget.after_model_response(usage)
            finally:
                await self._publish_locked(callback_context)

    async def on_model_error_callback(
        self,
        *,
        callback_context: Any,
        llm_request: Any,
        error: Exception,
    ) -> None:
        del llm_request
        async with self._lock:
            if not self._is_active(callback_context.invocation_id) or not self._pending_models:
                return
            span = self._pending_models.pop(0)
            if type(error).__name__ == "WorkerBudgetExceeded":
                _end_span(span, outcome="rejected", attributes={"error.type": type(error).__name__})
            else:
                _end_span(
                    span,
                    outcome="failed",
                    attributes={"error.type": _safe_error_type(error)},
                )
                self._metrics.record_model_error(error)
                self._require_reducer().record_model_error()
            await self._publish_locked(callback_context)

    async def before_result_finalizer_call(self, *, invocation_id: str) -> None:
        """Account the isolated ADK finalizer in the normal Worker budget."""

        async with self._lock:
            if not self._is_active(invocation_id):
                raise RuntimeError("Worker result finalizer invocation is stale")
            phase = "result_finalizer"
            if phase in self._pending_auxiliary_models:
                raise RuntimeError("Worker result finalizer already has a model call")
            budget = self._budget()
            if budget is not None:
                budget.before_model_call()
            self._metrics.record_model_call()
            self._require_reducer().record_model_call()
            self._pending_auxiliary_models[phase] = _start_span(
                self._instrumentation,
                "contractor.worker.model",
                {
                    "operation.kind": "model",
                    "model.alias": self._model_alias,
                    "model.phase": phase,
                },
            )
            await self._publish_locked(None)

    async def after_result_finalizer_call(self, *, invocation_id: str, usage: Any | None) -> None:
        async with self._lock:
            phase = "result_finalizer"
            if not self._is_active(invocation_id) or phase not in self._pending_auxiliary_models:
                raise RuntimeError("Worker result finalizer model completion is stale")
            span = self._pending_auxiliary_models.pop(phase)
            self._metrics.record_model_usage(usage)
            self._require_reducer().record_model_usage(usage)
            _end_span(
                span,
                outcome="succeeded",
                attributes={"model.phase": phase, **_usage_attributes(usage)},
            )
            try:
                budget = self._budget()
                if budget is not None:
                    budget.after_model_response(usage)
            finally:
                await self._publish_locked(None)

    async def capture_result_finalizer_content(
        self, *, invocation_id: str, input: Any = None, output: Any = None
    ) -> None:
        async with self._lock:
            if not self._is_active(invocation_id):
                return
            span = self._pending_auxiliary_models.get("result_finalizer")
            capture_span_content(
                span,
                input=(lambda: input) if input is not None else None,
                output=(lambda: output) if output is not None else None,
            )

    async def on_result_finalizer_error(self, *, invocation_id: str, error: BaseException) -> None:
        async with self._lock:
            phase = "result_finalizer"
            if not self._is_active(invocation_id) or phase not in self._pending_auxiliary_models:
                return
            span = self._pending_auxiliary_models.pop(phase)
            if isinstance(error, asyncio.CancelledError):
                _end_span(span, outcome="cancelled", attributes={"model.phase": phase})
            elif type(error).__name__ == "WorkerBudgetExceeded":
                _end_span(
                    span,
                    outcome="rejected",
                    attributes={
                        "error.type": type(error).__name__,
                        "model.phase": phase,
                    },
                )
            else:
                safe_error = error if isinstance(error, Exception) else RuntimeError("model failed")
                _end_span(
                    span,
                    outcome="failed",
                    attributes={
                        "error.type": _safe_error_type(safe_error),
                        "model.phase": phase,
                    },
                )
                self._metrics.record_model_error(safe_error)
                self._require_reducer().record_model_error()
            await self._publish_locked(None)

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> dict[str, Any] | None:
        async with self._lock:
            self._state.execution.check()
            invocation_id = tool_context.invocation_id
            if not self._is_active(invocation_id):
                return None
            budget = self._budget()
            if budget is not None:
                budget.before_tool_call()
            owner = getattr(tool, "func", tool)
            ordinal = self._next_tool_ordinal
            self._next_tool_ordinal += 1
            correlation_id = f"{invocation_id}:tool:{ordinal}"
            metric_token = bind_tool_metric_correlation(correlation_id)
            cursor = getattr(owner, "artifact_observation_cursor", None)
            pending = _PendingTool(
                correlation_id=correlation_id,
                invocation_id=invocation_id,
                name=str(tool.name),
                started_ns=time.perf_counter_ns(),
                span=_start_span(
                    self._instrumentation,
                    "contractor.worker.tool",
                    {"operation.kind": "tool", "tool.name": str(tool.name)},
                ),
                owner=owner,
                ordinal=ordinal,
                observation_cursor=cursor if type(cursor) is int and cursor >= 0 else None,
                metric_token=metric_token,
            )
            self._pending_tools[id(tool_context)] = pending
            capture_span_content(pending.span, input=lambda: tool_args)
            rejection = None
            raw_argument_error = getattr(owner, "contractor_raw_argument_error", None)
            if callable(raw_argument_error):
                try:
                    rejection = raw_argument_error(tool_args)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    rejection = _RecordedToolFailure(
                        "tool_argument_validation_failed", retryable=False
                    )
                if rejection is not None and not isinstance(rejection, Exception):
                    rejection = _RecordedToolFailure(
                        "tool_argument_validation_failed", retryable=False
                    )
            if rejection is not None:
                pending.error = rejection
            await self._publish_locked(tool_context)
            return _safe_tool_response(pending.name, rejection) if rejection is not None else None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: Any,
    ) -> None:
        del tool
        async with self._lock:
            pending = self._pending_tools.pop(id(tool_context), None)
            if pending is None:
                return
            failed = pending.error is not None or _result_is_failure(result)
            capture_span_content(pending.span, output=lambda: result)
            await self._finish_tool_locked(
                pending,
                failed=failed,
                error=pending.error,
                callback_context=tool_context,
                tool_args=tool_args,
                result=result,
            )

    async def on_tool_error_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        error: Exception,
    ) -> dict[str, Any]:
        del tool_args
        async with self._lock:
            pending = self._pending_tools.pop(id(tool_context), None)
            if pending is None:
                invocation_id = tool_context.invocation_id
                if not self._is_active(invocation_id):
                    return _safe_tool_response("unknown_tool", error)
                owner = getattr(tool, "func", tool)
                ordinal = self._next_tool_ordinal
                self._next_tool_ordinal += 1
                correlation_id = f"{invocation_id}:tool:{ordinal}"
                pending = _PendingTool(
                    correlation_id=correlation_id,
                    invocation_id=invocation_id,
                    # The absence of a matching before callback is the ADK
                    # unknown-tool path. Its name came from model output and
                    # must not become a State key or telemetry attribute.
                    name="unknown_tool",
                    started_ns=time.perf_counter_ns(),
                    span=_start_span(
                        self._instrumentation,
                        "contractor.worker.tool",
                        {"operation.kind": "tool", "tool.name": "unknown_tool"},
                    ),
                    owner=owner,
                    ordinal=ordinal,
                    observation_cursor=None,
                    metric_token=bind_tool_metric_correlation(correlation_id),
                )
            pending.error = error
            await self._finish_tool_locked(
                pending,
                failed=True,
                error=error,
                callback_context=tool_context,
                tool_args=None,
                result=None,
            )
            return _safe_tool_response(pending.name, error)

    async def record_unhandled_model_error(self, error: Exception) -> None:
        async with self._lock:
            if self._active_invocation_id is None:
                self._metrics.record_model_error(error)
                await self._state.sync_metrics()
                return
            self._metrics.record_model_error(error)
            self._require_reducer().record_model_error()
            await self._publish_locked(None)

    async def complete_invocation(
        self, *, invocation_id: str, phase: InvocationPhase
    ) -> dict[str, Any] | None:
        async with self._lock:
            if self._active_invocation_id is None:
                if self._prepared is not None and self._prepared[0] == invocation_id:
                    self._prepared = None
                return None
            if not self._is_active(invocation_id):
                raise RuntimeError("Worker instrumentation completion is stale")
            while self._pending_models:
                _end_span(self._pending_models.pop(0), outcome="cancelled")
            for phase, span in tuple(self._pending_auxiliary_models.items()):
                self._pending_auxiliary_models.pop(phase, None)
                _end_span(span, outcome="cancelled", attributes={"model.phase": phase})
            for key, pending in tuple(self._pending_tools.items()):
                self._pending_tools.pop(key, None)
                await self._finish_tool_locked(
                    pending,
                    failed=True,
                    error=_RecordedToolFailure("tool_call_cancelled", retryable=True),
                    callback_context=None,
                    tool_args=None,
                    result=None,
                )
            snapshot = await self._state.complete_invocation(
                invocation_id=invocation_id,
                phase=phase,
                metrics=self._require_reducer().snapshot(),
                workspace=self._workspace_snapshot(),
            )
            self._active_invocation_id = None
            self._active_session_identity = None
            self._continuation = None
            self._invocation_metrics = None
            self._workspace_observations = None
            return snapshot

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._prepared = None
            self._continuation = None
            while self._pending_models:
                _end_span(self._pending_models.pop(0), outcome="cancelled")
            for phase, span in tuple(self._pending_auxiliary_models.items()):
                self._pending_auxiliary_models.pop(phase, None)
                _end_span(span, outcome="cancelled", attributes={"model.phase": phase})
            for pending in self._pending_tools.values():
                _end_span(pending.span, outcome="cancelled")
                self._metrics.release_tool_correlation(pending.correlation_id)
                with contextlib.suppress(RuntimeError, ValueError):
                    reset_tool_metric_correlation(pending.metric_token)
            self._pending_tools.clear()
            self._active_invocation_id = None
            self._active_session_identity = None
            self._continuation = None
            self._invocation_metrics = None
            self._workspace_observations = None

    @property
    def projection_failed(self) -> bool:
        return self._projection_failed

    async def _finish_tool_locked(
        self,
        pending: _PendingTool,
        *,
        failed: bool,
        error: Exception | None,
        callback_context: Any | None,
        tool_args: Mapping[str, Any] | None,
        result: Any,
    ) -> None:
        recorded_failure = self._metrics.correlated_tool_outcome(pending.correlation_id)
        if recorded_failure is None:
            bounded_error: Exception | None = None
            if failed:
                code = _safe_error_code(error)
                bounded_error = _RecordedToolFailure(
                    code,
                    retryable=bool(getattr(error, "retryable", False)),
                )
            token = bind_tool_metric_correlation(pending.correlation_id)
            try:
                self._metrics.record_tool_call(
                    pending.name,
                    arguments={},
                    error=bounded_error,
                    duration_ms=max(
                        0,
                        int((time.perf_counter_ns() - pending.started_ns) / 1_000_000),
                    ),
                )
            finally:
                reset_tool_metric_correlation(token)
            recorded_failure = failed
        failed = failed or bool(recorded_failure)
        self._require_reducer().record_tool_call(pending.name, failed=failed)
        if not failed and pending.observation_cursor is not None:
            try:
                self._observe_artifacts(pending.owner, pending.observation_cursor)
            except Exception:
                self._projection_failed = True
        if not failed:
            extractor = getattr(pending.owner, "contractor_observation", None)
            workspace = self._workspace_observations
            if callable(extractor) and workspace is not None:
                try:
                    observation = extractor(tool_args or {}, result)
                    if observation is not None:
                        if not isinstance(observation, WorkspaceToolObservation):
                            raise TypeError("tool returned an invalid workspace observation")
                        workspace.record(observation, ordinal=pending.ordinal)
                except Exception:
                    workspace.mark_incomplete()
                    self._projection_failed = True
        _end_span(
            pending.span,
            outcome="failed" if failed else "succeeded",
            attributes=(
                {"error.type": _safe_error_type(error)} if failed and error is not None else None
            ),
        )
        self._metrics.release_tool_correlation(pending.correlation_id)
        # Parallel ADK tool callbacks may finish in a copied task context. A
        # token can only be reset in the context that created it; when this is
        # that context we clear it, otherwise the short-lived task owns it.
        with contextlib.suppress(RuntimeError, ValueError):
            reset_tool_metric_correlation(pending.metric_token)
        await self._publish_locked(callback_context)

    async def _publish_locked(self, callback_context: Any | None) -> None:
        active = self._active_invocation_id
        reducer = self._invocation_metrics
        if active is None or reducer is None:
            return
        try:
            snapshot = await self._state.publish_invocation_metrics(
                invocation_id=active,
                metrics=reducer.snapshot(),
                workspace=self._workspace_snapshot(),
            )
        except Exception:
            # Optional live projection cannot change a tool/model result. The
            # allocation report reducer remains authoritative and a later
            # lifecycle publication may recover.
            self._projection_failed = True
            return
        _install_session_snapshot(callback_context, snapshot)

    def _is_active(self, invocation_id: str) -> bool:
        return self._active_invocation_id == invocation_id

    def _require_reducer(self) -> InvocationMetricsReducer:
        reducer = self._invocation_metrics
        if reducer is None:
            raise RuntimeError("Worker invocation metrics are unavailable")
        return reducer

    def _workspace_snapshot(self) -> dict[str, Any] | None:
        reducer = self._workspace_observations
        return reducer.snapshot() if reducer is not None else None


def _session_identity(context: Any) -> tuple[Any, ...]:
    session = getattr(context, "session", None)
    identifier = getattr(session, "id", None)
    if identifier is None:
        return (id(session),)
    return (getattr(session, "app_name", None), getattr(session, "user_id", None), identifier)


def _install_session_snapshot(context: Any | None, snapshot: dict[str, Any]) -> None:
    if context is None:
        return
    session = getattr(context, "session", None)
    state = getattr(session, "state", None)
    if isinstance(state, dict):
        state["contractor"] = snapshot


def _safe_tool_response(tool_name: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": _safe_error_code(error),
            "message": f"{tool_name} failed ({type(error).__name__})",
            "retryable": bool(getattr(error, "retryable", False)),
        },
    }


def _safe_error_code(error: Exception | None) -> str:
    code = getattr(error, "code", "tool_call_failed")
    if not isinstance(code, str) or _SAFE_ERROR_CODE.fullmatch(code) is None:
        return "tool_call_failed"
    return code


def _result_is_failure(result: Any) -> bool:
    return isinstance(result, Mapping) and (
        result.get("ok") is False
        or isinstance(result.get("error_code"), str)
        or isinstance(result.get("error"), str | Mapping)
    )


def _safe_error_type(error: Exception) -> str:
    error_type = getattr(error, "provider_error_type", type(error).__name__)
    if not isinstance(error_type, str) or _SAFE_ERROR_CODE.fullmatch(error_type) is None:
        return type(error).__name__
    return error_type


def _start_span(
    instrumentation: RuntimeInstrumentation | None,
    name: str,
    attributes: Mapping[str, TelemetryAttribute] | None = None,
) -> RuntimeSpan | None:
    if instrumentation is None:
        return None
    try:
        return instrumentation.start_span(name, attributes=attributes)
    except Exception:
        return None


def _end_span(
    span: RuntimeSpan | None,
    *,
    outcome: str,
    attributes: Mapping[str, TelemetryAttribute] | None = None,
) -> None:
    if span is None:
        return
    try:
        span.end(outcome=outcome, attributes=attributes)
    except Exception:
        return


def _usage_attributes(usage: Any | None) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for source, target in (
        ("prompt_token_count", "tokens.input"),
        ("candidates_token_count", "tokens.output"),
        ("total_token_count", "tokens.total"),
        ("cached_content_token_count", "tokens.cached_input"),
    ):
        value = getattr(usage, source, None)
        if isinstance(value, int) and value >= 0:
            result[target] = value
    return result


def _metric_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized[:64] or "unknown"


def _saturating_add(current: int, increment: int) -> int:
    return min(_MAX_UINT64, current + increment)
