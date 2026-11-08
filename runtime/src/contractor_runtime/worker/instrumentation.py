"""Single ADK instrumentation plugin for one allocation-local Worker."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from google.adk.plugins import BasePlugin

from contractor_runtime.adapters import RuntimeInstrumentation, RuntimeSpan
from contractor_runtime.adapters.content import capture_span_content, model_request_content
from contractor_runtime.adapters.genai import usage_attributes
from contractor_runtime.adapters.instrumentation import end_isolated_span, start_isolated_span
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.tool_errors import ToolExecutionError, ToolResultDiagnostic
from contractor_runtime.llm.response import output_limit_reached
from contractor_runtime.observations import WorkspaceToolObservation
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer
from contractor_runtime.telemetry.metrics import (
    SAFE_ERROR_CODE,
    MetricsState,
    bind_tool_metric_correlation,
    reset_tool_metric_correlation,
    safe_error_code,
)
from contractor_runtime.toolsets.common.execution import WorkerExecutionState
from contractor_runtime.worker.observations import (
    WorkspaceObservationReducer,
    WorkspaceObservationSource,
)
from contractor_runtime.worker.state import InvocationPhase, WorkerStateStore


class InvocationBudget(Protocol):
    def before_model_call(self) -> None: ...

    def before_tool_call(self) -> None: ...

    def after_model_response(self, usage: Any | None, *, record_prompt: bool = True) -> None: ...

    def should_summarize(self) -> bool: ...


class ToolHistory(Protocol):
    """Invocation-local history bookkeeping, independent of plugin locking."""

    def observe_event(self, event: Any) -> None: ...

    def admitted(self, tool_context: Any) -> None: ...

    async def completed(self, tool_context: Any, result: Any) -> None: ...

    async def failed(self, tool_context: Any, result: Any, *, cancelled: bool = False) -> None: ...


class CompactionTrigger(Protocol):
    def should_compact(self, latest_prompt_tokens: int | None) -> bool: ...

    def after_main_response(self, latest_prompt_tokens: int | None) -> None: ...


class WorkerCompactionRequested(RuntimeError):
    """Safe control signal: the next main-model call crosses the boundary."""

    def __init__(self) -> None:
        super().__init__("Worker context compaction requested")


class WorkerSummarizationRequested(RuntimeError):
    """Content-free control signal used to unwind the normal ADK Runner."""

    def __init__(self) -> None:
        super().__init__("Worker terminal summarization requested")


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
    known: bool
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
        execution: WorkerExecutionState,
        budget: Callable[[], InvocationBudget | None],
        instrumentation: RuntimeInstrumentation | None,
        model_alias: str,
        observe_artifacts: Callable[[Any, int], None],
        allowed_tool_names: frozenset[str],
        check_admission: Callable[[], None] | None = None,
        history: ToolHistory | None = None,
        workspace_observation_source: WorkspaceObservationSource | None = None,
        summarizer_enabled: bool = False,
        compaction: CompactionTrigger | None = None,
    ) -> None:
        super().__init__(name="contractor_worker_instrumentation")
        self._state = state
        self._execution = execution
        self._metrics: MetricsState = state.metrics
        self._budget = budget
        self._instrumentation = instrumentation
        self._model_alias = model_alias
        self._observe_artifacts = observe_artifacts
        self._allowed_tool_names = frozenset(allowed_tool_names)
        self._check_admission = check_admission
        self._history = history
        self._workspace_observation_source = workspace_observation_source
        self._summarizer_enabled = summarizer_enabled
        self._compaction = compaction
        self._lock = asyncio.Lock()
        self._prepared: tuple[str, str] | None = None
        self._continuation: tuple[str, str] | None = None
        self._active_session_identity: tuple[Any, ...] | None = None
        self._active_invocation_id: str | None = None
        self._active_sdk_invocation_id: str | None = None
        self._used_sdk_invocation_ids: set[str] = set()
        self._invocation_metrics: InvocationMetricsReducer | None = None
        self._workspace_observations: WorkspaceObservationReducer | None = None
        self._pending_models: list[RuntimeSpan | None] = []
        self._pending_auxiliary_models: dict[str, RuntimeSpan | None] = {}
        self._pending_tools: dict[int, _PendingTool] = {}
        self._settling_tools: set[int] = set()
        self._next_tool_ordinal = 1
        self._projection_failed = False
        self._model_output_limited = False
        self._tool_cancellation_observed = False
        self._closed = False

    def prepare_invocation(self, *, invocation_id: str, subtask_id: str) -> None:
        if self._closed:
            raise RuntimeError("Worker instrumentation is closed")
        if self._prepared is not None or self._active_invocation_id is not None:
            raise RuntimeError("Worker instrumentation already has an invocation")
        self._projection_failed = False
        self._tool_cancellation_observed = False
        self._prepared = (invocation_id, subtask_id)

    def prepare_continuation(self, *, invocation_id: str, sdk_invocation_id: str) -> None:
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
            or self._settling_tools
            or not isinstance(sdk_invocation_id, str)
            or not sdk_invocation_id
            or sdk_invocation_id in self._used_sdk_invocation_ids
        ):
            raise RuntimeError("Worker instrumentation cannot continue this invocation")
        self._admit_call()
        self._continuation = (invocation_id, sdk_invocation_id)

    async def before_run_callback(self, *, invocation_context: Any) -> None:
        async with self._lock:
            if self._continuation is not None:
                if (
                    self._continuation[1] != invocation_context.invocation_id
                    or not self._is_active(self._continuation[0])
                    or self._active_session_identity != _session_identity(invocation_context)
                ):
                    raise RuntimeError("Worker instrumentation continuation is stale")
                self._execution.check()
                self._model_output_limited = False
                self._tool_cancellation_observed = False
                self._active_sdk_invocation_id = self._continuation[1]
                self._used_sdk_invocation_ids.add(self._continuation[1])
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
            self._model_output_limited = False
            self._tool_cancellation_observed = False
            self._active_invocation_id = prepared[0]
            self._active_sdk_invocation_id = invocation_context.invocation_id
            self._used_sdk_invocation_ids = {invocation_context.invocation_id}
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
            self._require_sdk_context(callback_context)
            self._admit_call()
            if self._pending_tools or self._settling_tools:
                raise RuntimeError("Worker model call overlapped unfinished tools")
            budget = self._budget()
            if budget is not None and budget.should_summarize():
                if self._pending_tools:
                    raise RuntimeError("Worker model call overlapped unfinished tools")
                snapshot = await self._state.request_summarization(
                    invocation_id=self._require_logical_invocation_id()
                )
                _install_session_snapshot(callback_context, snapshot)
                raise WorkerSummarizationRequested()
            if (
                budget is not None
                and self._compaction is not None
                and self._compaction.should_compact(budget.latest_prompt_tokens)
            ):
                # Raised before charging: the boundary call never starts.
                raise WorkerCompactionRequested()
            if budget is not None:
                budget.before_model_call()
            reducer = self._require_reducer()
            self._metrics.record_model_call()
            reducer.record_model_call()
            self._pending_models.append(
                start_isolated_span(
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
            self._require_sdk_context(callback_context)
            if not self._pending_models:
                return
            content = getattr(llm_response, "content", None)
            for part in getattr(content, "parts", None) or ():
                function_call = getattr(part, "function_call", None)
                if function_call is not None and not function_call.id:
                    function_call.id = f"contractor-tool-{uuid.uuid4().hex}"
            span = self._pending_models.pop(0)
            usage = llm_response.usage_metadata
            capture_span_content(span, output=lambda: llm_response.content)
            self._metrics.record_model_usage(usage)
            self._require_reducer().record_model_usage(usage)
            attributes = usage_attributes(usage)
            limited = output_limit_reached(llm_response)
            self._model_output_limited = self._model_output_limited or limited
            if limited:
                attributes["error.type"] = "OutputTokenLimitExceeded"
            end_isolated_span(
                span, outcome="failed" if limited else "succeeded", attributes=attributes
            )
            try:
                budget = self._budget()
                if budget is not None:
                    budget.after_model_response(usage)
                    if self._compaction is not None:
                        # Stops before this response's tools are dispatched.
                        self._compaction.after_main_response(budget.latest_prompt_tokens)
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
            self._require_sdk_context(callback_context)
            if not self._pending_models:
                return
            span = self._pending_models.pop(0)
            try:
                received, usage = _rejected_response_usage(error)
                end_isolated_span(
                    span,
                    outcome="failed",
                    attributes={
                        "error.type": span_error_type(error),
                        **usage_attributes(usage),
                    },
                )
                self._metrics.record_model_error(error)
                self._require_reducer().record_model_error()
                if received:
                    self._record_spent_usage_locked(usage)
            finally:
                await self._publish_locked(callback_context)

    async def before_result_finalizer_call(self, *, invocation_id: str) -> None:
        """Account the isolated ADK finalizer in the normal Worker budget."""

        await self._before_auxiliary_call("result_finalizer", invocation_id)

    async def after_result_finalizer_call(self, *, invocation_id: str, usage: Any | None) -> None:
        await self._after_auxiliary_call("result_finalizer", invocation_id, usage)

    async def on_result_finalizer_error(self, *, invocation_id: str, error: BaseException) -> None:
        await self._on_auxiliary_error("result_finalizer", invocation_id, error)

    async def before_context_compactor_call(self, *, invocation_id: str) -> None:
        """Account one compactor call in the owning invocation (WRK-R186)."""

        await self._before_auxiliary_call("context_compactor", invocation_id)

    async def after_context_compactor_call(self, *, invocation_id: str, usage: Any | None) -> None:
        await self._after_auxiliary_call("context_compactor", invocation_id, usage)

    async def on_context_compactor_error(self, *, invocation_id: str, error: BaseException) -> None:
        await self._on_auxiliary_error("context_compactor", invocation_id, error)

    async def _before_auxiliary_call(self, phase: str, invocation_id: str) -> None:
        async with self._lock:
            if not self._is_active(invocation_id):
                raise RuntimeError("Worker auxiliary model invocation is stale")
            self._admit_call()
            if phase in self._pending_auxiliary_models:
                raise RuntimeError("Worker auxiliary phase already has a model call")
            budget = self._budget()
            if budget is not None:
                budget.before_model_call()
            self._metrics.record_model_call()
            self._require_reducer().record_model_call()
            self._pending_auxiliary_models[phase] = start_isolated_span(
                self._instrumentation,
                "contractor.worker.model",
                {
                    "operation.kind": "model",
                    "model.alias": self._model_alias,
                    "model.phase": phase,
                },
            )
            try:
                await self._publish_locked(None)
            except asyncio.CancelledError:
                # Before-call has not returned ownership to the finalizer yet.
                # Close its partial registration under this lock, without another
                # publication/await that cancellation could interrupt again.
                span = self._pending_auxiliary_models.pop(phase)
                end_isolated_span(span, outcome="cancelled", attributes={"model.phase": phase})
                raise

    async def _after_auxiliary_call(
        self, phase: str, invocation_id: str, usage: Any | None
    ) -> None:
        async with self._lock:
            if not self._is_active(invocation_id) or phase not in self._pending_auxiliary_models:
                raise RuntimeError("Worker auxiliary model completion is stale")
            span = self._pending_auxiliary_models.pop(phase)
            self._metrics.record_model_usage(usage)
            self._require_reducer().record_model_usage(usage)
            end_isolated_span(
                span,
                outcome="succeeded",
                attributes={"model.phase": phase, **usage_attributes(usage)},
            )
            try:
                budget = self._budget()
                if budget is not None:
                    # Auxiliary usage never replaces the main prompt measurement.
                    budget.after_model_response(usage, record_prompt=False)
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

    async def _on_auxiliary_error(
        self, phase: str, invocation_id: str, error: BaseException
    ) -> None:
        async with self._lock:
            if not self._is_active(invocation_id) or phase not in self._pending_auxiliary_models:
                return
            span = self._pending_auxiliary_models.pop(phase)
            try:
                if isinstance(error, asyncio.CancelledError):
                    end_isolated_span(span, outcome="cancelled", attributes={"model.phase": phase})
                else:
                    safe_error = (
                        error if isinstance(error, Exception) else RuntimeError("model failed")
                    )
                    received, usage = _rejected_response_usage(safe_error)
                    end_isolated_span(
                        span,
                        outcome="failed",
                        attributes={
                            "error.type": span_error_type(safe_error),
                            "model.phase": phase,
                            **usage_attributes(usage),
                        },
                    )
                    self._metrics.record_model_error(safe_error)
                    self._require_reducer().record_model_error()
                    if received:
                        self._record_spent_usage_locked(usage)
            finally:
                await self._publish_locked(None)

    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
        async with self._lock:
            self._require_sdk_context(invocation_context)
            if self._history is not None:
                self._history.observe_event(event)

    def logical_invocation_id_for_tool(self, tool_context: Any) -> str:
        """Resolve one native tool context to its active Contractor invocation."""

        self._require_sdk_context(tool_context)
        return self._require_logical_invocation_id()

    async def before_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
    ) -> dict[str, Any] | None:
        async with self._lock:
            self._require_sdk_context(tool_context)
            self._admit_call()
            if id(tool_context) in self._pending_tools or id(tool_context) in self._settling_tools:
                raise RuntimeError("Worker tool callback was already admitted")
            budget = self._budget()
            if budget is not None:
                budget.before_tool_call()
            invocation_id = self._require_logical_invocation_id()
            owner = getattr(tool, "func", tool)
            raw_name = getattr(tool, "name", None)
            known = isinstance(raw_name, str) and raw_name in self._allowed_tool_names
            name = raw_name if known else "unknown_tool"
            ordinal = self._next_tool_ordinal
            self._next_tool_ordinal += 1
            correlation_id = f"{invocation_id}:tool:{ordinal}"
            metric_token = bind_tool_metric_correlation(correlation_id)
            cursor = getattr(owner, "artifact_observation_cursor", None) if known else None
            pending = _PendingTool(
                correlation_id=correlation_id,
                invocation_id=invocation_id,
                name=name,
                started_ns=time.perf_counter_ns(),
                span=start_isolated_span(
                    self._instrumentation,
                    "contractor.worker.tool",
                    {"operation.kind": "tool", "tool.name": name},
                ),
                owner=owner,
                ordinal=ordinal,
                observation_cursor=cursor if type(cursor) is int and cursor >= 0 else None,
                metric_token=metric_token,
                known=known,
            )
            self._pending_tools[id(tool_context)] = pending
            if self._history is not None:
                self._history.admitted(tool_context)
            if known:
                capture_span_content(pending.span, input=lambda: tool_args)
            rejection = None
            raw_argument_error = getattr(owner, "contractor_raw_argument_error", None)
            if known and callable(raw_argument_error):
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
            self._require_sdk_context(tool_context)
            pending = self._pending_tools.pop(id(tool_context), None)
            if pending is None:
                return
            self._settling_tools.add(id(tool_context))
        try:
            # Large result copies belong to the history owner. Its asynchronous
            # serialization must never hold the plugin's accounting lock.
            if self._history is not None:
                await self._history.completed(tool_context, result)
        finally:
            async with self._lock:
                self._settling_tools.discard(id(tool_context))
                failed = pending.error is not None or _result_is_failure(result)
                if pending.known:
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
        error: BaseException,
    ) -> dict[str, Any]:
        del tool_args
        cancelled = isinstance(error, asyncio.CancelledError)
        safe_error = (
            _RecordedToolFailure("tool_call_cancelled", retryable=True)
            if cancelled
            else error
            if isinstance(error, Exception)
            else RuntimeError("tool failed")
        )
        async with self._lock:
            self._require_sdk_context(tool_context)
            if cancelled:
                self._tool_cancellation_observed = True
            pending = self._pending_tools.pop(id(tool_context), None)
            if pending is None:
                # A duplicate/error cleanup callback cannot create another
                # semantic attempt, metric or span after its owner settled.
                if cancelled or isinstance(safe_error, ToolExecutionError):
                    raise error
                raw_name = getattr(tool, "name", None)
                known = isinstance(raw_name, str) and raw_name in self._allowed_tool_names
                name = raw_name if known else "unknown_tool"
                return _safe_tool_response(name, safe_error)
            pending.error = safe_error
            response = _safe_tool_response(pending.name, safe_error)
            self._settling_tools.add(id(tool_context))
        try:
            if self._history is not None:
                await self._history.failed(tool_context, response, cancelled=cancelled)
        except (asyncio.CancelledError, Exception):
            if not cancelled:
                raise
            # Cancellation owns this outcome. The history owner retains the
            # admitted call for conservative sealing when its cleanup fails;
            # neither a cleanup exception nor a second cancellation recovers it.
        finally:
            async with self._lock:
                self._settling_tools.discard(id(tool_context))
                await self._finish_tool_locked(
                    pending,
                    failed=True,
                    error=safe_error,
                    callback_context=tool_context,
                    tool_args=None,
                    result=None,
                    cancelled=cancelled,
                )
        if cancelled or isinstance(safe_error, ToolExecutionError):
            raise error
        return response

    def _record_spent_usage_locked(self, usage: Any | None) -> None:
        """Account a Gateway response that was received but rejected.

        The provider already spent these tokens, exactly as for an accepted
        response; the budget may therefore report exhaustion.
        """
        self._metrics.record_model_usage(usage)
        self._require_reducer().record_model_usage(usage)
        budget = self._budget()
        if budget is not None:
            budget.after_model_response(usage)

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
            if self._settling_tools:
                raise RuntimeError("Worker tool history is still settling")
            while self._pending_models:
                end_isolated_span(self._pending_models.pop(0), outcome="cancelled")
            for auxiliary_phase, span in tuple(self._pending_auxiliary_models.items()):
                self._pending_auxiliary_models.pop(auxiliary_phase, None)
                end_isolated_span(
                    span, outcome="cancelled", attributes={"model.phase": auxiliary_phase}
                )
            for key, pending in tuple(self._pending_tools.items()):
                self._pending_tools.pop(key, None)
                await self._finish_tool_locked(
                    pending,
                    failed=True,
                    error=_RecordedToolFailure("tool_call_cancelled", retryable=True),
                    callback_context=None,
                    tool_args=None,
                    result=None,
                    cancelled=True,
                )
            snapshot = await self._state.complete_invocation(
                invocation_id=invocation_id,
                phase=phase,
                metrics=self._require_reducer().snapshot(),
                workspace=self._workspace_snapshot(),
            )
            self._active_invocation_id = None
            self._active_sdk_invocation_id = None
            self._used_sdk_invocation_ids.clear()
            self._active_session_identity = None
            self._continuation = None
            self._invocation_metrics = None
            self._workspace_observations = None
            return snapshot

    async def record_completion(self, value) -> None:
        """Publish programmatic phase facts without model/tool accounting."""
        async with self._lock:
            self._metrics.record_completion(value)
            await self._state.sync_metrics()

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            if self._settling_tools:
                raise RuntimeError("Worker tool history is still settling")
            self._closed = True
            self._prepared = None
            self._continuation = None
            while self._pending_models:
                end_isolated_span(self._pending_models.pop(0), outcome="cancelled")
            for auxiliary_phase, span in tuple(self._pending_auxiliary_models.items()):
                self._pending_auxiliary_models.pop(auxiliary_phase, None)
                end_isolated_span(
                    span, outcome="cancelled", attributes={"model.phase": auxiliary_phase}
                )
            for pending in self._pending_tools.values():
                # A tool still running when the Worker closes was cancelled; the
                # allocation report records it like any other tool call.
                self._record_tool_metric_locked(
                    pending,
                    failed=True,
                    error=_RecordedToolFailure("tool_call_cancelled", retryable=True),
                )
                end_isolated_span(pending.span, outcome="cancelled")
                self._metrics.release_tool_correlation(pending.correlation_id)
                with contextlib.suppress(RuntimeError, ValueError):
                    reset_tool_metric_correlation(pending.metric_token)
            self._pending_tools.clear()
            self._active_invocation_id = None
            self._active_sdk_invocation_id = None
            self._used_sdk_invocation_ids.clear()
            self._active_session_identity = None
            self._continuation = None
            self._invocation_metrics = None
            self._workspace_observations = None

    @property
    def tool_cancellation_observed(self) -> bool:
        """An authorized native tool callback observed cancellation in this turn."""
        return self._tool_cancellation_observed

    @property
    def model_output_limited(self) -> bool:
        """Remember a limited response even when ADK emits no corresponding event."""
        return self._model_output_limited

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
        cancelled: bool = False,
    ) -> None:
        failed = self._record_tool_metric_locked(pending, failed=failed, error=error) or failed
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
        end_isolated_span(
            pending.span,
            outcome="cancelled" if cancelled else "failed" if failed else "succeeded",
            attributes=(
                {"error.type": span_error_type(error), "error.code": safe_error_code(error)}
                if failed and error is not None
                else None
            ),
        )
        self._metrics.release_tool_correlation(pending.correlation_id)
        # Parallel ADK tool callbacks may finish in a copied task context. A
        # token can only be reset in the context that created it; when this is
        # that context we clear it, otherwise the short-lived task owns it.
        with contextlib.suppress(RuntimeError, ValueError):
            reset_tool_metric_correlation(pending.metric_token)
        await self._publish_locked(callback_context)

    def _record_tool_metric_locked(
        self, pending: _PendingTool, *, failed: bool, error: Exception | None
    ) -> bool:
        """Record one allocation tool call unless the tool recorded it itself.

        Returns whether the recorded call failed.
        """

        recorded_failure = self._metrics.correlated_tool_outcome(pending.correlation_id)
        if recorded_failure is not None:
            return recorded_failure
        bounded_error: Exception | None = None
        if failed:
            bounded_error = _RecordedToolFailure(
                safe_error_code(error),
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
        return failed

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

    def _is_active_sdk_context(self, context: Any) -> bool:
        return (
            self._active_invocation_id is not None
            and self._continuation is None
            and self._active_sdk_invocation_id == getattr(context, "invocation_id", None)
            and self._active_session_identity == _session_identity(context)
        )

    def _require_sdk_context(self, context: Any) -> None:
        if not self._is_active_sdk_context(context):
            raise RuntimeError("Worker SDK callback is stale or belongs to another session")

    def _require_logical_invocation_id(self) -> str:
        invocation_id = self._active_invocation_id
        if invocation_id is None:
            raise RuntimeError("Worker invocation is unavailable")
        return invocation_id

    def _admit_call(self) -> None:
        self._execution.check()
        if self._check_admission is not None:
            self._check_admission()

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
    message = (
        error.diagnostic_message
        if isinstance(error, ToolInputError | ToolResultDiagnostic)
        else f"{tool_name} failed ({type(error).__name__})"
    )
    response = {
        "ok": False,
        "error": {
            "code": safe_error_code(error),
            "message": message.encode("utf-8", errors="replace")[:512].decode(
                "utf-8", errors="ignore"
            ),
            "retryable": bool(getattr(error, "retryable", False)),
        },
    }
    if isinstance(error, ToolResultDiagnostic):
        response["error"].update(error.model_details)
    return response


def _result_is_failure(result: Any) -> bool:
    return isinstance(result, Mapping) and (
        result.get("ok") is False
        or isinstance(result.get("error_code"), str)
        or isinstance(result.get("error"), str | Mapping)
    )


def span_error_type(error: Exception) -> str:
    """The ``error.type`` span attribute: a safe provider type or the class name."""

    error_type = getattr(error, "provider_error_type", type(error).__name__)
    if not isinstance(error_type, str) or SAFE_ERROR_CODE.fullmatch(error_type) is None:
        return type(error).__name__
    return error_type


def _rejected_response_usage(error: BaseException) -> tuple[bool, Any | None]:
    """Return whether a model error carries a received response and its usage."""

    if getattr(error, "response_received", False) is not True:
        return False, None
    return True, getattr(error, "usage_metadata", None)
