"""Google ADK-backed allocation-local Worker runtime."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from starlette.types import ASGIApp

from contractor_runtime.a2a_server import (
    agent_card_dict,
    build_agent_card,
    build_worker_a2a_application,
)
from contractor_runtime.adapters import RuntimeInstrumentation
from contractor_runtime.adapters.instrumentation import end_isolated_span, start_isolated_span
from contractor_runtime.agent_skills.runtime import (
    EXACT_SKILL_TOOL_NAMES,
    PreparedAgentSkills,
)
from contractor_runtime.contracts import (
    API_VERSION,
    AgentStateSnapshot,
    StageContentRequest,
    WorkerCompletion,
    WorkerFailure,
    WorkerResult,
    WorkerSessionMode,
)
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.tool_errors import ToolExecutionError
from contractor_runtime.core.worker_completion import (
    ContinueCompletion,
    WorkerCompletionError,
    bind_worker_completion,
)
from contractor_runtime.llm.factory import ModelFactory, gateway_model
from contractor_runtime.llm.openai import (
    GatewayModelError,
    OpenAICompatibleGatewayLlm,
)
from contractor_runtime.llm.response import output_limit_reached
from contractor_runtime.projectfs import (
    WorkspaceAutoExporter,
    WorkspaceExportError,
)
from contractor_runtime.telemetry.execution import (
    ContentFreeInstrumentation,
    declares_sensitive_output,
)
from contractor_runtime.worker.adk_features import require_supported_adk_features
from contractor_runtime.worker.adk_turn import AdkTurnRunner
from contractor_runtime.worker.budget import WorkerBudgetExceeded, _InvocationBudget
from contractor_runtime.worker.compaction import (
    CompactionCoordinator,
    WorkerCompactionError,
    append_marker,
    run_compactor,
)
from contractor_runtime.worker.completion import WorkerCompletionPipeline, _lean_observations
from contractor_runtime.worker.failures import (
    _chained_error,
    _compaction_failure,
    _failure,
    _gateway_worker_failure,
)
from contractor_runtime.worker.function_tools import ContractorFunctionTool
from contractor_runtime.worker.history import ToolHistory
from contractor_runtime.worker.instrumentation import (
    WorkerCompactionRequested,
    WorkerInstrumentationPlugin,
    WorkerSummarizationRequested,
    span_error_type,
)
from contractor_runtime.worker.invocation import (
    INVOCATION_CLEANUP_TIMEOUT_SECONDS,
    InvocationDeadline,
    InvocationLifecycle,
)
from contractor_runtime.worker.one_shot import candidate_text, generation_config
from contractor_runtime.worker.prompts import (
    _exposes_private_value,
    _summarizer_secrets,
    _task_prompt,
)
from contractor_runtime.worker.sessions import (
    WorkerSessionLifecycle,
    WorkerSessionLifecycleError,
)
from contractor_runtime.worker.state import InvocationPhase, WorkerStateStore
from contractor_runtime.worker.summarizer import (
    TranscriptRecorder,
)

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext

MAX_STAGE_REQUEST_JSON_BYTES = 256 * 1024


def requires_content_free_telemetry(
    tools: Mapping[str, Any], *, sensitive_output_tools: frozenset[str]
) -> bool:
    """Whether any selected tool makes the whole Worker's telemetry content-free.

    ``sensitive_output_tools`` are the tool names that the registered toolset
    factories declare as returning captured HTTP traffic or HTTP session
    material (every http-tools@1 and Caido tool, including tools added later).
    A tool may also declare ``contractor_sensitive_output`` itself.
    """

    return any(
        name in sensitive_output_tools or declares_sensitive_output(tool)
        for name, tool in tools.items()
    )


class AdkWorkerRuntime:
    def __init__(
        self,
        context: WorkerBuildContext,
        model: BaseLlm,
        *,
        model_factory: ModelFactory = gateway_model,
        workspace_exporter: WorkspaceAutoExporter | None = None,
        session_service: BaseSessionService | None = None,
    ) -> None:
        require_supported_adk_features()
        self.allocation_id = context.allocation_id
        self._context = context
        self._completion = bind_worker_completion(
            tools=context.tools,
            contract=context.completion_contract,
            summarizer=context.summarizer,
        )
        self._model: BaseLlm | None = model
        self._model_factory = model_factory
        self._metrics = context.state.metrics
        self._instrumentation = context.adapter_handles.instrumentation
        if self._instrumentation is not None and requires_content_free_telemetry(
            context.tools, sensitive_output_tools=context.sensitive_output_tools
        ):
            self._instrumentation = ContentFreeInstrumentation(self._instrumentation)
        self._app_name = "contractor_runtime_worker"
        self._user_id = "contractor_control_plane"
        self._session_lifecycle = WorkerSessionLifecycle(
            mode=context.worker_session_mode,
            app_name=self._app_name,
            user_id=self._user_id,
            service=session_service,
        )
        self._session_service = self._session_lifecycle.service
        self._accepting = True
        self._invocation = InvocationLifecycle(
            sessions=self._session_lifecycle,
            state=context.state,
            policy=context.model_policy,
            summarizer=context.summarizer,
            completion=self._completion,
            stop_accepting=self._stop_accepting,
        )
        self._runner: Runner | None = None
        self._agent: LlmAgent | None = None
        self._tool_history = ToolHistory(self._app_name, self._user_id)
        self._agent_skills: PreparedAgentSkills | None = context.agent_skills
        self._workspace_exporter = workspace_exporter
        if not isinstance(context.state, WorkerStateStore):
            raise TypeError("adk@1 requires WorkerStateStore")
        self._resources = context.resources
        self._worker_state = context.state

        policy = context.model_policy
        generation = generation_config(policy)
        adk_tools = [
            ContractorFunctionTool(tool, logical_invocation_id=self._logical_tool_invocation)
            for tool in context.tools.values()
        ]
        self._compaction: CompactionCoordinator | None = None
        if context.compaction is not None:
            if (
                context.worker_session_mode != WorkerSessionMode.ISOLATED
                or self._completion is not None
                or context.summarizer is not None
            ):
                raise RuntimeError("compact@1 requires an isolated ordinary Worker")
            self._compaction = CompactionCoordinator(
                context.compaction,
                worker_policy=policy,
                skill_tools=frozenset(EXACT_SKILL_TOOL_NAMES),
            )
            adk_tools.append(
                ContractorFunctionTool(
                    self._history_tool(), logical_invocation_id=self._logical_tool_invocation
                )
            )
        allowed_tool_names = frozenset(tool.name for tool in adk_tools)
        if self._agent_skills is not None:
            allowed_tool_names |= frozenset(EXACT_SKILL_TOOL_NAMES)
            adk_tools.append(
                self._agent_skills.build_adapter(
                    metrics=self._metrics,
                )
            )
        self._plugin = WorkerInstrumentationPlugin(
            state=self._worker_state,
            execution=self._resources.execution,
            budget=lambda: self._invocation.budget,
            instrumentation=self._instrumentation,
            model_alias=policy.model,
            observe_artifacts=lambda tool, cursor: self._pipeline.observe_artifacts(tool, cursor),
            workspace_observation_source=context.project_workspace,
            summarizer_enabled=context.summarizer is not None,
            allowed_tool_names=allowed_tool_names,
            check_admission=self._admit_semantic_work,
            history=self._tool_history,
            compaction=self._compaction,
        )
        if self._completion is not None:
            self._completion.phase_sink = self._plugin.record_completion
        self._pipeline = WorkerCompletionPipeline(
            context=context,
            state=self._worker_state,
            resources=self._resources,
            plugin=self._plugin,
            completion=self._completion,
            invocation=self._invocation,
            model=model,
            model_factory=self._model_factory,
            instrumentation=self._instrumentation,
            reserved_slots=workspace_exporter.reserved_slots if workspace_exporter else frozenset(),
            check_active=self._check_active,
            admit=self._admit_semantic_work,
        )
        instruction = context.instruction
        self._agent = LlmAgent(
            name="contractor_worker",
            description=context.description,
            model=model,
            # A provider delivers the authored text literally: ADK would
            # otherwise expand {name} from session State and fail on a
            # missing key before the first model call (WRK-R166).
            instruction=lambda _: instruction,
            tools=adk_tools,
            generate_content_config=generation,
            before_model_callback=(
                self._project_compacted_request if self._compaction is not None else None
            ),
        )
        self._app = App(
            name=self._app_name,
            root_agent=self._agent,
            plugins=[self._plugin],
        )
        self._runner = Runner(
            app=self._app,
            session_service=self._session_service,
        )
        self._turns = AdkTurnRunner(
            runner=self._runner,
            sessions=self._session_lifecycle,
            history=self._tool_history,
            app_name=self._app_name,
            user_id=self._user_id,
            max_model_calls=policy.max_model_calls,
            compaction=self._compaction,
            observe_cancellation=self._raise_observed_tool_cancellation,
            fence=self._fence_compacted_session,
        )
        endpoint = (
            f"{context.a2a_base_url.rstrip('/')}/private/v1/allocations/{context.allocation_id}/a2a"
        )
        self._card = build_agent_card(
            allocation_id=context.allocation_id,
            endpoint=endpoint,
            logical_agent_name=context.logical_agent_name,
            description=context.description,
            version=context.card_version,
        )
        self._agent_card = agent_card_dict(self._card)
        self._a2a_application = build_worker_a2a_application(self, self._card)

    async def start(self) -> None:
        # Sessions are invocation resources. Invalid, stale, draining and busy
        # A2A requests therefore cannot create one during allocation prepare.
        return None

    @property
    def agent_card(self) -> Mapping[str, Any]:
        return dict(self._agent_card)

    @property
    def a2a_application(self) -> ASGIApp:
        return self._a2a_application

    async def agent_state_snapshot(self) -> AgentStateSnapshot:
        return await self._worker_state.agent_state_snapshot()

    async def invoke(self, request: StageContentRequest) -> WorkerCompletion:
        span = start_isolated_span(
            self._instrumentation,
            "contractor.worker.a2a_task",
            {"operation.kind": "a2a_task"},
        )
        try:
            result, attributes = await self._invoke(request)
        except asyncio.CancelledError:
            end_isolated_span(span, outcome="cancelled")
            raise
        except Exception as error:
            error_type = span_error_type(error)
            end_isolated_span(
                span,
                outcome="failed",
                attributes={"error.type": error_type},
            )
            _record_worker_error(self._instrumentation, error_type)
            raise
        if result.failure is not None:
            attributes["error.type"] = result.failure.code
            _record_worker_error(self._instrumentation, result.failure.code)
        end_isolated_span(
            span,
            outcome="failed" if result.failure is not None else "succeeded",
            attributes=attributes,
        )
        return result

    async def failure_completion(
        self, code: str, message: str, *, retryable: bool = False
    ) -> WorkerCompletion:
        """Create a bounded Runtime-owned failure for an A2A adapter rejection."""

        return await self._untracked_failure_completion(code, message, retryable)

    async def _invoke(
        self, request: StageContentRequest
    ) -> tuple[WorkerCompletion, dict[str, int]]:
        if self._resources.execution.failure is not None:
            self._accepting = False
            return (
                await self._untracked_failure_completion(
                    self._resources.execution.failure.value, "Sandbox execution failed", False
                ),
                {},
            )
        if not self._accepting:
            return (
                await self._untracked_failure_completion(
                    "worker_draining", "Worker is no longer accepting A2A work", True
                ),
                {},
            )
        if self._invocation.lock.locked():
            return (
                await self._untracked_failure_completion(
                    "worker_busy", "Worker already has an active A2A invocation", True
                ),
                {},
            )
        encoded_request = request.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
        if len(encoded_request) > MAX_STAGE_REQUEST_JSON_BYTES:
            return (
                await self._untracked_failure_completion(
                    "stage_content_too_large",
                    "StageContentRequest exceeds the Worker limit",
                    False,
                ),
                {},
            )
        completion: WorkerCompletion | None = None
        attributes: dict[str, int] = {}
        try:
            async with self._invocation.own() as (session_id, budget):
                completion, attributes = await self._execute_invocation(request, session_id, budget)
            return completion, attributes
        except WorkerSessionLifecycleError as error:
            self._accepting = False
            if completion is not None:
                # Cleanup failure replaces success, but retains the invocation
                # identity. Publish the corrected completion diagnostics as a new
                # State revision without completing the invocation twice.
                failure = completion.failure or _failure(
                    "worker_session_lifecycle_failed",
                    f"Worker session lifecycle failed ({error.code})",
                    True,
                )
                revision = completion.state_revision
                if self._completion is not None:
                    await self._completion.record_phase("failed", failure.code)
                    revision = (await self._worker_state.snapshot())["stateRevision"]
                return (
                    completion.model_copy(
                        update={
                            "result": None,
                            "failure": failure,
                            "state_revision": revision,
                        }
                    ),
                    attributes,
                )
            return (
                await self._untracked_failure_completion(
                    "worker_session_lifecycle_failed",
                    f"Worker session lifecycle failed ({error.code})",
                    True,
                ),
                {},
            )

    async def _execute_invocation(
        self, request: StageContentRequest, session_id: str, budget: _InvocationBudget
    ) -> tuple[WorkerCompletion, dict[str, int]]:
        model_errors_before = self._metrics.counters.get("llm_errors", 0)
        invocation_id = f"worker-{uuid.uuid4().hex}"
        invocation_phase: InvocationPhase = "failed"
        outcome: WorkerResult | WorkerFailure
        state_snapshot: dict[str, Any] | None = None
        try:
            budget.start()
            if self._completion is not None:
                self._completion.reset_diagnostics()
            self._plugin.prepare_invocation(
                invocation_id=invocation_id,
                subtask_id=request.subtask_id,
            )
            if not self._accepting:
                outcome = _failure(
                    "worker_draining", "Worker is no longer accepting A2A work", True
                )
                exportable = False
            else:
                outcome, exportable = await self._run_adk(request, invocation_id, session_id)
            self._resources.execution.check()
            exporter = self._workspace_exporter
            if exportable and exporter is not None:
                try:
                    if not (isinstance(outcome, WorkerResult)):
                        raise RuntimeError("Expected isinstance(outcome, WorkerResult)")
                    exported = await exporter.export(outcome)
                except WorkspaceExportError as error:
                    self._metrics.record_workspace_export(error=error)
                    outcome = _failure(
                        "workspace_export_failed",
                        f"Workspace export failed ({error.cause})",
                        error.retryable,
                    )
                else:
                    self._metrics.record_workspace_export(
                        state_bytes=exported.state_bytes,
                        diff_bytes=exported.diff_bytes,
                    )
                    outcome = exported.result
            invocation_phase = "succeeded" if isinstance(outcome, WorkerResult) else "failed"
            self._metrics.record_outcome(invocation_phase)
        except asyncio.CancelledError:
            self._metrics.record_outcome("cancelled")
            invocation_phase = "cancelled"
            raise
        except Exception as error:
            sandbox_failure = self._resources.execution.failure
            gateway_error = _chained_error(error, GatewayModelError)
            if (
                sandbox_failure is None
                and gateway_error is not None
                and self._metrics.counters.get("llm_errors", 0) == model_errors_before
            ):
                # Only a Gateway failure that bypassed the model callbacks is a
                # model error. Exporter, State and instrumentation defects stay
                # worker_execution_failed without inflating modelErrors.
                await self._plugin.record_unhandled_model_error(gateway_error)
            if sandbox_failure is not None:
                self._accepting = False
                outcome = _failure(sandbox_failure.value, "Sandbox execution failed", False)
            elif (session_error := _chained_error(error, WorkerSessionLifecycleError)) is not None:
                self._accepting = False
                outcome = _failure(
                    "worker_session_lifecycle_failed",
                    f"Worker session lifecycle failed ({session_error.code})",
                    session_error.code == "tool_history_unavailable",
                )
            elif gateway_error is not None:
                outcome = _gateway_worker_failure(gateway_error)
            elif (tool_error := _chained_error(error, ToolExecutionError)) is not None:
                self._accepting = False
                outcome = _failure(
                    tool_error.code, "HTTP execution infrastructure unavailable", False
                )
            else:
                outcome = _failure("worker_execution_failed", "Worker execution failed", True)
            self._metrics.record_outcome("failed")
            invocation_phase = "failed"
        finally:
            try:
                if self._completion is not None and invocation_phase != "succeeded":
                    code = "worker_cancelled" if invocation_phase == "cancelled" else outcome.code
                    await self._completion.record_phase("failed", code)
                state_snapshot = await self._finish_invocation_state(
                    invocation_id,
                    request.subtask_id,
                    invocation_phase,
                    session_id,
                )
            except WorkerSessionLifecycleError as error:
                self._accepting = False
                if invocation_phase != "cancelled":
                    state_snapshot = await self._worker_state.snapshot()
                    outcome = _failure(
                        "worker_session_lifecycle_failed",
                        f"Worker session lifecycle failed ({error.code})",
                        True,
                    )
            except Exception:
                # State teardown must never replace cancellation with an error
                # or a successful completion.
                self._accepting = False
                if invocation_phase != "cancelled":
                    raise
        if state_snapshot is None:
            raise RuntimeError("Worker State did not produce a terminal revision")
        if self._resources.execution.failure is not None:
            self._accepting = False
            outcome = _failure(
                self._resources.execution.failure.value, "Sandbox execution failed", False
            )
        if isinstance(outcome, WorkerFailure) and self._completion is not None:
            await self._completion.record_phase("failed", outcome.code)
            state_snapshot = await self._worker_state.snapshot()
        if isinstance(outcome, WorkerResult):
            outcome = outcome.model_copy(
                update={
                    "observations": _lean_observations(
                        state_snapshot,
                        invocation_id,
                        projection_failed=self._plugin.projection_failed,
                    )
                }
            )
        completion = WorkerCompletion(
            apiVersion=API_VERSION,
            **({"result": outcome} if isinstance(outcome, WorkerResult) else {"failure": outcome}),
            invocationId=invocation_id,
            stateRevision=state_snapshot["stateRevision"],
        )
        return completion, _invocation_count_attributes(state_snapshot, invocation_id)

    def cancel_active(self, owner: asyncio.Task[Any]) -> None:
        task = self._invocation.task
        if (
            task is not None
            and task is owner
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    async def finalize(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def abort(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def _run_adk(
        self,
        request: StageContentRequest,
        invocation_id: str,
        session_id: str,
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        runner = self._runner
        if runner is None:
            return _failure(
                "worker_draining", "Worker is no longer accepting A2A work", True
            ), False
        prompt = _task_prompt(request)
        # Only an enabled terminal summarizer consumes the normal-loop
        # transcript (WRK-R147); without one no transcript is built.
        summary_secrets: tuple[str, ...] = ()
        transcript: TranscriptRecorder | None = None
        if self._context.summarizer is not None:
            summary_secrets = _summarizer_secrets(self._context)
            transcript = TranscriptRecorder(
                include_reasoning=self._context.summarizer.include_reasoning,
                secrets=summary_secrets,
            )
        self._pipeline.refs.clear()
        completion = self._completion
        # The Planner's deadline covers the whole logical invocation, ordinary
        # or Audit (WRK-R158, WRK-R167). The request timeout still applies
        # independently inside model/tool clients (WRK-R157).
        deadline = InvocationDeadline(request.deadline)
        self._invocation.deadline = deadline
        sdk_invocation_id = invocation_id
        try:
            if completion is not None:
                failure = completion.validate_request(request)
                if failure is not None:
                    return failure, False
                completion.begin(invocation_id)
            if self._compaction is not None:
                self._compaction.begin(
                    invocation_id=invocation_id,
                    task_prompt=prompt,
                    secrets=_summarizer_secrets(self._context),
                )
            async with deadline.scope:
                # An expired request starts no model or tool work.
                deadline.check()
                while True:
                    self._admit_semantic_work()
                    candidate: str | None = None
                    compacted = False
                    try:
                        # An early return must close ADK's generator in this task:
                        # its tracing contexts cannot be detached by GC in another one.
                        async with contextlib.aclosing(
                            self._turns.events(invocation_id, sdk_invocation_id, session_id, prompt)
                        ) as events:
                            async for event in events:
                                if transcript is not None:
                                    transcript.record(event)
                                if completion is not None:
                                    self._check_active()
                                if output_limit_reached(event):
                                    return _failure(
                                        "worker_output_limit_exceeded",
                                        "Worker model response reached its output token limit",
                                        True,
                                    ), False
                                if completion is not None and getattr(
                                    event, "error_code", None
                                ) not in (None, "MODEL_RETURNED_NO_CONTENT"):
                                    return _failure(
                                        "worker_execution_failed",
                                        "Worker model returned an error",
                                        True,
                                    ), False
                                text = candidate_text(event)
                                if text is not None:
                                    candidate = text
                    except Exception as error:
                        self._resources.execution.check()
                        # Work that ends after the deadline, including a
                        # summarization request, ends with the deadline.
                        deadline.check()
                        if _chained_error(error, WorkerSessionLifecycleError) is not None:
                            raise
                        if self._plugin.model_output_limited:
                            return _failure(
                                "worker_output_limit_exceeded",
                                "Worker model response reached its output token limit",
                                True,
                            ), False
                        if _chained_error(error, WorkerSummarizationRequested) is not None:
                            return await self._pipeline.summarize(
                                request=request,
                                invocation_id=invocation_id,
                                transcript=transcript,
                                observed_refs=tuple(self._pipeline.refs),
                                secrets=summary_secrets,
                            )
                        compaction_error = _chained_error(error, WorkerCompactionError)
                        budget_error = _chained_error(error, WorkerBudgetExceeded)
                        if (
                            compaction_error is None
                            and budget_error is None
                            and self._compaction is not None
                            and _chained_error(error, WorkerCompactionRequested) is not None
                        ):
                            try:
                                await self._compact(invocation_id, sdk_invocation_id, session_id)
                            except WorkerCompactionError as failure:
                                compaction_error = failure
                            except WorkerBudgetExceeded as exhausted:
                                # The compactor call counts against the same
                                # hard budgets; exhaustion keeps its own code.
                                budget_error = exhausted
                            else:
                                compacted = True
                        if compaction_error is not None:
                            return _compaction_failure(compaction_error), False
                        if not compacted:
                            if budget_error is None:
                                raise
                            self._metrics.record_worker_budget_exhausted(budget_error.dimension)
                            return (
                                _failure(
                                    "worker_budget_exhausted",
                                    "Worker invocation budget exhausted "
                                    f"({budget_error.dimension})",
                                    True,
                                ),
                                False,
                            )
                    if compacted and self._compaction is not None:
                        # The same logical invocation continues in a new SDK
                        # turn whose requests carry the checkpoint (WRK-R192).
                        self._check_active()
                        sdk_invocation_id = f"{invocation_id}-turn-{uuid.uuid4().hex}"
                        self._plugin.prepare_continuation(
                            invocation_id=invocation_id, sdk_invocation_id=sdk_invocation_id
                        )
                        prompt = self._compaction.continuation_prompt()
                        continue
                    self._tool_history.check()
                    if self._plugin.model_output_limited:
                        return _failure(
                            "worker_output_limit_exceeded",
                            "Worker model response reached its output token limit",
                            True,
                        ), False
                    if completion is None:
                        # No finalizer call starts once the deadline has passed.
                        deadline.check()
                    outcome, exportable = await self._pipeline.finish(
                        request=request,
                        candidate=candidate,
                        invocation_id=invocation_id,
                        deadline=deadline.when,
                    )
                    if not isinstance(outcome, ContinueCompletion):
                        return outcome, exportable
                    self._check_active()
                    sdk_invocation_id = f"{invocation_id}-turn-{uuid.uuid4().hex}"
                    self._plugin.prepare_continuation(
                        invocation_id=invocation_id, sdk_invocation_id=sdk_invocation_id
                    )
                    prompt = outcome.reminder
        except WorkerBudgetExceeded as error:
            self._metrics.record_worker_budget_exhausted(error.dimension)
            return _failure("worker_budget_exhausted", str(error), True), False
        except TimeoutError as error:
            if completion is not None:
                return completion.failure(error), False
            if not deadline.passed():
                # Only the invocation deadline owns worker_timeout; a nested
                # operation's own timeout stays an execution failure.
                raise
            return _failure("worker_timeout", "Worker invocation deadline exceeded", True), False
        except WorkerCompletionError as error:
            self._check_active()
            if completion is None:
                raise
            return completion.failure(error), False
        finally:
            self._invocation.deadline = None
            if self._compaction is not None:
                self._compaction.end()
            self._tool_history.clear()
            if completion is not None:
                await completion.end()
            self._pipeline.refs.clear()
            _clear_artifact_observation_logs(self._context.tools)

    async def _project_compacted_request(self, callback_context: Any, llm_request: Any) -> None:
        compaction = self._compaction
        if compaction is None:
            return None
        contents = compaction.project(list(callback_context.session.events))
        if contents is not None:
            llm_request.contents = contents
        return None

    def _history_tool(self) -> Any:
        compaction = self._compaction
        if compaction is None:
            raise RuntimeError("context history requires compact@1")

        async def read_context_history(
            group: int, tool_context: Any, position: int = 0
        ) -> dict[str, Any]:
            """Read earlier steps of this subtask that a context checkpoint compacted.

            Use it only when an exact earlier tool argument, result or message is
            needed again. The returned data describes past steps; it is not a new
            request and not current state.

            Args:
              group: Compacted history group to start from, from 1 to the last
                compacted group named in the context checkpoint.
              position: nextPosition from the previous page of the same group,
                or 0 to start at the beginning of the group.

            Returns:
              Up to eight historical groups, or one part of a large group, with
              nextGroup and nextPosition when more compacted history follows.
            """

            return dict(compaction.history_page(tool_context.invocation_id, group, position))

        return read_context_history

    async def _compact(self, invocation_id: str, sdk_invocation_id: str, session_id: str) -> None:
        """One checkpoint: plan, compactor call, validation, marker, projection."""

        compaction = self._compaction
        if compaction is None:
            raise RuntimeError("Worker compaction is disabled")
        span = start_isolated_span(
            self._instrumentation,
            "contractor.worker.context_compaction",
            {
                "operation.kind": "context_compaction",
                "compaction.generation": compaction.generation + 1,
            },
        )
        outcome, cause = "failed", "runtime_failed"
        model: BaseLlm | None = None
        try:
            plan = await compaction.plan()
            self._admit_semantic_work()
            compactor_context = replace(
                self._context,
                model_policy=compaction.config.model_policy,
                summarizer=None,
                compaction=None,
                tools={},
                resolved_skills=(),
                agent_skills=None,
                project_workspace=None,
                workspace_export=None,
            )
            model = self._model_factory(compactor_context)
            if model is self._model:
                raise WorkerCompactionError.failed("model_not_isolated")
            try:
                candidate = await run_compactor(
                    model=model,
                    config=compaction.config,
                    document=plan.document,
                    invocation_id=invocation_id,
                    observer=self._plugin,
                )
            except WorkerCompactionError:
                raise
            except Exception as error:
                if self._invocation.deadline is not None:
                    self._invocation.deadline.check()
                budget_error = _chained_error(error, WorkerBudgetExceeded)
                if budget_error is not None:
                    raise budget_error from None
                inner = _chained_error(error, WorkerCompactionError)
                if inner is not None:
                    raise inner from None
                gateway_error = _chained_error(error, GatewayModelError)
                if gateway_error is not None:
                    raise WorkerCompactionError.failed(
                        "gateway", retryable=_gateway_worker_failure(gateway_error).retryable
                    ) from None
                raise WorkerCompactionError.failed("runtime_failed") from None
            summary = compaction.validate_summary(plan, candidate)
            if _exposes_private_value(summary, self._context):
                raise WorkerCompactionError.failed("unsafe_output")
            self._admit_semantic_work()
            compaction.ensure_current(plan)
            marker = compaction.marker(plan, sdk_invocation_id=sdk_invocation_id)
            await append_marker(
                self._session_service,
                app_name=self._app_name,
                user_id=self._user_id,
                session_id=session_id,
                marker=marker,
                fence=self._fence_compacted_session,
            )
            compaction.commit(plan, summary, marker)
            if self._invocation.budget is not None:
                # A new context generation needs its own measured main usage.
                self._invocation.budget.latest_prompt_tokens = None
            outcome, cause = "succeeded", ""
        except WorkerCompactionError as error:
            cause = error.cause
            raise
        except asyncio.CancelledError:
            outcome, cause = "cancelled", ""
            raise
        finally:
            if isinstance(model, OpenAICompatibleGatewayLlm) and model is not self._model:
                await model.close()
            end_isolated_span(
                span,
                outcome=outcome,
                attributes={"compaction.cause": cause} if cause else {},
            )

    def _stop_accepting(self) -> None:
        self._accepting = False

    def _fence_compacted_session(self) -> None:
        self._accepting = False
        self._session_lifecycle.fence()

    def _logical_tool_invocation(self, tool_context: Any) -> str:
        return self._plugin.logical_invocation_id_for_tool(tool_context)

    def _admit_semantic_work(self) -> None:
        self._check_active()
        self._tool_history.check()
        if self._invocation.deadline is None:
            raise RuntimeError("Worker has no active invocation deadline")
        self._invocation.deadline.check()
        self._raise_observed_tool_cancellation()

    def _raise_observed_tool_cancellation(self) -> None:
        if not self._plugin.tool_cancellation_observed:
            return
        self._resources.execution.check()
        if self._invocation.deadline is not None:
            self._invocation.deadline.check()
        # A refused budget can cause the SDK to cancel admitted siblings.
        # That secondary cancellation must not replace its owning failure.
        if self._invocation.budget is not None and self._invocation.budget.failure is not None:
            raise self._invocation.budget.failure
        raise asyncio.CancelledError

    def _check_active(self):
        self._resources.execution.check()
        task = asyncio.current_task()
        if not self._accepting or (task is not None and task.cancelling()):
            raise asyncio.CancelledError
        if self._invocation.budget is not None and self._invocation.budget.failure is not None:
            raise self._invocation.budget.failure

    async def _stop(self, deadline: datetime) -> None:
        self._accepting = False
        task = self._invocation.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            remaining = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
            if remaining <= 0:
                raise TimeoutError("Worker stop deadline expired")
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if not done:
                raise TimeoutError("active ADK invocation did not stop")
        runner = self._runner
        self._runner = None
        failures: list[Exception] = []
        try:
            if runner is not None:
                try:
                    await runner.close()
                except Exception as error:
                    failures.append(error)
            try:
                await self._session_lifecycle.close()
            except Exception as error:
                failures.append(error)
            agent_skills = self._agent_skills
            if agent_skills is not None:
                try:
                    await agent_skills.close()
                except Exception as error:
                    failures.append(error)
        finally:
            exporter = self._workspace_exporter
            self._workspace_exporter = None
            if exporter is not None:
                exporter.release()
            model = self._model
            self._model = None
            if isinstance(model, OpenAICompatibleGatewayLlm):
                try:
                    await model.close()
                except Exception as error:
                    failures.append(error)
            self._agent = None
            self._pipeline.release()
            self._turns = None
            self._instrumentation = None
            self._agent_skills = None
            self._context = None  # type: ignore[assignment]
        if failures:
            raise failures[0]

    async def _finish_invocation_state(
        self,
        invocation_id: str,
        subtask_id: str,
        phase: InvocationPhase,
        session_id: str,
    ) -> dict[str, Any]:
        async def finish() -> dict[str, Any]:
            snapshot = await self._plugin.complete_invocation(
                invocation_id=invocation_id,
                phase=phase,
            )
            if snapshot is None:
                metrics = _empty_invocation_metrics()
                await self._worker_state.begin_invocation(
                    invocation_id=invocation_id,
                    subtask_id=subtask_id,
                    metrics=metrics,
                    summarizer_enabled=self._context.summarizer is not None,
                )
                snapshot = await self._worker_state.complete_invocation(
                    invocation_id=invocation_id,
                    phase=phase,
                    metrics=metrics,
                )
            await self._sync_worker_state(snapshot, session_id)
            return snapshot

        async def checkpoint_and_finish() -> dict[str, Any]:
            # Resource I/O has its own bounded allowance, separate from the
            # five-second State finalization budget.
            if self._resources.before_completion is not None:
                await self._resources.before_completion()
            return await asyncio.wait_for(finish(), timeout=INVOCATION_CLEANUP_TIMEOUT_SECONDS)

        task = asyncio.create_task(
            checkpoint_and_finish(),
            name="worker-invocation-state-finalize",
        )
        if await join_owned_task(task):
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
            raise asyncio.CancelledError
        return task.result()

    async def _untracked_failure_completion(
        self, code: str, message: str, retryable: bool
    ) -> WorkerCompletion:
        self._metrics.record_outcome("failed")
        snapshot = await self._worker_state.sync_metrics()
        return WorkerCompletion(
            apiVersion=API_VERSION,
            failure=_failure(code, message, retryable),
            invocationId=f"worker-rejected-{uuid.uuid4().hex}",
            stateRevision=snapshot["stateRevision"],
        )

    async def _sync_worker_state(
        self,
        snapshot: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        if snapshot is None:
            snapshot = await self._worker_state.sync_metrics()
        if session_id is None:
            session_id = self._session_lifecycle.active_session_id
        if session_id is None:
            return
        session = await self._session_service.get_session(
            app_name=self._app_name, user_id=self._user_id, session_id=session_id
        )
        if session is None:
            return
        await self._session_service.append_event(
            session,
            Event(
                invocationId=f"state-{uuid.uuid4().hex}",
                author="contractor_runtime",
                actions=EventActions(stateDelta={"contractor": snapshot}),
            ),
        )


def _clear_artifact_observation_logs(tools: Mapping[str, Any]) -> None:
    for tool in tools.values():
        clear = getattr(tool, "clear_artifact_observations", None)
        if callable(clear):
            clear()


def _record_worker_error(
    instrumentation: RuntimeInstrumentation | None,
    error_type: str,
) -> None:
    span = start_isolated_span(
        instrumentation,
        "contractor.worker.error",
        {"operation.kind": "worker_error", "error.type": error_type},
    )
    end_isolated_span(span, outcome="failed")


def _invocation_count_attributes(snapshot: Mapping[str, Any], invocation_id: str) -> dict[str, int]:
    invocation = snapshot.get("currentInvocation")
    if not isinstance(invocation, Mapping) or invocation.get("invocationId") != invocation_id:
        invocation = snapshot.get("lastCompletedInvocation")
    if not isinstance(invocation, Mapping) or invocation.get("invocationId") != invocation_id:
        return {}
    metrics = invocation.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    result: dict[str, int] = {}
    for source, target in (
        ("modelCalls", "counts.model_calls"),
        ("toolCalls", "counts.tool_calls"),
    ):
        value = metrics.get(source)
        if type(value) is int and value >= 0:
            result[target] = value
    return result


def _empty_invocation_metrics() -> dict[str, Any]:
    return {
        "modelCalls": 0,
        "modelErrors": 0,
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "cachedInputTokens": 0,
        "tokenUsageUnavailable": 0,
        "toolCalls": 0,
        "toolErrors": 0,
        "tools": {},
        "truncated": False,
    }
