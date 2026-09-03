"""Google ADK-backed allocation-local Worker runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import re
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from pydantic import ValidationError
from starlette.types import ASGIApp

from contractor_runtime.a2a_server import (
    agent_card_dict,
    build_agent_card,
    build_worker_a2a_application,
)
from contractor_runtime.adapters import RuntimeInstrumentation, RuntimeSpan, TelemetryAttribute
from contractor_runtime.agent_skills.runtime import (
    PreparedAgentSkills,
    prepare_agent_skills,
    probe_native_agent_skills,
)
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import (
    API_VERSION,
    AgentStateSnapshot,
    ArtifactRef,
    ResolvedModelPolicy,
    StageContentRequest,
    ToolObservationCount,
    WorkerCompletion,
    WorkerFailure,
    WorkerModelResult,
    WorkerObservations,
    WorkerResult,
    WorkerSummarizerConfig,
)
from contractor_runtime.instrumentation import (
    WorkerInstrumentationPlugin,
    WorkerSummarizationRequested,
)
from contractor_runtime.model_client import (
    clear_gateway_client_options,
    gateway_client_options,
)
from contractor_runtime.observations import lean_workspace_summary
from contractor_runtime.projectfs import (
    MAX_EXPORTED_RESULT_ARTIFACTS,
    MAX_EXPORTED_RESULT_JSON_BYTES,
    OverlayWorkspaceSession,
    WorkspaceAutoExporter,
    WorkspaceExportError,
)
from contractor_runtime.summarizer import (
    SummarizerFailure,
    SummarizerUsage,
    TerminalSummarizer,
    TranscriptRecorder,
    build_summarizer_prompt,
)
from contractor_runtime.toolsets.artifact_visibility import is_reserved_memory_binding
from contractor_runtime.worker_state import InvocationPhase, WorkerStateStore

if TYPE_CHECKING:
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    from contractor_runtime.factories import WorkerBuildContext

MAX_STAGE_REQUEST_JSON_BYTES = 256 * 1024
MAX_STAGE_RESULT_JSON_BYTES = MAX_EXPORTED_RESULT_JSON_BYTES
MAX_RESULT_ARTIFACTS = MAX_EXPORTED_RESULT_ARTIFACTS
SAFE_TOOL_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ModelFactory(Protocol):
    def __call__(self, context: WorkerBuildContext) -> BaseLlm: ...


class GatewayModelError(RuntimeError):
    """Secret-free boundary error for failures below the LLM Gateway adapter."""

    def __init__(self, provider_error_type: str) -> None:
        self.provider_error_type = provider_error_type
        super().__init__(f"LLM gateway call failed ({provider_error_type})")


class WorkerBudgetExceeded(RuntimeError):
    """Safe internal control signal for one exhausted invocation dimension."""

    def __init__(self, dimension: str, limit: int, observed: int) -> None:
        self.dimension = dimension
        self.limit = limit
        self.observed = observed
        super().__init__(f"Worker invocation budget exhausted ({dimension})")


@dataclass(slots=True)
class _InvocationBudget:
    max_model_calls: int
    max_tool_calls: int
    max_total_tokens: int
    metrics: Any
    cumulative_budget: int | None = None
    summary_prompt_boundary: int | None = None
    model_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    token_usage_unavailable: int = 0
    latest_prompt_tokens: int | None = None

    def start(self) -> None:
        self.metrics.start_worker_budget(
            max_model_calls=self.max_model_calls,
            max_tool_calls=self.max_tool_calls,
            max_total_tokens=self.max_total_tokens,
        )
        self._sync()

    def before_model_call(self) -> None:
        self._require_token_capacity()
        if self.model_calls >= self.max_model_calls:
            raise WorkerBudgetExceeded("model_calls", self.max_model_calls, self.model_calls)
        self.model_calls += 1
        self._sync()

    def before_tool_call(self) -> None:
        self._require_token_capacity()
        if self.tool_calls >= self.max_tool_calls:
            raise WorkerBudgetExceeded("tool_calls", self.max_tool_calls, self.tool_calls)
        self.tool_calls += 1
        self._sync()

    def after_model_response(self, usage: Any | None) -> None:
        prompt = getattr(usage, "prompt_token_count", None) if usage is not None else None
        self.latest_prompt_tokens = prompt if isinstance(prompt, int) and prompt >= 0 else None
        total = getattr(usage, "total_token_count", None) if usage is not None else None
        if not isinstance(total, int) or total < 0:
            self.token_usage_unavailable += 1
            self._sync()
            return
        self.total_tokens += total
        self._sync()
        if self.total_tokens > self.max_total_tokens:
            raise WorkerBudgetExceeded("total_tokens", self.max_total_tokens, self.total_tokens)

    def _require_token_capacity(self) -> None:
        if self.total_tokens >= self.max_total_tokens:
            raise WorkerBudgetExceeded("total_tokens", self.max_total_tokens, self.total_tokens)

    def should_summarize(self) -> bool:
        return (
            self.cumulative_budget is not None and self.total_tokens >= self.cumulative_budget
        ) or (
            self.summary_prompt_boundary is not None
            and self.latest_prompt_tokens is not None
            and self.latest_prompt_tokens >= self.summary_prompt_boundary
        )

    def _sync(self) -> None:
        self.metrics.observe_worker_budget(
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            total_tokens=self.total_tokens,
            token_usage_unavailable=self.token_usage_unavailable,
        )


class GatewayLiteLlm(LiteLlm):
    """LiteLLM client whose allocation credential can be explicitly erased."""

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        provider_error_type: str | None = None
        try:
            async for response in super().generate_content_async(llm_request, stream=stream):
                yield response
            return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Do not retain the provider exception: it may contain request headers or the token.
            provider_error_type = type(error).__name__
        raise GatewayModelError(provider_error_type or "UnknownProviderError") from None

    def clear_credentials(self) -> None:
        clear_gateway_client_options(self._additional_args)


def gateway_model(context: WorkerBuildContext) -> BaseLlm:
    policy = context.model_policy
    return GatewayLiteLlm(
        model=f"openai/{policy.model}",
        **gateway_client_options(context),
    )


class AdkWorkerRuntimeFactory:
    ref = "adk@1"
    supports_agent_skills = True

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        artifact_client_factory: Callable[[str, Any], ArtifactClient] | None = None,
    ) -> None:
        self._model_factory = model_factory or gateway_model
        self._artifact_client_factory = artifact_client_factory

    async def probe(self) -> bool:
        # Gateway/model availability remains allocation-scoped, but advertising
        # adk@1 also promises the exact native Agent Skills APIs used below.
        return await probe_native_agent_skills()

    async def create(self, context: WorkerBuildContext) -> AdkWorkerRuntime:
        prepared = await prepare_agent_skills(
            context.resolved_skills,
            allocation_id=context.allocation_id,
            runtime_settings=context.runtime_settings,
            workspace=context.workspace,
            artifact_client_factory=self._artifact_client_factory,
        )
        if prepared is not None:
            context = replace(context, agent_skills=prepared)
        runtime: AdkWorkerRuntime | None = None
        try:
            workspace_exporter: WorkspaceAutoExporter | None = None
            if context.workspace_export is not None:
                if (
                    not isinstance(context.project_workspace, OverlayWorkspaceSession)
                    or self._artifact_client_factory is None
                ):
                    raise RuntimeError("overlay workspace export dependencies are unavailable")
                workspace_exporter = WorkspaceAutoExporter(
                    workspace=context.project_workspace,
                    client=self._artifact_client_factory(
                        context.allocation_id, context.runtime_settings
                    ),
                    namespace=context.namespace,
                    slots=context.workspace_export,
                )
            runtime = AdkWorkerRuntime(
                context,
                self._model_factory(context),
                model_factory=self._model_factory,
                workspace_exporter=workspace_exporter,
            )
            await runtime.start()
            return runtime
        except asyncio.CancelledError:
            if prepared is not None:
                await prepared.close()
            raise
        except Exception:
            if runtime is not None:
                with contextlib.suppress(Exception):
                    await runtime.abort(datetime.now(UTC) + timedelta(seconds=1))
            elif prepared is not None:
                await prepared.close()
            raise


class AdkWorkerRuntime:
    def __init__(
        self,
        context: WorkerBuildContext,
        model: BaseLlm,
        *,
        model_factory: ModelFactory = gateway_model,
        workspace_exporter: WorkspaceAutoExporter | None = None,
    ) -> None:
        self.allocation_id = context.allocation_id
        self._context = context
        self._model: BaseLlm | None = model
        self._model_factory = model_factory
        self._metrics = context.state.metrics
        self._instrumentation = context.adapter_handles.instrumentation
        self._session_service = InMemorySessionService()
        self._session_id = context.allocation_id
        self._app_name = "contractor_runtime_worker"
        self._user_id = "contractor_control_plane"
        self._accepting = True
        self._invoke_lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._runner: Runner | None = None
        self._agent: LlmAgent | None = None
        self._active_budget: _InvocationBudget | None = None
        self._invocation_observed_refs: list[ArtifactRef] = []
        self._agent_skills: PreparedAgentSkills | None = context.agent_skills
        self._workspace_exporter = workspace_exporter
        if not isinstance(context.state, WorkerStateStore):
            raise TypeError("adk@1 requires WorkerStateStore")
        self._worker_state = context.state

        policy = context.model_policy
        generation = types.GenerateContentConfig(max_output_tokens=policy.max_output_tokens)
        if policy.temperature is not None:
            generation.temperature = policy.temperature
        adk_tools = [FunctionTool(tool) for tool in context.tools.values()]
        if self._agent_skills is not None:
            adk_tools.append(
                self._agent_skills.build_adapter(
                    metrics=self._metrics,
                )
            )
        self._plugin = WorkerInstrumentationPlugin(
            state=self._worker_state,
            budget=lambda: self._active_budget,
            instrumentation=self._instrumentation,
            model_alias=policy.model,
            observe_artifacts=self._observe_tool_artifacts,
            workspace_observation_source=context.project_workspace,
            summarizer_enabled=context.summarizer is not None,
        )
        self._agent = LlmAgent(
            name="contractor_worker",
            description=context.description,
            model=model,
            instruction=context.instruction,
            tools=adk_tools,
            output_schema=WorkerModelResult,
            generate_content_config=generation,
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
        state = await self._worker_state.snapshot()
        await self._session_service.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=self._session_id,
            state={"contractor": state},
        )

    @property
    def agent_card(self) -> Mapping[str, Any]:
        return dict(self._agent_card)

    @property
    def a2a_application(self) -> ASGIApp:
        return self._a2a_application

    async def agent_state_snapshot(self) -> AgentStateSnapshot:
        return await self._worker_state.agent_state_snapshot()

    async def invoke(self, request: StageContentRequest) -> WorkerCompletion:
        span = _start_span(
            self._instrumentation,
            "contractor.worker.a2a_task",
            {"operation.kind": "a2a_task"},
        )
        try:
            result = await self._invoke(request)
        except asyncio.CancelledError:
            _end_span(span, outcome="cancelled")
            raise
        except Exception as error:
            error_type = _safe_error_type(error)
            _end_span(
                span,
                outcome="failed",
                attributes={"error.type": error_type},
            )
            _record_worker_error(self._instrumentation, error_type)
            raise
        attributes = _aggregate_count_attributes(self._metrics.counters)
        if result.failure is not None:
            attributes["error.type"] = result.failure.code
            _record_worker_error(self._instrumentation, result.failure.code)
        _end_span(
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

    async def _invoke(self, request: StageContentRequest) -> WorkerCompletion:
        if not self._accepting:
            return await self._untracked_failure_completion(
                "worker_draining", "Worker is no longer accepting A2A work", True
            )
        if self._invoke_lock.locked():
            return await self._untracked_failure_completion(
                "worker_busy", "Worker already has an active A2A invocation", True
            )
        await self._invoke_lock.acquire()
        self._active_task = asyncio.current_task()
        policy = self._context.model_policy
        budget = _InvocationBudget(
            max_model_calls=policy.max_model_calls,
            max_tool_calls=policy.max_tool_calls,
            max_total_tokens=policy.max_total_tokens,
            metrics=self._metrics,
            cumulative_budget=(
                self._context.summarizer.cumulative_budget
                if self._context.summarizer is not None
                else None
            ),
            summary_prompt_boundary=_summary_prompt_boundary(policy, self._context.summarizer),
        )
        self._active_budget = budget
        model_errors_before = self._metrics.counters.get("llm_errors", 0)
        invocation_id = f"worker-{uuid.uuid4().hex}"
        invocation_phase: InvocationPhase = "failed"
        outcome: WorkerResult | WorkerFailure
        state_snapshot: dict[str, Any] | None = None
        try:
            budget.start()
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
                encoded_request = request.model_dump_json(by_alias=True, exclude_none=True).encode(
                    "utf-8"
                )
                if len(encoded_request) > MAX_STAGE_REQUEST_JSON_BYTES:
                    outcome = _failure(
                        "stage_content_too_large",
                        "StageContentRequest exceeds the Worker limit",
                        False,
                    )
                    exportable = False
                else:
                    outcome, exportable = await self._run_adk(request, invocation_id)
            exporter = self._workspace_exporter
            if exportable and exporter is not None:
                try:
                    assert isinstance(outcome, WorkerResult)
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
            if self._metrics.counters.get("llm_errors", 0) == model_errors_before:
                await self._plugin.record_unhandled_model_error(error)
            gateway_error = _gateway_model_error(error)
            if gateway_error is not None:
                outcome = _failure(
                    "worker_gateway_unavailable", "Worker LLM Gateway request failed", True
                )
            else:
                outcome = _failure("worker_execution_failed", "Worker execution failed", True)
            self._metrics.record_outcome("failed")
            invocation_phase = "failed"
        finally:
            try:
                state_snapshot = await self._finish_invocation_state(
                    invocation_id,
                    request.subtask_id,
                    invocation_phase,
                )
            finally:
                self._active_budget = None
                self._active_task = None
                self._invoke_lock.release()
        if state_snapshot is None:
            raise RuntimeError("Worker State did not produce a terminal revision")
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
        return WorkerCompletion(
            apiVersion=API_VERSION,
            result=outcome if isinstance(outcome, WorkerResult) else None,
            failure=outcome if isinstance(outcome, WorkerFailure) else None,
            invocationId=invocation_id,
            stateRevision=state_snapshot["stateRevision"],
        )

    def cancel_active(self) -> None:
        task = self._active_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def finalize(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def abort(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def _run_adk(
        self,
        request: StageContentRequest,
        invocation_id: str,
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        runner = self._runner
        if runner is None:
            return _failure(
                "worker_draining", "Worker is no longer accepting A2A work", True
            ), False
        prompt = _task_prompt(request)
        wrapped_gateway_token = self._context.runtime_settings.llm_gateway_token
        gateway_token = (
            wrapped_gateway_token.get_secret_value() if wrapped_gateway_token is not None else ""
        )
        summary_secrets = _summarizer_secrets(self._context, gateway_token)
        transcript = TranscriptRecorder(secrets=summary_secrets)
        self._invocation_observed_refs.clear()
        candidate: str | None = None
        try:
            try:
                async for event in runner.run_async(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    invocation_id=invocation_id,
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                ):
                    transcript.record(event)
                    text = _candidate_text(event)
                    if text is not None:
                        candidate = text
            except Exception as error:
                if _worker_summarization_request(error) is not None:
                    return await self._run_terminal_summarizer(
                        request=request,
                        invocation_id=invocation_id,
                        transcript=transcript,
                        observed_refs=tuple(self._invocation_observed_refs),
                        secrets=summary_secrets,
                    )
                budget_error = _worker_budget_error(error)
                if budget_error is None:
                    raise
                self._metrics.record_worker_budget_exhausted(budget_error.dimension)
                return (
                    _failure(
                        "worker_budget_exhausted",
                        f"Worker invocation budget exhausted ({budget_error.dimension})",
                        True,
                    ),
                    False,
                )
            return self._build_runtime_result(
                request, candidate, tuple(self._invocation_observed_refs)
            )
        finally:
            self._invocation_observed_refs.clear()
            _clear_artifact_observation_logs(self._context.tools)

    async def _run_terminal_summarizer(
        self,
        *,
        request: StageContentRequest,
        invocation_id: str,
        transcript: TranscriptRecorder,
        observed_refs: tuple[ArtifactRef, ...],
        secrets: tuple[str, ...],
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        config = self._context.summarizer
        if config is None:
            raise RuntimeError("Worker summarization was requested while disabled")
        usage = SummarizerUsage()
        summarizer: TerminalSummarizer | None = None
        try:
            groups = transcript.finish()
            live_state = await self._worker_state.snapshot()
            observations = _lean_observations(
                live_state,
                invocation_id,
                projection_failed=self._plugin.projection_failed,
            )
            prompt = build_summarizer_prompt(
                request,
                observations,
                groups,
                transcript_truncated=transcript.truncated,
                secrets=secrets,
            )
            summary_context = replace(
                self._context,
                model_policy=config.model_policy,
                summarizer=None,
                tools={},
                resolved_skills=(),
                agent_skills=None,
                project_workspace=None,
                workspace_export=None,
            )
            summary_model = self._model_factory(summary_context)
            if summary_model is self._model:
                raise SummarizerFailure("model_not_isolated")
            summarizer = TerminalSummarizer(model=summary_model, policy=config.model_policy)
            candidate = await summarizer.run(prompt=prompt, invocation_id=invocation_id)
            usage = summarizer.usage
            result, exportable = self._build_runtime_result(request, candidate, observed_refs)
            if not isinstance(result, WorkerResult):
                failure_code = _summarizer_result_failure_code(result.code)
                await self._complete_summarizer_attempt(
                    invocation_id=invocation_id,
                    succeeded=False,
                    usage=usage,
                    failure_code=failure_code,
                )
                return _summarizer_failure(failure_code), False
            await self._complete_summarizer_attempt(
                invocation_id=invocation_id,
                succeeded=True,
                usage=usage,
            )
            return result.model_copy(update={"summarized": True}), exportable
        except asyncio.CancelledError:
            usage = summarizer.usage if summarizer is not None else usage
            await _await_safely(
                self._complete_summarizer_attempt(
                    invocation_id=invocation_id,
                    succeeded=False,
                    usage=usage,
                    failure_code="cancelled",
                )
            )
            raise
        except SummarizerFailure as error:
            usage = summarizer.usage if summarizer is not None else usage
            failure_code = error.code
        except Exception:
            usage = summarizer.usage if summarizer is not None else usage
            failure_code = "runtime_failed"

        await self._complete_summarizer_attempt(
            invocation_id=invocation_id,
            succeeded=False,
            usage=usage,
            failure_code=failure_code,
        )
        return _summarizer_failure(failure_code), False

    async def _complete_summarizer_attempt(
        self,
        *,
        invocation_id: str,
        succeeded: bool,
        usage: SummarizerUsage,
        failure_code: str | None = None,
    ) -> None:
        await self._worker_state.complete_summarization(
            invocation_id=invocation_id,
            succeeded=succeeded,
            model_calls=usage.model_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            token_usage_unavailable=usage.token_usage_unavailable,
            failure_code=failure_code,
        )
        self._metrics.record_summarizer_attempt(
            succeeded=succeeded,
            model_calls=usage.model_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            token_usage_unavailable=usage.token_usage_unavailable,
            failure_code=failure_code,
        )

    def _build_runtime_result(
        self,
        request: StageContentRequest,
        candidate: str | None,
        observed_refs: tuple[ArtifactRef, ...],
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        if candidate is None:
            return _failure(
                "worker_result_missing", "Worker returned no structured result", True
            ), False
        if len(candidate.encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
            return _failure(
                "worker_result_too_large", "Worker structured result exceeds its limit", False
            ), False
        try:
            raw_candidate = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            return _failure(
                "worker_result_invalid", "Worker returned an invalid structured result", True
            ), False
        if (
            isinstance(raw_candidate, dict)
            and isinstance(raw_candidate.get("result"), str)
            and len(raw_candidate["result"].encode("utf-8")) > 64 * 1024
        ):
            return _failure(
                "worker_result_too_large", "Worker structured result exceeds its limit", False
            ), False
        try:
            model_result = WorkerModelResult.model_validate(raw_candidate)
        except ValidationError:
            return _failure(
                "worker_result_invalid", "Worker returned an invalid structured result", True
            ), False
        if model_result.subtask_id != request.subtask_id:
            return _failure(
                "worker_result_subtask_mismatch",
                "Worker result does not match the requested subtask",
                True,
            ), False
        wrapped_gateway_token = self._context.runtime_settings.llm_gateway_token
        gateway_token = (
            wrapped_gateway_token.get_secret_value() if wrapped_gateway_token is not None else ""
        )
        if gateway_token and gateway_token in model_result.result:
            return _failure(
                "unsafe_worker_result", "Worker returned content blocked by Runtime policy", False
            ), False
        observed = {
            (ref.namespace, ref.name): ref for ref in _latest_observed_exact_refs(observed_refs)
        }
        artifacts: dict[str, ArtifactRef] = {}
        exporter = self._workspace_exporter
        reserved_slots = exporter.reserved_slots if exporter is not None else frozenset()
        for slot, binding in request.result_artifacts.items():
            if slot in reserved_slots:
                continue
            if binding.namespace in {"inputs", "outputs", "skills"} or is_reserved_memory_binding(
                binding.namespace, binding.name
            ):
                return _failure(
                    "invalid_worker_result_binding",
                    "Worker result binding is reserved by Runtime policy",
                    False,
                ), False
            exact = observed.get((binding.namespace, binding.name))
            if exact is not None:
                artifacts[slot] = exact
        if len(artifacts) > MAX_RESULT_ARTIFACTS - len(reserved_slots):
            return _failure(
                "worker_result_too_large", "Worker result exceeds its limit", False
            ), False
        result = WorkerResult(
            subtaskId=request.subtask_id,
            result=model_result.result,
            observations=WorkerObservations(
                profile="lean@1", tools={}, workspace=None, truncated=False
            ),
            artifacts=artifacts,
            summarized=False,
        )
        if len(result.model_dump_json(by_alias=True).encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
            return _failure(
                "worker_result_too_large", "Worker result exceeds its limit", False
            ), False
        return result, True

    def _observe_tool_artifacts(self, tool: Any, cursor: int) -> None:
        observations = getattr(tool, "observed_exact_refs_since", None)
        if not callable(observations):
            return
        try:
            refs = observations(cursor)
            for ref in refs:
                exact = ref.require_exact()
                if not is_reserved_memory_binding(exact.namespace, exact.name):
                    self._invocation_observed_refs.append(exact)
        except Exception:
            # A provenance adapter defect must not turn a completed tool side
            # effect into an invented result. Omitting the ref fails the Stage
            # result contract closed at the Planner boundary.
            return

    async def _stop(self, deadline: datetime) -> None:
        self._accepting = False
        task = self._active_task
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
                await self._session_service.delete_session(
                    app_name=self._app_name,
                    user_id=self._user_id,
                    session_id=self._session_id,
                )
            except Exception as error:
                failures.append(error)
            agent_skills = self._agent_skills
            if agent_skills is not None:
                try:
                    await agent_skills.close()
                except Exception as error:
                    failures.append(error)
        finally:
            model = self._model
            self._model = None
            if isinstance(model, GatewayLiteLlm):
                model.clear_credentials()
            self._agent = None
            self._instrumentation = None
            self._agent_skills = None
            self._workspace_exporter = None
            self._context = None  # type: ignore[assignment]
        if failures:
            raise failures[0]

    async def _finish_invocation_state(
        self,
        invocation_id: str,
        subtask_id: str,
        phase: InvocationPhase,
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
            await self._sync_worker_state(snapshot)
            return snapshot

        task = asyncio.create_task(finish(), name="worker-invocation-state-finalize")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task
            raise

    async def _untracked_failure_completion(
        self, code: str, message: str, retryable: bool
    ) -> WorkerCompletion:
        self._metrics.record_outcome("failed")
        snapshot = await self._worker_state.sync_metrics()
        await self._sync_worker_state(snapshot)
        return WorkerCompletion(
            apiVersion=API_VERSION,
            failure=_failure(code, message, retryable),
            invocationId=f"worker-rejected-{uuid.uuid4().hex}",
            stateRevision=snapshot["stateRevision"],
        )

    async def _sync_worker_state(self, snapshot: dict[str, Any] | None = None) -> None:
        if snapshot is None:
            snapshot = await self._worker_state.sync_metrics()
        session = await self._session_service.get_session(
            app_name=self._app_name, user_id=self._user_id, session_id=self._session_id
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


def _candidate_text(event: Event) -> str | None:
    content = event.content
    if content is None or content.role != "model" or event.partial:
        return None
    text: list[str] = []
    for part in content.parts or []:
        if getattr(part, "thought", False):
            continue
        if part.text is None:
            return None
        text.append(part.text)
    return "".join(text) if text else None


def _latest_observed_exact_refs(refs: tuple[ArtifactRef, ...]) -> list[ArtifactRef]:
    latest: dict[tuple[str, str], ArtifactRef] = {}
    for ref in refs:
        exact = ref.require_exact()
        if is_reserved_memory_binding(exact.namespace, exact.name):
            continue
        latest[(exact.namespace, exact.name)] = exact
    return [latest[key] for key in sorted(latest)]


def _clear_artifact_observation_logs(tools: Mapping[str, Any]) -> None:
    for tool in tools.values():
        clear = getattr(tool, "clear_artifact_observations", None)
        if callable(clear):
            clear()


def _task_prompt(request: StageContentRequest) -> str:
    """Render Planner-supplied task data; wire/lifecycle contracts stay Runtime-private."""

    inputs = {
        name: ref.model_dump(mode="json", by_alias=True)
        for name, ref in sorted(request.artifacts.items())
    }
    return (
        f"Subtask ID:\n{request.subtask_id}\n\n"
        f"Objective:\n{request.objective}\n\n"
        f"Task instructions:\n{request.instructions}\n\n"
        "String parameters:\n"
        + json.dumps(request.parameters, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nNamed input artifacts:\n"
        + json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=2)
    )


def _summary_prompt_boundary(
    policy: ResolvedModelPolicy,
    config: WorkerSummarizerConfig | None,
) -> int | None:
    """Derive the normal-loop prompt boundary from pinned model metadata."""

    if config is None:
        return None
    if policy.context_window_tokens is None or policy.max_output_tokens is None:
        raise RuntimeError("summarized Worker policy has no context-window metadata")
    ratio_boundary = math.floor(policy.context_window_tokens * config.context_window_ratio)
    output_safe_boundary = policy.context_window_tokens - policy.max_output_tokens
    return min(ratio_boundary, output_safe_boundary)


def _summarizer_secrets(context: WorkerBuildContext, gateway_token: str) -> tuple[str, ...]:
    """Return allocation-private values that must not enter summarizer input."""

    candidates = (
        gateway_token,
        str(context.workspace.path),
        str(context.workspace.root),
    )
    # Replacing '/' would destroy every path-like value rather than protect a
    # useful host path, so only non-root concrete paths are admitted.
    return tuple(dict.fromkeys(value for value in candidates if len(value) > 1))


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


def _safe_error_type(error: Exception) -> str:
    error_type = getattr(error, "provider_error_type", type(error).__name__)
    if not isinstance(error_type, str) or SAFE_TOOL_ERROR_CODE.fullmatch(error_type) is None:
        return type(error).__name__
    return error_type


def _worker_budget_error(error: BaseException) -> WorkerBudgetExceeded | None:
    """Find a budget signal through ADK's plugin callback wrappers."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, WorkerBudgetExceeded):
            return current
        current = current.__cause__ or current.__context__
    return None


def _worker_summarization_request(
    error: BaseException,
) -> WorkerSummarizationRequested | None:
    """Find the Runtime control signal through ADK callback wrappers."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, WorkerSummarizationRequested):
            return current
        current = current.__cause__ or current.__context__
    return None


def _record_worker_error(
    instrumentation: RuntimeInstrumentation | None,
    error_type: str,
) -> None:
    span = _start_span(
        instrumentation,
        "contractor.worker.error",
        {"operation.kind": "worker_error", "error.type": error_type},
    )
    _end_span(span, outcome="failed")


def _aggregate_count_attributes(counters: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for source, target in (
        ("llm_calls", "counts.model_calls"),
        ("tool_calls", "counts.tool_calls"),
    ):
        value = counters.get(source)
        if isinstance(value, int) and value >= 0:
            result[target] = value
    return result


def _failure(code: str, message: str, retryable: bool) -> WorkerFailure:
    return WorkerFailure(code=code, message=message, retryable=retryable)


def _summarizer_failure(cause: str) -> WorkerFailure:
    return _failure(
        "worker_summarization_failed",
        f"Worker terminal summarization failed ({cause})",
        True,
    )


def _summarizer_result_failure_code(code: str) -> str:
    return {
        "worker_result_missing": "result_missing",
        "worker_result_invalid": "result_invalid",
        "worker_result_subtask_mismatch": "result_subtask_mismatch",
        "worker_result_too_large": "result_too_large",
        "unsafe_worker_result": "unsafe_result",
        "invalid_worker_result_binding": "invalid_result_binding",
    }.get(code, "result_rejected")


async def _await_safely(awaitable: Any) -> Any:
    task = asyncio.create_task(awaitable, name="worker-summarizer-state-finalize")
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await task
        return None


def _gateway_model_error(error: BaseException) -> GatewayModelError | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GatewayModelError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _empty_invocation_metrics() -> dict[str, Any]:
    return {
        "modelCalls": 0,
        "modelErrors": 0,
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "cachedInputTokens": 0,
        "tokenUsageUnavailable": 0,
        "latestPromptTokens": None,
        "toolCalls": 0,
        "toolErrors": 0,
        "tools": {},
        "truncated": False,
    }


def _lean_observations(
    snapshot: Mapping[str, Any],
    invocation_id: str,
    *,
    projection_failed: bool,
) -> WorkerObservations:
    invocation = snapshot.get("currentInvocation")
    if not isinstance(invocation, Mapping) or invocation.get("invocationId") != invocation_id:
        invocation = snapshot.get("lastCompletedInvocation")
    if not isinstance(invocation, Mapping) or invocation.get("invocationId") != invocation_id:
        raise RuntimeError("Worker State invocation correlation is invalid")
    metrics = invocation.get("metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("Worker invocation metrics are unavailable")
    raw_tools = metrics.get("tools")
    if not isinstance(raw_tools, Mapping):
        raise RuntimeError("Worker invocation tool metrics are unavailable")
    tools: dict[str, ToolObservationCount] = {}
    for name in sorted(raw_tools):
        aggregate = raw_tools[name]
        if not isinstance(name, str) or not isinstance(aggregate, Mapping):
            raise RuntimeError("Worker invocation tool metrics are invalid")
        tools[name] = ToolObservationCount(
            calls=aggregate.get("calls"),
            failures=aggregate.get("failures"),
        )
    workspace, workspace_truncated = lean_workspace_summary(invocation.get("workspace"))
    return WorkerObservations(
        profile="lean@1",
        tools=tools,
        workspace=workspace,
        truncated=(bool(metrics.get("truncated")) or workspace_truncated or projection_failed),
    )
