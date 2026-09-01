"""Google ADK-backed allocation-local Worker runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from google.adk.agents import LlmAgent
from google.adk.events import Event, EventActions
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
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
    ArtifactRef,
    StageContentRequest,
    StageContentResult,
    StageOutcome,
    TerminationError,
)
from contractor_runtime.model_client import (
    clear_gateway_client_options,
    gateway_client_options,
)
from contractor_runtime.projectfs import (
    MAX_EXPORTED_RESULT_ARTIFACTS,
    MAX_EXPORTED_RESULT_JSON_BYTES,
    OverlayWorkspaceSession,
    WorkspaceAutoExporter,
    WorkspaceExportError,
)
from contractor_runtime.toolsets.artifact_visibility import is_reserved_memory_binding

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    from contractor_runtime.factories import WorkerBuildContext

MAX_STAGE_REQUEST_JSON_BYTES = 256 * 1024
MAX_STAGE_RESULT_JSON_BYTES = MAX_EXPORTED_RESULT_JSON_BYTES
MAX_RESULT_ARTIFACTS = MAX_EXPORTED_RESULT_ARTIFACTS
MAX_RESULT_SUMMARY_CHARS = 64 * 1024
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
    model_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    token_usage_unavailable: int = 0

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

    def _sync(self) -> None:
        self.metrics.observe_worker_budget(
            model_calls=self.model_calls,
            tool_calls=self.tool_calls,
            total_tokens=self.total_tokens,
            token_usage_unavailable=self.token_usage_unavailable,
        )


class WorkerFunctionTool(FunctionTool):
    """Return bounded tool failures to the model so it can correct or terminate."""

    def __init__(
        self,
        function: Callable[..., Any],
        budget: Callable[[], _InvocationBudget | None],
        instrumentation: RuntimeInstrumentation | None,
        observe_artifacts: Callable[[Any, int], None],
    ):
        super().__init__(function)
        self._budget = budget
        self._instrumentation = instrumentation
        self._observe_artifacts = observe_artifacts

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        observation_cursor = getattr(self.func, "artifact_observation_cursor", None)
        completed_successfully = False
        budget = self._budget()
        if budget is not None:
            budget.before_tool_call()
        span = _start_span(
            self._instrumentation,
            "contractor.worker.tool",
            {"operation.kind": "tool", "tool.name": self.name},
        )
        try:
            raw_argument_error = getattr(self.func, "contractor_raw_argument_error", None)
            if callable(raw_argument_error):
                rejection = raw_argument_error(args)
                if rejection is not None:
                    raise rejection
            result = await super().run_async(args=args, tool_context=tool_context)
            completed_successfully = not (isinstance(result, Mapping) and result.get("ok") is False)
        except asyncio.CancelledError:
            _end_span(span, outcome="cancelled")
            raise
        except Exception as error:
            code = getattr(error, "code", "tool_call_failed")
            if not isinstance(code, str) or SAFE_TOOL_ERROR_CODE.fullmatch(code) is None:
                code = "tool_call_failed"
            _end_span(
                span,
                outcome="failed",
                attributes={"error.type": _safe_error_type(error)},
            )
            return {
                "ok": False,
                "error": {
                    "code": code,
                    "message": f"{self.name} failed ({type(error).__name__})",
                    "retryable": bool(getattr(error, "retryable", False)),
                },
            }
        finally:
            if (
                completed_successfully
                and type(observation_cursor) is int
                and observation_cursor >= 0
            ):
                self._observe_artifacts(self.func, observation_cursor)
        outcome = (
            "failed" if isinstance(result, Mapping) and result.get("ok") is False else "succeeded"
        )
        _end_span(span, outcome=outcome)
        return result


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
        workspace_exporter: WorkspaceAutoExporter | None = None,
    ) -> None:
        self.allocation_id = context.allocation_id
        self._context = context
        self._model: BaseLlm | None = model
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
        self._model_spans: list[RuntimeSpan] = []
        self._agent_skills: PreparedAgentSkills | None = context.agent_skills
        self._workspace_exporter = workspace_exporter

        policy = context.model_policy
        generation = types.GenerateContentConfig(max_output_tokens=policy.max_output_tokens)
        if policy.temperature is not None:
            generation.temperature = policy.temperature
        adk_tools = [
            WorkerFunctionTool(
                tool,
                lambda: self._active_budget,
                self._instrumentation,
                self._observe_tool_artifacts,
            )
            for tool in context.tools.values()
        ]
        if self._agent_skills is not None:
            adk_tools.append(
                self._agent_skills.build_adapter(
                    budget=lambda: self._active_budget,
                    metrics=self._metrics,
                )
            )
        self._agent = LlmAgent(
            name="contractor_worker",
            description=context.description,
            model=model,
            instruction=context.instruction,
            tools=adk_tools,
            # The model owns task work and a human-readable summary only. Runtime
            # projects that summary and invocation-local trusted tool observations
            # into the private A2A response below.
            generate_content_config=generation,
            before_model_callback=self._before_model,
            after_model_callback=self._after_model,
            on_model_error_callback=self._on_model_error,
        )
        self._runner = Runner(
            app_name=self._app_name,
            agent=self._agent,
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
        await self._session_service.create_session(
            app_name=self._app_name,
            user_id=self._user_id,
            session_id=self._session_id,
            state={"metrics": self._metrics.snapshot()},
        )

    @property
    def agent_card(self) -> Mapping[str, Any]:
        return dict(self._agent_card)

    @property
    def a2a_application(self) -> ASGIApp:
        return self._a2a_application

    async def invoke(self, request: StageContentRequest) -> StageContentResult:
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
        if result.error is not None:
            attributes["error.type"] = result.error.code
            _record_worker_error(self._instrumentation, result.error.code)
        _end_span(span, outcome=result.outcome.value, attributes=attributes)
        return result

    async def _invoke(self, request: StageContentRequest) -> StageContentResult:
        if not self._accepting:
            return _failure("worker_draining", "Worker is no longer accepting A2A work", True)
        if self._invoke_lock.locked():
            return _failure("worker_busy", "Worker already has an active A2A invocation", True)
        await self._invoke_lock.acquire()
        self._active_task = asyncio.current_task()
        policy = self._context.model_policy
        budget = _InvocationBudget(
            max_model_calls=policy.max_model_calls,
            max_tool_calls=policy.max_tool_calls,
            max_total_tokens=policy.max_total_tokens,
            metrics=self._metrics,
        )
        self._active_budget = budget
        model_errors_before = self._metrics.counters.get("llm_errors", 0)
        try:
            budget.start()
            if not self._accepting:
                return _failure("worker_draining", "Worker is no longer accepting A2A work", True)
            encoded_request = request.model_dump_json(by_alias=True, exclude_none=True).encode(
                "utf-8"
            )
            if len(encoded_request) > MAX_STAGE_REQUEST_JSON_BYTES:
                result = _failure(
                    "stage_content_too_large", "StageContentRequest exceeds the Worker limit", False
                )
                exportable = False
            else:
                result, exportable = await self._run_adk(request)
            exporter = self._workspace_exporter
            if exportable and exporter is not None:
                try:
                    exported = await exporter.export(result)
                except WorkspaceExportError as error:
                    self._metrics.record_workspace_export(error=error)
                    result = _failure(
                        "workspace_export_failed",
                        f"Workspace export failed ({error.cause})",
                        error.retryable,
                    )
                else:
                    self._metrics.record_workspace_export(
                        state_bytes=exported.state_bytes,
                        diff_bytes=exported.diff_bytes,
                    )
                    result = exported.result
            self._metrics.record_outcome(result.outcome.value)
            return result
        except asyncio.CancelledError:
            self._metrics.record_outcome("cancelled")
            raise
        except Exception as error:
            if self._metrics.counters.get("llm_errors", 0) == model_errors_before:
                self._metrics.record_model_error(error)
            result = _failure("worker_execution_failed", "Worker execution failed", True)
            self._metrics.record_outcome(result.outcome.value)
            return result
        finally:
            try:
                await asyncio.shield(self._sync_metrics())
            finally:
                self._active_budget = None
                self._active_task = None
                self._invoke_lock.release()

    def cancel_active(self) -> None:
        task = self._active_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def finalize(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def abort(self, deadline: datetime) -> None:
        await self._stop(deadline)

    async def _run_adk(self, request: StageContentRequest) -> tuple[StageContentResult, bool]:
        runner = self._runner
        if runner is None:
            return _failure(
                "worker_draining", "Worker is no longer accepting A2A work", True
            ), False
        prompt = _task_prompt(request)
        self._invocation_observed_refs.clear()
        candidate: str | None = None
        try:
            try:
                async for event in runner.run_async(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    invocation_id=f"worker-{uuid.uuid4().hex}",
                    new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
                ):
                    text = _candidate_text(event)
                    if text is not None:
                        candidate = text
            except WorkerBudgetExceeded as error:
                self._metrics.record_worker_budget_exhausted(error.dimension)
                return (
                    _failure(
                        "worker_budget_exhausted",
                        f"Worker invocation budget exhausted ({error.dimension})",
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

    def _build_runtime_result(
        self,
        request: StageContentRequest,
        candidate: str | None,
        observed_refs: tuple[ArtifactRef, ...],
    ) -> tuple[StageContentResult, bool]:
        if candidate is None:
            return _failure(
                "worker_result_missing", "Worker returned no final summary", True
            ), False
        if len(candidate.encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
            return _failure(
                "worker_result_too_large", "Worker final summary exceeds its limit", False
            ), False
        wrapped_gateway_token = self._context.runtime_settings.llm_gateway_token
        gateway_token = (
            wrapped_gateway_token.get_secret_value() if wrapped_gateway_token is not None else ""
        )
        if gateway_token and gateway_token in candidate:
            return _failure(
                "unsafe_worker_result", "Worker returned content blocked by Runtime policy", False
            ), False
        summary = candidate.strip()
        if not summary:
            return _failure(
                "worker_result_missing", "Worker returned no final summary", True
            ), False
        if len(summary) > MAX_RESULT_SUMMARY_CHARS:
            return _failure(
                "worker_result_too_large", "Worker final summary exceeds its limit", False
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
            if is_reserved_memory_binding(binding.namespace, binding.name):
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
        result = StageContentResult(
            apiVersion=API_VERSION,
            outcome=StageOutcome.SUCCEEDED,
            summary=summary,
            artifacts=artifacts,
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
            while self._model_spans:
                _end_span(self._model_spans.pop(), outcome="cancelled")
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

    async def _before_model(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        del callback_context, llm_request
        budget = self._active_budget
        if budget is not None:
            budget.before_model_call()
        self._metrics.record_model_call()
        span = _start_span(
            self._instrumentation,
            "contractor.worker.model",
            {
                "operation.kind": "model",
                "model.alias": self._context.model_policy.model,
            },
        )
        if span is not None:
            self._model_spans.append(span)

    async def _after_model(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        del callback_context
        usage_attributes = _usage_attributes(llm_response.usage_metadata)
        if llm_response.usage_metadata is not None:
            self._metrics.record_model_usage(llm_response.usage_metadata)
        _end_span(self._pop_model_span(), outcome="succeeded", attributes=usage_attributes)
        budget = self._active_budget
        if budget is not None:
            budget.after_model_response(llm_response.usage_metadata)

    async def _on_model_error(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        del callback_context, llm_request
        span = self._pop_model_span()
        if isinstance(error, WorkerBudgetExceeded):
            _end_span(span, outcome="rejected", attributes={"error.type": type(error).__name__})
            return
        _end_span(
            span,
            outcome="failed",
            attributes={"error.type": _safe_error_type(error)},
        )
        self._metrics.record_model_error(error)

    def _pop_model_span(self) -> RuntimeSpan | None:
        return self._model_spans.pop() if self._model_spans else None

    async def _sync_metrics(self) -> None:
        session = await self._session_service.get_session(
            app_name=self._app_name, user_id=self._user_id, session_id=self._session_id
        )
        if session is None:
            return
        await self._session_service.append_event(
            session,
            Event(
                invocationId=f"metrics-{uuid.uuid4().hex}",
                author="contractor_runtime",
                actions=EventActions(stateDelta={"metrics": self._metrics.snapshot()}),
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
        f"Objective:\n{request.objective}\n\n"
        f"Task instructions:\n{request.instructions}\n\n"
        "String parameters:\n"
        + json.dumps(request.parameters, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n\nNamed input artifacts:\n"
        + json.dumps(inputs, ensure_ascii=False, sort_keys=True, indent=2)
    )


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


def _failure(code: str, summary: str, retryable: bool) -> StageContentResult:
    return StageContentResult(
        apiVersion=API_VERSION,
        outcome=StageOutcome.FAILED,
        summary=summary,
        artifacts={},
        error=TerminationError(code=code, message=summary, retryable=retryable),
    )
