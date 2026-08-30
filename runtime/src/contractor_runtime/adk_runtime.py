"""Google ADK-backed allocation-local Worker runtime."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncGenerator, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from google.adk.agents import LlmAgent
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
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    StageContentRequest,
    StageContentResult,
    StageOutcome,
    TerminationError,
)

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse

    from contractor_runtime.factories import WorkerBuildContext

MAX_STAGE_REQUEST_JSON_BYTES = 256 * 1024
MAX_STAGE_RESULT_JSON_BYTES = 256 * 1024
MAX_RESULT_ARTIFACTS = 128
MAX_RESULT_SUMMARY_CHARS = 64 * 1024
SAFE_TOOL_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ModelFactory(Protocol):
    def __call__(self, context: WorkerBuildContext) -> BaseLlm: ...


class GatewayModelError(RuntimeError):
    """Secret-free boundary error for failures below the LLM Gateway adapter."""

    def __init__(self, provider_error_type: str) -> None:
        self.provider_error_type = provider_error_type
        super().__init__(f"LLM gateway call failed ({provider_error_type})")


class WorkerFunctionTool(FunctionTool):
    """Return bounded tool failures to the model so it can correct or terminate."""

    async def run_async(self, *, args: dict[str, Any], tool_context: Any) -> Any:
        try:
            return await super().run_async(args=args, tool_context=tool_context)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            code = getattr(error, "code", "tool_call_failed")
            if not isinstance(code, str) or SAFE_TOOL_ERROR_CODE.fullmatch(code) is None:
                code = "tool_call_failed"
            return {
                "ok": False,
                "error": {
                    "code": code,
                    "message": f"{self.name} failed ({type(error).__name__})",
                    "retryable": bool(getattr(error, "retryable", False)),
                },
            }


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
        self._additional_args.clear()


def gateway_model(context: WorkerBuildContext) -> BaseLlm:
    policy = context.agent_template.model_policy
    settings = context.runtime_settings
    return GatewayLiteLlm(
        model=f"openai/{policy.model}",
        api_base=settings.llm_gateway_url,
        api_key=settings.llm_gateway_token.get_secret_value(),
        timeout=float(settings.request_timeout_seconds),
    )


class AdkWorkerRuntimeFactory:
    ref = "adk@1"

    def __init__(self, model_factory: ModelFactory | None = None) -> None:
        self._model_factory = model_factory or gateway_model

    async def create(self, context: WorkerBuildContext) -> AdkWorkerRuntime:
        runtime = AdkWorkerRuntime(context, self._model_factory(context))
        await runtime.start()
        return runtime


class AdkWorkerRuntime:
    def __init__(self, context: WorkerBuildContext, model: BaseLlm) -> None:
        self.allocation_id = context.allocation_id
        self._context = context
        self._model: BaseLlm | None = model
        self._metrics = context.state.metrics
        self._session_service = InMemorySessionService()
        self._session_id = context.allocation_id
        self._app_name = "contractor_runtime_worker"
        self._user_id = "contractor_control_plane"
        self._accepting = True
        self._invoke_lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._runner: Runner | None = None
        self._finalizer_runner: Runner | None = None
        self._agent: LlmAgent | None = None
        self._finalizer_agent: LlmAgent | None = None

        policy = context.agent_template.model_policy
        generation = types.GenerateContentConfig(max_output_tokens=policy.max_output_tokens)
        if policy.temperature is not None:
            generation.temperature = policy.temperature
        adk_tools = [WorkerFunctionTool(tool) for tool in context.tools.values()]
        self._agent = LlmAgent(
            name="contractor_worker",
            description=context.agent_template.description,
            model=model,
            instruction=context.agent_template.instructions.text,
            tools=adk_tools,
            # Do not combine ADK output_schema with function tools. OpenAI-compatible
            # local backends turn that schema into a grammar on every turn, including
            # tool-selection turns, and some reject the resulting grammar. The final
            # free-text candidate is validated strictly below before it crosses the
            # Worker boundary.
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
        self._finalizer_agent = LlmAgent(
            name="contractor_worker_result_finalizer",
            description="Serialize one already completed Contractor Worker result",
            model=model,
            instruction=(
                "You are a result serializer, not a task executor. Tools are unavailable. "
                "Return exactly one raw StageContentResult JSON object using only an exact "
                "ArtifactRef explicitly supplied in the finalization request. Never invent, "
                "shorten, or alter a revision."
            ),
            tools=[],
            generate_content_config=generation.model_copy(deep=True),
            before_model_callback=self._before_model,
            after_model_callback=self._after_model,
            on_model_error_callback=self._on_model_error,
        )
        self._finalizer_runner = Runner(
            app_name=self._app_name,
            agent=self._finalizer_agent,
            session_service=self._session_service,
        )
        endpoint = (
            f"{context.a2a_base_url.rstrip('/')}/private/v1/allocations/{context.allocation_id}/a2a"
        )
        self._card = build_agent_card(
            allocation_id=context.allocation_id,
            endpoint=endpoint,
            logical_agent_name=context.logical_agent_name,
            description=context.agent_template.description,
            version=context.agent_template.ref.version,
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
        if not self._accepting:
            return _failure("worker_draining", "Worker is no longer accepting A2A work", True)
        if self._invoke_lock.locked():
            return _failure("worker_busy", "Worker already has an active A2A invocation", True)
        await self._invoke_lock.acquire()
        self._active_task = asyncio.current_task()
        model_errors_before = self._metrics.counters.get("llm_errors", 0)
        try:
            if not self._accepting:
                return _failure("worker_draining", "Worker is no longer accepting A2A work", True)
            encoded_request = request.model_dump_json(by_alias=True, exclude_none=True).encode(
                "utf-8"
            )
            if len(encoded_request) > MAX_STAGE_REQUEST_JSON_BYTES:
                result = _failure(
                    "stage_content_too_large", "StageContentRequest exceeds the Worker limit", False
                )
            else:
                result = await self._run_adk(encoded_request.decode("utf-8"))
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

    async def _run_adk(self, request_json: str) -> StageContentResult:
        runner = self._runner
        if runner is None:
            return _failure("worker_draining", "Worker is no longer accepting A2A work", True)
        prompt = (
            "Execute the following Contractor StageContentRequest. Durable data is represented "
            "only by ArtifactRef values.\n"
            + request_json
            + "\nReturn raw JSON without Markdown fences. A successful final "
            'response has shape {"apiVersion":"contractor/v1alpha1","outcome":"succeeded",'
            '"summary":"...","artifacts":{"result_slot":{"namespace":"...","name":"...",'
            '"revision":"..."}}}. A failed response uses outcome "failed", may use an empty '
            'artifacts object, and must add {"error":{"code":"...","message":"...",'
            '"retryable":true}}. Return exactly one StageContentResult JSON object.'
        )
        candidate: str | None = None
        async for event in runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            invocation_id=f"worker-{uuid.uuid4().hex}",
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            text = _candidate_text(event)
            if text is not None:
                candidate = text
        if candidate is None:
            candidate = await self._recover_missing_candidate(request_json)
            self._metrics.record_worker_result_recovery(succeeded=candidate is not None)
        if candidate is None:
            self._metrics.record_worker_result_error("missing")
            return _failure("invalid_worker_result", "Worker returned no bounded JSON result", True)
        if len(candidate.encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
            self._metrics.record_worker_result_error("oversized")
            return _failure(
                "invalid_worker_result", "Worker returned an oversized StageContentResult", True
            )
        gateway_token = self._context.runtime_settings.llm_gateway_token.get_secret_value()
        if gateway_token and gateway_token in candidate:
            return _failure(
                "unsafe_worker_result", "Worker returned content blocked by Runtime policy", False
            )
        try:
            result = StageContentResult.model_validate_json(_unwrap_json_fence(candidate))
        except ValidationError as error:
            error_types = sorted(
                {
                    item_type
                    for item in error.errors(
                        include_url=False, include_context=False, include_input=False
                    )
                    if isinstance((item_type := item.get("type")), str)
                    and SAFE_TOOL_ERROR_CODE.fullmatch(item_type) is not None
                }
            )
            classification = "schema_" + "_".join(error_types[:8])
            self._metrics.record_worker_result_error(classification)
            return _failure(
                "invalid_worker_result", "Worker returned an invalid StageContentResult", True
            )
        if (
            len(result.summary) > MAX_RESULT_SUMMARY_CHARS
            or len(result.artifacts) > MAX_RESULT_ARTIFACTS
        ):
            return _failure(
                "invalid_worker_result", "Worker returned an oversized StageContentResult", False
            )
        known = _known_exact_refs(self._context.tools)
        if any(_ref_key(ref) not in known for ref in result.artifacts.values()):
            return _failure(
                "unverified_artifact_ref",
                "Worker result contains an artifact revision not observed through ArtifactClient",
                True,
            )
        return result

    async def _recover_missing_candidate(self, request_json: str) -> str | None:
        """Request one tool-free envelope after a tool loop ends without text."""

        runner = self._finalizer_runner
        if runner is None:
            return None
        exact_refs = [
            ref.model_dump(mode="json", by_alias=True)
            for ref in _latest_known_exact_refs(self._context.tools)
        ]
        prompt = (
            "The Worker tool phase ended without a model-visible final response. Do not perform "
            "more analysis. Serialize its result now.\nStageContentRequest:\n"
            + request_json
            + "\nLatest exact ArtifactRefs observed through trusted tools:\n"
            + json.dumps(exact_refs, ensure_ascii=False, separators=(",", ":"))
            + "\nReturn raw JSON without Markdown fences. For success use "
            '{"apiVersion":"contractor/v1alpha1","outcome":"succeeded",'
            '"summary":"...","artifacts":{"result_slot":{"namespace":"...",'
            '"name":"...","revision":"..."}}}. For failure use outcome "failed", an '
            'empty artifacts object if appropriate, and add {"error":{"code":"...",'
            '"message":"...","retryable":true}}. Use only supplied exact refs and return '
            "exactly one object."
        )
        candidate: str | None = None
        async for event in runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            invocation_id=f"worker-finalizer-{uuid.uuid4().hex}",
            new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
        ):
            text = _candidate_text(event)
            if text is not None:
                candidate = text
        return candidate

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
        finalizer_runner = self._finalizer_runner
        self._finalizer_runner = None
        try:
            if runner is not None:
                await runner.close()
            if finalizer_runner is not None:
                await finalizer_runner.close()
            await self._session_service.delete_session(
                app_name=self._app_name, user_id=self._user_id, session_id=self._session_id
            )
        finally:
            model = self._model
            self._model = None
            if isinstance(model, GatewayLiteLlm):
                model.clear_credentials()
            self._agent = None
            self._finalizer_agent = None

    async def _before_model(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> None:
        del callback_context, llm_request
        self._metrics.record_model_call()

    async def _after_model(
        self, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> None:
        del callback_context
        if llm_response.usage_metadata is not None:
            self._metrics.record_model_usage(llm_response.usage_metadata)

    async def _on_model_error(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> None:
        del callback_context, llm_request
        self._metrics.record_model_error(error)

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


def _unwrap_json_fence(candidate: str) -> str:
    stripped = candidate.strip()
    for prefix in ("```json\n", "```JSON\n", "```\n"):
        if stripped.startswith(prefix) and stripped.endswith("\n```"):
            return stripped[len(prefix) : -4].strip()
    return stripped


def _known_exact_refs(tools: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for tool in tools.values():
        for ref in getattr(tool, "known_exact_refs", ()):
            result.add(_ref_key(ref))
    return result


def _latest_known_exact_refs(tools: Mapping[str, Any]) -> list[ArtifactRef]:
    latest: dict[tuple[str, str], ArtifactRef] = {}
    for tool in tools.values():
        for ref in getattr(tool, "known_exact_refs", ()):
            exact = ref.require_exact()
            latest[(exact.namespace, exact.name)] = exact
    return [latest[key] for key in sorted(latest)]


def _ref_key(ref: ArtifactRef) -> tuple[str, str, str]:
    revision = ref.require_exact().revision
    assert revision is not None
    return ref.namespace, ref.name, revision


def _failure(code: str, summary: str, retryable: bool) -> StageContentResult:
    return StageContentResult(
        apiVersion=API_VERSION,
        outcome=StageOutcome.FAILED,
        summary=summary,
        artifacts={},
        error=TerminationError(code=code, message=summary, retryable=retryable),
    )
