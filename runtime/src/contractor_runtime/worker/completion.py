"""Worker completion, finalization, summarization and result assembly."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from google.adk.models.base_llm import BaseLlm
from pydantic import ValidationError

from contractor_runtime.adapters import RuntimeInstrumentation
from contractor_runtime.contracts import (
    ArtifactRef,
    StageContentRequest,
    ToolObservationCount,
    WorkerFailure,
    WorkerModelResult,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.contracts.artifacts import (
    is_reserved_memory_binding,
    is_reserved_result_binding,
)
from contractor_runtime.contracts.base import is_blank
from contractor_runtime.core.async_tasks import join_owned_task
from contractor_runtime.core.worker_completion import (
    ContinueCompletion,
    FailCompletion,
    PreparedWorkerCompletion,
)
from contractor_runtime.llm.cost import model_cost_phase
from contractor_runtime.llm.factory import ModelFactory
from contractor_runtime.llm.openai import (
    GatewayModelError,
    OpenAICompatibleGatewayLlm,
)
from contractor_runtime.projectfs import (
    MAX_EXPORTED_RESULT_ARTIFACTS,
    MAX_EXPORTED_RESULT_JSON_BYTES,
)
from contractor_runtime.shared_config import get_shared_config
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.worker.budget import WorkerBudgetExceeded
from contractor_runtime.worker.failures import (
    _chained_error,
    _failure,
    _gateway_worker_failure,
    _summarizer_failure,
    _summarizer_result_failure_code,
)
from contractor_runtime.worker.finalizer import (
    ResultFinalizerFailure,
    WorkerResultFinalizer,
)
from contractor_runtime.worker.instrumentation import (
    WorkerInstrumentationPlugin,
)
from contractor_runtime.worker.invocation import InvocationLifecycle
from contractor_runtime.worker.observations import lean_workspace_summary
from contractor_runtime.worker.prompts import _exposes_private_value
from contractor_runtime.worker.state import WorkerStateStore
from contractor_runtime.worker.summarizer import (
    SummarizerFailure,
    SummarizerUsage,
    TerminalSummarizer,
    TranscriptRecorder,
    build_summarizer_prompt,
)

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext

MAX_STAGE_RESULT_JSON_BYTES = MAX_EXPORTED_RESULT_JSON_BYTES
MAX_RESULT_ARTIFACTS = MAX_EXPORTED_RESULT_ARTIFACTS


class WorkerCompletionPipeline:
    """Complete one model result under Runtime-owned identity and artifact policy."""

    def __init__(
        self,
        *,
        context: WorkerBuildContext,
        state: WorkerStateStore,
        resources: ToolResources,
        plugin: WorkerInstrumentationPlugin,
        completion: PreparedWorkerCompletion | None,
        invocation: InvocationLifecycle,
        model: BaseLlm,
        model_factory: ModelFactory,
        instrumentation: RuntimeInstrumentation | None,
        reserved_slots: frozenset[str],
        check_active: Callable[[], None],
        admit: Callable[[], None],
    ) -> None:
        self._context = context
        self._worker_state = state
        self._resources = resources
        self._plugin = plugin
        self._completion = completion
        self._invocation = invocation
        self._metrics = state.metrics
        self._model = model
        self._model_factory = model_factory
        self._instrumentation = instrumentation
        self._reserved_slots = reserved_slots
        self._check_active = check_active
        self._admit_semantic_work = admit
        self.refs: list[ArtifactRef] = []
        self.finalizer = (
            None
            if completion
            else WorkerResultFinalizer(model=model, policy=context.model_policy, observer=plugin)
        )

    def release(self) -> None:
        self.finalizer = None
        self._model = None
        self._context = None
        self.refs.clear()

    async def finish(self, *, request, candidate, invocation_id, deadline):
        """One semantic dispatch before terminal State or session cleanup."""
        if self._completion is not None:
            try:
                decision = await self._completion.finish(
                    request=request,
                    deadline=deadline,
                    check_active=self._check_active,
                )
            except Exception as error:
                self._check_active()
                return self._completion.failure(error), False
            if isinstance(decision, ContinueCompletion):
                return decision, False
            if isinstance(decision, FailCompletion):
                return decision.failure, False
            # Completion owns the published text/receipts, while Runtime owns
            # identity, artifact slots and observations. Validate fresh fields:
            # the completion's mutable WorkerResult is not the final projection.
            fields = _validate_result_fields(
                {"subtaskId": request.subtask_id, "result": decision.result.result},
                expected_subtask_id=request.subtask_id,
            )
            if isinstance(fields, WorkerFailure):
                return fields, False
            return self.assemble_result(
                request,
                fields,
                tuple(self.refs),
                verified_finalizer_refs=tuple(decision.result.artifacts.values()),
            )
        return await self._complete_ordinary_finish(request, candidate, invocation_id)

    async def _complete_ordinary_finish(self, request, candidate, invocation_id):
        self._resources.execution.check()
        if candidate is None:
            return self.build_result(request, candidate, tuple(self.refs))
        try:
            return await self._run_result_finalizer(
                request=request,
                candidate=candidate,
                invocation_id=invocation_id,
                observed_refs=tuple(self.refs),
            )
        except Exception as error:
            # ADK wraps callback exceptions. The absolute invocation deadline
            # still owns a refusal in the finalizer's actual model callback.
            if self._invocation.deadline is not None:
                self._invocation.deadline.check()
            budget_error = _chained_error(error, WorkerBudgetExceeded)
            if budget_error is not None:
                self._metrics.record_worker_budget_exhausted(budget_error.dimension)
                return (
                    _failure(
                        "worker_budget_exhausted",
                        f"Worker invocation budget exhausted ({budget_error.dimension})",
                        True,
                    ),
                    False,
                )
            gateway_error = _chained_error(error, GatewayModelError)
            if gateway_error is not None:
                return _gateway_worker_failure(gateway_error), False
            finalizer_error = _chained_error(error, ResultFinalizerFailure)
            if finalizer_error is not None:
                if finalizer_error.code == "output_limit_exceeded":
                    return _failure(
                        "worker_output_limit_exceeded",
                        "Worker result finalizer reached its output token limit",
                        True,
                    ), False
                if finalizer_error.code == "input_too_large":
                    return _failure(
                        "worker_result_too_large",
                        "Worker result finalization input exceeds its limit",
                        False,
                    ), False
                return _failure(
                    "worker_result_finalizer_failed",
                    f"Worker result finalizer failed ({finalizer_error.code})",
                    True,
                ), False
            return _failure(
                "worker_result_finalizer_failed",
                "Worker result finalizer failed (runtime_failed)",
                True,
            ), False

    async def _run_result_finalizer(
        self,
        *,
        request: StageContentRequest,
        candidate: str,
        invocation_id: str,
        observed_refs: tuple[ArtifactRef, ...],
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        candidate_bytes = len(candidate.encode("utf-8"))
        if candidate_bytes > get_shared_config().worker_result_max_bytes:
            return _failure(
                "worker_result_too_large", "Worker terminal result exceeds its limit", False
            ), False
        if is_blank(candidate):
            return _failure(
                "worker_result_invalid", "Worker returned an invalid terminal result", True
            ), False
        if _exposes_private_value(candidate, self._context):
            return _failure(
                "unsafe_worker_result", "Worker returned content blocked by Runtime policy", False
            ), False
        finalizer = self.finalizer
        if finalizer is None:
            return _failure(
                "worker_draining", "Worker is no longer accepting A2A work", True
            ), False
        with model_cost_phase("finalization"):
            structured_candidate = await finalizer.run(
                subtask_id=request.subtask_id,
                result_text=candidate,
                invocation_id=invocation_id,
            )
        result, exportable = self.build_result(
            request,
            structured_candidate,
            observed_refs,
        )
        if isinstance(result, WorkerResult) and result.result != candidate:
            return _failure(
                "worker_result_finalizer_mismatch",
                "Worker result finalizer changed the terminal result",
                True,
            ), False
        return result, exportable

    async def summarize(
        self,
        *,
        request: StageContentRequest,
        invocation_id: str,
        transcript: TranscriptRecorder | None,
        observed_refs: tuple[ArtifactRef, ...],
        secrets: tuple[str, ...],
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        config = self._context.summarizer
        if config is None or transcript is None:
            raise RuntimeError("Worker summarization was requested while disabled")
        usage = SummarizerUsage()
        retryable = True
        summarizer: TerminalSummarizer | None = None
        summary_model: BaseLlm | None = None
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
                include_reasoning=config.include_reasoning,
                reasoning_truncated=transcript.reasoning_truncated,
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
            summarizer = TerminalSummarizer(
                model=summary_model,
                policy=config.model_policy,
                instructions=config.instructions.text if config.instructions is not None else None,
                include_reasoning=config.include_reasoning,
                instrumentation=self._instrumentation,
                check_admission=self._admit_semantic_work,
            )
            with model_cost_phase("summarization"):
                candidate = await summarizer.run(prompt=prompt, invocation_id=invocation_id)
            usage = summarizer.usage
            result, exportable = self.build_result(request, candidate, observed_refs)
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
        except TimeoutError:
            usage = summarizer.usage if summarizer is not None else usage
            if self._invocation.deadline is not None and self._invocation.deadline.passed():
                await _await_safely(
                    self._complete_summarizer_attempt(
                        invocation_id=invocation_id,
                        succeeded=False,
                        usage=usage,
                        failure_code="cancelled",
                    )
                )
                raise
            failure_code = "runtime_failed"
        except SummarizerFailure as error:
            usage = summarizer.usage if summarizer is not None else usage
            failure_code = error.code
            retryable = error.retryable
        except Exception:
            usage = summarizer.usage if summarizer is not None else usage
            failure_code = "runtime_failed"
        finally:
            if (
                isinstance(summary_model, OpenAICompatibleGatewayLlm)
                and summary_model is not self._model
            ):
                await summary_model.close()

        await self._complete_summarizer_attempt(
            invocation_id=invocation_id,
            succeeded=False,
            usage=usage,
            failure_code=failure_code,
        )
        return _summarizer_failure(failure_code, retryable=retryable), False

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

    def build_result(
        self,
        request: StageContentRequest,
        candidate: str | None,
        observed_refs: tuple[ArtifactRef, ...],
        *,
        verified_finalizer_refs: tuple[ArtifactRef, ...] = (),
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        """Model-authored JSON boundary shared by finalizer and summarizer."""
        fields = _decode_model_result(candidate, expected_subtask_id=request.subtask_id)
        if isinstance(fields, WorkerFailure):
            return fields, False
        return self.assemble_result(
            request, fields, observed_refs, verified_finalizer_refs=verified_finalizer_refs
        )

    def assemble_result(
        self,
        request: StageContentRequest,
        validated_fields: WorkerModelResult,
        observed_refs: tuple[ArtifactRef, ...],
        *,
        verified_finalizer_refs: tuple[ArtifactRef, ...] = (),
    ) -> tuple[WorkerResult | WorkerFailure, bool]:
        """Project freshly validated fields through Runtime-owned result policy.

        Both private callers validate immediately before this synchronous call;
        no model/strategy-supplied identity, observations or slots are inherited.
        """
        if _exposes_private_value(validated_fields.result, self._context):
            return _failure(
                "unsafe_worker_result", "Worker returned content blocked by Runtime policy", False
            ), False
        observed = {
            (ref.namespace, ref.name): ref
            for ref in _latest_observed_exact_refs((*observed_refs, *verified_finalizer_refs))
        }
        artifacts: dict[str, ArtifactRef] = {}
        reserved_slots = self._reserved_slots
        for slot, binding in request.result_artifacts.items():
            if slot in reserved_slots:
                continue
            if is_reserved_result_binding(binding.namespace, binding.name):
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
            result=validated_fields.result,
            observations=WorkerObservations(profile="lean@1", tools={}, truncated=False),
            artifacts=artifacts,
            summarized=False,
        )
        if len(result.model_dump_json(by_alias=True).encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
            return _failure(
                "worker_result_too_large", "Worker result exceeds its limit", False
            ), False
        return result, True

    def observe_artifacts(self, tool: Any, cursor: int) -> None:
        observations = getattr(tool, "observed_exact_refs_since", None)
        if not callable(observations):
            return
        try:
            refs = observations(cursor)
            for ref in refs:
                exact = ref.require_exact()
                if not is_reserved_memory_binding(exact.namespace, exact.name):
                    self.refs.append(exact)
        except Exception:
            # A provenance adapter defect must not turn a completed tool side
            # effect into an invented result. Omitting the ref fails the Stage
            # result contract closed at the Planner boundary.
            return


def _decode_model_result(
    candidate: str | None, *, expected_subtask_id: str
) -> WorkerModelResult | WorkerFailure:
    """Decode actual model output without changing validation/error precedence."""
    if candidate is None:
        return _failure("worker_result_missing", "Worker returned no structured result", True)
    if len(candidate.encode("utf-8")) > MAX_STAGE_RESULT_JSON_BYTES:
        return _failure(
            "worker_result_too_large", "Worker structured result exceeds its limit", False
        )
    try:
        raw_candidate = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return _failure(
            "worker_result_invalid", "Worker returned an invalid structured result", True
        )
    return _validate_result_fields(raw_candidate, expected_subtask_id=expected_subtask_id)


def _validate_result_fields(
    raw_candidate: Any, *, expected_subtask_id: str
) -> WorkerModelResult | WorkerFailure:
    """Validate fresh fields from decoded JSON or a trusted completion decision."""
    if (
        isinstance(raw_candidate, dict)
        and isinstance(raw_candidate.get("result"), str)
        and len(raw_candidate["result"].encode("utf-8"))
        > get_shared_config().worker_result_max_bytes
    ):
        return _failure(
            "worker_result_too_large", "Worker structured result exceeds its limit", False
        )
    try:
        fields = WorkerModelResult.model_validate(raw_candidate)
    except ValidationError:
        return _failure(
            "worker_result_invalid", "Worker returned an invalid structured result", True
        )
    if fields.subtask_id != expected_subtask_id:
        return _failure(
            "worker_result_subtask_mismatch",
            "Worker result does not match the requested subtask",
            True,
        )
    return fields


def _latest_observed_exact_refs(refs: tuple[ArtifactRef, ...]) -> list[ArtifactRef]:
    latest: dict[tuple[str, str], ArtifactRef] = {}
    for ref in refs:
        exact = ref.require_exact()
        if is_reserved_memory_binding(exact.namespace, exact.name):
            continue
        latest[(exact.namespace, exact.name)] = exact
    return [latest[key] for key in sorted(latest)]


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
    for name in sorted(raw_tools)[: get_shared_config().worker_observation_max_tools]:
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
        **({"workspace": workspace} if workspace is not None else {}),
        truncated=(
            bool(metrics.get("truncated"))
            or workspace_truncated
            or projection_failed
            or len(raw_tools) > get_shared_config().worker_observation_max_tools
        ),
    )


async def _await_safely(awaitable: Any) -> Any:
    task = asyncio.create_task(awaitable, name="worker-summarizer-state-finalize")
    if await join_owned_task(task) or task.cancelled():
        with contextlib.suppress(Exception, asyncio.CancelledError):
            task.result()
        return None
    return task.result()
