"""Model-free Worker using exact inputs and durable create-before-launch receipts."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, get_type_hints

import jcs
from pydantic import ConfigDict, ValidationError, create_model

from contractor_runtime.a2a_server import (
    agent_card_dict,
    build_agent_card,
    build_worker_a2a_application,
)
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    StageContentRequest,
    WorkerCompletion,
    WorkerFailure,
    WorkerObservations,
    WorkerResult,
)
from contractor_runtime.toolsets.common.input_errors import ToolInputError

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext
    from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory

MAX_INPUT_BYTES = 64 * 1024
MAX_REPORT_BYTES = 512 * 1024
MAX_ADDITIONAL_ARTIFACTS = 16
MAX_ADDITIONAL_REF_BYTES = 8192
MAX_RECEIPT_BYTES = 16 * 1024
RECEIPT_PREFIX = "tool-invocation."
FAILURE_CODES = frozenset(
    {
        "tool_input_invalid",
        "tool_input_conflict",
        "tool_execution_failed",
        "tool_output_invalid",
        "tool_timeout",
        "tool_report_failed",
        "tool_outcome_unknown",
    }
)


class ToolWorkerError(Exception):
    """Fixed, content-free execution diagnostic."""


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(jcs.canonicalize(value)).hexdigest()


def _metrics(tool: str, calls: int = 0, failed: bool = False) -> dict:
    return {
        **dict.fromkeys(
            (
                "modelCalls",
                "modelErrors",
                "inputTokens",
                "outputTokens",
                "totalTokens",
                "cachedInputTokens",
                "tokenUsageUnavailable",
            ),
            0,
        ),
        "toolCalls": calls,
        "toolErrors": int(failed and calls > 0),
        "latestPromptTokens": None,
        "tools": {tool: {"calls": calls, "failures": int(failed)}} if calls else {},
        "truncated": False,
    }


class ToolWorkerRuntimeFactory:
    ref = "tool@1"
    supports_agent_skills = False
    supports_worker_completion = True

    def __init__(self, artifact_client_factory: ArtifactClientFactory | None = None):
        self._clients = artifact_client_factory

    async def probe(self) -> bool:
        return True

    async def create(self, context: WorkerBuildContext) -> ToolWorkerRuntime:
        if self._clients is None:
            raise ValueError("tool@1 requires an Artifact client")
        return ToolWorkerRuntime(
            context, self._clients(context.allocation_id, context.runtime_settings)
        )


class ToolWorkerRuntime:
    def __init__(self, context: WorkerBuildContext, client: ArtifactClient):
        execution = context.execution
        if (
            execution is None
            or context.template_ref is None
            or set(context.tools) != {execution.tool}
        ):
            raise ValueError("tool@1 requires pinned execution and one selected callable")
        if (
            context.model_policy is not None
            or context.summarizer is not None
            or context.project_workspace is not None
            or context.completion_contract is not None
            or context.resolved_skills
        ):
            raise ValueError("tool@1 received incompatible Worker configuration")
        self._context = context
        self._execution = execution
        self._client = client
        self._tool = context.tools[execution.tool]
        if not callable(self._tool):
            raise ValueError("tool@1 selected tool is not callable")
        signature = inspect.signature(self._tool)
        hints = get_type_hints(
            self._tool if inspect.isfunction(self._tool) else self._tool.__call__
        )
        fields = {}
        for name, parameter in signature.parameters.items():
            if (
                parameter.kind not in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
                or name not in hints
            ):
                raise ValueError("tool@1 requires annotated named arguments")
            fields[name] = (
                hints[name],
                ... if parameter.default is parameter.empty else parameter.default,
            )
        self._input = create_model(
            "ToolWorkerInput", __config__=ConfigDict(strict=True, extra="forbid"), **fields
        )
        self._state = context.state
        self.allocation_id = context.allocation_id
        self._active: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._accepting = True
        self._cache: OrderedDict[str, tuple[str, WorkerCompletion]] = OrderedDict()
        card = build_agent_card(
            allocation_id=self.allocation_id,
            endpoint=f"{context.a2a_base_url.rstrip('/')}/private/v1/allocations/{self.allocation_id}/a2a",
            logical_agent_name=context.logical_agent_name,
            description=context.description,
            version=context.card_version,
        )
        self.agent_card = agent_card_dict(card)
        self.a2a_application = build_worker_a2a_application(self, card)

    async def agent_state_snapshot(self):
        return await self._state.agent_state_snapshot()

    async def failure_completion(self, code: str, message: str, *, retryable: bool = False):
        self._state.metrics.record_outcome("failed")
        snapshot = await self._state.sync_metrics()
        return WorkerCompletion(
            apiVersion=API_VERSION,
            invocationId=f"tool-rejected-{uuid.uuid4().hex}",
            stateRevision=snapshot["stateRevision"],
            failure=WorkerFailure(code=code, message=message, retryable=retryable),
        )

    def _resolve(self, request: StageContentRequest):
        arguments = {}
        for name, binding in self._execution.arguments.items():
            if binding.source == "parameter":
                arguments[name] = request.parameters[binding.name]
            elif binding.source == "artifact":
                arguments[name] = request.artifacts[binding.name].model_dump(by_alias=True)
            else:
                arguments[name] = binding.value
        arguments = self._input.model_validate(arguments).model_dump()
        target = request.result_artifacts[self._execution.result_artifact]
        if target.namespace != self._context.namespace or target.name.startswith(RECEIPT_PREFIX):
            raise ValueError("invalid tool output namespace or reserved binding")
        identity = {
            "template": self._context.template_ref.model_dump(by_alias=True),
            "arguments": arguments,
            "output": target.model_dump(by_alias=True),
        }
        additional = {
            slot: ref
            for slot, ref in request.result_artifacts.items()
            if slot != self._execution.result_artifact
        }
        if len(additional) > MAX_ADDITIONAL_ARTIFACTS:
            raise ValueError("too many tool output bindings")
        names = {target.name}
        for ref in additional.values():
            if (
                ref.namespace != self._context.namespace
                or ref.name.startswith((RECEIPT_PREFIX, "memory."))
                or ref.name in names
            ):
                raise ValueError("invalid additional tool output binding")
            names.add(ref.name)
        if additional:
            # Preserve pre-existing single-report invocation digests.
            identity["additionalOutputs"] = {
                slot: ref.model_dump(by_alias=True) for slot, ref in additional.items()
            }
        canonical = jcs.canonicalize(identity)
        if len(canonical) > MAX_INPUT_BYTES:
            raise ValueError("tool input exceeds its bound")
        key = _digest(
            {
                "runId": self._context.run_id,
                "stageExecutionId": self._context.stage_execution_id,
                "logicalAgentName": self._context.logical_agent_name,
                "subtaskId": request.subtask_id,
            }
        ).removeprefix("sha256:")
        return arguments, target, key, "sha256:" + hashlib.sha256(canonical).hexdigest()

    def _additional_artifacts(self, value, request, *, complete):
        """Accept only exact refs for the declared, Worker-owned output bindings."""
        expected = set(request.result_artifacts) - {self._execution.result_artifact}
        if (
            not isinstance(value, dict)
            or len(value) > MAX_ADDITIONAL_ARTIFACTS
            or set(value) - expected
            or (complete and set(value) != expected)
            or len(_json(value)) > MAX_ADDITIONAL_REF_BYTES
        ):
            raise ValueError("invalid additional tool artifacts")
        refs = {}
        for slot, raw in value.items():
            if not isinstance(raw, dict):
                raise ValueError("invalid additional artifact reference")
            ref = ArtifactRef.model_validate(raw).require_exact()
            binding = request.result_artifacts[slot]
            if ref.namespace != binding.namespace or ref.name != binding.name:
                raise ValueError("additional artifact does not match its binding")
            refs[slot] = ref
        return refs

    async def _verify_additional_artifacts(self, refs):
        # Tools have their own Artifact clients, so a ref in a JSON observation
        # is not proof of publication. Exact reads verify existence and revision.
        for ref in refs.values():
            await self._client.read_artifact(ref, max_bytes=MAX_REPORT_BYTES)

    async def invoke(self, request: StageContentRequest) -> WorkerCompletion:
        if not self._accepting:
            return await self.failure_completion(
                "worker_draining", "Worker is no longer accepting work"
            )
        if self._lock.locked():
            return await self.failure_completion(
                "worker_busy", "Worker has an active tool invocation"
            )
        try:
            arguments, target, key, digest = self._resolve(request)
        except (KeyError, ValueError, TypeError, ValidationError):
            return await self.failure_completion(
                "tool_input_invalid", "Tool input binding or type is invalid"
            )
        if key in self._cache:
            previous, completion = self._cache[key]
            if previous != digest:
                return await self.failure_completion(
                    "tool_input_conflict", "Tool invocation input changed"
                )
            return completion.model_copy(deep=True)
        async with self._lock:
            self._active = asyncio.create_task(self._run(request, arguments, target, key, digest))
            try:
                result = await self._active
                self._cache[key] = (digest, result.model_copy(deep=True))
                if len(self._cache) > 128:
                    self._cache.popitem(last=False)
                return result
            finally:
                self._active = None

    async def _run(self, request, arguments, target, key, digest):
        invocation_id = "tool-" + key
        calls, error, report = 0, None, None
        artifacts = {}
        cancelled, truncated, tool_failed = False, False, False
        receipt = ArtifactRef(namespace=self._context.namespace, name=RECEIPT_PREFIX + key)
        owned = None
        beginning = asyncio.create_task(
            self._state.begin_invocation(
                invocation_id=invocation_id,
                subtask_id=request.subtask_id,
                metrics=_metrics(self._execution.tool),
            )
        )
        try:
            await asyncio.shield(beginning)
            try:
                value = await self._client.read_artifact(receipt, max_bytes=MAX_RECEIPT_BYTES)
            except ArtifactAPIError as failure:
                if failure.status_code != 404:
                    raise ToolWorkerError("tool_outcome_unknown") from None
                value = None
            if value is not None:
                saved = json.loads(value.data)
                if (
                    not isinstance(saved, dict)
                    or type(saved.get("schemaVersion")) is not int
                    or saved.get("schemaVersion") != 1
                    or set(saved)
                    - {
                        "schemaVersion",
                        "inputDigest",
                        "phase",
                        "report",
                        "errorCode",
                        "truncated",
                        "artifacts",
                    }
                    or type(saved.get("truncated", False)) is not bool
                ):
                    raise ToolWorkerError("tool_outcome_unknown")
                if saved.get("inputDigest") != digest:
                    raise ToolWorkerError("tool_input_conflict")
                if saved.get("phase") not in {"completed", "failed"}:
                    raise ToolWorkerError("tool_outcome_unknown")
                artifacts = self._additional_artifacts(
                    saved.get("artifacts", {}), request, complete=saved["phase"] == "completed"
                )
                if saved.get("report") is not None:
                    report = ArtifactRef.model_validate(saved["report"]).require_exact()
                    if report.namespace != target.namespace or report.name != target.name:
                        raise ToolWorkerError("tool_outcome_unknown")
                if saved["phase"] == "failed":
                    code = saved.get("errorCode")
                    raise ToolWorkerError(code if code in FAILURE_CODES else "tool_outcome_unknown")
                if report is None:
                    raise ToolWorkerError("tool_outcome_unknown")
                await self._verify_additional_artifacts(artifacts)
                truncated = saved.get("truncated", False)
            else:
                owned = await self._client.write_artifact(
                    receipt,
                    data=_json({"schemaVersion": 1, "inputDigest": digest, "phase": "started"}),
                    media_type="application/json",
                    expected_revision=None,
                )
                calls = 1
                try:
                    async with asyncio.timeout(self._execution.timeout_seconds):
                        observation = await self._tool(**arguments)
                except TimeoutError:
                    tool_failed = True
                    raise ToolWorkerError("tool_timeout") from None
                except ToolInputError:
                    tool_failed = True
                    raise ToolWorkerError("tool_input_invalid") from None
                except Exception:
                    tool_failed = True
                    raise ToolWorkerError("tool_execution_failed") from None
                if (
                    not isinstance(observation, dict)
                    or not isinstance(observation.get("status"), str)
                    or observation["status"] not in {"completed", "failed"}
                ):
                    tool_failed = True
                    raise ToolWorkerError("tool_output_invalid")
                tool_failed = observation["status"] == "failed"
                truncated = any(
                    value is True
                    for name, value in observation.items()
                    if name.endswith("Truncated")
                )
                try:
                    artifacts = self._additional_artifacts(
                        observation.get("artifacts", {}),
                        request,
                        complete=observation["status"] == "completed",
                    )
                    payload = _json(
                        {
                            "schemaVersion": 1,
                            "tool": self._execution.tool,
                            "inputDigest": digest,
                            "inputArtifacts": {
                                name: ref.model_dump(by_alias=True)
                                for name, ref in request.artifacts.items()
                            },
                            "observation": observation,
                        }
                    )
                    if len(payload) > MAX_REPORT_BYTES:
                        raise ValueError
                except (ValueError, TypeError, RecursionError):
                    tool_failed = True
                    raise ToolWorkerError("tool_output_invalid") from None
                try:
                    await self._verify_additional_artifacts(artifacts)
                    published = await self._client.write_artifact(
                        target, data=payload, media_type="application/json", expected_revision=None
                    )
                    report = published.artifact
                except Exception:
                    raise ToolWorkerError("tool_report_failed") from None
                if observation["status"] == "failed":
                    raise ToolWorkerError("tool_execution_failed")
        except asyncio.CancelledError:
            error = "tool_cancelled"
            cancelled = True
            tool_failed = calls > 0
        except ToolWorkerError as failure:
            error = str(failure)
        except Exception:
            error = "tool_outcome_unknown"
        finally:
            # State admission may have been cancelled after its mutation won.
            # Join it before the matching terminal revision is published.
            await beginning
            # Cancellation leaves the durable started receipt intact. It never
            # authorizes another scan after lease fencing or a lost response.
            if owned is not None and error != "tool_cancelled":
                terminal = {
                    "schemaVersion": 1,
                    "inputDigest": digest,
                    "phase": "failed" if error else "completed",
                    "truncated": truncated,
                }
                if error:
                    terminal["errorCode"] = error
                if report is not None:
                    terminal["report"] = report.model_dump(by_alias=True)
                if artifacts:
                    terminal["artifacts"] = {
                        slot: ref.model_dump(by_alias=True) for slot, ref in artifacts.items()
                    }
                try:
                    await self._client.write_artifact(
                        receipt,
                        data=_json(terminal),
                        media_type="application/json",
                        expected_revision=owned.artifact.revision,
                    )
                except asyncio.CancelledError:
                    error, cancelled = "tool_cancelled", True
                except Exception:
                    error = "tool_outcome_unknown"
            if error == "tool_report_failed":
                self._state.metrics.record_tool_report_failure()
            self._state.metrics.record_outcome("failed" if error else "succeeded")
            finishing = asyncio.create_task(
                self._state.complete_invocation(
                    invocation_id=invocation_id,
                    phase="cancelled"
                    if error == "tool_cancelled"
                    else "failed"
                    if error
                    else "succeeded",
                    metrics=_metrics(self._execution.tool, calls, tool_failed),
                )
            )
            try:
                snapshot = await asyncio.shield(finishing)
            except asyncio.CancelledError:
                await finishing
                raise
        if cancelled:
            raise asyncio.CancelledError
        if error:
            return WorkerCompletion(
                apiVersion=API_VERSION,
                invocationId=invocation_id,
                stateRevision=snapshot["stateRevision"],
                failure=WorkerFailure(
                    code=error,
                    message="Tool execution did not complete successfully",
                    retryable=False,
                ),
            )
        return WorkerCompletion(
            apiVersion=API_VERSION,
            invocationId=invocation_id,
            stateRevision=snapshot["stateRevision"],
            result=WorkerResult(
                subtaskId=request.subtask_id,
                result="Tool completed; inspect the scan report for evidence and coverage limits.",
                observations=WorkerObservations(
                    profile="lean@1",
                    tools=_metrics(self._execution.tool, calls)["tools"],
                    truncated=truncated,
                ),
                artifacts={self._execution.result_artifact: report, **artifacts},
                summarized=False,
            ),
        )

    def cancel_active(self) -> None:
        if self._active is not None:
            self._active.cancel()

    async def finalize(self, deadline: datetime) -> None:
        self._accepting = False
        task = self._active
        if task is not None:
            task.cancel()
            async with asyncio.timeout(max(0.001, (deadline - datetime.now(UTC)).total_seconds())):
                await asyncio.gather(task, return_exceptions=True)

    async def abort(self, deadline: datetime) -> None:
        await self.finalize(deadline)
