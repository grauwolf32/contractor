"""A narrow model tool for publishing one canonical Audit check result package."""

import time
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NotRequired, TypedDict

from google.adk.tools.tool_context import ToolContext

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.audit_packages import (
    IDENTIFIER,
    MAX_EVIDENCE,
    MAX_SUMMARY_BYTES,
    PACKAGE_MEDIA_TYPE,
    _build_result_package,
    _decode_execution_manifest,
    _decode_task_input,
    _digest,
    _match_trusted_inputs,
    _validate_values,
)
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.artifact_visibility import (
    artifact_observation_cursor,
    clear_artifact_observations,
    model_visible_exact_refs,
    model_visible_observations_since,
)
from contractor_runtime.toolsets.run_artifacts import (
    ArtifactClientFactory,
    ToolMetrics,
    gateway_secrets,
)
from contractor_runtime.workspace import AllocationWorkspace

ASSESSMENTS = frozenset(
    {
        "satisfied",
        "violated",
        "supported",
        "refuted",
        "blocked",
        "inconclusive",
        "not-tested",
    }
)


class EvidenceArgument(TypedDict):
    kind: str
    summary: str


class BatchResultArgument(TypedDict):
    assessment: str
    summary: str
    completed: list[str]
    gaps: list[str]
    evidence: NotRequired[list[EvidenceArgument]]
    proposal_keys: NotRequired[list[str]]


class AuditResultsToolsetFactory:
    """Build the selected result publisher without exposing Audit authority."""

    ref = "audit-results@1"
    exported_tools = frozenset({"read_audit_task", "submit_check_result"})
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles, project_workspace
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("audit-results@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        secrets = gateway_secrets(runtime_settings)
        builders: dict[str, Callable[[], Any]] = {
            "read_audit_task": lambda: ReadAuditTaskTool(client, metrics, secrets),
            "submit_check_result": lambda: SubmitCheckResultTool(
                client, metrics, secrets, namespace
            ),
        }
        return {name: builders[name]() for name in selected}


class ReadAuditTaskTool:
    name = "read_audit_task"
    description = """Read the immutable Audit task set assigned to this Worker.

    Validates each exact task package against the ordered execution manifest.
    Use the returned task order and requested coverage when submitting results.

    Returns:
        batchSize, ordered tasks and taskPackageIds, the exact taskArtifact and
        executionManifestDigest. A single-task result also includes task and
        taskPackageId. Raw package bytes are not returned.
    """

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return model_visible_exact_refs(getattr(self._client, "known_exact_refs", ()))

    @property
    def artifact_observation_cursor(self) -> int:
        return artifact_observation_cursor(self._client)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return model_visible_observations_since(self._client, cursor)

    def clear_artifact_observations(self) -> None:
        clear_artifact_observations(self._client)

    async def close(self) -> None:
        self._secrets = ()

    async def __call__(self) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            task_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="task")
            )
            execution_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="execution_manifest")
            )
            task_records = _decode_task_input(task_value.data, task_value.media_type)
            execution = _decode_execution_manifest(execution_value.data, execution_value.media_type)
            _match_trusted_inputs(task_records, execution)
            tasks = [record[0] for record in task_records]
            result = {
                "batchSize": len(tasks),
                "tasks": tasks,
                "taskPackageIds": [record[1] for record in task_records],
                "taskArtifact": task_value.artifact.model_dump(by_alias=True),
                "executionManifestDigest": _digest(execution_value.data),
            }
            if len(tasks) == 1:
                result["task"] = tasks[0]
                result["taskPackageId"] = task_records[0][1]
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                result={
                    "artifact": result["taskArtifact"],
                    "itemCount": len(tasks),
                    "itemKeys": [task["item_key"] for task in tasks],
                },
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            raise


class SubmitCheckResultTool:
    name = "submit_check_result"
    description = """Publish one complete result set for the assigned Audit checks.

    Call read_audit_task first. For one task, provide assessment, summary, completed
    and gaps. For a batch, provide results in task order and omit all individual result fields.
    Task identity and requested coverage come from the validated assignment.

    Args:
        assessment: satisfied, violated, supported, refuted, blocked, inconclusive
            or not-tested; required in single-task mode.
        summary: Non-empty explanation of the outcome; required in single-task mode.
        completed: Sorted unique requested coverage identifiers completed for the
            task; required in single-task mode, and may be empty.
        gaps: Sorted unique gap identifiers; required in single-task mode, and may
            be empty.
        evidence: Optional evidence objects containing only kind and summary.
        proposal_keys: Optional sorted unique client keys of finding proposals.
        results: One object per assigned task, in order, with assessment, summary,
            completed and gaps, plus optional evidence and proposal_keys. Use this
            for batch mode without the individual result arguments.

    Returns:
        Saved result package metadata with its exact artifact revision, mediaType
        and size.
    """

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        namespace: str,
    ) -> None:
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self._namespace = namespace
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return model_visible_exact_refs(getattr(self._client, "known_exact_refs", ()))

    @property
    def artifact_observation_cursor(self) -> int:
        return artifact_observation_cursor(self._client)

    def observed_exact_refs_since(self, cursor: int) -> tuple[ArtifactRef, ...]:
        return model_visible_observations_since(self._client, cursor)

    def clear_artifact_observations(self) -> None:
        clear_artifact_observations(self._client)

    async def close(self) -> None:
        self._secrets = ()

    async def __call__(
        self,
        tool_context: ToolContext,
        assessment: str | None = None,
        summary: str | None = None,
        completed: list[str] | None = None,
        gaps: list[str] | None = None,
        evidence: list[EvidenceArgument] | None = None,
        proposal_keys: list[str] | None = None,
        results: list[BatchResultArgument] | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        metric_arguments = _result_metric_arguments(
            assessment, summary, completed, gaps, evidence, proposal_keys, results
        )
        try:
            task_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="task")
            )
            execution_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="execution_manifest")
            )
            task_records = _decode_task_input(task_value.data, task_value.media_type)
            execution = _decode_execution_manifest(execution_value.data, execution_value.media_type)
            _match_trusted_inputs(task_records, execution)
            normalized_results = _normalize_results(
                [record[0] for record in task_records],
                assessment=assessment,
                summary=summary,
                completed=completed,
                gaps=gaps,
                evidence=evidence,
                proposal_keys=proposal_keys,
                results=results,
                invocation_id=tool_context.invocation_id,
            )
            payload = _build_result_package(
                tasks=[record[0] for record in task_records],
                execution_bytes=execution_value.data,
                results=normalized_results,
            )
            written = await self._client.write_artifact(
                ArtifactRef(namespace=self._namespace, name="result"),
                data=payload,
                media_type=PACKAGE_MEDIA_TYPE,
                expected_revision=None,
            )
            result = written.model_dump(by_alias=True)
            self._metrics.record_tool_call(
                self.name,
                arguments=metric_arguments,
                result={
                    "artifact": result["artifact"],
                    "mediaType": result["mediaType"],
                    "size": result["size"],
                },
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments=metric_arguments,
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            raise


def _result_metric_arguments(
    assessment: str | None,
    summary: str | None,
    completed: list[str] | None,
    gaps: list[str] | None,
    evidence: list[EvidenceArgument] | None,
    proposal_keys: list[str] | None,
    results: list[BatchResultArgument] | None,
) -> dict[str, Any]:
    if isinstance(results, list):
        content_bytes = 0
        completed_count = 0
        gap_count = 0
        evidence_count = 0
        proposal_count = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            value = item.get("summary")
            content_bytes += len(value.encode("utf-8")) if isinstance(value, str) else 0
            for field, target in (
                ("completed", "completed"),
                ("gaps", "gaps"),
                ("evidence", "evidence"),
                ("proposal_keys", "proposals"),
            ):
                values = item.get(field)
                count = len(values) if isinstance(values, list) else 0
                if target == "completed":
                    completed_count += count
                elif target == "gaps":
                    gap_count += count
                elif target == "evidence":
                    evidence_count += count
                else:
                    proposal_count += count
        return {
            "mode": "batch",
            "item_count": len(results),
            "content_bytes": content_bytes,
            "completed_count": completed_count,
            "gap_count": gap_count,
            "evidence_count": evidence_count,
            "proposal_count": proposal_count,
        }
    return {
        "assessment": assessment,
        "content_bytes": len(summary.encode("utf-8")) if isinstance(summary, str) else 0,
        "completed_count": len(completed) if isinstance(completed, list) else 0,
        "gap_count": len(gaps) if isinstance(gaps, list) else 0,
        "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "proposal_count": len(proposal_keys) if isinstance(proposal_keys, list) else 0,
    }


def _normalize_results(
    tasks: list[dict[str, Any]],
    *,
    assessment: str | None,
    summary: str | None,
    completed: list[str] | None,
    gaps: list[str] | None,
    evidence: list[EvidenceArgument] | None,
    proposal_keys: list[str] | None,
    results: list[BatchResultArgument] | None,
    invocation_id: str,
) -> list[dict[str, Any]]:
    if results is None:
        if len(tasks) != 1:
            raise ValueError("a batch requires one complete ordered results array")
        if assessment is None or summary is None or completed is None or gaps is None:
            raise ValueError("single-result fields are incomplete")
        source: list[dict[str, Any]] = [
            {
                "assessment": assessment,
                "summary": summary,
                "completed": completed,
                "gaps": gaps,
                "evidence": [] if evidence is None else evidence,
                "proposal_keys": [] if proposal_keys is None else proposal_keys,
            }
        ]
    else:
        if any(
            value is not None
            for value in (assessment, summary, completed, gaps, evidence, proposal_keys)
        ):
            raise ValueError("batch results cannot be combined with single-result fields")
        if not isinstance(results, list) or len(results) != len(tasks):
            raise ValueError("batch results must exactly match the trusted task order")
        source = [dict(item) for item in results]

    normalized: list[dict[str, Any]] = []
    required = {"assessment", "summary", "completed", "gaps"}
    allowed = required | {"evidence", "proposal_keys"}
    for index, candidate in enumerate(source):
        if (
            not isinstance(candidate, dict)
            or not required.issubset(candidate)
            or not set(candidate).issubset(allowed)
        ):
            raise ValueError(f"result {index} has invalid fields")
        candidate_evidence = _validate_arguments(
            candidate["assessment"],
            candidate["summary"],
            candidate["completed"],
            candidate["gaps"],
            candidate.get("evidence"),
        )
        candidate_proposals = _validate_proposal_keys(candidate.get("proposal_keys"))
        normalized.append(
            {
                "assessment": candidate["assessment"],
                "summary": candidate["summary"],
                "completed": candidate["completed"],
                "gaps": candidate["gaps"],
                "evidence": candidate_evidence,
                "proposals": [
                    {"invocation_id": invocation_id, "client_key": key}
                    for key in candidate_proposals
                ],
            }
        )
    if sum(len(item["evidence"]) for item in normalized) > MAX_EVIDENCE:
        raise ValueError("batch evidence exceeds its aggregate bound")
    return normalized


def _validate_arguments(
    assessment: str,
    summary: str,
    completed: list[str],
    gaps: list[str],
    evidence: list[EvidenceArgument] | None,
) -> list[dict[str, str]]:
    if assessment not in ASSESSMENTS:
        raise ValueError("assessment is not supported")
    if (
        not isinstance(summary, str)
        or not summary.strip()
        or len(summary.encode("utf-8")) > MAX_SUMMARY_BYTES
    ):
        raise ValueError("summary must be bounded non-empty UTF-8 text")
    _validate_values("completed", completed)
    _validate_values("gaps", gaps)
    normalized = [] if evidence is None else evidence
    if not isinstance(normalized, list) or len(normalized) > MAX_EVIDENCE:
        raise ValueError("evidence exceeds its bound")
    result: list[dict[str, str]] = []
    for item in normalized:
        if not isinstance(item, dict) or set(item) != {"kind", "summary"}:
            raise ValueError("each evidence item requires only kind and summary")
        kind, item_summary = item.get("kind"), item.get("summary")
        if not isinstance(kind, str) or IDENTIFIER.fullmatch(kind) is None:
            raise ValueError("evidence kind is invalid")
        if (
            not isinstance(item_summary, str)
            or not item_summary.strip()
            or len(item_summary.encode("utf-8")) > MAX_SUMMARY_BYTES
        ):
            raise ValueError("evidence summary is invalid")
        result.append({"kind": kind, "summary": item_summary})
    return result


def _validate_proposal_keys(value: list[str] | None) -> list[str]:
    result = [] if value is None else value
    _validate_values("proposal_keys", result)
    if len(result) > 128:
        raise ValueError("proposal_keys exceeds its bound")
    return result


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
