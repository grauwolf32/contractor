"""A narrow model tool for publishing one canonical Audit check result package."""

import hashlib
import io
import json
import re
import stat
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NotRequired, TypedDict

import jcs
from google.adk.tools.tool_context import ToolContext

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
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

PACKAGE_SCHEMA = "contractor.audit.package.v1"
TASK_SCHEMA = "contractor.audit.item-task.v1"
EXECUTION_SCHEMA = "contractor.audit.execution-manifest.v1"
RESULT_SCHEMA = "contractor.audit.check-results.v1"
EVIDENCE_SCHEMA = "contractor.audit.evidence.v1"
PACKAGE_MEDIA_TYPE = "application/zip"
JSON_MEDIA_TYPE = "application/json"
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_MEMBER_BYTES = 16 * 1024 * 1024
MAX_BATCH_ITEMS = 64
MAX_SUMMARY_BYTES = 16 * 1024
MAX_VALUES = 512
MAX_EVIDENCE = 256
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
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


def _validate_values(field: str, values: list[str]) -> None:
    if not isinstance(values, list) or len(values) > MAX_VALUES:
        raise ValueError(f"{field} exceeds its bound")
    if values != sorted(set(values)):
        raise ValueError(f"{field} must be sorted and unique")
    for value in values:
        if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
            raise ValueError(f"{field} contains an invalid value")


def _validate_proposal_keys(value: list[str] | None) -> list[str]:
    result = [] if value is None else value
    _validate_values("proposal_keys", result)
    if len(result) > 128:
        raise ValueError("proposal_keys exceeds its bound")
    return result


def _decode_task_package(payload: bytes, media_type: str) -> tuple[dict[str, Any], str]:
    if media_type != PACKAGE_MEDIA_TYPE or not 0 < len(payload) <= MAX_PACKAGE_BYTES:
        raise ValueError("task input is not an Audit package")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            if sorted(archive.namelist()) != ["manifest.json", "task.json"]:
                raise ValueError("task package members are invalid")
            manifest_bytes = _read_bounded(archive, "manifest.json")
            task_bytes = _read_bounded(archive, "task.json")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("task package is invalid") from error
    manifest = _canonical_object(manifest_bytes, "task package manifest")
    if (
        set(manifest) != {"schema", "package_id", "kind", "members"}
        or manifest.get("schema") != PACKAGE_SCHEMA
        or manifest.get("kind") != "item-task"
        or not isinstance(manifest.get("package_id"), str)
        or IDENTIFIER.fullmatch(manifest["package_id"]) is None
    ):
        raise ValueError("task package manifest is invalid")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != 1 or not isinstance(members[0], dict):
        raise ValueError("task package manifest is invalid")
    member = members[0]
    if (
        member.get("id") != "task-document"
        or member.get("path") != "task.json"
        or member.get("media_type") != JSON_MEDIA_TYPE
        or member.get("size") != len(task_bytes)
        or member.get("digest") != _digest(task_bytes)
    ):
        raise ValueError("task package member is invalid")
    task = _canonical_object(task_bytes, "task document")
    if task.get("schema") != TASK_SCHEMA:
        raise ValueError("task document schema is invalid")
    return task, manifest["package_id"]


def _decode_task_input(payload: bytes, media_type: str) -> list[tuple[dict[str, Any], str, bytes]]:
    try:
        task, package_id = _decode_task_package(payload, media_type)
        return [(task, package_id, payload)]
    except ValueError:
        try:
            return _decode_task_set(payload, media_type)
        except ValueError as batch_error:
            raise ValueError("task input is not a valid Audit task or task set") from batch_error


def _decode_task_set(payload: bytes, media_type: str) -> list[tuple[dict[str, Any], str, bytes]]:
    if media_type != PACKAGE_MEDIA_TYPE or not 0 < len(payload) <= MAX_PACKAGE_BYTES:
        raise ValueError("task set input is not an Audit package")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            manifest_bytes = _read_bounded(archive, "manifest.json")
            manifest = _canonical_object(manifest_bytes, "task set manifest")
            members = manifest.get("members")
            if (
                set(manifest) != {"schema", "package_id", "kind", "members"}
                or manifest.get("schema") != PACKAGE_SCHEMA
                or manifest.get("kind") != "item-task-set"
                or not isinstance(manifest.get("package_id"), str)
                or IDENTIFIER.fullmatch(manifest["package_id"]) is None
                or not isinstance(members, list)
                or not 2 <= len(members) <= MAX_BATCH_ITEMS
            ):
                raise ValueError("task set manifest is invalid")
            expected_paths = ["manifest.json"] + [
                f"tasks/{index:03d}.zip" for index in range(len(members))
            ]
            if sorted(archive.namelist()) != expected_paths:
                raise ValueError("task set package members are invalid")
            result: list[tuple[dict[str, Any], str, bytes]] = []
            for index, member in enumerate(members):
                path = f"tasks/{index:03d}.zip"
                if (
                    not isinstance(member, dict)
                    or set(member) != {"id", "path", "media_type", "size", "digest"}
                    or member.get("id") != f"task-{index:03d}"
                    or member.get("path") != path
                    or member.get("media_type") != PACKAGE_MEDIA_TYPE
                ):
                    raise ValueError("task set member is invalid")
                nested = _read_bounded(archive, path, MAX_PACKAGE_MEMBER_BYTES)
                if member.get("size") != len(nested) or member.get("digest") != _digest(nested):
                    raise ValueError("task set member digest is invalid")
                task, package_id = _decode_task_package(nested, PACKAGE_MEDIA_TYPE)
                result.append((task, package_id, nested))
            return result
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("task set package is invalid") from error


def _decode_execution_manifest(payload: bytes, media_type: str) -> dict[str, Any]:
    if media_type != JSON_MEDIA_TYPE:
        raise ValueError("execution manifest input is not JSON")
    manifest = _canonical_object(payload, "execution manifest")
    items = manifest.get("items")
    if (
        set(manifest) != {"schema", "items"}
        or manifest.get("schema") != EXECUTION_SCHEMA
        or not isinstance(items, list)
        or not 1 <= len(items) <= MAX_BATCH_ITEMS
    ):
        raise ValueError("execution manifest membership is invalid")
    seen_items: set[str] = set()
    seen_packages: set[str] = set()
    for index, item in enumerate(items):
        if (
            not isinstance(item, dict)
            or item.get("ordinal") != index
            or not isinstance(item.get("item_key"), str)
            or IDENTIFIER.fullmatch(item["item_key"]) is None
            or not isinstance(item.get("subject_key"), str)
            or IDENTIFIER.fullmatch(item["subject_key"]) is None
            or not isinstance(item.get("task_package_id"), str)
            or IDENTIFIER.fullmatch(item["task_package_id"]) is None
            or not isinstance(item.get("task_package_digest"), str)
            or not item["task_package_digest"].startswith("sha256:")
        ):
            raise ValueError("execution manifest item is invalid")
        if item["item_key"] in seen_items or item["task_package_id"] in seen_packages:
            raise ValueError("execution manifest identity is duplicated")
        seen_items.add(item["item_key"])
        seen_packages.add(item["task_package_id"])
    return manifest


def _match_trusted_inputs(
    tasks: list[tuple[dict[str, Any], str, bytes]],
    execution: dict[str, Any],
) -> None:
    if len(tasks) != len(execution["items"]):
        raise ValueError("task and execution manifest membership differs")
    for index, (task, task_package_id, task_package) in enumerate(tasks):
        item = execution["items"][index]
        if (
            item.get("item_key") != task.get("item_key")
            or item.get("subject_key") != task.get("subject_key")
            or item.get("task_package_id") != task_package_id
            or item.get("task_package_digest") != _digest(task_package)
        ):
            raise ValueError("task and execution manifest identities differ")
        if not isinstance(task.get("item_key"), str) or not isinstance(
            task.get("subject_key"), str
        ):
            raise ValueError("task identity is invalid")


def _build_result_package(
    *,
    tasks: list[dict[str, Any]],
    execution_bytes: bytes,
    results: list[dict[str, Any]],
) -> bytes:
    if len(tasks) == 0 or len(tasks) != len(results):
        raise ValueError("result membership differs from trusted tasks")
    evidence_values: list[dict[str, str]] = []
    evidence_members: list[tuple[str, str, str, bytes]] = []
    result_values: list[dict[str, Any]] = []
    for task_index, (task, result) in enumerate(zip(tasks, results, strict=True)):
        requested = _requested_coverage(task)
        completed = result["completed"]
        if any(value not in requested for value in completed):
            raise ValueError("completed coverage is outside the trusted request")
        evidence_ids: list[str] = []
        for evidence_index, evidence in enumerate(result["evidence"]):
            if len(tasks) == 1:
                evidence_id = f"ev-{evidence_index + 1}"
            else:
                evidence_id = f"ev-{task_index + 1}-{evidence_index + 1}"
            content_id = evidence_id.replace("ev-", "ev-content-", 1)
            evidence_ids.append(evidence_id)
            evidence_values.append(
                {
                    "id": evidence_id,
                    "kind": evidence["kind"],
                    "summary": evidence["summary"],
                    "content_member_id": content_id,
                }
            )
            evidence_members.append(
                (
                    content_id,
                    f"evidence/{evidence_id}.txt",
                    "text/plain",
                    evidence["summary"].encode("utf-8"),
                )
            )
        result_values.append(
            {
                "item_key": task["item_key"],
                "subject_key": task["subject_key"],
                "assessment": result["assessment"],
                "summary": result["summary"],
                "evidence_ids": evidence_ids,
                "coverage": {
                    "requested": requested,
                    "completed": completed,
                    "gaps": result["gaps"],
                },
                "proposals": result["proposals"],
            }
        )
    result_document = {
        "schema": RESULT_SCHEMA,
        "execution_manifest_digest": _digest(execution_bytes),
        "results": result_values,
    }
    members: list[tuple[str, str, str, bytes]] = [
        ("check-results", "check-results.json", JSON_MEDIA_TYPE, jcs.canonicalize(result_document))
    ]
    if evidence_values:
        members.append(
            (
                "evidence",
                "evidence.json",
                JSON_MEDIA_TYPE,
                jcs.canonicalize({"schema": EVIDENCE_SCHEMA, "evidence": evidence_values}),
            )
        )
        members.extend(evidence_members)
    package_identity = hashlib.sha256(b"".join(member[3] for member in members)).hexdigest()[:32]
    package_manifest = {
        "schema": PACKAGE_SCHEMA,
        "package_id": "result-" + package_identity,
        "kind": "check-results",
        "members": [
            {
                "id": member_id,
                "path": path,
                "media_type": media_type,
                "size": len(data),
                "digest": _digest(data),
            }
            for member_id, path, media_type, data in sorted(members, key=lambda item: item[1])
        ],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _write_member(archive, "manifest.json", jcs.canonicalize(package_manifest))
        for _, path, _, data in sorted(members, key=lambda item: item[1]):
            _write_member(archive, path, data)
    payload = output.getvalue()
    if len(payload) > MAX_PACKAGE_BYTES:
        raise ValueError("result package exceeds its aggregate bound")
    return payload


def _requested_coverage(task: dict[str, Any]) -> list[str]:
    checklist = task.get("checklist")
    operation = task.get("operation")
    finding = task.get("finding")
    if isinstance(checklist, dict) and operation is None:
        requested = checklist.get("required_evidence")
        if not isinstance(requested, list):
            raise ValueError("checklist requested coverage is invalid")
        _validate_values("requested coverage", requested)
        return requested
    if isinstance(operation, dict) and checklist is None:
        return ["operation-resolution"]
    if isinstance(finding, dict) and checklist is None and operation is None:
        method = finding.get("method")
        if not isinstance(method, str) or IDENTIFIER.fullmatch(method) is None:
            raise ValueError("finding requested coverage is invalid")
        return [method]
    raise ValueError("task kind is invalid")


def _canonical_object(payload: bytes, name: str) -> dict[str, Any]:
    if not payload or len(payload) > MAX_DOCUMENT_BYTES:
        raise ValueError(f"{name} exceeds its bound")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or jcs.canonicalize(value) != payload:
        raise ValueError(f"{name} is not a canonical object")
    return value


def _read_bounded(archive: zipfile.ZipFile, name: str, limit: int = MAX_DOCUMENT_BYTES) -> bytes:
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size > limit:
        raise ValueError("Audit package member exceeds its bound")
    with archive.open(info, mode="r") as source:
        result = source.read(limit + 1)
    if len(result) > limit or len(result) != info.file_size:
        raise ValueError("Audit package member exceeds its bound")
    return result


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del runtime_settings
    return ArtifactClient(allocation_id, _UnavailableTransport())


class _UnavailableTransport:
    async def request(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
