"""A narrow model tool for publishing one canonical Audit check result package."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

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


class AuditResultsToolsetFactory:
    """Build the selected result publisher without exposing Audit authority."""

    ref = "audit-results@1"
    exported_tools = frozenset({"submit_check_result"})
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
        builders: dict[str, Callable[[], Any]] = {
            "submit_check_result": lambda: SubmitCheckResultTool(
                client, metrics, gateway_secrets(runtime_settings), namespace
            )
        }
        return {name: builders[name]() for name in selected}


class SubmitCheckResultTool:
    name = "submit_check_result"
    description = (
        "Publish the result for the one assigned Audit check. Identity and requested "
        "coverage are read from trusted task and execution_manifest inputs."
    )

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
        assessment: str,
        summary: str,
        completed: list[str],
        gaps: list[str],
        tool_context: ToolContext,
        evidence: list[dict[str, str]] | None = None,
        proposal_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        metric_arguments = {
            "assessment": assessment,
            "content_bytes": len(summary.encode("utf-8")) if isinstance(summary, str) else 0,
            "completed_count": len(completed) if isinstance(completed, list) else 0,
            "gap_count": len(gaps) if isinstance(gaps, list) else 0,
            "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
            "proposal_count": len(proposal_keys) if isinstance(proposal_keys, list) else 0,
        }
        try:
            normalized_evidence = _validate_arguments(
                assessment, summary, completed, gaps, evidence
            )
            normalized_proposals = _validate_proposal_keys(proposal_keys)
            task_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="task")
            )
            execution_value = await self._client.read_artifact(
                ArtifactRef(namespace="inputs", name="execution_manifest")
            )
            task, task_package_id = _decode_task_package(task_value.data, task_value.media_type)
            execution = _decode_execution_manifest(execution_value.data, execution_value.media_type)
            _match_trusted_inputs(task, task_package_id, task_value.data, execution)
            payload = _build_result_package(
                task=task,
                execution_bytes=execution_value.data,
                assessment=assessment,
                summary=summary,
                completed=completed,
                gaps=gaps,
                evidence=normalized_evidence,
                proposals=[
                    {"invocation_id": tool_context.invocation_id, "client_key": key}
                    for key in normalized_proposals
                ],
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


def _validate_arguments(
    assessment: str,
    summary: str,
    completed: list[str],
    gaps: list[str],
    evidence: list[dict[str, str]] | None,
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
    if media_type != PACKAGE_MEDIA_TYPE:
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


def _decode_execution_manifest(payload: bytes, media_type: str) -> dict[str, Any]:
    if media_type != JSON_MEDIA_TYPE:
        raise ValueError("execution manifest input is not JSON")
    manifest = _canonical_object(payload, "execution manifest")
    items = manifest.get("items")
    if manifest.get("schema") != EXECUTION_SCHEMA or not isinstance(items, list) or len(items) != 1:
        raise ValueError("execution manifest is not a one-item manifest")
    if not isinstance(items[0], dict):
        raise ValueError("execution manifest item is invalid")
    return manifest


def _match_trusted_inputs(
    task: dict[str, Any],
    task_package_id: str,
    task_package: bytes,
    execution: dict[str, Any],
) -> None:
    item = execution["items"][0]
    if (
        item.get("item_key") != task.get("item_key")
        or item.get("subject_key") != task.get("subject_key")
        or item.get("task_package_id") != task_package_id
        or item.get("task_package_digest") != _digest(task_package)
    ):
        raise ValueError("task and execution manifest identities differ")
    if not isinstance(task.get("item_key"), str) or not isinstance(task.get("subject_key"), str):
        raise ValueError("task identity is invalid")


def _build_result_package(
    *,
    task: dict[str, Any],
    execution_bytes: bytes,
    assessment: str,
    summary: str,
    completed: list[str],
    gaps: list[str],
    evidence: list[dict[str, str]],
    proposals: list[dict[str, str]],
) -> bytes:
    requested = _requested_coverage(task)
    if any(value not in requested for value in completed):
        raise ValueError("completed coverage is outside the trusted request")
    evidence_values = [
        {
            "id": f"ev-{index + 1}",
            "kind": item["kind"],
            "summary": item["summary"],
            "content_member_id": f"ev-content-{index + 1}",
        }
        for index, item in enumerate(evidence)
    ]
    result_document = {
        "schema": RESULT_SCHEMA,
        "execution_manifest_digest": _digest(execution_bytes),
        "results": [
            {
                "item_key": task["item_key"],
                "subject_key": task["subject_key"],
                "assessment": assessment,
                "summary": summary,
                "evidence_ids": [item["id"] for item in evidence_values],
                "coverage": {
                    "requested": requested,
                    "completed": completed,
                    "gaps": gaps,
                },
                "proposals": proposals,
            }
        ],
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
        for index, item in enumerate(evidence):
            members.append(
                (
                    f"ev-content-{index + 1}",
                    f"evidence/ev-{index + 1}.txt",
                    "text/plain",
                    item["summary"].encode("utf-8"),
                )
            )
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
    return output.getvalue()


def _requested_coverage(task: dict[str, Any]) -> list[str]:
    checklist = task.get("checklist")
    operation = task.get("operation")
    if isinstance(checklist, dict) and operation is None:
        requested = checklist.get("required_evidence")
        if not isinstance(requested, list):
            raise ValueError("checklist requested coverage is invalid")
        _validate_values("requested coverage", requested)
        return requested
    if isinstance(operation, dict) and checklist is None:
        return ["operation-resolution"]
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


def _read_bounded(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size > MAX_DOCUMENT_BYTES:
        raise ValueError("Audit package member exceeds its bound")
    with archive.open(info, mode="r") as source:
        result = source.read(MAX_DOCUMENT_BYTES + 1)
    if len(result) > MAX_DOCUMENT_BYTES or len(result) != info.file_size:
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
