"""Allocation-bound candidate finding intake with no Audit authority."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import API_VERSION, ArtifactRef, RuntimeSettings
from contractor_runtime.toolsets.run_artifacts import (
    ArtifactClientFactory,
    ToolMetrics,
    gateway_secrets,
)
from contractor_runtime.workspace import AllocationWorkspace

FINDING_SCHEMA = "contractor.audit.finding-proposal.v1"
MAX_TEXT_BYTES = 64 * 1024
MAX_VALUES = 512
MAX_EVIDENCE = 256
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
SEVERITIES = frozenset({"informational", "low", "medium", "high", "critical"})


class SecurityFindingsToolsetFactory:
    """Build the selected candidate publisher for one allocation."""

    ref = "security-findings@1"
    exported_tools = frozenset({"finding"})
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
        del run_id, namespace, workspace, adapter_handles, project_workspace
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("security-findings@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        return {
            name: FindingTool(client, metrics, gateway_secrets(runtime_settings))
            for name in selected
        }


class FindingTool:
    name = "finding"
    description = """Submit a security finding proposal with exact evidence references.

    The server records a proposal receipt; this does not confirm the vulnerability
    or assign its final assessment.

    Args:
        client_key: Stable proposal key for deduplication within this invocation.
        title: Non-empty finding title.
        description: Non-empty explanation of the observed issue and impact.
        subject: Object containing only kind and key identifiers for the subject.
        evidence_refs: Unique exact artifact references, each with namespace, name
            and revision, supporting the proposal.
        hypothesis: Optional hypothesis requiring verification.
        proposed_checks: Optional objects with objective text and method identifier.
        standard_refs: Optional objects with scheme, version and requirement_id.
        severity_suggestion: informational, low, medium, high or critical;
            omit when no severity is proposed.

    Returns:
        proposal_id and receipt_id for the submitted proposal.
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

    async def close(self) -> None:
        self._secrets = ()

    async def __call__(
        self,
        client_key: str,
        title: str,
        description: str,
        subject: dict[str, str],
        evidence_refs: list[dict[str, str]],
        tool_context: ToolContext,
        hypothesis: str | None = None,
        proposed_checks: list[dict[str, str]] | None = None,
        standard_refs: list[dict[str, str]] | None = None,
        severity_suggestion: str | None = None,
    ) -> dict[str, str]:
        started_ns = time.perf_counter_ns()
        metric_arguments = {
            "client_key": client_key if isinstance(client_key, str) else "invalid",
            "description_bytes": _text_size(description),
            "evidence_count": len(evidence_refs) if isinstance(evidence_refs, list) else 0,
            "proposed_check_count": (
                len(proposed_checks) if isinstance(proposed_checks, list) else 0
            ),
            "standard_ref_count": len(standard_refs) if isinstance(standard_refs, list) else 0,
        }
        try:
            request = _build_request(
                invocation_id=tool_context.invocation_id,
                client_key=client_key,
                title=title,
                description=description,
                subject=subject,
                evidence_refs=evidence_refs,
                hypothesis=hypothesis,
                proposed_checks=proposed_checks,
                standard_refs=standard_refs,
                severity_suggestion=severity_suggestion,
            )
            receipt = await self._client.submit_finding_proposal(request)
            result = {
                "proposal_id": str(receipt["proposalId"]),
                "receipt_id": str(receipt["receiptId"]),
            }
            self._metrics.record_tool_call(
                self.name,
                arguments=metric_arguments,
                result=result,
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


def _build_request(
    *,
    invocation_id: str,
    client_key: str,
    title: str,
    description: str,
    subject: dict[str, str],
    evidence_refs: list[dict[str, str]],
    hypothesis: str | None,
    proposed_checks: list[dict[str, str]] | None,
    standard_refs: list[dict[str, str]] | None,
    severity_suggestion: str | None,
) -> dict[str, object]:
    _identifier("client_key", client_key)
    _text("title", title, required=True)
    _text("description", description, required=True)
    _text("hypothesis", hypothesis or "", required=False)
    if not isinstance(invocation_id, str) or not invocation_id.strip() or len(invocation_id) > 128:
        raise ValueError("Worker invocation identity is invalid")
    if not isinstance(subject, dict) or set(subject) != {"kind", "key"}:
        raise ValueError("subject requires only kind and key")
    _identifier("subject.kind", subject["kind"])
    _identifier("subject.key", subject["key"])

    if not isinstance(evidence_refs, list) or len(evidence_refs) > MAX_EVIDENCE:
        raise ValueError("evidence_refs exceeds its bound")
    refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str, str]] = set()
    for raw in evidence_refs:
        try:
            ref = ArtifactRef.model_validate(raw).require_exact()
        except (ValidationError, ValueError) as error:
            raise ValueError("evidence_refs contains an invalid exact ArtifactRef") from error
        assert ref.revision is not None
        key = (ref.namespace, ref.name, ref.revision)
        if key in seen_refs:
            raise ValueError("evidence_refs contains a duplicate")
        seen_refs.add(key)
        refs.append(ref.model_dump(by_alias=True, exclude_none=True))
    refs.sort(key=lambda value: (value["namespace"], value["name"], value["revision"]))

    checks = _object_list("proposed_checks", proposed_checks, {"objective", "method"}, MAX_VALUES)
    for check in checks:
        _text("proposed_checks.objective", check["objective"], required=True)
        _identifier("proposed_checks.method", check["method"])
    standards = _object_list(
        "standard_refs", standard_refs, {"scheme", "version", "requirement_id"}, MAX_VALUES
    )
    for reference in standards:
        _identifier("standard_refs.scheme", reference["scheme"])
        _text("standard_refs.version", reference["version"], required=True)
        _identifier("standard_refs.requirement_id", reference["requirement_id"])
    severity = severity_suggestion or ""
    if severity not in SEVERITIES and severity != "":
        raise ValueError("severity_suggestion is invalid")

    submission_digest = hashlib.sha256(
        b"contractor.finding.submission.v1\0"
        + invocation_id.encode("utf-8")
        + b"\0"
        + client_key.encode("utf-8")
    ).hexdigest()
    proposal: dict[str, object] = {
        "schema": FINDING_SCHEMA,
        "client_key": client_key,
        "title": title,
        "description": description,
        "subject": subject,
        "preconditions": [],
        "standard_refs": standards,
        "evidence_ids": [f"evidence-{index + 1}" for index in range(len(refs))],
        "proposed_checks": checks,
        "severity_suggestion": severity,
        "limitations": [],
    }
    if hypothesis:
        proposal["hypothesis"] = hypothesis
    return {
        "apiVersion": API_VERSION,
        "invocationId": invocation_id,
        "submissionId": f"finding-{submission_digest}",
        "proposal": proposal,
        "evidenceRefs": refs,
    }


def _object_list(
    field: str,
    value: list[dict[str, str]] | None,
    keys: set[str],
    maximum: int,
) -> list[dict[str, str]]:
    result = [] if value is None else value
    if not isinstance(result, list) or len(result) > maximum:
        raise ValueError(f"{field} exceeds its bound")
    for item in result:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError(f"{field} contains an invalid object")
        if any(not isinstance(candidate, str) for candidate in item.values()):
            raise ValueError(f"{field} values must be strings")
    return result


def _identifier(field: str, value: str) -> None:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")


def _text(field: str, value: str, *, required: bool) -> None:
    if not isinstance(value, str) or "\x00" in value or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError(f"{field} is invalid")
    if required and not value.strip():
        raise ValueError(f"{field} is required")


def _text_size(value: object) -> int:
    return len(value.encode("utf-8")) if isinstance(value, str) else 0


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)


def _unconfigured_client(_allocation_id: str, _settings: RuntimeSettings) -> ArtifactClient:
    raise RuntimeError("security-findings@1 has no ArtifactClient factory")
