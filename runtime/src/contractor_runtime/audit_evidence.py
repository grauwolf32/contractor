"""Allocation-bound normalized evidence publication through the Audit owner."""

import json
from typing import ClassVar, Literal

import jcs
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import BaseModel, ConfigDict, ValidationError

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactTransportError,
    allocation_path,
    raise_for_status,
    require_json,
)
from contractor_runtime.contracts import API_VERSION, ArtifactRef
from contractor_runtime.contracts.audit_documents import validate_audit_material
from contractor_runtime.core.tool_errors import ToolResultDiagnostic
from contractor_runtime.digests import sha256_digest
from contractor_runtime.strict_json import strict_json_loads


def evidence_snapshot_ref(source: ArtifactRef) -> ArtifactRef:
    source.require_exact()
    digest = sha256_digest(jcs.canonicalize(source.model_dump(exclude_none=True)))
    return ArtifactRef(namespace="inputs", name="selected-" + digest.removeprefix("sha256:"))


class EvidenceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    apiVersion: Literal["contractor/v1alpha1"]
    material: dict
    sources: list[dict]
    selectedSources: list[dict]
    itemKey: str
    replayed: bool


class EvidencePersistenceUnknown(ArtifactTransportError, ToolResultDiagnostic):
    code = "audit_evidence_persistence_unknown"
    retryable = False
    diagnostic_message = (
        "Evidence acceptance is unconfirmed after bounded receipt reconciliation. "
        "Earlier records remain saved; do not repeat target traffic."
    )
    model_details: ClassVar[dict] = {"persistence": "unknown"}


class AuditEvidenceClient:
    def __init__(self, client, *, run_id, audit_id):
        self.client = client
        self.run_id, self.audit_id = run_id, audit_id

    async def submit(self, value):
        try:
            return await self._submit(value)
        except ArtifactTransportError:
            raise EvidencePersistenceUnknown() from None

    async def _submit(self, value):
        value = {"apiVersion": API_VERSION, **value}
        body = await threads.to_thread_until_done(
            jcs.canonicalize, value, name="audit-evidence-request"
        )
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Audit evidence request exceeds its size limit")
        for attempt in range(2):
            try:
                response = await self.client.transport.request(
                    "POST",
                    allocation_path(self.client.allocation_id, "audit-evidence"),
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=1 << 20,
                )
                break
            except ArtifactTransportError:
                if attempt:
                    raise
        raise_for_status(response)
        if response.status_code not in {200, 201}:
            raise ArtifactTransportError("Audit evidence API returned an invalid status")
        require_json(response.headers)
        try:
            receipt = EvidenceReceipt.model_validate(strict_json_loads(response.body))
        except (ValidationError, ValueError):
            raise ArtifactTransportError("Audit evidence API returned an invalid receipt") from None
        result = json.loads(receipt.model_dump_json())
        try:
            for material in [result["material"], *result["sources"], *result["selectedSources"]]:
                await threads.to_thread_until_done(
                    validate_audit_material, material, name="audit-evidence-receipt"
                )
        except (SchemaValidationError, ValidationError, ValueError, KeyError, TypeError):
            raise ArtifactTransportError(
                "Audit evidence API returned invalid material metadata"
            ) from None
        if (
            receipt.itemKey != value["itemKey"]
            or receipt.material["scope_kind"] != "audit"
            or receipt.material["scope_id"] != self.audit_id
            or len(receipt.selectedSources) != len(value["sources"])
            or len(receipt.sources) != len(value["sources"])
        ):
            raise ArtifactTransportError("Audit evidence API returned an inconsistent receipt")
        for retained, selected in zip(receipt.sources, receipt.selectedSources, strict=True):
            if (
                retained["scope_kind"] != "audit"
                or retained["scope_id"] != self.audit_id
                or any(
                    retained[field] != selected[field]
                    for field in ("digest", "media_type", "size_bytes")
                )
            ):
                raise ArtifactTransportError("Audit evidence API returned inconsistent sources")
        for selected, original in zip(result["selectedSources"], value["sources"], strict=True):
            expected = evidence_snapshot_ref(ArtifactRef.model_validate(original))
            if (
                selected["scope_kind"] != "run"
                or selected["scope_id"] != self.run_id
                or selected["ref"]["namespace"] != expected.namespace
                or selected["ref"]["name"] != expected.name
            ):
                raise ArtifactTransportError("Audit evidence API returned another source selection")
        return result
