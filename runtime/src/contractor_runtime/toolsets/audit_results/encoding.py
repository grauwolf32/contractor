"""Deterministic ordinary JSON results over a single exact trusted context."""

import threading
from dataclasses import dataclass, field

import jcs
from jsonschema import ValidationError

from contractor_runtime.contracts.audit_documents import decode_audit_document
from contractor_runtime.core.worker_completion import WorkerCompletionError
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_DOCUMENT_BYTES,
    AuditEncodedResult,
    AuditSnapshot,
    AuditTrustedInputs,
)


class AuditResultError(WorkerCompletionError):
    """Stable safe failure; never include submitted content or transport messages."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        if code not in {
            "audit_result_invalid",
            "audit_result_size_exceeded",
            "audit_result_publication_conflict",
            "audit_result_publication_failed",
        }:
            raise ValueError("unknown Audit result failure")
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True, eq=False)
class AuditContextRecords:
    """Allocation-owned decoding; model-facing reads always receive copies."""

    data: bytes = field(repr=False)
    items: dict[str, dict] = field(repr=False)

    @property
    def item_keys(self):
        return tuple(self.items)


def evidence_handles(member: dict) -> dict[str, dict]:
    return {f"evidence-{index:03d}": value for index, value in enumerate(member["evidence"])}


class CanonicalAuditResultEncoder:
    """Decode one bounded context per allocation, then encode prospective results.

    Each member can cite only evidence already granted to that member. Audit,
    Run, Task, Hypothesis and Finding identities never come from tool arguments.
    The Server applies authoritative normalization after ordinary publication.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._records: AuditContextRecords | None = None

    def context_records(self, context: bytes) -> AuditContextRecords:
        with self._lock:
            if self._records is None or self._records.data != context:
                try:
                    value = decode_audit_document(context, schema="contractor.audit.context.v1")
                    self._records = AuditContextRecords(
                        context, {item["item_key"]: item for item in value["items"]}
                    )
                except (ValueError, TypeError, RecursionError, UnicodeError, ValidationError):
                    raise AuditResultError("audit_result_invalid") from None
            return self._records

    def trusted_inputs(self, inputs: AuditTrustedInputs) -> AuditContextRecords:
        if not isinstance(inputs, AuditTrustedInputs):
            raise AuditResultError("audit_result_invalid")
        records = self.context_records(inputs.context)
        if inputs.owner.item_keys != records.item_keys:
            raise AuditResultError("audit_result_invalid")
        return records

    def encode(self, snapshot: AuditSnapshot, *, inputs: AuditTrustedInputs) -> AuditEncodedResult:
        if not isinstance(snapshot, AuditSnapshot) or snapshot.owner != inputs.owner:
            raise AuditResultError("audit_result_invalid")
        context = self.trusted_inputs(inputs)
        items = []
        for record in snapshot.items:
            value = record.value
            member = context.items[value.item_key]
            checking = "hypothesis" in member
            if checking != (value.verdict is not None):
                raise AuditResultError("audit_result_invalid")
            if value.verdict in {"supported", "refuted"} and (
                not value.tested_conditions or not value.evidence
            ):
                raise AuditResultError("audit_result_invalid")
            if value.verdict == "inconclusive" and not value.limitations:
                raise AuditResultError("audit_result_invalid")
            allowed = evidence_handles(member)
            if any(handle not in allowed for handle in value.evidence):
                raise AuditResultError("audit_result_invalid")
            items.append(
                {
                    "item_key": value.item_key,
                    "rationale": value.rationale,
                    "tested_conditions": list(value.tested_conditions),
                    "limitations": list(value.limitations),
                    "evidence": [allowed[handle] for handle in value.evidence],
                    **({"verdict": value.verdict} if checking else {}),
                }
            )
        data = jcs.canonicalize({"schema": "contractor.audit.conclusions.v1", "items": items})
        if len(data) > MAX_DOCUMENT_BYTES:
            raise AuditResultError("audit_result_size_exceeded")
        # Prospective admission may be partial; only a complete seal is published.
        return AuditEncodedResult(snapshot.owner, data)
