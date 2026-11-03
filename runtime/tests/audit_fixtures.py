"""Common exact Audit context fixtures for real completion and transport tests."""

from copy import deepcopy

import jcs

from contractor_runtime.digests import sha256_digest as digest
from contractor_runtime.toolsets.audit_results.contracts import (
    AuditInvocationOwner,
    AuditTrustedInputs,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)


def context_document(*, count=1, checking=True):
    items = []
    for index in range(count):
        key = ("check-authz", "check-input")[index] if index < 2 else f"item-{index:02d}"
        item = {
            "item_key": key,
            "task_id": f"task-{index}",
            "objective": "Inspect selected source",
            "evidence": [
                {
                    "scope_kind": "run",
                    "scope_id": "run-1",
                    "ref": {"namespace": "inputs", "name": "source", "revision": "source-r1"},
                    "digest": "sha256:" + "a" * 64,
                    "media_type": "text/plain",
                    "size_bytes": 3,
                }
            ],
        }
        if checking:
            item["hypothesis"] = {
                "schema": "contractor.audit.hypothesis.v1",
                "id": f"hypothesis-{index}",
                "revision": 1,
                "statement": "A reader crosses the selected access boundary",
                "scope": {"actor": "selected reader"},
                "grounds": "Selected source suggests a path",
                "origin": {"kind": "manual", "audit_id": "audit-1"},
            }
        items.append(item)
    return {
        "schema": "contractor.audit.context.v1",
        "audit_id": "audit-1",
        "run_id": "run-1",
        "workflow_ref": "audit-contract-check@1",
        "workflow_digest": "sha256:" + "b" * 64,
        "items": items,
    }


def assigned(*, batch=False, invocation="invocation-1", count=None, checking=True):
    doc = context_document(count=count or (2 if batch else 1), checking=checking)
    data = jcs.canonicalize(doc)
    keys = tuple(item["item_key"] for item in doc["items"])
    owner = AuditInvocationOwner("allocation-1", invocation, digest(data), keys)
    inputs = AuditTrustedInputs(owner, data)
    items = tuple(
        RecordedAuditItem(
            NormalizedAuditItem(
                key,
                "Verified the selected boundary at app.py:12.",
                ("selected source trace",),
                (),
                ("evidence-000",),
                "supported" if checking else None,
            ),
            1,
        )
        for key in keys
    )
    return SealedAuditSnapshot(owner, items), inputs


def arguments(item):
    return {
        "rationale": item.rationale,
        "tested_conditions": list(item.tested_conditions),
        "limitations": list(item.limitations),
        "evidence": list(item.evidence),
        **({"verdict": item.verdict} if item.verdict is not None else {}),
    }


def conclusions(snapshot, inputs):
    context = context_document(count=len(snapshot.items))
    result = []
    for record, member in zip(snapshot.items, context["items"], strict=True):
        value = arguments(record.value)
        value.update(item_key=record.value.item_key, evidence=deepcopy(member["evidence"]))
        result.append(value)
    return {"schema": "contractor.audit.conclusions.v1", "items": result}
