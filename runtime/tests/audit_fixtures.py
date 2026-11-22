"""Common exact Audit context fixtures for real completion and transport tests."""

import json
from copy import deepcopy

import jcs

from contractor_runtime.audit_evidence import evidence_snapshot_ref
from contractor_runtime.contracts import ArtifactRef
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
                verdict="supported" if checking else None,
                selected_evidence=tuple(
                    jcs.canonicalize(material) for material in doc["items"][index]["evidence"]
                ),
            ),
            1,
        )
        for index, key in enumerate(keys)
    )
    return SealedAuditSnapshot(owner, items), inputs


def arguments(item, *, keyed=False):
    return {
        **({"item_key": item.item_key} if keyed else {}),
        "rationale": item.rationale,
        "limitations": list(item.limitations),
        **({"verdict": item.verdict} if item.verdict is not None else {}),
    }


def conclusions(snapshot, inputs):
    context = context_document(count=len(snapshot.items))
    result = []
    for record, member in zip(snapshot.items, context["items"], strict=True):
        value = arguments(record.value)
        value.update(
            item_key=record.value.item_key,
            tested_conditions=list(record.value.tested_conditions),
            evidence=deepcopy(member["evidence"]),
        )
        result.append(value)
    return {"schema": "contractor.audit.conclusions.v1", "items": result}


async def accept_fixture_sources(
    collector,
    *,
    item_key=None,
    conditions=("selected source trace",),
    limitations=(),
    suffix="initial",
):
    """Explicitly select fixture material and simulate its accepted owner receipt.

    Unit completion tests use this for an already accepted prefix. Actual source
    preparation, allocation intake and retention are covered by real process/PG tests.
    Calling assigned/read_tasks alone deliberately performs no association.
    """
    context = json.loads(collector.inputs.context)
    for index, member in enumerate(context["items"]):
        key = member["item_key"]
        if item_key is not None and key != item_key:
            continue
        sources = []
        for material in member["evidence"]:
            source = deepcopy(material)
            source.update(scope_kind="audit", scope_id=context["audit_id"])
            source["ref"] = {
                "namespace": "audit-fixture",
                "name": f"source-{index}",
                "revision": "source-r1",
            }
            sources.append(source)
        note = {
            "scope_kind": "audit",
            "scope_id": context["audit_id"],
            "ref": {
                "namespace": "audit-fixture",
                "name": f"note-{index}-{suffix}",
                "revision": "note-r1",
            },
            "digest": digest(b"fixture accepted note"),
            "media_type": "application/json",
            "size_bytes": 21,
        }
        selected = deepcopy(member["evidence"])
        for material in selected:
            snapshot = evidence_snapshot_ref(ArtifactRef.model_validate(material["ref"]))
            material["ref"] = {**snapshot.model_dump(exclude_none=True), "revision": "snapshot-r1"}
        receipt = {
            "apiVersion": "contractor/v1alpha1",
            "itemKey": key,
            "material": note,
            "sources": sources,
            "selectedSources": selected,
            "replayed": False,
        }

        async def save(receipt=receipt):
            return receipt

        await collector.record_evidence(
            collector.owner.invocation_id,
            key,
            save,
            tested_conditions=conditions,
            limitations=limitations,
        )
