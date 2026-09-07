"""Bounded, invocation-owned Audit drafts. No artifacts or ADK State are mutated."""

import asyncio
from copy import deepcopy
from dataclasses import replace

from contractor_runtime.audit_completion_contracts import (
    MAX_EVIDENCE,
    MAX_REVISION,
    AuditRecordReceipt,
    AuditSnapshot,
    AuditTrustedInputs,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)
from contractor_runtime.audit_packages import _decode_task_input, _requested_coverage
from contractor_runtime.audit_result_encoding import AuditResultError, CanonicalAuditPackageEncoder


class AuditCollectionError(ValueError):
    """Only Runtime-authored reasons and trusted identifiers may enter diagnostics."""

    code = "audit_result_invalid"

    def __init__(
        self,
        field: str,
        reason: str,
        *,
        item_key: str | None = None,
        current_revision: int | None = None,
    ):
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason
        self.item_key = item_key
        self.current_revision = current_revision

    def as_dict(self):
        result = {"code": self.code, "field": self.field, "message": self.reason}
        if self.item_key is not None:
            result["itemKey"] = self.item_key
        if self.current_revision is not None:
            result["currentRevision"] = self.current_revision
        return result


class InvocationAuditCollector:
    """Create afresh for each invocation; the owner and input bytes never change.

    Acceptance, sealing and discard are serialized by one lock. The integration
    owner must discard in its failure/cancellation cleanup; there is no durable
    state to recover after process loss. Explicit-update replay retains one
    revision number per item, never an unbounded history of submitted content.
    """

    def __init__(self, inputs: AuditTrustedInputs):
        self._inputs = inputs
        self._encoder = CanonicalAuditPackageEncoder()
        # The actual encoder checks task/manifest membership, digests and order.
        self._encoder.encode(AuditSnapshot(inputs.owner, ()), inputs=inputs)
        records = _decode_task_input(inputs.task_package, "application/zip")
        self._tasks = {task["item_key"]: task for task, _, _ in records}
        self._package_ids = tuple(package_id for _, package_id, _ in records)
        for task in self._tasks.values():
            _requested_coverage(task)
        self._items: dict[str, RecordedAuditItem] = {}
        self._replayed_from: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._sealed = False
        self._discarded = False

    @property
    def owner(self):
        return self._inputs.owner

    @property
    def inputs(self):
        return self._inputs

    def _check_live(self, *, writing=False):
        if self._discarded or (writing and self._sealed):
            raise AuditCollectionError("collection", "Invocation collection is closed.")

    async def read_tasks(self, invocation_id: str):
        async with self._lock:
            self._check_live()
            self.check_invocation(invocation_id)
            tasks = deepcopy(list(self._tasks.values()))
            result = {
                "batchSize": len(tasks),
                "tasks": tasks,
                "taskPackageIds": list(self._package_ids),
                "executionManifestDigest": self.inputs.execution_manifest_sha256,
            }
            if len(tasks) == 1:
                result.update(task=tasks[0], taskPackageId=self._package_ids[0])
            return result

    def check_invocation(self, invocation_id: str):
        if invocation_id != self.owner.invocation_id:
            raise AuditCollectionError("invocation", "Tool belongs to a different invocation.")

    def _validate(self, item: NormalizedAuditItem):
        if not isinstance(item, NormalizedAuditItem):
            raise AuditCollectionError("result", "Provide a normalized result.")
        if item.item_key not in self._tasks:
            raise AuditCollectionError("item_key", "Choose an item key from read_audit_task.")
        key = item.item_key
        task = self._tasks[key]
        item = replace(
            item,
            completed=tuple(sorted(item.completed)),
            gaps=tuple(sorted(item.gaps)),
            proposal_keys=tuple(sorted(item.proposal_keys)),
        )
        requested = _requested_coverage(task)
        if not set(item.completed) <= set(requested):
            raise AuditCollectionError(
                "completed",
                "Use only requested coverage from read_audit_task.",
                item_key=key,
            )
        if task.get("operation") is not None:
            if item.assessment == "not-tested" and item.completed:
                raise AuditCollectionError(
                    "completed",
                    "A not-tested operation must have empty completed coverage.",
                    item_key=key,
                )
            return item
        conclusive = item.assessment in {"satisfied", "violated", "supported", "refuted"}
        standard = task.get("standard")
        if standard is not None:
            contract = standard["evidence_contract"]
            if item.assessment not in contract["assessments"]:
                raise AuditCollectionError(
                    "assessment",
                    "Use an assessment allowed by the pinned standard.",
                    item_key=key,
                )
            if len(item.evidence) > contract["maximum_evidence"] or (
                conclusive and len(item.evidence) < contract["minimum_evidence"]
            ):
                raise AuditCollectionError(
                    "evidence",
                    "Match the pinned standard's minimum/maximum evidence count.",
                    item_key=key,
                )
            if any(value.kind not in contract["evidence_kinds"] for value in item.evidence):
                raise AuditCollectionError(
                    "evidence.kind",
                    "Use evidence kinds allowed by the pinned standard.",
                    item_key=key,
                )
        if conclusive and task.get("checklist") is not None:
            missing = sorted(set(requested) - {value.kind for value in item.evidence})
            if missing:
                # Trusted, bounded identifiers only; cap large checklists' diagnostics.
                names = ", ".join(missing[:8])
                raise AuditCollectionError(
                    "evidence",
                    f"Missing required evidence kinds ({len(missing)}): {names}.",
                    item_key=key,
                )
        return item

    def _snapshot(self, items=None):
        selected = self._items if items is None else items
        return AuditSnapshot(
            self.owner,
            tuple(selected[key] for key in self.owner.item_keys if key in selected),
        )

    def _admit(self, prospective):
        if sum(len(item.value.evidence) for item in prospective.values()) > MAX_EVIDENCE:
            raise AuditCollectionError(
                "evidence", "Collection allows at most 256 evidence records."
            )
        try:
            self._encoder.encode(self._snapshot(prospective), inputs=self.inputs)
        except AuditResultError as error:
            reason = (
                "Prospective collection exceeds canonical data, member or ZIP bounds; reduce data."
                if error.code == "audit_result_size_exceeded"
                else "Prospective result cannot be encoded; check the assigned task and result."
            )
            raise AuditCollectionError("collection", reason) from None

    def _receipt(self):
        return AuditRecordReceipt(
            self.owner,
            tuple(
                (key, self._items[key].revision)
                for key in self.owner.item_keys
                if key in self._items
            ),
            tuple(key for key in self.owner.item_keys if key not in self._items),
        )

    async def record(self, item: NormalizedAuditItem, *, expected_revision=None):
        async with self._lock:
            self._check_live(writing=True)
            item = self._validate(item)
            key = item.item_key
            current = self._items.get(key)
            if expected_revision is not None and (
                type(expected_revision) is not int or not 1 <= expected_revision <= MAX_REVISION
            ):
                raise AuditCollectionError(
                    "expected_revision",
                    "Use a positive integer revision from the receipt.",
                    item_key=key,
                )
            if (
                current is not None
                and current.value == item
                and (expected_revision is None or expected_revision == self._replayed_from.get(key))
            ):
                return self._receipt()
            if (current is None and expected_revision is not None) or (
                current is not None and expected_revision != current.revision
            ):
                raise AuditCollectionError(
                    "expected_revision",
                    "Result conflicts; use the current recorded revision.",
                    item_key=key,
                    current_revision=current.revision if current else None,
                )
            revision = 1 if current is None else current.revision + 1
            if revision > MAX_REVISION:
                raise AuditCollectionError(
                    "expected_revision", "Revision limit reached.", item_key=key
                )
            prospective = {**self._items, key: RecordedAuditItem(item, revision)}
            self._admit(prospective)
            # No suspension between prospective validation and the atomic replacement.
            self._items = prospective
            if expected_revision is not None:
                self._replayed_from[key] = expected_revision
            return self._receipt()

    async def record_batch(self, items: tuple[NormalizedAuditItem, ...]):
        async with self._lock:
            self._check_live(writing=True)
            if not isinstance(items, tuple) or len(items) != len(self.owner.item_keys):
                raise AuditCollectionError("results", "Provide the complete batch in task order.")
            values = tuple(self._validate(item) for item in items)
            if tuple(item.item_key for item in values) != self.owner.item_keys:
                raise AuditCollectionError("results", "Provide the complete batch in task order.")
            prospective = dict(self._items)
            for item in values:
                current = prospective.get(item.item_key)
                if current is not None and current.value != item:
                    raise AuditCollectionError(
                        "results",
                        "Existing result differs; correct it with scalar expected_revision.",
                        item_key=item.item_key,
                        current_revision=current.revision,
                    )
                if current is None:
                    prospective[item.item_key] = RecordedAuditItem(item, 1)
            self._admit(prospective)
            self._items = prospective
            return self._receipt()

    async def snapshot(self):
        async with self._lock:
            self._check_live()
            return self._snapshot()

    async def seal(self):
        async with self._lock:
            self._check_live()
            if len(self._items) != len(self.owner.item_keys):
                raise AuditCollectionError(
                    "collection", "Record every missing item before sealing."
                )
            result = SealedAuditSnapshot(self.owner, self._snapshot().items)
            self._sealed = True
            return result

    async def discard(self):
        async with self._lock:
            self._discarded = True
            self._items.clear()
            self._replayed_from.clear()
