"""Bounded invocation-local drafts for ordinary Audit conclusions."""

import asyncio
from copy import deepcopy
from functools import partial

from contractor_runtime import threads
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.worker_completion import WorkerCompletionError
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_ITEMS,
    MAX_PROPOSAL_KEYS,
    MAX_REVISION,
    AuditRecordReceipt,
    AuditSnapshot,
    AuditTrustedInputs,
    NormalizedAuditItem,
    RecordedAuditItem,
    SealedAuditSnapshot,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditResultEncoder,
    evidence_handles,
)
from contractor_runtime.toolsets.common.cwe import CWEReferenceError, check_cwe_reference


class AuditCollectionError(WorkerCompletionError, ToolInputError):
    """Safe field errors contain Runtime-authored explanations only."""

    code = "audit_result_invalid"

    def __init__(self, field, reason, *, item_key=None, current_revision=None):
        super().__init__(f"{field}: {reason}")
        self.field, self.reason = field, reason
        self.item_key, self.current_revision = item_key, current_revision
        self.details = {}

    def as_dict(self):
        result = {"code": self.code, "field": self.field, "message": self.reason}
        if self.item_key is not None:
            result["itemKey"] = self.item_key
        if self.current_revision is not None:
            result["currentRevision"] = self.current_revision
        return result


class InvocationAuditCollector:
    """One invocation owns drafts, revisions and a final immutable seal.

    Admission encodes at most 64 members and 8 MiB off the event loop. The lock
    serializes updates, sealing and discard; cancellation joins owned encoding
    before releasing it. Nothing survives the invocation except normal artifacts.
    """

    def __init__(self, inputs: AuditTrustedInputs, encoder: CanonicalAuditResultEncoder):
        self._inputs, self._encoder = inputs, encoder
        self._tasks = encoder.trusted_inputs(inputs).items
        self._items: dict[str, RecordedAuditItem] = {}
        self._replayed_from: dict[str, int] = {}
        self._proposals: dict[str, frozenset] = {}
        self._lock = asyncio.Lock()
        self._sealed = self._discarded = False

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
            for task in tasks:
                task["evidence"] = [
                    {"handle": handle, "material": material}
                    for handle, material in evidence_handles(task).items()
                ]
            return {"batchSize": len(tasks), "tasks": tasks}

    def check_invocation(self, invocation_id: str):
        if invocation_id != self.owner.invocation_id:
            raise AuditCollectionError("invocation", "Tool belongs to a different invocation.")

    def check_finding_item(self, invocation_id, item_key):
        self._check_live(writing=True)
        self.check_invocation(invocation_id)
        if item_key is None and len(self.owner.item_keys) == 1:
            return
        if not isinstance(item_key, str) or item_key not in self.owner.item_keys:
            raise AuditCollectionError(
                "item_key", "Choose the assigned Audit task for this finding."
            )

    def check_proposal_refs(self, invocation_id, standard_refs):
        # Finding intake owns standards and publication authority. Audit completion
        # only checks invocation lifetime and the existing CWE contract locally.
        self._check_live(writing=True)
        self.check_invocation(invocation_id)
        refs = frozenset(
            (ref["scheme"], ref["version"], ref["requirement_id"]) for ref in standard_refs
        )
        if len(refs) != len(standard_refs):
            raise AuditCollectionError("standard_refs", "Remove duplicate standard references.")
        for reference in standard_refs:
            try:
                check_cwe_reference(reference)
            except CWEReferenceError as error:
                raise AuditCollectionError("standard_refs", error.reason) from None
        return refs

    async def register_proposal(self, invocation_id, client_key, standard_refs):
        async with self._lock:
            refs = self.check_proposal_refs(invocation_id, standard_refs)
            existing = self._proposals.get(client_key)
            if existing is not None and existing != refs:
                raise AuditCollectionError("client_key", "Finding client key changed content.")
            if existing is None and len(self._proposals) >= MAX_ITEMS * MAX_PROPOSAL_KEYS:
                raise AuditCollectionError("client_key", "Invocation finding limit reached.")
            self._proposals[client_key] = refs

    def _validate(self, item: NormalizedAuditItem):
        if not isinstance(item, NormalizedAuditItem) or item.item_key not in self._tasks:
            raise AuditCollectionError("item_key", "Choose an item key from read_audit_task.")
        member = self._tasks[item.item_key]
        if ("hypothesis" in member) != (item.verdict is not None):
            raise AuditCollectionError(
                "verdict", "Supply a verdict only for a task with an assigned hypothesis."
            )
        if item.verdict in {"supported", "refuted"} and (
            not item.tested_conditions or not item.evidence
        ):
            raise AuditCollectionError(
                "evidence", "A conclusive check needs tested conditions and selected evidence."
            )
        if item.verdict == "inconclusive" and not item.limitations:
            raise AuditCollectionError("limitations", "Explain why the check is inconclusive.")
        if any(handle not in evidence_handles(member) for handle in item.evidence):
            raise AuditCollectionError("evidence", "Use evidence handles from this assigned task.")
        return item

    def _snapshot(self, items=None):
        selected = self._items if items is None else items
        return AuditSnapshot(
            self.owner,
            tuple(selected[key] for key in self.owner.item_keys if key in selected),
        )

    async def _admit(self, prospective):
        try:
            await threads.to_thread_until_done(
                partial(self._encoder.encode, self._snapshot(prospective), inputs=self.inputs),
                name="audit-admission-encode",
            )
        except AuditResultError:
            raise AuditCollectionError(
                "collection", "Prospective conclusions exceed the JSON document contract."
            ) from None

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
            await self._admit(prospective)
            # The lock is still held: nothing changed the collection while
            # the prospective set was encoded, so the replacement is atomic.
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
            await self._admit(prospective)
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
            self._proposals.clear()
