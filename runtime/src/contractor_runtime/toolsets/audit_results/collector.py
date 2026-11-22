"""Bounded invocation-local drafts for ordinary Audit conclusions."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from functools import partial

import jcs

from contractor_runtime import threads
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.worker_completion import WorkerCompletionError
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_COLLECTED_BYTES,
    MAX_EVIDENCE,
    MAX_ITEMS,
    MAX_PROPOSAL_KEYS,
    MAX_REVISION,
    MAX_VALUES,
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


_CURRENT = object()


class InvocationAuditCollector:
    """One invocation owns drafts, revisions and a final immutable seal.

    Admission encodes at most 64 members and 8 MiB off the event loop. The lock
    serializes updates, sealing and discard; cancellation joins owned encoding
    before releasing it. Nothing survives the invocation except normal artifacts.
    """

    def __init__(
        self,
        inputs: AuditTrustedInputs,
        encoder: CanonicalAuditResultEncoder,
    ):
        self._inputs, self._encoder = inputs, encoder
        self._tasks = encoder.trusted_inputs(inputs).items
        self._items: dict[str, RecordedAuditItem] = {}
        self._evidence: dict[str, dict] = {}
        self._response_revisions: dict[str, int] | None = None
        self._proposals: dict[str, frozenset] = {}
        self._lock = asyncio.Lock()
        self._sealed = self._discarded = False

    @property
    def audit_id(self):
        return self._encoder.trusted_inputs(self.inputs).audit_id

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
                task["materials"] = [
                    {
                        "material": material,
                        "source": {
                            "kind": "artifact",
                            "namespace": material["ref"]["namespace"],
                            "name": material["ref"]["name"],
                        },
                    }
                    for material in task.pop("evidence")
                ]
            return {"batchSize": len(tasks), "tasks": tasks}

    def check_invocation(self, invocation_id: str):
        if invocation_id != self.owner.invocation_id:
            raise AuditCollectionError("invocation", "Tool belongs to a different invocation.")

    def check_finding_item(self, invocation_id, item_key, *, incidental=False):
        self._check_live(writing=True)
        self.check_invocation(invocation_id)
        if item_key is None and (incidental or len(self.owner.item_keys) == 1):
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

    async def record_evidence(
        self, invocation_id, item_key, save, *, tested_conditions=(), limitations=()
    ):
        async with self._lock:
            self._check_live(writing=True)
            self.check_invocation(invocation_id)
            if item_key not in self._tasks:
                raise AuditCollectionError("item_key", "Choose an assigned check.")
            for field, values in (
                ("tested_conditions", tested_conditions),
                ("limitations", limitations),
            ):
                combined = {
                    value
                    for r in self._evidence.values()
                    if r["itemKey"] == item_key
                    for value in r[field]
                }
                if len(combined.union(values)) > MAX_VALUES:
                    raise AuditCollectionError(
                        field, "Combined evidence notes exceed the result limit."
                    )
            receipt = await save()
            key = jcs.canonicalize(receipt["material"]).decode()
            if key not in self._evidence and len(self._evidence) >= MAX_EVIDENCE:
                raise AuditCollectionError("sources", "Selected evidence limit reached.")
            if receipt["itemKey"] != item_key:
                raise RuntimeError("Evidence receipt has another item")
            self._evidence[key] = {
                **receipt,
                "tested_conditions": tuple(tested_conditions),
                "limitations": tuple(limitations),
            }
            return receipt

    def _with_evidence(self, item):
        if not isinstance(item, NormalizedAuditItem) or item.item_key not in self._tasks:
            raise AuditCollectionError("item_key", "Choose an item key from read_audit_task.")
        records = [r for r in self._evidence.values() if r["itemKey"] == item.item_key]
        selected = {jcs.canonicalize(source) for r in records for source in r["sources"]}
        # Exact annotations preserve authorship and applicability. Their conditions
        # and gaps are assembled without promoting grounds or incidental Findings.
        selected.update(jcs.canonicalize(r["material"]) for r in records)
        conditions = tuple(
            dict.fromkeys(value for r in records for value in r["tested_conditions"])
        )
        gaps = tuple(
            dict.fromkeys(
                (*item.limitations, *(value for r in records for value in r["limitations"]))
            )
        )
        if len(conditions) > MAX_VALUES or len(gaps) > MAX_VALUES or len(selected) > MAX_EVIDENCE:
            raise AuditCollectionError(
                "limitations", "Combined evidence and judgment exceed the result limits."
            )
        return replace(
            item,
            tested_conditions=conditions,
            limitations=gaps,
            selected_evidence=tuple(sorted(selected)),
        )

    def _retained_evidence_bytes(self):
        return sum(
            record["material"]["size_bytes"]
            + sum(source["size_bytes"] for source in record["sources"])
            for record in self._evidence.values()
        )

    def _validate(self, item: NormalizedAuditItem):
        if not isinstance(item, NormalizedAuditItem) or item.item_key not in self._tasks:
            raise AuditCollectionError("item_key", "Choose an item key from read_audit_task.")
        member = self._tasks[item.item_key]
        if ("hypothesis" in member) != (item.verdict is not None):
            raise AuditCollectionError(
                "verdict", "Supply a verdict only for a task with an assigned hypothesis."
            )
        if item.verdict in {"supported", "refuted"} and (
            not item.tested_conditions or not item.selected_evidence
        ):
            raise AuditCollectionError(
                "verdict",
                "Call evidence with sources and tested conditions before a conclusive judgment.",
            )
        if item.verdict == "inconclusive" and not item.limitations:
            raise AuditCollectionError("limitations", "Explain why the check is inconclusive.")
        return item

    def _snapshot(self, items=None):
        selected = self._items if items is None else items
        return AuditSnapshot(
            self.owner,
            tuple(selected[key] for key in self.owner.item_keys if key in selected),
        )

    async def _admit(self, prospective):
        try:
            encoded = await threads.to_thread_until_done(
                partial(self._encoder.encode, self._snapshot(prospective), inputs=self.inputs),
                name="audit-admission-encode",
            )
            if len(encoded.data) + self._retained_evidence_bytes() > MAX_COLLECTED_BYTES:
                raise AuditResultError("audit_result_size_exceeded")
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

    def begin_response(self) -> None:
        self._response_revisions = {key: item.revision for key, item in self._items.items()}

    def item_base(self, item_key: str) -> int | None:
        if not isinstance(item_key, str):
            return None
        if self._response_revisions is not None:
            return self._response_revisions.get(item_key)
        item = self._items.get(item_key)
        return item.revision if item else None

    async def record(self, item: NormalizedAuditItem, *, expected_revision=_CURRENT):
        if expected_revision is _CURRENT:
            expected_revision = self.item_base(item.item_key)
        async with self._lock:
            self._check_live(writing=True)
            item = self._validate(self._with_evidence(item))
            key = item.item_key
            current = self._items.get(key)
            if current is not None and current.value == item:
                return self._receipt()
            if (current.revision if current else None) != expected_revision:
                raise AuditCollectionError(
                    "item_key",
                    "This item changed in the same response; submit again in a later turn.",
                    item_key=key,
                    current_revision=current.revision if current else None,
                )
            revision = 1 if current is None else current.revision + 1
            if revision > MAX_REVISION:
                raise AuditCollectionError("item_key", "Revision limit reached.", item_key=key)
            prospective = {**self._items, key: RecordedAuditItem(item, revision)}
            await self._admit(prospective)
            # The lock is still held: nothing changed the collection while
            # the prospective set was encoded, so the replacement is atomic.
            self._items = prospective
            return self._receipt()

    async def record_batch(self, items: tuple[NormalizedAuditItem, ...]):
        async with self._lock:
            self._check_live(writing=True)
            if not isinstance(items, tuple) or len(items) != len(self.owner.item_keys):
                raise AuditCollectionError("items", "Provide every assigned item key exactly once.")
            values = tuple(self._validate(self._with_evidence(item)) for item in items)
            keyed = {item.item_key: item for item in values}
            if len(keyed) != len(values) or set(keyed) != set(self.owner.item_keys):
                raise AuditCollectionError("items", "Provide every assigned item key exactly once.")
            values = tuple(keyed[key] for key in self.owner.item_keys)
            prospective = dict(self._items)
            for item in values:
                current = prospective.get(item.item_key)
                if current is not None and current.value != item:
                    raise AuditCollectionError(
                        "results",
                        "Existing result differs; correct it with a single-item submission.",
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
            self._response_revisions = None
            self._proposals.clear()
            self._evidence.clear()
