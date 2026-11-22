"""Immutable value contracts for invocation-local Audit collection and publication.

Only trusted Runtime code constructs these values. They are not ADK State or
model arguments. Concrete admission, encoding, collection and publication
behavior lives in the corresponding implementation modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, get_args

from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, is_blank
from contractor_runtime.digests import sha256_digest

MAX_ITEMS = 64
MAX_SUMMARY_BYTES = 65536
MAX_VALUES = 512
MAX_PROPOSAL_KEYS = 128
MAX_EVIDENCE = 256
MAX_COLLECTED_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = MAX_COLLECTED_BYTES
MAX_REMINDERS = 2
MAX_REVISION = MAX_EXACT_INTEGER
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
type AuditVerdict = Literal["supported", "refuted", "inconclusive"]
VERDICTS = frozenset(get_args(AuditVerdict.__value__))


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid bounded Audit identifier")


def _tuple(values, maximum):
    if not isinstance(values, tuple) or len(values) > maximum:
        raise ValueError("Audit collection must be an immutable bounded tuple")


def _text(value: str) -> None:
    if (
        not isinstance(value, str)
        or is_blank(value)
        or "\x00" in value
        or len(value.encode()) > MAX_SUMMARY_BYTES
    ):
        raise ValueError("Audit summary must be nonempty and bounded")


@dataclass(frozen=True, slots=True)
class AuditInvocationOwner:
    allocation_id: str
    invocation_id: str
    task_set_sha256: str
    item_keys: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.allocation_id)
        _identifier(self.invocation_id)
        if not isinstance(self.task_set_sha256, str) or not DIGEST.fullmatch(self.task_set_sha256):
            raise ValueError("Audit owner requires an exact task-set digest")
        _tuple(self.item_keys, MAX_ITEMS)
        for key in self.item_keys:
            _identifier(key)
        if not self.item_keys or len(set(self.item_keys)) != len(self.item_keys):
            raise ValueError("Audit assignment needs distinct ordered items")


@dataclass(frozen=True, slots=True)
class AuditTrustedInputs:
    """Exact artifact bytes pinned before tools; encoder needs requested task coverage.

    Run-scoped revision/grant checks belong to preparation. The collector and
    publisher share this immutable value, never reread mutable input aliases.
    """

    owner: AuditInvocationOwner
    context: bytes = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner):
            raise ValueError("trusted Audit inputs require invocation ownership")
        if not isinstance(self.context, bytes) or not 0 < len(self.context) <= MAX_DOCUMENT_BYTES:
            raise ValueError("trusted Audit input bytes exceed bounds")
        if self.owner.task_set_sha256 != sha256_digest(self.context):
            raise ValueError("trusted Audit context changed digest")


@dataclass(frozen=True, slots=True)
class NormalizedAuditItem:
    item_key: str
    rationale: str = field(repr=False)
    tested_conditions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    verdict: str | None = None
    selected_evidence: tuple[bytes, ...] = field(default=(), repr=False)

    def __post_init__(self):
        _identifier(self.item_key)
        if self.verdict is not None and self.verdict not in VERDICTS:
            raise ValueError("unknown Audit verdict")
        _tuple(self.selected_evidence, MAX_EVIDENCE)
        if any(not isinstance(value, bytes) for value in self.selected_evidence):
            raise ValueError("invalid selected evidence")
        _text(self.rationale)
        for values in (self.tested_conditions, self.limitations):
            _tuple(values, MAX_VALUES)
            for value in values:
                _text(value)
            if len(set(values)) != len(values):
                raise ValueError("duplicate Audit collection text")
        if any(len(value.encode()) > 512 for value in self.limitations):
            raise ValueError("Audit limitation bytes exceeded")


@dataclass(frozen=True, slots=True)
class RecordedAuditItem:
    value: NormalizedAuditItem
    revision: int

    def __post_init__(self):
        if (
            not isinstance(self.value, NormalizedAuditItem)
            or type(self.revision) is not int
            or not 1 <= self.revision <= MAX_REVISION
        ):
            raise ValueError("recorded Audit item needs a normalized value and positive revision")


@dataclass(frozen=True, slots=True)
class AuditSnapshot:
    owner: AuditInvocationOwner
    items: tuple[RecordedAuditItem, ...]

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner):
            raise ValueError("Audit snapshot requires trusted invocation ownership")
        _tuple(self.items, MAX_ITEMS)
        if any(not isinstance(item, RecordedAuditItem) for item in self.items):
            raise ValueError("Audit snapshot requires normalized revisioned items")
        keys = tuple(item.value.item_key for item in self.items)
        if keys != tuple(key for key in self.owner.item_keys if key in keys):
            raise ValueError("Audit snapshot must use distinct trusted assignment order")


@dataclass(frozen=True, slots=True)
class SealedAuditSnapshot(AuditSnapshot):
    def __post_init__(self):
        super(SealedAuditSnapshot, self).__post_init__()
        if tuple(item.value.item_key for item in self.items) != self.owner.item_keys:
            raise ValueError("only a complete trusted-order Audit snapshot can be sealed")


@dataclass(frozen=True, slots=True)
class AuditRecordReceipt:
    """Acknowledged Audit ownership; all-item completion and Run outcome are separate."""

    owner: AuditInvocationOwner
    revisions: tuple[tuple[str, int], ...]
    missing_item_keys: tuple[str, ...]
    status: Literal["recorded"] = "recorded"
    materials: tuple[bytes, ...] = field(default=(), repr=False)

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner) or self.status != "recorded":
            raise ValueError("invalid Audit collection receipt")
        _tuple(self.revisions, MAX_ITEMS)
        _tuple(self.materials, MAX_ITEMS)
        if any(not isinstance(value, bytes) for value in self.materials):
            raise ValueError("Audit receipts require exact retained materials")
        _tuple(self.missing_item_keys, MAX_ITEMS)
        for pair in self.revisions:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or pair[0] not in self.owner.item_keys
                or type(pair[1]) is not int
                or not 1 <= pair[1] <= MAX_REVISION
            ):
                raise ValueError("Audit receipt contains invalid revision")
        accepted = tuple(pair[0] for pair in self.revisions)
        if accepted != tuple(key for key in self.owner.item_keys if key in accepted):
            raise ValueError("Audit receipt revisions must follow trusted order")
        if self.missing_item_keys != tuple(
            key for key in self.owner.item_keys if key not in accepted
        ):
            raise ValueError("Audit receipt missing set is inconsistent")

    @property
    def accepted_count(self):
        return len(self.revisions)

    @property
    def total_count(self):
        return len(self.owner.item_keys)

    @property
    def complete(self):
        return not self.missing_item_keys


@dataclass(frozen=True, slots=True)
class AuditEncodedResult:
    """Publisher encoder's exact bytes and sizes used for prospective admission."""

    owner: AuditInvocationOwner
    data: bytes = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner) or not isinstance(self.data, bytes):
            raise ValueError("Audit encoding requires immutable bytes and ownership")
        if not 0 < len(self.data) <= MAX_DOCUMENT_BYTES:
            raise ValueError("Audit encoded result size exceeded")


@dataclass(frozen=True, slots=True)
class AuditPublicationReceipt:
    owner: AuditInvocationOwner
    namespace: str
    name: str
    revision: str
    sha256: str
    size_bytes: int
    media_type: Literal["application/json"] = "application/json"

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner):
            raise ValueError("Audit publication requires trusted ownership")
        if any(
            not isinstance(value, str) or not ARTIFACT_NAME.fullmatch(value)
            for value in (self.namespace, self.name)
        ):
            raise ValueError("invalid Audit publication binding")
        if (
            not isinstance(self.revision, str)
            or not self.revision
            or len(self.revision.encode()) > 128
        ):
            raise ValueError("Audit publication requires an exact revision")
        if not isinstance(self.sha256, str) or not DIGEST.fullmatch(self.sha256):
            raise ValueError("Audit publication requires a content digest")
        if (
            type(self.size_bytes) is not int
            or not 0 < self.size_bytes <= MAX_DOCUMENT_BYTES
            or self.media_type != "application/json"
        ):
            raise ValueError("invalid Audit publication size/media")
