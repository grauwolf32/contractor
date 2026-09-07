"""Inert immutable interfaces for invocation-local Audit collection and publication.

Only trusted Runtime code constructs these values. They are not ADK State or
model arguments. Concrete admission/encoder/collector/publication behavior is
implemented separately; defining these interfaces advertises no capability.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol

from contractor_runtime.contracts import WorkerFailure, WorkerResult

MAX_ITEMS = 64
MAX_SUMMARY_BYTES = 16 * 1024
MAX_VALUES = 512
MAX_PROPOSAL_KEYS = 128
MAX_EVIDENCE = 256
MAX_COLLECTED_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_BYTES = 16 * 1024 * 1024
MAX_REMINDERS = 2
MAX_REVISION = 2**53 - 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ASSESSMENTS = frozenset(
    {"satisfied", "violated", "supported", "refuted", "blocked", "inconclusive", "not-tested"}
)


def _identifier(value: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid bounded Audit identifier")


def _tuple(values, maximum):
    if not isinstance(values, tuple) or len(values) > maximum:
        raise ValueError("Audit collection must be an immutable bounded tuple")


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > MAX_SUMMARY_BYTES:
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
    task_package: bytes = field(repr=False)
    execution_manifest: bytes = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner):
            raise ValueError("trusted Audit inputs require invocation ownership")
        if (
            not isinstance(self.task_package, bytes)
            or not 0 < len(self.task_package) <= MAX_PACKAGE_BYTES
            or not isinstance(self.execution_manifest, bytes)
            or not 0 < len(self.execution_manifest) <= MAX_COLLECTED_BYTES
        ):
            raise ValueError("trusted Audit input bytes exceed bounds")
        if self.owner.task_set_sha256 != "sha256:" + hashlib.sha256(self.task_package).hexdigest():
            raise ValueError("trusted Audit task package changed digest")

    @property
    def execution_manifest_sha256(self):
        return "sha256:" + hashlib.sha256(self.execution_manifest).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    kind: str
    summary: str = field(repr=False)

    def __post_init__(self):
        _identifier(self.kind)
        _text(self.summary)


@dataclass(frozen=True, slots=True)
class NormalizedAuditItem:
    item_key: str
    assessment: str
    summary: str = field(repr=False)
    completed: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    evidence: tuple[AuditEvidence, ...] = ()
    proposal_keys: tuple[str, ...] = ()

    def __post_init__(self):
        _identifier(self.item_key)
        if not isinstance(self.assessment, str) or self.assessment not in ASSESSMENTS:
            raise ValueError("unknown Audit assessment")
        _text(self.summary)
        for values in (self.completed, self.gaps, self.proposal_keys):
            _tuple(values, MAX_VALUES)
            for value in values:
                _identifier(value)
            if len(set(values)) != len(values):
                raise ValueError("duplicate Audit collection identifier")
        if len(self.proposal_keys) > MAX_PROPOSAL_KEYS:
            raise ValueError("Audit proposal keys exceed the existing importer limit")
        _tuple(self.evidence, MAX_EVIDENCE)
        if any(not isinstance(value, AuditEvidence) for value in self.evidence):
            raise ValueError("Audit evidence must be normalized and immutable")


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
        if sum(len(item.value.evidence) for item in self.items) > MAX_EVIDENCE:
            raise ValueError("Audit snapshot exceeds aggregate evidence limit")


@dataclass(frozen=True, slots=True)
class SealedAuditSnapshot(AuditSnapshot):
    def __post_init__(self):
        super(SealedAuditSnapshot, self).__post_init__()
        if tuple(item.value.item_key for item in self.items) != self.owner.item_keys:
            raise ValueError("only a complete trusted-order Audit snapshot can be sealed")


@dataclass(frozen=True, slots=True)
class AuditRecordReceipt:
    """Local acceptance only; this is neither an artifact nor importer acceptance."""

    owner: AuditInvocationOwner
    revisions: tuple[tuple[str, int], ...]
    missing_item_keys: tuple[str, ...]
    status: Literal["recorded"] = "recorded"

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner) or self.status != "recorded":
            raise ValueError("invalid Audit collection receipt")
        _tuple(self.revisions, MAX_ITEMS)
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
class AuditEncodedPackage:
    """Publisher encoder's exact bytes and sizes used for prospective admission."""

    owner: AuditInvocationOwner
    data: bytes = field(repr=False)
    collected_bytes: int
    member_bytes: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.owner, AuditInvocationOwner) or not isinstance(self.data, bytes):
            raise ValueError("Audit encoding requires immutable bytes and ownership")
        if (
            type(self.collected_bytes) is not int
            or not 0 <= self.collected_bytes <= MAX_COLLECTED_BYTES
        ):
            raise ValueError("Audit collected-data budget exceeded")
        # Evidence plus manifests/check-result records; bounded without approximating ZIP overhead.
        _tuple(self.member_bytes, MAX_EVIDENCE + 3)
        if any(
            type(size) is not int or not 0 <= size <= MAX_MEMBER_BYTES for size in self.member_bytes
        ):
            raise ValueError("Audit package member size exceeded")
        if not 0 < len(self.data) <= MAX_PACKAGE_BYTES:
            raise ValueError("Audit encoded package size exceeded")


@dataclass(frozen=True, slots=True)
class AuditPublicationReceipt:
    owner: AuditInvocationOwner
    namespace: str
    name: str
    revision: str
    sha256: str
    size_bytes: int
    media_type: Literal["application/zip"] = "application/zip"

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
            or not 0 < self.size_bytes <= MAX_PACKAGE_BYTES
            or self.media_type != "application/zip"
        ):
            raise ValueError("invalid Audit publication size/media")


class AuditPackageEncoder(Protocol):
    def encode(self, snapshot: AuditSnapshot, *, inputs: AuditTrustedInputs) -> AuditEncodedPackage:
        """Return canonical package bytes with actual overhead or reject bounds.

        Admission calls this same pure function on prospective state. It must
        verify matching ownership, derive requested coverage from pinned task bytes,
        and use the pinned manifest/order, never clocks or randomness.
        """
        ...


class AuditCollector(Protocol):
    owner: AuditInvocationOwner
    inputs: AuditTrustedInputs

    async def record(
        self, item: NormalizedAuditItem, *, expected_revision: int | None = None
    ) -> AuditRecordReceipt:
        """Create/replay without revision; replace only with the current revision.

        Identical replay never increments. The immediately preceding explicit
        update may replay only with matching expected revision and exact content.
        Validate task-local evidence and prospective encoder sizes before mutation.
        """
        ...

    async def record_batch(self, items: tuple[NormalizedAuditItem, ...]) -> AuditRecordReceipt:
        """Accept one complete trusted-order batch atomically; changed existing items conflict."""
        ...

    async def snapshot(self) -> AuditSnapshot: ...
    async def seal(self) -> SealedAuditSnapshot:
        """Atomically seal a complete set; all later writes fail, including replays."""
        ...

    async def discard(self) -> None:
        """Release invocation-local state; never persist/recover partial acceptance."""
        ...


class AuditResultPublisher(Protocol):
    async def publish(
        self, snapshot: SealedAuditSnapshot, *, inputs: AuditTrustedInputs, deadline: float
    ) -> AuditPublicationReceipt:
        """Verify owner/binding and bytes under deadline and bounded CAS/read-back policy."""
        ...


@dataclass(frozen=True, slots=True)
class ContinueCompletion:
    reminder: str
    kind: Literal["continue"] = "continue"

    def __post_init__(self):
        if (
            not isinstance(self.reminder, str)
            or not self.reminder
            or len(self.reminder.encode()) > MAX_SUMMARY_BYTES
            or self.kind != "continue"
        ):
            raise ValueError("completion reminder must be bounded Runtime-authored text")


@dataclass(frozen=True, slots=True)
class CompleteCompletion:
    result: WorkerResult
    kind: Literal["complete"] = "complete"

    def __post_init__(self):
        if self.kind != "complete" or not isinstance(self.result, WorkerResult):
            raise ValueError("complete boundary requires a trusted WorkerResult")


@dataclass(frozen=True, slots=True)
class FailCompletion:
    failure: WorkerFailure
    kind: Literal["fail"] = "fail"

    def __post_init__(self):
        if self.kind != "fail" or not isinstance(self.failure, WorkerFailure):
            raise ValueError("failed boundary requires a bounded WorkerFailure")


CompletionDecision = ContinueCompletion | CompleteCompletion | FailCompletion


class WorkerCompletionBoundary(Protocol):
    async def finish(self) -> CompletionDecision:
        """Select continue/complete/fail before publishing terminal invocation State."""
        ...
