"""Pure deterministic encoding shared by Audit admission and final publication."""

import io
import threading
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from contractor_runtime.core.worker_completion import WorkerCompletionError
from contractor_runtime.digests import sha256_digest
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_COLLECTED_BYTES,
    MAX_MEMBER_BYTES,
    MAX_PACKAGE_BYTES,
    AuditEncodedPackage,
    AuditSnapshot,
    AuditTrustedInputs,
)
from contractor_runtime.toolsets.audit_results.packages import (
    JSON_MEDIA_TYPE,
    PACKAGE_MEDIA_TYPE,
    _build_result_package,
    _decode_execution_manifest,
    _decode_task_input,
    _match_trusted_inputs,
    _validate_values,
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
class AuditTaskRecords:
    """One task input decoded from its exact bytes.

    The task mappings are shared by every reader in the allocation and are
    never mutated; tools hand the model deep copies.
    """

    task_package: bytes = field(repr=False)
    tasks: tuple[dict[str, Any], ...] = field(repr=False)
    package_ids: tuple[str, ...]
    package_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class DecodedAuditInputs:
    """Task records matched once with the exact execution manifest bytes."""

    records: AuditTaskRecords
    execution_manifest: bytes = field(repr=False)
    execution_manifest_digest: str
    tasks: Mapping[str, dict[str, Any]] = field(repr=False)

    @property
    def item_keys(self) -> tuple[str, ...]:
        return tuple(self.tasks)


class CanonicalAuditPackageEncoder:
    """Canonical result encoder that decodes its trusted inputs once (AWC-R125).

    Prepared completion owns one encoder per allocation and shares it with
    every invocation's collector and the finalizer: the pinned task and
    manifest bytes are decoded and matched once, while every admission still
    encodes its whole prospective set. The retained decoding is keyed by the
    exact input bytes and holds at most one input pair; other bytes replace
    it. Methods may run in worker threads, so a lock guards the retained value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: AuditTaskRecords | None = None
        self._decoded: DecodedAuditInputs | None = None

    def task_records(self, task_package: bytes) -> AuditTaskRecords:
        """Decode a task input or return its retained decoding.

        Invalid input raises the codec's ValueError unchanged; preparation
        derives the trusted owner from the returned task item keys.
        """
        with self._lock:
            return self._task_records(task_package)

    def trusted_inputs(self, inputs: AuditTrustedInputs) -> DecodedAuditInputs:
        """Return the matched decoding of ``inputs`` in its owner's exact item order."""
        if not isinstance(inputs, AuditTrustedInputs):
            raise AuditResultError("audit_result_invalid")
        try:
            with self._lock:
                decoded = self._decoded
                if (
                    decoded is None
                    or decoded.records.task_package != inputs.task_package
                    or decoded.execution_manifest != inputs.execution_manifest
                ):
                    records = self._task_records(inputs.task_package)
                    execution = _decode_execution_manifest(
                        inputs.execution_manifest, JSON_MEDIA_TYPE
                    )
                    _match_trusted_inputs(
                        list(
                            zip(
                                records.tasks,
                                records.package_ids,
                                records.package_digests,
                                strict=True,
                            )
                        ),
                        execution,
                    )
                    decoded = DecodedAuditInputs(
                        records,
                        inputs.execution_manifest,
                        sha256_digest(inputs.execution_manifest),
                        MappingProxyType({task["item_key"]: task for task in records.tasks}),
                    )
                    self._decoded = decoded
        except (ValueError, KeyError, TypeError, OverflowError):
            raise AuditResultError("audit_result_invalid") from None
        if decoded.item_keys != inputs.owner.item_keys:
            raise AuditResultError("audit_result_invalid")
        return decoded

    def _task_records(self, task_package: bytes) -> AuditTaskRecords:
        cached = self._records
        if cached is not None and cached.task_package == task_package:
            return cached
        decoded = _decode_task_input(task_package, PACKAGE_MEDIA_TYPE)
        records = AuditTaskRecords(
            task_package,
            tuple(task for task, _, _ in decoded),
            tuple(package_id for _, package_id, _ in decoded),
            tuple(sha256_digest(package) for _, _, package in decoded),
        )
        self._records = records
        return records

    def encode(self, snapshot: AuditSnapshot, *, inputs: AuditTrustedInputs) -> AuditEncodedPackage:
        """Encode prospective trusted-order data without transport.

        Empty/partial snapshots support admission only. Publication requires the
        independently sealed full set. Revision history and acceptance order are
        excluded; proposal invocation IDs retain their existing semantic meaning.
        Collected bytes count all content members, including rendered evidence;
        package/member limits additionally account for the manifest and ZIP headers.
        """
        if (
            not isinstance(snapshot, AuditSnapshot)
            or not isinstance(inputs, AuditTrustedInputs)
            or snapshot.owner != inputs.owner
        ):
            raise AuditResultError("audit_result_invalid")
        decoded = self.trusted_inputs(inputs)
        if (
            sum(
                len(value.http_capture or b"")
                for item in snapshot.items
                for value in item.value.evidence
            )
            > MAX_COLLECTED_BYTES
        ):
            raise AuditResultError("audit_result_size_exceeded")
        try:
            selected_tasks, results = [], []
            for recorded in snapshot.items:
                item = recorded.value
                # Importer requires sorted unique coverage and proposal identities.
                for name in ("completed", "gaps", "proposal_keys"):
                    _validate_values(name, list(getattr(item, name)))
                selected_tasks.append(decoded.tasks[item.item_key])
                results.append(
                    {
                        "assessment": item.assessment,
                        "summary": item.summary,
                        "completed": list(item.completed),
                        "gaps": list(item.gaps),
                        "evidence": [
                            {
                                "kind": value.kind,
                                "summary": value.summary,
                                **(
                                    {
                                        "http_capture": value.http_capture,
                                        "observed_at": value.observed_at,
                                        **(
                                            {"deployment_marker": value.deployment_marker}
                                            if value.deployment_marker is not None
                                            else {}
                                        ),
                                    }
                                    if value.http_capture is not None
                                    else {}
                                ),
                            }
                            for value in item.evidence
                        ],
                        "proposals": [
                            {"invocation_id": snapshot.owner.invocation_id, "client_key": key}
                            for key in item.proposal_keys
                        ],
                    }
                )
            payload = _build_result_package(
                tasks=selected_tasks,
                execution_digest=decoded.execution_manifest_digest,
                results=results,
            )
        except ValueError as error:
            # The builder's sole aggregate size failure is stable and local.
            code = (
                "audit_result_size_exceeded"
                if str(error) == "result package exceeds its aggregate bound"
                else "audit_result_invalid"
            )
            raise AuditResultError(code) from None
        except (KeyError, TypeError, OverflowError):
            raise AuditResultError("audit_result_invalid") from None
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            sizes = tuple(info.file_size for info in archive.infolist())
            collected = sum(
                info.file_size for info in archive.infolist() if info.filename != "manifest.json"
            )
        if (
            collected > MAX_COLLECTED_BYTES
            or any(size > MAX_MEMBER_BYTES for size in sizes)
            or len(payload) > MAX_PACKAGE_BYTES
        ):
            raise AuditResultError("audit_result_size_exceeded")
        return AuditEncodedPackage(snapshot.owner, payload, collected, sizes)
