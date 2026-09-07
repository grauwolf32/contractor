"""Pure deterministic encoding shared by Audit admission and final publication."""

import io
import zipfile

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
from contractor_runtime.worker.completion import WorkerCompletionError


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


class CanonicalAuditPackageEncoder:
    def encode(self, snapshot: AuditSnapshot, *, inputs: AuditTrustedInputs) -> AuditEncodedPackage:
        """Encode prospective trusted-order data without transport or mutable state.

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
        try:
            records = _decode_task_input(inputs.task_package, PACKAGE_MEDIA_TYPE)
            execution = _decode_execution_manifest(inputs.execution_manifest, JSON_MEDIA_TYPE)
            _match_trusted_inputs(records, execution)
            if tuple(task["item_key"] for task, _, _ in records) != snapshot.owner.item_keys:
                raise AuditResultError("audit_result_invalid")
            tasks = {task["item_key"]: task for task, _, _ in records}
            selected_tasks, results = [], []
            for recorded in snapshot.items:
                item = recorded.value
                # Importer requires sorted unique coverage and proposal identities.
                for field in ("completed", "gaps", "proposal_keys"):
                    _validate_values(field, list(getattr(item, field)))
                selected_tasks.append(tasks[item.item_key])
                results.append(
                    {
                        "assessment": item.assessment,
                        "summary": item.summary,
                        "completed": list(item.completed),
                        "gaps": list(item.gaps),
                        "evidence": [
                            {"kind": value.kind, "summary": value.summary}
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
                execution_bytes=inputs.execution_manifest,
                results=results,
                allow_empty=True,
            )
        except AuditResultError:
            raise
        except ValueError as error:
            # The legacy builder's sole aggregate size failure is stable and local.
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
