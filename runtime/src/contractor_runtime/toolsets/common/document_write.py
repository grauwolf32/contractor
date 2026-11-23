"""CAS reconciliation for single-binding, allocation-owned documents."""

from __future__ import annotations

from contractor_runtime.artifacts import (
    ARTIFACT_HISTORY_MAINTENANCE_PENDING,
    ArtifactAPIError,
    ArtifactClient,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef, ArtifactWriteResult
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.toolsets.common.artifact_write import (
    AUTHORITY_DENIALS,
    ArtifactReconciliationFailed,
    definite_write_rejection,
    write_artifact_exact,
)


class DocumentWriteUnconfirmed(ToolInputError):
    """A CAS conflict is distinct from unresolved storage acknowledgement loss."""

    def __init__(self, message: str, *, conflict: bool) -> None:
        super().__init__(message, code="document_changed")
        self.conflict = conflict


async def write_document_exact(
    client: ArtifactClient,
    target: ArtifactRef,
    *,
    data: bytes,
    media_type: str,
    expected_revision: str | None,
    max_bytes: int,
    reload_tool: str,
    read_tool: str,
) -> ArtifactWriteResult | ArtifactValue:
    """Replay an ambiguous PUT once, then adopt only byte-identical current state."""

    def changed(*, conflict: bool = False) -> DocumentWriteUnconfirmed:
        return DocumentWriteUnconfirmed(
            f"Document revision needs reconciliation; call {reload_tool}, then "
            f"{read_tool}, before changing it",
            conflict=conflict,
        )

    try:
        return await write_artifact_exact(
            client,
            target,
            data=data,
            media_type=media_type,
            expected_revision=expected_revision,
            max_bytes=max_bytes,
            reject_first=definite_write_rejection,
            reject_replay=lambda error: (
                error.code == ARTIFACT_HISTORY_MAINTENANCE_PENDING
                or (definite_write_rejection(error) and error.code != "artifact_conflict")
            ),
        )
    except ArtifactAPIError as error:
        if error.code == "artifact_conflict":
            raise changed(conflict=True) from None
        raise
    except ArtifactReconciliationFailed as error:
        read_error = error.read_error
        if isinstance(read_error, ArtifactAPIError) and read_error.code in AUTHORITY_DENIALS:
            raise read_error from None
        raise changed(
            conflict=error.current is not None
            and error.current.artifact.revision != expected_revision
        ) from None
