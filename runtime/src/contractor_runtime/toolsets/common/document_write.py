"""CAS reconciliation for single-binding, allocation-owned documents."""

from __future__ import annotations

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef, ArtifactWriteResult
from contractor_runtime.toolsets.common.input_errors import ToolInputError

_DEFINITE_DENIALS = frozenset({"allocation_write_fenced", "artifact_access_denied"})


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

    def changed() -> ToolInputError:
        return ToolInputError(
            f"Document revision needs reconciliation; call {reload_tool}, then "
            f"{read_tool}, before changing it",
            code="document_changed",
        )

    async def put() -> ArtifactWriteResult:
        return await client.write_artifact(
            target,
            data=data,
            media_type=media_type,
            expected_revision=expected_revision,
        )

    try:
        return await put()
    except ArtifactAPIError as error:
        if error.code == "artifact_conflict":
            raise changed() from None
        if _definite_rejection(error):
            raise
    except ArtifactTransportError:
        pass

    # The first PUT may have committed before the response was lost. Never
    # alter its bytes or precondition for this one allowed replay.
    try:
        return await put()
    except ArtifactAPIError as error:
        if _definite_rejection(error) and error.code != "artifact_conflict":
            raise
    except ArtifactTransportError:
        pass

    try:
        current = await client.read_artifact(target, max_bytes=max_bytes)
    except ArtifactAPIError as error:
        if error.code in _DEFINITE_DENIALS:
            raise
        raise changed() from None
    except ArtifactTransportError:
        raise changed() from None
    if current.media_type == media_type and current.data == data:
        return current
    raise changed()


def _definite_rejection(error: ArtifactAPIError) -> bool:
    return error.code in _DEFINITE_DENIALS or not 500 <= error.status_code < 600
