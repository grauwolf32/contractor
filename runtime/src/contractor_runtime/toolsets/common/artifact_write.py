"""Bounded exact-byte CAS recovery; domain errors remain with each consumer."""

from __future__ import annotations

from collections.abc import Callable

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef, ArtifactWriteResult

AUTHORITY_DENIALS = frozenset({"allocation_write_fenced", "artifact_access_denied"})


def definite_write_rejection(error: ArtifactAPIError) -> bool:
    return error.code in AUTHORITY_DENIALS or not (error.server_failure or error.transient_failure)


class ArtifactReconciliationFailed(Exception):
    """Keep a failed current read distinct from a proven payload mismatch."""

    def __init__(
        self,
        *,
        current: ArtifactValue | None = None,
        read_error: ArtifactAPIError | ArtifactTransportError | None = None,
    ) -> None:
        super().__init__("Artifact write could not be reconciled")
        self.current = current
        self.read_error = read_error


async def write_artifact_exact(
    client: ArtifactClient,
    target: ArtifactRef,
    *,
    data: bytes,
    media_type: str,
    expected_revision: str | None,
    max_bytes: int,
    reject_first: Callable[[ArtifactAPIError], bool],
    reject_replay: Callable[[ArtifactAPIError], bool],
) -> ArtifactWriteResult | ArtifactValue:
    """PUT once, replay ambiguous bytes once, then inspect bounded current state.

    A definite first rejection propagates directly. Consumers decide whether a
    rejected replay is authoritative or requires a current read, and translate
    reconciliation failures into their own errors. Cancellation propagates.
    """

    async def put() -> ArtifactWriteResult:
        return await client.write_artifact(
            target, data=data, media_type=media_type, expected_revision=expected_revision
        )

    try:
        return await put()
    except ArtifactAPIError as error:
        if reject_first(error):
            raise
    except ArtifactTransportError:
        pass

    try:
        return await put()
    except ArtifactAPIError as error:
        if reject_replay(error):
            raise
    except ArtifactTransportError:
        pass

    try:
        current = await client.read_artifact(target, max_bytes=max_bytes)
    except (ArtifactAPIError, ArtifactTransportError) as error:
        raise ArtifactReconciliationFailed(read_error=error) from None
    if current.media_type == media_type and current.data == data:
        return current
    raise ArtifactReconciliationFailed(current=current)
