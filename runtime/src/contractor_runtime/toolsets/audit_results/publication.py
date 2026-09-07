"""Bounded create-only publication of an independently sealed Audit result set."""

import asyncio
import hashlib
import math
from collections.abc import Callable
from typing import Protocol

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactResponseLimitError,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef, ArtifactWriteResult
from contractor_runtime.toolsets.audit_results.contracts import (
    MAX_PACKAGE_BYTES,
    AuditInvocationOwner,
    AuditPublicationReceipt,
    AuditTrustedInputs,
    SealedAuditSnapshot,
)
from contractor_runtime.toolsets.audit_results.encoding import (
    AuditResultError,
    CanonicalAuditPackageEncoder,
)


class PublicationArtifactClient(Protocol):
    @property
    def allocation_id(self) -> str: ...

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> ArtifactWriteResult: ...

    async def read_artifact(self, ref: ArtifactRef, *, max_bytes: int) -> ArtifactValue: ...


class DeterministicAuditResultPublisher:
    def __init__(
        self,
        *,
        owner: AuditInvocationOwner,
        result_artifact: ArtifactRef,
        client: PublicationArtifactClient,
        check_active: Callable[[], None],
    ) -> None:
        """Pin the Server-validated binding and allocation-scoped transport.

        check_active is the invocation's fatal-state/fencing check. It must raise
        the existing cancellation/budget/sandbox signal unchanged. Construction
        does not grant output access; preparation and the Artifact API enforce it.
        """
        target = ArtifactRef.model_validate(result_artifact.model_dump(by_alias=True))
        if (
            not isinstance(owner, AuditInvocationOwner)
            or client.allocation_id != owner.allocation_id
            or target.revision is not None
            or target.namespace == "inputs"
            or not callable(check_active)
        ):
            raise AuditResultError("audit_result_invalid")
        self._owner = owner
        self._target = target
        self._client = client
        self._check_active = check_active
        self._encoder = CanonicalAuditPackageEncoder()

    async def publish(
        self, snapshot: SealedAuditSnapshot, *, inputs: AuditTrustedInputs, deadline: float
    ) -> AuditPublicationReceipt:
        """Return exact verified metadata or fail within two writes and two reads.

        Only a create conflict or ambiguous transport outcome permits read-back.
        An authoritative success response binds the bytes sent to its exact
        revision. A lost reply is reconciled by comparing bounded returned bytes
        and media type. Authority failures terminate without retry or read-back.
        """
        if (
            not isinstance(snapshot, SealedAuditSnapshot)
            or snapshot.owner != self._owner
            or type(deadline) not in (int, float)
            or not math.isfinite(deadline)
        ):
            raise AuditResultError("audit_result_invalid")
        try:
            self._check(deadline)
            encoded = self._encoder.encode(snapshot, inputs=inputs)
            self._check(deadline)
            async with asyncio.timeout_at(deadline):
                for _ in range(2):
                    self._check(deadline)
                    try:
                        written = await self._client.write_artifact(
                            self._target.model_copy(deep=True),
                            data=encoded.data,
                            media_type="application/zip",
                            expected_revision=None,
                        )
                        self._check(deadline)
                        if (
                            not isinstance(written, ArtifactWriteResult)
                            or not self._matches(written.artifact)
                            or written.media_type != "application/zip"
                            or written.size != len(encoded.data)
                        ):
                            raise ArtifactTransportError("invalid Audit publication receipt")
                        return await self._receipt(written.artifact, encoded.data, deadline)
                    except ArtifactResponseLimitError:
                        raise AuditResultError("audit_result_size_exceeded") from None
                    except ArtifactAPIError as error:
                        if not _reconcilable(error):
                            raise AuditResultError("audit_result_publication_failed") from None
                    except (ArtifactTransportError, TimeoutError):
                        pass
                    self._check(deadline)
                    try:
                        value = await self._client.read_artifact(
                            self._target.model_copy(deep=True),
                            max_bytes=MAX_PACKAGE_BYTES,
                        )
                        self._check(deadline)
                        if not isinstance(value, ArtifactValue) or not self._matches(
                            value.artifact
                        ):
                            raise ArtifactTransportError("invalid Audit reconciliation receipt")
                        if len(value.data) > MAX_PACKAGE_BYTES:
                            raise ArtifactResponseLimitError("Audit read-back exceeds bound")
                        if value.data != encoded.data or value.media_type != "application/zip":
                            raise AuditResultError("audit_result_publication_conflict")
                        return await self._receipt(value.artifact, encoded.data, deadline)
                    except ArtifactResponseLimitError:
                        raise AuditResultError("audit_result_size_exceeded") from None
                    except ArtifactAPIError as error:
                        if not _retryable(error) and not (
                            error.status_code == 404 and error.code == "artifact_not_found"
                        ):
                            raise AuditResultError("audit_result_publication_failed") from None
                    except (ArtifactTransportError, TimeoutError):
                        pass
                raise AuditResultError("audit_result_publication_failed", retryable=True)
        except TimeoutError:
            raise AuditResultError("audit_result_publication_failed", retryable=True) from None

    def _check(self, deadline: float) -> None:
        self._check_active()
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError

    def _matches(self, ref: ArtifactRef) -> bool:
        try:
            return (
                isinstance(ref, ArtifactRef)
                and ref.namespace == self._target.namespace
                and ref.name == self._target.name
                and isinstance(ref.revision, str)
                and 0 < len(ref.revision.encode()) <= 128
            )
        except UnicodeError:
            return False

    async def _receipt(
        self, ref: ArtifactRef, data: bytes, deadline: float
    ) -> AuditPublicationReceipt:
        receipt = AuditPublicationReceipt(
            self._owner,
            ref.namespace,
            ref.name,
            ref.revision,
            "sha256:" + hashlib.sha256(data).hexdigest(),
            len(data),
        )
        # Deliver pending cancellation even when a local transport completes inline.
        await asyncio.sleep(0)
        self._check(deadline)
        return receipt


def _retryable(error: ArtifactAPIError) -> bool:
    # An untrusted retryable flag cannot turn authority rejection into a retry.
    return (
        error.code
        not in {
            "allocation_write_fenced",
            "allocation_not_found",
            "artifact_access_denied",
            "unauthorized",
            "forbidden",
        }
        and error.status_code in (429, 500, 502, 503, 504)
        and error.retryable
    )


def _reconcilable(error: ArtifactAPIError) -> bool:
    return (error.status_code == 409 and error.code == "artifact_conflict") or _retryable(error)
