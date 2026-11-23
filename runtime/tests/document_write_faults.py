"""CAS-faithful lost-response fault injection for document toolset tests."""

from __future__ import annotations

from typing import Any

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError
from contractor_runtime.contracts import ArtifactRef


class FaultInjectingArtifactClient:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.failure: str | None = None
        self.concurrent_data = b"concurrent document"
        self.attempts: list[tuple[bytes, str, str | None]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def read_artifact(self, ref: ArtifactRef, *, max_bytes: int) -> Any:
        return await self.inner.read_artifact(ref, max_bytes=max_bytes)

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        self.attempts.append((data, media_type, expected_revision))
        failure = self.failure
        if failure in {"fenced", "forbidden", "invalid"}:
            self.failure = None
            status, code = {
                "fenced": (409, "allocation_write_fenced"),
                "forbidden": (403, "artifact_access_denied"),
                "invalid": (400, "artifact_invalid"),
            }[failure]
            raise ArtifactAPIError(status, code, False)
        if failure == "conflict":
            self.failure = None
            self.inner.seed(target.namespace, target.name, media_type, self.concurrent_data)
            raise ArtifactAPIError(409, "artifact_conflict", True)
        try:
            written = await self.inner.write_artifact(
                target,
                data=data,
                media_type=media_type,
                expected_revision=expected_revision,
            )
        except ValueError:
            raise ArtifactAPIError(409, "artifact_conflict", True) from None
        if failure in {"lost_transport", "lost_500", "diverged"}:
            self.failure = None
            if failure == "diverged":
                self.inner.seed(target.namespace, target.name, media_type, self.concurrent_data)
            if failure == "lost_500":
                raise ArtifactAPIError(500, "artifact_write_failed", True)
            raise ArtifactTransportError("write response was lost")
        return written
