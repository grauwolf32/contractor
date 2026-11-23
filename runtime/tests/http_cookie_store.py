"""Strict test-only Artifact CAS for HTTP unit tests; process tests use PostgreSQL."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactResponseLimitError
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.http.tools import (
    HTTPToolsetFactory as ProductionHTTPToolsetFactory,
)


class CookieArtifactStore:
    def __init__(self) -> None:
        self.current: dict[tuple[str, str], Any] = {}
        self.writes = 0
        self.reads = 0

    async def read_artifact(self, ref: ArtifactRef, *, max_bytes: int) -> Any:
        self.reads += 1
        value = self.current.get((ref.namespace, ref.name))
        if value is None or (ref.revision is not None and ref.revision != value.artifact.revision):
            raise ArtifactAPIError(404, "artifact_not_found", False)
        if len(value.data) > max_bytes:
            raise ArtifactResponseLimitError("response too large")
        return value

    async def write_artifact(
        self,
        target: ArtifactRef,
        *,
        data: bytes,
        media_type: str,
        expected_revision: str | None,
    ) -> Any:
        if target.revision is not None:
            raise ValueError("write target must be versionless")
        previous = self.current.get((target.namespace, target.name))
        actual = None if previous is None else previous.artifact.revision
        if actual != expected_revision:
            raise ArtifactAPIError(409, "artifact_conflict", False)
        self.writes += 1
        value = SimpleNamespace(
            artifact=target.model_copy(update={"revision": f"cookie-{self.writes}"}),
            data=data,
            media_type=media_type,
        )
        self.current[(target.namespace, target.name)] = value
        return value


class HTTPToolsetFactory(ProductionHTTPToolsetFactory):
    """Each ordinary unit-test factory owns an isolated, strict Artifact store."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        store = CookieArtifactStore()
        kwargs.setdefault("artifact_client_factory", lambda *_: store)
        super().__init__(*args, **kwargs)
