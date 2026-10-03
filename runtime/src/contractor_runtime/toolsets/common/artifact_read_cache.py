"""One bounded, allocation-local payload for paging immutable Artifact revisions."""

from __future__ import annotations

import asyncio
from typing import Any

from contractor_runtime.artifacts import MAX_ARTIFACT_BYTES, ArtifactClient, ArtifactValue
from contractor_runtime.contracts import ArtifactRef

# The Artifact API accepts payloads up to 64 MiB. Retain at most one such
# payload across Run and text artifact readers in an allocation.
MAX_CACHED_ARTIFACT_BYTES = MAX_ARTIFACT_BYTES


class ExactArtifactReadCache:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._value: ArtifactValue | None = None

    async def read(self, client: ArtifactClient, ref: ArtifactRef) -> ArtifactValue:
        if ref.revision is None:
            # A versionless binding can advance at any time. Always resolve it
            # through the API, but retain the exact revision returned for pages.
            value = await client.read_artifact(ref)
            if _cacheable(ref, value):
                async with self._lock:
                    self._value = value
            return value

        async with self._lock:
            cached = self._value
            if cached is not None and cached.artifact == ref:
                observe = getattr(client, "observe_cached_read", None)
                if callable(observe):
                    observe(ref)
                return cached
            self._value = None
            value = await client.read_artifact(ref)
            if _cacheable(ref, value):
                self._value = value
            return value

    async def clear(self) -> None:
        async with self._lock:
            self._value = None


def allocation_artifact_read_cache(state: Any) -> ExactArtifactReadCache:
    cache = getattr(state, "_artifact_read_cache", None)
    if cache is None:
        cache = ExactArtifactReadCache()
        state._artifact_read_cache = cache
    return cache


def _cacheable(ref: ArtifactRef, value: ArtifactValue) -> bool:
    exact = value.artifact
    return (
        exact.revision is not None
        and exact.namespace == ref.namespace
        and exact.name == ref.name
        and (ref.revision is None or exact.revision == ref.revision)
        and len(value.data) <= MAX_CACHED_ARTIFACT_BYTES
    )
