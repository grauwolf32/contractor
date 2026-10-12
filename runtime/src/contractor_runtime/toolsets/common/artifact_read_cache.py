"""One bounded, allocation-local payload for paging immutable Artifact revisions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from contractor_runtime.artifacts import MAX_ARTIFACT_BYTES, ArtifactClient, ArtifactValue
from contractor_runtime.contracts import ArtifactRef

# The Artifact API accepts payloads up to 64 MiB. Retain at most one such
# payload across Run and text artifact readers in an allocation.
MAX_CACHED_ARTIFACT_BYTES = MAX_ARTIFACT_BYTES


class ExactArtifactReadCache:
    """Serve pages of one exact revision from one download (ART-R117).

    A read that the cache cannot serve first resolves the ref's exact revision,
    media type and size through the metadata read: ``accept`` refuses the
    media type before any payload is downloaded, and a versionless ref whose
    current revision is already cached is served without downloading again.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._value: ArtifactValue | None = None

    async def read(
        self,
        client: ArtifactClient,
        ref: ArtifactRef,
        *,
        accept: Callable[[str], None],
    ) -> ArtifactValue:
        if ref.revision is not None:
            cached = await self._cached(client, ref, accept)
            if cached is not None:
                return cached
        metadata = await client.read_artifact_metadata(ref)
        accept(metadata.media_type)
        exact = metadata.artifact
        async with self._lock:
            cached = self._value
            if cached is not None and cached.artifact == exact:
                _observe(client, exact)
                return cached
            self._value = None
            value = await client.read_artifact(exact)
            if _cacheable(exact, value):
                self._value = value
            return value

    async def _cached(
        self, client: ArtifactClient, ref: ArtifactRef, accept: Callable[[str], None]
    ) -> ArtifactValue | None:
        async with self._lock:
            cached = self._value
        if cached is None or cached.artifact != ref:
            return None
        accept(cached.media_type)
        _observe(client, ref)
        return cached

    async def clear(self) -> None:
        async with self._lock:
            self._value = None


def allocation_artifact_read_cache(state: Any) -> ExactArtifactReadCache:
    cache = getattr(state, "_artifact_read_cache", None)
    if cache is None:
        cache = ExactArtifactReadCache()
        state._artifact_read_cache = cache
    return cache


def _observe(client: ArtifactClient, ref: ArtifactRef) -> None:
    observe = getattr(client, "observe_cached_read", None)
    if callable(observe):
        observe(ref)


def _cacheable(ref: ArtifactRef, value: ArtifactValue) -> bool:
    exact = value.artifact
    return (
        exact.revision is not None
        and exact.namespace == ref.namespace
        and exact.name == ref.name
        and (ref.revision is None or exact.revision == ref.revision)
        and len(value.data) <= MAX_CACHED_ARTIFACT_BYTES
    )
