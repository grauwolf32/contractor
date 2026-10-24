"""Private publisher preparation over explicitly selected immutable evidence."""

from __future__ import annotations

import asyncio
from typing import Any

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactReadResult, ArtifactRef
from contractor_runtime.toolsets.common.finding_anchors import (
    MAX_SOURCE_BYTES,
    derive_zip,
    unavailable,
)


class ExactSourceCache:
    """Allocation-owned bytes of the last selected exact source ZIP (FND-R184).

    At most one payload of at most 8 MiB (LIM-FND-38) is retained. Each
    finding still resolves the metadata of every selected ref; the bytes are
    reused only for the same exact ref with the media type and size that this
    finding's metadata reports, and the reuse is recorded as a content
    observation of that exact revision like a download. Concurrent findings of
    one ref share one download. A failed or cancelled download retains
    nothing, so the next finding reads the source itself. Parsed trees and
    derivations are never retained.
    """

    def __init__(self) -> None:
        self._value: ArtifactValue | None = None
        self._loading: dict[tuple[str, str, str | None], asyncio.Event] = {}
        self._closed = False

    async def read(self, client: ArtifactClient, selected: ArtifactReadResult) -> ArtifactValue:
        exact = selected.artifact
        key = (exact.namespace, exact.name, exact.revision)
        while True:
            value = self._value
            if (
                value is not None
                and value.artifact == exact
                and value.media_type == selected.media_type
                and len(value.data) == selected.size
            ):
                client.observe_cached_read(value.artifact)
                return value
            loading = self._loading.get(key)
            if loading is None:
                break
            # Another finding is downloading this exact source; use its outcome.
            await loading.wait()
        done = asyncio.Event()
        self._loading[key] = done
        try:
            payload = await client.read_artifact(exact, max_bytes=MAX_SOURCE_BYTES)
            if payload.media_type != selected.media_type or len(payload.data) != selected.size:
                raise ArtifactTransportError(
                    "finding anchor source metadata differs from exact content"
                )
            if payload.artifact == exact and not self._closed:
                # One entry: replacing it releases any other exact source.
                self._value = payload
            return payload
        finally:
            del self._loading[key]
            done.set()

    def close(self) -> None:
        """Release the retained bytes; a closed cache never retains again."""
        self._closed = True
        self._value = None


async def prepare_anchors(
    client: ArtifactClient, request: dict[str, Any], sources: ExactSourceCache
) -> dict[str, Any]:
    preparation = asyncio.timeout(30)
    try:
        async with preparation:
            return await _prepare_anchors(client, request, sources)
    except TimeoutError:
        if not preparation.expired():
            raise
        return unavailable("input_limit")


async def _prepare_anchors(
    client: ArtifactClient, request: dict[str, Any], sources: ExactSourceCache
) -> dict[str, Any]:
    proposal = request["proposal"]
    locations = proposal.get("locations", [])
    if not locations:
        return unavailable("not_provided")
    if any(
        "file" not in location or ("line" not in location and "range" not in location)
        for location in locations
    ):
        return unavailable("selector_unavailable")
    refs = request["evidenceRefs"]
    if len(refs) > 32 or len(locations) > 32:
        return unavailable("input_limit")
    selected = None
    for raw in refs:
        ref = ArtifactRef.model_validate(raw).require_exact()
        metadata = await client.read_artifact_metadata(ref)
        if metadata.media_type == "application/zip":
            if selected is not None:
                return unavailable("asset_unavailable")
            selected = metadata
    if selected is None:
        return unavailable("asset_unavailable")
    if selected.size > MAX_SOURCE_BYTES:
        return unavailable("input_limit")
    try:
        payload = await sources.read(client, selected)
    except ArtifactResponseLimitError:
        return unavailable("input_limit")
    standards = proposal["standard_refs"]
    qualify = (
        len(locations) == 1
        and len(standards) == 1
        and standards[0]["scheme"] == "CWE"
        and standards[0]["requirement_id"] == "CWE-321"
        and "httpEvidence" not in request
    )
    return await threads.to_thread_until_done(
        lambda: derive_zip(
            payload.data, selected.artifact.model_dump(), locations, qualify_local=qualify
        ),
        name="finding-anchor-derive",
    )
