"""Private publisher preparation over explicitly selected immutable evidence."""

from __future__ import annotations

import asyncio
from typing import Any

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.finding_anchors import (
    MAX_SOURCE_BYTES,
    derive_zip,
    unavailable,
)


async def prepare_anchors(client: ArtifactClient, request: dict[str, Any]) -> dict[str, Any]:
    preparation = asyncio.timeout(30)
    try:
        async with preparation:
            return await _prepare_anchors(client, request)
    except TimeoutError:
        if not preparation.expired():
            raise
        return unavailable("input_limit")


async def _prepare_anchors(client: ArtifactClient, request: dict[str, Any]) -> dict[str, Any]:
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
        payload = await client.read_artifact(selected.artifact, max_bytes=MAX_SOURCE_BYTES)
    except ArtifactResponseLimitError:
        return unavailable("input_limit")
    if payload.media_type != selected.media_type or len(payload.data) != selected.size:
        raise ArtifactTransportError("finding anchor source metadata differs from exact content")
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
