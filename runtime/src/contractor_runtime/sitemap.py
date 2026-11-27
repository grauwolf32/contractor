"""Narrow allocation-bound Project sitemap command client.

The shared Artifact transport owns private HTTPS, request IDs and joined I/O.
Ordinary ArtifactClient reads serve the map. No command accepts a Project ID
or foreign Run scope.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactTransportError,
)
from contractor_runtime.contracts import decode_private
from contractor_runtime.contracts.sitemap import (
    SitemapMutationResponse,
    SitemapMutationResult,
    SitemapSnapshotResponse,
    SitemapSnapshotResult,
)

MAX_SITEMAP_BYTES = 32 << 20
MAX_SITEMAP_COMMAND_BYTES = 256 << 10
MAX_SITEMAP_METADATA_BYTES = 16 << 10
SITEMAP_MEDIA_TYPE = "application/vnd.contractor.project-sitemap+json"


class SitemapClient:
    def __init__(self, client: ArtifactClient) -> None:
        self._allocation_id = client.allocation_id
        self._transport = client.transport

    async def collect(
        self, batch: Mapping[str, object], key: str, *, automatic: bool = True
    ) -> SitemapMutationResult:
        response = await self._post("observations", batch, key, automatic=automatic)
        return decode_private(SitemapMutationResponse, response).result

    async def mutate(self, command: Mapping[str, object], key: str) -> SitemapMutationResult:
        response = await self._post("mutations", command, key)
        return decode_private(SitemapMutationResponse, response).result

    async def snapshot(self, selection: Mapping[str, object], key: str) -> SitemapSnapshotResult:
        response = await self._post("snapshots", selection, key)
        return decode_private(SitemapSnapshotResponse, response).result

    async def _post(
        self, operation: str, value: Mapping[str, object], key: str, *, automatic: bool = False
    ) -> bytes:
        body = await threads.to_thread_until_done(
            lambda: json.dumps(
                {
                    "apiVersion": "contractor/v1alpha1",
                    {
                        "mutations": "mutation",
                        "snapshots": "selection",
                        "observations": "input",
                    }[operation]: value,
                    **({"automatic": automatic} if operation == "observations" else {}),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
            name="sitemap-command-encode",
        )
        if len(body) > MAX_SITEMAP_COMMAND_BYTES:
            raise ArtifactAPIError(400, "invalid_request", False)
        # Reconcile one ambiguous transport loss with the same bytes and key.
        # The Server owns storage retry budgets and definite API failures.
        async with asyncio.timeout(15):
            response = None
            collected = automatic
            for attempt in range(1 if collected else 2):
                try:
                    rpc = {
                        "mutations": self._transport.mutate_sitemap,
                        "snapshots": self._transport.snapshot_sitemap,
                        "observations": self._transport.collect_sitemap,
                    }[operation]
                    response = await rpc(
                        self._allocation_id, {**json.loads(body), "idempotencyKey": key}
                    )
                    break
                except ArtifactTransportError:
                    if collected or attempt != 0:
                        raise
            if response is None:
                raise RuntimeError("sitemap operation did not produce a response")
            raw = await threads.to_thread_until_done(
                lambda: json.dumps(
                    response, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode(),
                name="sitemap-response-validate",
            )
            if len(raw) > MAX_SITEMAP_METADATA_BYTES:
                raise ArtifactTransportError("sitemap API returned invalid mutation metadata")
            return raw
