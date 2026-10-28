"""Narrow allocation-bound Project sitemap API client.

The shared Artifact transport owns private HTTPS, request IDs and joined I/O.
No method accepts a Project ID or foreign Run scope.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlencode

from contractor_runtime import threads
from contractor_runtime.adapters.otlp_retry import retry_after_seconds
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
    allocation_path,
    raise_for_status,
    require_json,
)
from contractor_runtime.contracts import decode_private
from contractor_runtime.contracts.base import RESOURCE_ID_PATTERN
from contractor_runtime.contracts.sitemap import (
    SitemapAccess,
    SitemapMutationResponse,
    SitemapMutationResult,
    SitemapSnapshotResponse,
    SitemapSnapshotResult,
)

MAX_SITEMAP_BYTES = 32 << 20
MAX_SITEMAP_COMMAND_BYTES = 256 << 10
MAX_SITEMAP_METADATA_BYTES = 16 << 10
SITEMAP_MEDIA_TYPE = "application/vnd.contractor.project-sitemap+json"


@dataclass(frozen=True, slots=True)
class SitemapViewBytes:
    revision: str
    data: bytes = field(repr=False)


class SitemapClient:
    def __init__(self, client: ArtifactClient) -> None:
        self._root = allocation_path(client.allocation_id, "sitemap")
        self._transport = client.transport

    async def access(self) -> SitemapAccess:
        path = self._root + "/access"
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": "application/json"},
            body=b"",
            max_response_bytes=MAX_SITEMAP_METADATA_BYTES,
        )
        raise_for_status(response)
        require_json(response.headers)
        if response.status_code != 200 or len(response.body) > MAX_SITEMAP_METADATA_BYTES:
            raise ArtifactTransportError("sitemap API returned invalid access metadata")
        return decode_private(SitemapAccess, response.body)

    async def view(self, revision: str | None = None) -> SitemapViewBytes:
        path = self._root
        if revision is not None:
            if RESOURCE_ID_PATTERN.fullmatch(revision) is None or revision == "absent":
                raise ValueError("invalid exact sitemap revision")
            path += "?" + urlencode({"revision": revision})
        response = await self._transport.request(
            "GET",
            path,
            headers={"Accept": SITEMAP_MEDIA_TYPE},
            body=b"",
            max_response_bytes=MAX_SITEMAP_BYTES,
        )
        raise_for_status(response)
        actual = response.headers.get("x-contractor-sitemap-revision", "")
        if (
            response.status_code != 200
            or response.headers.get("content-type") != SITEMAP_MEDIA_TYPE
            or RESOURCE_ID_PATTERN.fullmatch(actual) is None
            or (revision is not None and revision != actual)
        ):
            raise ArtifactTransportError("sitemap API returned invalid view metadata")
        if len(response.body) > MAX_SITEMAP_BYTES:
            raise ArtifactResponseLimitError("sitemap view exceeds its byte bound")
        return SitemapViewBytes(actual, response.body)

    async def collect(self, batch: Mapping[str, object], key: str) -> SitemapMutationResult:
        response = await self._post("observations", batch, key)
        return decode_private(SitemapMutationResponse, response).result

    async def mutate(self, command: Mapping[str, object], key: str) -> SitemapMutationResult:
        response = await self._post("mutations", command, key)
        return decode_private(SitemapMutationResponse, response).result

    async def snapshot(self, selection: Mapping[str, object], key: str) -> SitemapSnapshotResult:
        response = await self._post("snapshots", selection, key)
        return decode_private(SitemapSnapshotResponse, response).result

    async def _post(self, operation: str, value: Mapping[str, object], key: str) -> bytes:
        body = await threads.to_thread_until_done(
            lambda: json.dumps(
                {
                    "apiVersion": "contractor/v1alpha1",
                    {"mutations": "mutation", "snapshots": "selection", "observations": "input"}[
                        operation
                    ]: value,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode(),
            name="sitemap-command-encode",
        )
        if len(body) > MAX_SITEMAP_COMMAND_BYTES:
            raise ValueError("sitemap command exceeds its byte bound")
        # Reconcile one ambiguous transport loss with the same bytes and key.
        # The Server owns storage retry budgets and definite API failures.
        async with asyncio.timeout(15):
            response = None
            for attempt in range(1 if operation == "observations" else 2):
                try:
                    response = await self._transport.request(
                        "POST",
                        self._root + "/" + operation,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                            "Idempotency-Key": key,
                        },
                        body=body,
                        max_response_bytes=MAX_SITEMAP_METADATA_BYTES,
                    )
                    break
                except ArtifactTransportError:
                    if operation == "observations" or attempt != 0:
                        raise
            if response is None:
                raise RuntimeError("sitemap operation did not produce a response")
            try:
                raise_for_status(response)
            except ArtifactAPIError as error:
                error.retry_after_seconds = retry_after_seconds(response.headers.get("retry-after"))
                raise
            require_json(response.headers)
            if response.status_code != 200 or len(response.body) > MAX_SITEMAP_METADATA_BYTES:
                raise ArtifactTransportError("sitemap API returned invalid mutation metadata")
            return response.body
