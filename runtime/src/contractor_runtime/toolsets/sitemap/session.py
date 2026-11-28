"""One allocation-owned, joined Project view and signed pagination sequence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import grpc

from contractor_runtime import threads
from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactReadTarget,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.sitemap import MAX_SITEMAP_BYTES, SITEMAP_MEDIA_TYPE, SitemapClient
from contractor_runtime.toolsets.common.cursors import (
    decode_cursor,
    encode_cursor,
    query_digest,
    require_limit,
)
from contractor_runtime.toolsets.sitemap.index import (
    MAX_PAGE_BYTES,
    MAX_PAGE_ITEMS,
    Entry,
    ViewIndex,
)


class SitemapSession:
    def __init__(
        self,
        artifacts: ArtifactClient,
        before_read: Callable[[], Awaitable[None]] | None = None,
        *,
        namespace: str = "",
        deployment: str = "",
    ) -> None:
        self.client = SitemapClient(artifacts)
        self._artifacts = artifacts
        self._before_read = before_read
        self._lock = asyncio.Lock()
        self._key = bytearray(secrets.token_bytes(32))
        self._view: ViewIndex | None = None
        self._identity = ""
        self._revision = ""
        self._closed = False
        self._editable = False
        self._pending_edit: tuple[str, dict, dict] | None = None
        self._source_id = manual_source_id(namespace, deployment)

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._pending_edit = None
            self._drop()
            self._key[:] = b"\x00" * len(self._key)

    def _drop(self) -> None:
        self._view = None
        self._editable = False
        self._identity = self._revision = ""

    @staticmethod
    def expired() -> ArtifactAPIError:
        return ArtifactAPIError(
            grpc.StatusCode.FAILED_PRECONDITION, "sitemap_cursor_expired", False
        )

    @asynccontextmanager
    async def _reading(
        self, operation: str, refresh: bool, cursor: str | None, query: str
    ) -> AsyncIterator[tuple[ViewIndex, int]]:
        async with asyncio.timeout(15):
            await self._lock.acquire()
        try:
            if self._closed or (refresh and cursor is not None):
                raise self.expired()
            if cursor is None and (refresh or self._view is None):
                # Joined indexing and the allocation lock serialize replacement
                # with release. Free the old view before building its successor.
                self._drop()
                if self._before_read is not None:
                    # Automatic publication owns its bounded checkpoint. Its
                    # exhaustion must leave the explicit read its own allowance.
                    await self._before_read()
            async with asyncio.timeout(15):
                yield await self._ready(operation, refresh, cursor, query)
        finally:
            self._lock.release()

    async def _ready(
        self, operation: str, refresh: bool, cursor: str | None, query: str
    ) -> tuple[ViewIndex, int]:
        if self._closed or (refresh and cursor is not None):
            raise self.expired()
        target = ArtifactReadTarget(scope="project", operation=operation)
        ref = ArtifactRef(namespace="sitemap", name="map")
        offset = 0
        if cursor is not None:
            try:
                value = decode_cursor(
                    self._key, cursor, {"view": str, "revision": str, "query": str, "offset": int}
                )
            except ValueError:
                raise self.expired() from None
            if (value["view"], value["revision"], value["query"]) != (
                self._identity,
                self._revision,
                query,
            ) or self._view is None:
                raise self.expired()
            offset = value["offset"]
        elif refresh or self._view is None:
            try:
                raw = await self._artifacts.read_artifact(
                    ref, max_bytes=MAX_SITEMAP_BYTES, target=target
                )
            except ArtifactAPIError as error:
                # Only a missing current binding is an empty domain view.
                # Project/operation failures have separate Server error codes.
                if error.code != "artifact_not_found":
                    raise
                data = (
                    b'{"version":"project-sitemap@1","projection":"sitemap-safe-projection@1",'
                    b'"groups":[],"sources":[],"rules":[],"markers":[],"gaps":[]}'
                )
                revision = "absent"
            else:
                if raw.media_type != SITEMAP_MEDIA_TYPE:
                    raise ArtifactTransportError("sitemap artifact has an invalid media type")
                data, revision = raw.data, raw.artifact.require_exact().revision
            view = await threads.to_thread_until_done(
                lambda: ViewIndex(data), name="sitemap-view-index"
            )
            self._view = view
            self._editable = True
            self._identity, self._revision = str(uuid4()), revision
        await self._artifacts.check_artifact_access(ref, target=target)
        if self._view is None:
            raise self.expired()
        return self._view, offset

    def _cursor(self, query: str, offset: int) -> str:
        return encode_cursor(
            self._key,
            {"view": self._identity, "revision": self._revision, "query": query, "offset": offset},
        )

    def _page(self, rows: list[dict], query: str, offset: int | None) -> dict[str, Any]:
        result = {
            "revision": self._revision,
            "entries": rows,
            "nextCursor": self._cursor(query, offset) if offset is not None else None,
        }
        if (
            len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
            > MAX_PAGE_BYTES
        ):
            raise ToolInputError("sitemap page exceeds its byte bound")
        return result

    def _parts_page(
        self,
        query: str,
        offset: int,
        count: int,
        limit: int,
        load: Callable[[int], dict | None],
    ) -> dict:
        if offset > count:
            raise self.expired()
        rows, scanned = [], 0
        budget = MAX_PAGE_BYTES - 4096
        while offset < count and len(rows) < limit and scanned < MAX_PAGE_ITEMS:
            row = load(offset)
            if row is not None:
                size = len(json.dumps(row, ensure_ascii=False).encode())
                if size > budget:
                    break
                rows.append(row)
                budget -= size
            offset += 1
            scanned += 1
        return self._page(rows, query, offset if offset < count else None)

    async def list(
        self,
        *,
        cursor: str | None,
        refresh: bool,
        limit: int,
        method: str | None,
        origin: str | None,
        path_prefix: str | None,
        include_suppressed: bool,
        scope: str = "entries",
    ) -> dict:
        limit = require_limit(limit, MAX_PAGE_ITEMS, lambda: ToolInputError("limit must be 1..100"))
        if scope not in ("entries", "metadata"):
            raise ToolInputError("scope must be entries or metadata")
        if scope == "metadata" and (
            any(value is not None for value in (method, origin, path_prefix)) or include_suppressed
        ):
            raise ToolInputError("metadata scope does not accept entry filters")
        for text in (method, origin, path_prefix):
            if text is not None and (not isinstance(text, str) or len(text.encode()) > 8192):
                raise ToolInputError("sitemap filters must be bounded strings")
        filters = {
            "operation": "list",
            "scope": scope,
            "method": method,
            "origin": origin,
            "pathPrefix": path_prefix,
            "includeSuppressed": include_suppressed,
            "limit": limit,
        }
        query = query_digest(filters)
        async with self._reading("list_sitemap_entries", refresh, cursor, query) as (view, offset):
            if scope == "metadata":
                return self._parts_page(
                    query,
                    offset,
                    len(view.metadata),
                    limit,
                    lambda position: {
                        "kind": view.metadata[position][0],
                        "value": view.load(view.metadata[position][1]),
                    },
                )
            # Indexed headers only, and at most 100 inspected candidates per page.
            identities = view.entry_ids
            if offset > len(identities):
                raise self.expired()
            rows: list[dict] = []
            budget = MAX_PAGE_BYTES - 4096
            scanned = 0
            while offset < len(identities) and len(rows) < limit and scanned < MAX_PAGE_ITEMS:
                identity = identities[offset]
                entry = view.entries[identity]
                row = {key: view.load(span) for key, span in entry.fields.items()}
                row["memberCount"] = len(entry.members)
                row["activeMemberCount"] = sum(
                    member not in view.suppressed for member in entry.members
                )
                eligible = (
                    (include_suppressed or not entry.members or row["activeMemberCount"] > 0)
                    and (method is None or row["method"] == method)
                    and (origin is None or row["origin"] == origin)
                    and (path_prefix is None or row["route"].startswith(path_prefix))
                )
                if eligible:
                    size = len(json.dumps(row, ensure_ascii=False).encode())
                    if size > budget:
                        break
                    rows.append(row)
                    budget -= size
                scanned += 1
                offset += 1
            return self._page(rows, query, offset if offset < len(identities) else None)

    async def read(
        self,
        identity: str,
        *,
        cursor: str | None,
        refresh: bool,
        limit: int,
        include_suppressed: bool,
    ) -> dict:
        limit = require_limit(limit, MAX_PAGE_ITEMS, lambda: ToolInputError("limit must be 1..100"))
        if not isinstance(identity, str) or len(identity) > 128:
            raise ToolInputError("entry_id must be an ID returned by list_sitemap_entries")
        query = query_digest(
            {
                "operation": "read",
                "entry": identity,
                "limit": limit,
                "includeSuppressed": include_suppressed,
            }
        )
        async with self._reading("read_sitemap_entry", refresh, cursor, query) as (view, offset):
            entry = view.entries.get(identity)
            if entry is None:
                raise ArtifactAPIError(grpc.StatusCode.NOT_FOUND, "artifact_not_found", False)
            return self._parts_page(
                query,
                offset,
                len(entry.parts) // 4 + len(entry.markers) // 2,
                limit,
                lambda position: self._read_row(view, entry, position, include_suppressed),
            )

    @staticmethod
    def _read_row(
        view: ViewIndex, entry: Entry, offset: int, include_suppressed: bool
    ) -> dict | None:
        part_count = len(entry.parts) // 4
        if offset >= part_count:
            marker = (offset - part_count) * 2
            span = (entry.markers[marker], entry.markers[marker + 1])
            return {"kind": "marker", "value": view.load(span)}
        kind, member, start, end = entry.parts[offset * 4 : offset * 4 + 4]
        identity = None if member == 0xFFFFFFFF else entry.members[member]
        if identity is not None and identity in view.suppressed and not include_suppressed:
            return None
        names = ("member", "summary", "annotation", "example", "evidence", "declaration")
        return {
            "kind": names[kind],
            "memberId": identity,
            "suppressed": identity in view.suppressed,
            "value": {"path": view.load((start, end)), "omitted": entry.member_omitted[member]}
            if kind == 0
            else view.load((start, end)),
        }

    async def edit(self, command: Mapping[str, object], key: str | None) -> dict:
        async with self._lock:
            if self._closed:
                raise self.expired()
            pending = self._pending_edit
            if pending is not None:
                if (key is not None and pending[0] != key) or pending[1] != command:
                    raise ToolInputError(
                        "Reconcile the pending edit using the same arguments and key"
                    )
                key, prepared = pending[0], pending[2]
            else:
                if self._view is None or not self._editable:
                    raise ToolInputError("Read/list the sitemap with refresh=true before editing")
                key = key or str(uuid4())
                prepared = dict(command, expectedRevision=self._revision)
                if command["operation"] == "replace-source":
                    source = dict(command["source"])
                    prior = self._view.sources.get(self._source_id)
                    if prior is not None:
                        source["expected"] = self._view.load(prior)
                    prepared["source"] = source
                self._pending_edit = (key, dict(command), prepared)
            try:
                result = await self.client.mutate(prepared, key)
            except ArtifactAPIError as error:
                if not error.retryable and not error.server_failure:
                    self._pending_edit = None
                raise
            else:
                self._pending_edit = None
            finally:
                # Old cursors page their immutable view, not new edit authority.
                self._editable = False
            return result.model_dump(by_alias=True)

    async def mutate(self, command: Mapping[str, object], key: str) -> dict:
        async with self._lock:
            if self._closed:
                raise self.expired()
            result = await self.client.mutate(command, key)
            return result.model_dump(by_alias=True)

    async def dispatch_import(self, operation: Callable[[], Awaitable[dict]]) -> dict:
        async with self._lock:
            if self._closed:
                raise self.expired()
            return await operation()

    async def snapshot(
        self, groups: list[str], members: list[str], require_inputs: bool, key: str
    ) -> dict:
        async with self._lock:
            if self._closed or self._view is None or self._revision == "absent":
                raise self.expired()
            result = await self.client.snapshot(
                {
                    "revision": self._revision,
                    "groups": groups,
                    "members": members,
                    "requireReferencedInputs": require_inputs,
                },
                key,
            )
            return result.model_dump(by_alias=True)


def manual_source_id(namespace: str, deployment: str = "") -> str:
    # Shared identity fixture: api/testdata/sitemap-editor-source-id.json.
    digest = hashlib.sha256()
    for part in ("manual", deployment, namespace):
        encoded = part.encode("utf-8")
        digest.update(str(len(encoded)).encode() + b":" + encoded)
    return digest.hexdigest()
