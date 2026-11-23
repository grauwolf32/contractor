"""Prepare findings documents in the current Run and expose a bounded snapshot reader."""

from __future__ import annotations

import asyncio
import base64
import copy
import re
from typing import Any

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
    ArtifactValue,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics
from contractor_runtime.digests import sha256_digest
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.security_findings.collection import (
    COLLECTION_MEDIA_TYPE,
    MAX_ARCHIVE_BYTES,
    FindingCollection,
    FindingsError,
    canonical,
    decode_collection,
    identifier,
    require,
    strict_json,
)

MAX_PAGE_BYTES = 256 * 1024


async def prepare_reader(
    client: ArtifactClient,
    metrics: ToolMetrics,
    secrets: tuple[str, ...],
    *,
    source_ref: ArtifactRef | None = None,
) -> ListFindingsTool:
    try:
        source = await client.read_artifact(
            source_ref or ArtifactRef(namespace="inputs", name="findings"),
            max_bytes=MAX_ARCHIVE_BYTES,
        )
    except ArtifactResponseLimitError as error:
        raise FindingsError("findings_limit_exceeded") from error
    except (ArtifactAPIError, ArtifactTransportError) as error:
        raise FindingsError(
            "findings_collection_unavailable", retryable=_retryable(error)
        ) from error
    require(source.media_type == COLLECTION_MEDIA_TYPE)
    # Validate every member before creating any destination or exposing any tool.
    collection = await to_thread_until_done(
        decode_collection, source.data, name="findings-collection-decode"
    )
    documents = {}
    for document in collection.metadata["documents"]:
        target = ArtifactRef(namespace=collection.namespace, name=document["id"])
        data = collection.contents[document["id"]]
        for attempt in range(2):
            try:
                ambiguous_write = False
                try:
                    written = await client.write_artifact(
                        target,
                        data=data,
                        media_type=document["media_type"],
                        expected_revision=None,
                    )
                except ArtifactResponseLimitError:
                    raise
                except ArtifactAPIError as error:
                    if error.status_code not in {409, 412}:
                        raise
                except ArtifactTransportError:
                    # A write may have committed before its response was lost.
                    ambiguous_write = True
                else:
                    ref = written.artifact.require_exact()
                    require(
                        (ref.namespace, ref.name) == (target.namespace, target.name),
                        "findings_document_conflict",
                    )
                    break

                try:
                    retained = await client.read_artifact(target, max_bytes=MAX_ARCHIVE_BYTES)
                except ArtifactAPIError as error:
                    if (
                        ambiguous_write
                        and attempt == 0
                        and error.status_code == 404
                        and error.code == "artifact_not_found"
                    ):
                        # The first write did not commit; one create retry is safe.
                        continue
                    raise
                ref = await to_thread_until_done(
                    _verify_document,
                    retained,
                    target,
                    document,
                    data,
                    name="findings-document-verify",
                )
                # Confirm the discovered revision before publishing its exact ref.
                confirmed = await client.read_artifact(ref, max_bytes=MAX_ARCHIVE_BYTES)
                await to_thread_until_done(
                    _verify_document,
                    confirmed,
                    ref,
                    document,
                    data,
                    name="findings-exact-document-verify",
                )
                break
            except ArtifactResponseLimitError as error:
                raise FindingsError("findings_limit_exceeded") from error
            except (ArtifactAPIError, ArtifactTransportError) as error:
                raise FindingsError(
                    "findings_document_unavailable", retryable=_retryable(error)
                ) from error
        else:
            raise FindingsError("findings_document_unavailable")
        documents[document["id"]] = {
            "ref": ref.model_dump(exclude_none=True),
            "digest": document["digest"],
            "media_type": document["media_type"],
            "size_bytes": document["size_bytes"],
            "source": {"scope": document["scope"], "ref": document["ref"]},
        }
    items = await to_thread_until_done(
        _project_items, collection, documents, name="findings-reader-projection"
    )
    # Only previews and metadata survive preparation. Full bytes stay in artifacts.
    return ListFindingsTool(items, collection.digest, metrics, secrets)


def _verify_document(
    value: ArtifactValue, expected: ArtifactRef, document: dict[str, Any], data: bytes
) -> ArtifactRef:
    ref = value.artifact.require_exact()
    require(
        (ref.namespace, ref.name) == (expected.namespace, expected.name)
        and (expected.revision is None or ref == expected)
        and value.media_type == document["media_type"]
        and len(value.data) == document["size_bytes"]
        and sha256_digest(value.data) == document["digest"]
        and value.data == data,
        "findings_document_conflict",
    )
    return ref


def _project_items(
    collection: FindingCollection, documents: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    items = []
    for entry in collection.metadata["entries"]:
        proposal = collection.proposals[entry["proposal_document_id"]]
        item = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "proposal_document_id",
                "evidence",
            }
        }
        item.update(
            {
                "subject": proposal["subject"],
                "title_preview": _preview(proposal["title"], 512),
                "description_preview": _preview(proposal["description"], 2048),
                "has_hypothesis": bool(proposal.get("hypothesis")),
                "proposal": documents[entry["proposal_document_id"]],
                "evidence": [
                    {"evidence_id": link["evidence_id"], **documents[link["document_id"]]}
                    for link in entry["evidence"]
                ],
            }
        )
        if "audit_origin" in item:
            item["audit_origin"] = dict(item["audit_origin"])
            context_id = item["audit_origin"].pop("hypothesis_snapshot_document_id", None)
            if context_id is not None:
                item["audit_origin"]["hypothesis_snapshot"] = documents[context_id]
        items.append(item)
    return tuple(items)


class ListFindingsTool:
    name = "list_findings"
    description = """List finding proposals from this Run's prepared collection.

    Returns bounded previews and exact current-Run artifact references. Use
    read_artifact with proposal.ref or evidence[].ref for full content. Source
    refs are provenance only. Captured Project Finding metadata describes the collection snapshot;
    listing does not confirm, deduplicate or change a finding.

    Args:
        subject_kind: Optional exact, case-sensitive subject kind identifier.
        subject_key: Optional exact subject key; requires subject_kind.
        limit: Maximum items to return, from 1 to 100; defaults to 20. The byte
            limit may shorten a page. Follow next_cursor until it is null.
        cursor: Opaque next_cursor from this collection with the same filters.
            Omit for the first page. The limit may change between pages.

    Returns:
        items with receipt identity, subject, title and description previews,
        hypothesis presence, proposal and evidence refs, and the captured
        Project Finding when published;
        next_cursor is null when no matching entries remain. No matches returns
        an empty items list. Invalid filters or cursors and oversized individual
        items fail explicitly without dropping entries.
    """

    def __init__(
        self,
        items: tuple[dict[str, Any], ...],
        collection_digest: str,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._items = items
        self._digest = collection_digest
        self._metrics = metrics
        self._secrets = secrets
        self._closed = False
        self._lock = asyncio.Lock()
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._items = ()
            self._secrets = ()

    async def __call__(
        self,
        subject_kind: str | None = None,
        subject_key: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "has_subject_kind": subject_kind is not None,
            "has_subject_key": subject_key is not None,
            "has_cursor": cursor is not None,
            "limit": limit if type(limit) is int and 1 <= limit <= 100 else None,
        }
        async with self._lock:
            with RecordedToolCall(
                self._metrics, self.name, arguments, secrets=self._secrets
            ) as call:
                require(not self._closed, "findings_reader_unavailable")
                require(
                    subject_kind is None or identifier(subject_kind), "findings_arguments_invalid"
                )
                require(
                    subject_key is None or (subject_kind is not None and identifier(subject_key)),
                    "findings_arguments_invalid",
                )
                page_limit = 20 if limit is None else limit
                require(
                    type(page_limit) is int and 1 <= page_limit <= 100, "findings_arguments_invalid"
                )
                result = await to_thread_until_done(
                    self._page,
                    subject_kind,
                    subject_key,
                    page_limit,
                    cursor,
                    name="findings-reader-page",
                )
                call.succeed(
                    {
                        "item_count": len(result["items"]),
                        "has_more": result["next_cursor"] is not None,
                    }
                )
                return result

    def _page(
        self,
        subject_kind: str | None,
        subject_key: str | None,
        page_limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        matching = [
            item
            for item in self._items
            if (subject_kind is None or (item["subject"] or {}).get("kind") == subject_kind)
            and (subject_key is None or (item["subject"] or {}).get("key") == subject_key)
        ]
        start = 0
        if cursor is not None:
            after = self._decode_cursor(cursor, subject_kind, subject_key)
            positions = [i for i, item in enumerate(matching) if item["receipt_id"] == after]
            require(len(positions) == 1, "findings_cursor_invalid")
            start = positions[0] + 1
        page: list[dict[str, Any]] = []
        page_cursor: str | None = None
        item_bytes = 0
        for index in range(start, min(len(matching), start + page_limit)):
            item = matching[index]
            next_cursor = (
                None
                if index + 1 == len(matching)
                else self._cursor(item["receipt_id"], subject_kind, subject_key)
            )
            # Each candidate is canonicalized once; the page size adds it to
            # the admitted items, its separator and the envelope with its cursor.
            size = len(canonical(item))
            if _page_bytes(item_bytes + size, len(page) + 1, next_cursor) > MAX_PAGE_BYTES:
                require(page, "findings_limit_exceeded")
                break
            page.append(item)
            item_bytes += size
            page_cursor = next_cursor
        result: dict[str, Any] = {"items": page, "next_cursor": page_cursor}
        return copy.deepcopy(result)

    def _cursor(self, after: str, kind: str | None, key: str | None) -> str:
        return (
            base64.urlsafe_b64encode(
                canonical(
                    {
                        "version": 1,
                        "collection_digest": self._digest,
                        "subject_kind": kind,
                        "subject_key": key,
                        "after_receipt_id": after,
                    }
                )
            )
            .rstrip(b"=")
            .decode("ascii")
        )

    def _decode_cursor(self, cursor: str, kind: str | None, key: str | None) -> str:
        try:
            require(
                isinstance(cursor, str)
                and 0 < len(cursor) <= 1366
                and re.fullmatch(r"[A-Za-z0-9_-]+", cursor)
            )
            raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            require(len(raw) <= 1024)
            value = strict_json(raw)
            require(
                isinstance(value, dict)
                and type(value.get("version")) is int
                and identifier(value.get("after_receipt_id"))
            )
            after = value["after_receipt_id"]
            require(cursor == self._cursor(after, kind, key))
            return after
        except (ValueError, TypeError, KeyError) as error:
            raise FindingsError("findings_cursor_invalid") from error


def _page_bytes(item_bytes: int, item_count: int, next_cursor: str | None) -> int:
    """Return the exact canonical size of a page from its items' canonical sizes.

    RFC 8785 serializes each nested value independently of its container and an
    array as its members joined by commas, so a page is its empty envelope with
    the canonical items and one separator between each adjacent pair inserted.
    """
    envelope = len(canonical({"items": [], "next_cursor": next_cursor}))
    return envelope + item_bytes + max(item_count - 1, 0)


def _retryable(error: ArtifactAPIError | ArtifactTransportError) -> bool:
    # A lost transport may be transient; the Artifact API states its own retryability.
    return not isinstance(error, ArtifactAPIError) or error.retryable


def _preview(text: str, limit: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "text": encoded[:limit].decode("utf-8", errors="ignore"),
        "truncated": len(encoded) > limit,
    }
