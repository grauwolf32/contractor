"""Prepare findings documents in the current Run and expose a bounded snapshot reader."""

from __future__ import annotations

import base64
import copy
import re
import time
from typing import Any

from contractor_runtime.artifacts import (
    ArtifactAPIError,
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
)
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.toolsets.common.metrics import ToolMetrics
from contractor_runtime.toolsets.security_findings.collection import (
    COLLECTION_MEDIA_TYPE,
    MAX_ARCHIVE_BYTES,
    FindingsError,
    canonical,
    decode_collection,
    digest,
    identifier,
    require,
    strict_json,
)

MAX_PAGE_BYTES = 256 * 1024


async def prepare_reader(
    client: ArtifactClient, metrics: ToolMetrics, secrets: tuple[str, ...]
) -> ListFindingsTool:
    try:
        source = await client.read_artifact(
            ArtifactRef(namespace="inputs", name="findings"), max_bytes=MAX_ARCHIVE_BYTES
        )
    except ArtifactResponseLimitError as error:
        raise FindingsError("findings_limit_exceeded") from error
    except (ArtifactAPIError, ArtifactTransportError) as error:
        raise FindingsError("findings_collection_unavailable") from error
    require(source.media_type == COLLECTION_MEDIA_TYPE)
    # Validate every member before creating any destination or exposing any tool.
    collection = decode_collection(source.data)
    documents = {}
    for document in collection.metadata["documents"]:
        target = ArtifactRef(namespace=collection.namespace, name=document["id"])
        try:
            try:
                written = await client.write_artifact(
                    target,
                    data=collection.contents[document["id"]],
                    media_type=document["media_type"],
                    expected_revision=None,
                )
                exact = written.artifact.require_exact()
                require(
                    (exact.namespace, exact.name) == (target.namespace, target.name),
                    "findings_document_conflict",
                )
            except ArtifactAPIError as error:
                if error.status_code not in {409, 412}:
                    raise
                exact = target
            except ArtifactTransportError:
                # A write may have committed before its response was lost. Read
                # and verify the deterministic binding; never overwrite it.
                exact = target
            retained = await client.read_artifact(exact, max_bytes=MAX_ARCHIVE_BYTES)
        except ArtifactResponseLimitError as error:
            raise FindingsError("findings_limit_exceeded") from error
        except (ArtifactAPIError, ArtifactTransportError) as error:
            raise FindingsError("findings_document_unavailable") from error
        ref = retained.artifact.require_exact()
        require(
            (ref.namespace, ref.name) == (target.namespace, target.name)
            and (exact.revision is None or ref.revision == exact.revision)
            and retained.media_type == document["media_type"]
            and len(retained.data) == document["size_bytes"]
            and digest(retained.data) == document["digest"],
            "findings_document_conflict",
        )
        documents[document["id"]] = {
            "ref": ref.model_dump(exclude_none=True),
            "digest": document["digest"],
            "media_type": document["media_type"],
            "size_bytes": document["size_bytes"],
            "source": {"scope": document["scope"], "ref": document["ref"]},
        }
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
        items.append(item)
    # Only previews and metadata survive preparation. Full bytes stay in artifacts.
    return ListFindingsTool(tuple(items), collection.digest, metrics, secrets)


class ListFindingsTool:
    name = "list_findings"
    description = """List finding proposals from this Run's prepared collection.

    Returns bounded previews and exact current-Run artifact references. Use
    read_artifact with proposal.ref or evidence[].ref for full content. Source
    refs are provenance only. Captured reviews describe the collection snapshot;
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
        hypothesis presence, proposal and evidence refs, and captured reviews;
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
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
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
        started = time.perf_counter_ns()
        arguments = {
            "has_subject_kind": subject_kind is not None,
            "has_subject_key": subject_key is not None,
            "has_cursor": cursor is not None,
            "limit": limit if type(limit) is int and 1 <= limit <= 100 else None,
        }
        try:
            require(not self._closed, "findings_reader_unavailable")
            require(subject_kind is None or identifier(subject_kind), "findings_arguments_invalid")
            require(
                subject_key is None or (subject_kind is not None and identifier(subject_key)),
                "findings_arguments_invalid",
            )
            page_limit = 20 if limit is None else limit
            require(
                type(page_limit) is int and 1 <= page_limit <= 100, "findings_arguments_invalid"
            )
            matching = [
                item
                for item in self._items
                if (subject_kind is None or item["subject"]["kind"] == subject_kind)
                and (subject_key is None or item["subject"]["key"] == subject_key)
            ]
            start = 0
            if cursor is not None:
                after = self._decode_cursor(cursor, subject_kind, subject_key)
                positions = [i for i, item in enumerate(matching) if item["receipt_id"] == after]
                require(len(positions) == 1, "findings_cursor_invalid")
                start = positions[0] + 1
            result: dict[str, Any] = {"items": [], "next_cursor": None}
            for index in range(start, min(len(matching), start + page_limit)):
                item = matching[index]
                next_cursor = (
                    None
                    if index + 1 == len(matching)
                    else self._cursor(item["receipt_id"], subject_kind, subject_key)
                )
                candidate = {"items": [*result["items"], item], "next_cursor": next_cursor}
                if len(canonical(candidate)) > MAX_PAGE_BYTES:
                    require(result["items"], "findings_limit_exceeded")
                    break
                result = candidate
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                result={
                    "item_count": len(result["items"]),
                    "has_more": result["next_cursor"] is not None,
                },
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started),
            )
            return copy.deepcopy(result)
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started),
            )
            raise

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


def _preview(text: str, limit: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "text": encoded[:limit].decode("utf-8", errors="ignore"),
        "truncated": len(encoded) > limit,
    }


def _elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)
