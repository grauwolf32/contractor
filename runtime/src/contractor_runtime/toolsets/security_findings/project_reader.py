"""Live authorized Project discovery and explicit retained finding selections."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, TypedDict
from urllib.parse import urlencode

from contractor_runtime.artifacts import (
    ArtifactClient,
    ArtifactResponseLimitError,
    ArtifactTransportError,
    allocation_path,
    raise_for_status,
    require_json,
)
from contractor_runtime.contracts import DIGEST_PATTERN, MEDIA_TYPE_PATTERN, ArtifactRef
from contractor_runtime.contracts.base import MAX_EXACT_INTEGER, RESOURCE_ID_PATTERN
from contractor_runtime.core.tool_metrics import RecordedToolCall, ToolMetrics
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.artifact_visibility import ArtifactObservingTool
from contractor_runtime.toolsets.common.versions import MaterialSelector, SelectedVersions
from contractor_runtime.toolsets.security_findings.collection import (
    COLLECTION_MEDIA_TYPE,
    MAX_ARCHIVE_BYTES,
    FindingsError,
    canonical,
    object_fields,
    require,
    strict_json,
)
from contractor_runtime.toolsets.security_findings.reader import ListFindingsTool, prepare_reader


class FindingSelector(TypedDict):
    findingId: str


def _identity(value: object) -> bool:
    return isinstance(value, str) and RESOURCE_ID_PATTERN.fullmatch(value) is not None


def _revision(value: object) -> bool:
    return type(value) is int and 1 <= value <= MAX_EXACT_INTEGER


def _exact(value: Any) -> ArtifactRef:
    object_fields(value, "ref digest mediaType sizeBytes")
    try:
        ref = ArtifactRef.model_validate(value["ref"]).require_exact()
    except (ValueError, TypeError) as error:
        raise FindingsError("findings_reference_invalid") from error
    require(
        isinstance(value["digest"], str)
        and DIGEST_PATTERN.fullmatch(value["digest"])
        and isinstance(value["mediaType"], str)
        and MEDIA_TYPE_PATTERN.fullmatch(value["mediaType"])
        and type(value["sizeBytes"]) is int
        and 0 <= value["sizeBytes"] <= MAX_ARCHIVE_BYTES
    )
    return ref


def _current(value: Any, *, detail: bool = False) -> None:
    shared = (
        "findingId projectId title state analystVerdict analystSeverity "
        "duplicateTargetId revision createdAt updatedAt"
    )
    fields = (
        "firstReceiptId firstProposal rationale actorId assessedAt"
        if detail
        else "titleTruncated subjectKind subjectKey subjectTruncated severitySuggestion"
    )
    object_fields(value, shared + " " + fields)
    require(
        _identity(value["findingId"])
        and _identity(value["projectId"])
        and _revision(value["revision"])
        and isinstance(value["title"], str)
    )
    require(all(isinstance(value[key], str) for key in ("createdAt", "updatedAt")))
    if detail:
        require(_identity(value["firstReceiptId"]))
        _exact(value["firstProposal"])
    else:
        require(
            type(value["titleTruncated"]) is bool
            and type(value["subjectTruncated"]) is bool
            and all(
                isinstance(value[key], str)
                for key in ("subjectKind", "subjectKey", "severitySuggestion")
            )
        )


def _selection_request(value: dict[str, Any]) -> dict:
    object_fields(value, "clientKey findings receiptIds")
    selection_key, findings, receipt_ids = (
        value["clientKey"],
        value["findings"],
        value["receiptIds"],
    )
    try:
        ArtifactRef(namespace="finding-collections", name=selection_key)
    except (ValueError, TypeError) as error:
        raise FindingsError("findings_arguments_invalid") from error
    receipts = receipt_ids
    require(
        isinstance(findings, list)
        and isinstance(receipts, list)
        and len(findings) + len(receipts) <= 128,
        "findings_arguments_invalid",
    )
    for item in findings:
        object_fields(item, "findingId revision")
        require(
            isinstance(item["findingId"], str)
            and RESOURCE_ID_PATTERN.fullmatch(item["findingId"])
            and len(item["findingId"]) <= 128
            and type(item["revision"]) is int
            and 1 <= item["revision"] <= MAX_EXACT_INTEGER,
            "findings_arguments_invalid",
        )
    require(
        all(
            isinstance(item, str) and len(item) <= 128 and RESOURCE_ID_PATTERN.fullmatch(item)
            for item in receipts
        ),
        "findings_arguments_invalid",
    )
    require(
        len({item["findingId"] for item in findings}) == len(findings)
        and len(set(receipts)) == len(receipts),
        "findings_arguments_invalid",
    )
    return {"clientKey": selection_key, "findings": findings, "receiptIds": receipts}


class ProjectFindingTools:
    name = "project-findings"

    def __init__(
        self,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
        namespace: str,
        versions=None,
    ):
        self.client, self.metrics, self.secrets = client, metrics, secrets
        self.versions = versions if versions is not None else SelectedVersions()
        self.namespace = namespace
        self.closed = False
        self.__name__ = self.name

    async def close(self) -> None:
        self.closed = True
        self.secrets = ()

    async def request(self, method: str, suffix: str, data: dict | None = None) -> dict:
        body = (
            b""
            if data is None
            else await to_thread_until_done(canonical, data, name="finding-selection-encode")
        )
        for attempt in range(2):
            try:
                response = await self.client.transport.request(
                    method,
                    allocation_path(self.client.allocation_id, suffix),
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=256 * 1024,
                )
                break
            except ArtifactResponseLimitError:
                raise
            except ArtifactTransportError:
                if attempt:
                    raise
        require(len(response.body) <= 256 * 1024, "findings_limit_exceeded")
        raise_for_status(response)
        require(response.status_code == 200, "findings_reader_unavailable")
        require_json(response.headers)
        value = await to_thread_until_done(
            partial(strict_json, canonical_required=False),
            response.body,
            name="project-findings-response",
        )
        require(isinstance(value, dict), "findings_reader_unavailable")
        return value

    async def invoke(self, tool: str, method: str, suffix: str, data: dict | None = None) -> dict:
        require(not self.closed, "findings_reader_unavailable")
        with RecordedToolCall(self.metrics, tool, {}, secrets=self.secrets) as call:
            try:
                result = await self.request(method, suffix, data)
            except ArtifactResponseLimitError as error:
                raise FindingsError("findings_limit_exceeded") from error
            except ArtifactTransportError as error:
                raise FindingsError("findings_collection_unavailable", retryable=True) from error
            call.succeed({})
            return result

    async def list_project_findings(self, limit: int = 20, cursor: str | None = None) -> dict:
        """List current findings in this Run's authorized Project.

        Live pages carry current Finding revisions and a Project revision fence;
        they retain no evidence and are not a frozen snapshot. Follow nextCursor
        until null, including empty progress pages. If the Project changed,
        restart discovery; never mix pages as an exact retained selection.

        Args:
            limit: Maximum findings per page, 1 to 100; defaults to 20.
            cursor: Opaque nextCursor from the same Project, or omit to begin.

        Returns:
            Current finding summaries, Project revision, and continuation cursor.
        """
        require(type(limit) is int and 1 <= limit <= 100, "findings_arguments_invalid")
        query: dict[str, str | int] = {"limit": limit}
        if cursor is not None:
            require(
                isinstance(cursor, str) and 0 < len(cursor) <= 1366, "findings_arguments_invalid"
            )
            query["cursor"] = cursor
        value = await self.invoke("list_project_findings", "GET", "findings?" + urlencode(query))
        object_fields(value, "items projectRevision nextCursor")
        require(isinstance(value["items"], list) and len(value["items"]) <= limit)
        require(_revision(value["projectRevision"]))
        require(
            value["nextCursor"] is None
            or (isinstance(value["nextCursor"], str) and 0 < len(value["nextCursor"]) <= 1366)
        )
        for item in value["items"]:
            _current(item)
        self.versions.discover_findings(value["items"])
        return value

    async def read_project_finding(self, finding_id: str) -> dict:
        """Read current canonical finding metadata at a discovered revision.

        Args:
            finding_id: Finding identity returned by list_project_findings.

        Returns:
            Current analyst assessment and original claim metadata. Source refs
            record provenance and grant no Artifact access. Explicit selection
            is required to retain and read proposal/evidence bytes in this Run.
        """
        require(
            isinstance(finding_id, str) and RESOURCE_ID_PATTERN.fullmatch(finding_id),
            "findings_arguments_invalid",
        )
        revision = self.versions.finding_revision(finding_id)
        value = await self.invoke(
            "read_project_finding", "GET", f"findings/{finding_id}?revision={revision}"
        )
        _current(value, detail=True)
        require(value["findingId"] == finding_id and value["revision"] == revision)
        return value

    async def select_project_findings(
        self,
        selection_key: str,
        findings: list[FindingSelector],
        receipt_ids: list[str] | None = None,
    ) -> dict:
        """Retain an explicit exact finding/evidence selection in this Run.

        Selecting a Finding includes all its independent sources within the
        package bounds. For a large source history, select explicit receipt IDs
        instead. A foreign, stale, restricted or unavailable source rejects the
        entire selection. Reuse the same key and exact selection after response
        loss; a changed selection needs a new key. No target work is executed.

        Args:
            selection_key: Portable Artifact name identifying this selection.
            findings: Objects with findingId from discovery; Runtime supplies the version.
            receipt_ids: Optional explicit Project source receipts; absent means none.
                At most 128 selectors total. Empty arrays explicitly select nothing.

        Returns:
            Exact immutable collection Artifact, snapshot time, entry count and replay marker.
            Use read_finding_selection with artifact.ref to page its retained contents.
        """
        require(isinstance(findings, list) and len(findings) <= 128, "findings_arguments_invalid")
        resolved = []
        for item in findings:
            object_fields(item, "findingId")
            require(_identity(item["findingId"]), "findings_arguments_invalid")
            resolved.append(
                {
                    "findingId": item["findingId"],
                    "revision": self.versions.finding_revision(item["findingId"]),
                }
            )
        request = _selection_request(
            {
                "clientKey": selection_key,
                "findings": resolved,
                "receiptIds": [] if receipt_ids is None else receipt_ids,
            }
        )
        value = await self.invoke(
            "select_project_findings",
            "POST",
            "finding-selections",
            request,
        )
        object_fields(value, "artifact snapshotAt entryCount replayed")
        ref = _exact(value["artifact"])
        require(
            ref.namespace == self.namespace
            and ref.name == selection_key
            and value["artifact"]["mediaType"] == COLLECTION_MEDIA_TYPE
            and type(value["entryCount"]) is int
            and 0 <= value["entryCount"] <= 256
            and type(value["replayed"]) is bool
            and isinstance(value["snapshotAt"], str)
        )
        self.versions.observe(ref)
        return value


class ListProjectFindingsTool(ProjectFindingTools):
    name = "list_project_findings"
    __call__ = ProjectFindingTools.list_project_findings
    __doc__ = ProjectFindingTools.list_project_findings.__doc__


class ReadProjectFindingTool(ProjectFindingTools):
    name = "read_project_finding"
    __call__ = ProjectFindingTools.read_project_finding
    __doc__ = ProjectFindingTools.read_project_finding.__doc__


class SelectProjectFindingsTool(ProjectFindingTools):
    name = "select_project_findings"
    __call__ = ProjectFindingTools.select_project_findings
    __doc__ = ProjectFindingTools.select_project_findings.__doc__


class ReadFindingSelection(ArtifactObservingTool):
    name = "read_finding_selection"
    description = """Page one explicitly selected exact finding collection.

    Args:
        collection_ref: Namespace/name of the collection returned by select_project_findings.
            A declared frozen collection input may also be used explicitly.
        limit: Maximum entries, 1 to 100; defaults to 20.
        cursor: next_cursor from the same exact collection, or omit to begin.

    Returns:
        Bounded retained proposal previews and exact current-Run proposal/evidence
        refs. Read full bytes with read_artifact. Source refs preserve lineage and
        do not grant foreign access. Follow next_cursor until null. Captured
        finding state describes the snapshot and grants no mutation authority.
    """

    def __init__(self, owner: ProjectFindingTools):
        self.owner = owner
        self._client = owner.client
        self.reader: ListFindingsTool | None = None
        self.ref: ArtifactRef | None = None
        self.lock = asyncio.Lock()
        self.closed = False
        self.__name__, self.__doc__ = self.name, self.description

    async def __call__(
        self, collection_ref: MaterialSelector, limit: int = 20, cursor: str | None = None
    ) -> dict:
        try:
            ref = self.owner.versions.resolve(collection_ref)
        except (ValueError, TypeError) as error:
            raise FindingsError("findings_arguments_invalid") from error
        require(ref.namespace in {self.owner.namespace, "inputs"}, "findings_arguments_invalid")
        async with self.lock:
            require(not self.closed, "findings_reader_unavailable")
            if ref != self.ref or self.reader is None:
                if self.reader is not None:
                    await self.reader.close()
                    self.reader = None
                self.reader = await prepare_reader(
                    self.owner.client, self.owner.metrics, self.owner.secrets, source_ref=ref
                )
                self.ref = ref
            if self.reader is None:
                raise RuntimeError("prepared finding reader is missing")
            return await self.reader(limit=limit, cursor=cursor)

    async def close(self) -> None:
        async with self.lock:
            self.closed = True
            await self.owner.close()
            if self.reader is not None:
                await self.reader.close()
                self.reader = None
