"""Explicit sitemap operations. Ordinary target tools never depend on these."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactAPIError, ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.sitemap import SitemapClient
from contractor_runtime.toolsets.common.artifact_visibility import require_model_visible_binding
from contractor_runtime.toolsets.common.artifacts import ArtifactClientFactory, _unconfigured_client
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.report_observations import REPORT_FORMATS, prepare_report
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.tool_base import SessionArtifactTool
from contractor_runtime.toolsets.common.versions import MaterialSelector, ModelArtifacts
from contractor_runtime.toolsets.sitemap.session import SitemapSession
from contractor_runtime.workspace import AllocationWorkspace

ImportDispatcher = Callable[[ArtifactClient, Mapping[str, Any], str], Awaitable[dict[str, Any]]]


async def import_sitemap_report(
    client: ArtifactClient, selection: Mapping[str, Any], key: str
) -> dict[str, Any]:
    if (
        not isinstance(selection, Mapping)
        or set(selection) != {"format", "artifact"}
        or selection["format"] not in REPORT_FORMATS.values()
    ):
        raise ToolInputError("selection requires a supported report format and exact artifact")
    try:
        artifact = ArtifactRef.model_validate(selection["artifact"]).require_exact()
        require_model_visible_binding(artifact.namespace, artifact.name)
    except (ValueError, TypeError):
        raise ToolInputError("artifact must be an accessible exact current-Run ref") from None
    batch = await prepare_report(client, selection["format"], artifact)
    result = await SitemapClient(client).collect(batch, key, automatic=False)
    return result.model_dump(by_alias=True)


class SitemapToolsetFactory:
    ref = "sitemap@1"
    exported_tools = frozenset(
        {
            "list_sitemap_entries",
            "read_sitemap_entry",
            "upsert_sitemap_entry",
            "remove_sitemap_entry",
            "restore_sitemap_entry",
            "configure_sitemap_grouping",
            "snapshot_sitemap_context",
            "import_sitemap_source",
        }
    )
    infrastructure_channels = MappingProxyType({})

    def __init__(
        self,
        client_factory: ArtifactClientFactory | None = None,
        import_dispatcher: ImportDispatcher | None = None,
    ) -> None:
        self._clients = client_factory or _unconfigured_client
        self._import = import_dispatcher

    async def probe(self) -> frozenset[str]:
        # An uninstalled format dispatcher is never a positive capability.
        return (
            self.exported_tools
            if self._import is not None
            else self.exported_tools - {"import_sitemap_source"}
        )

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
        deployment_marker: str | None = None,
    ) -> Mapping[str, Any]:
        del run_id, workspace, adapter_handles, project_workspace
        require_selected_tools(selected, await self.probe())
        metrics = require_metrics(resources, self.ref)
        artifact = ModelArtifacts(
            self._clients(allocation_id, runtime_settings), resources.selected_versions
        )
        session = SitemapSession(
            artifact,
            resources.sitemap_read_flush,
            namespace=namespace,
            deployment=deployment_marker or "",
        )
        secrets = runtime_secrets(runtime_settings)
        classes = (
            ListSitemapEntries,
            ReadSitemapEntry,
            UpsertSitemapEntry,
            RemoveSitemapEntry,
            RestoreSitemapEntry,
            ConfigureSitemapGrouping,
            SnapshotSitemapContext,
            ImportSitemapSource,
        )
        builders = {tool.name: tool for tool in classes}
        return {
            name: builders[name](session, artifact, metrics, secrets, self._import)
            for name in selected
        }


class _SitemapTool(SessionArtifactTool[SitemapSession]):
    def __init__(self, session, artifact: ArtifactClient, metrics, secrets, dispatcher) -> None:
        super().__init__(session, artifact, metrics, secrets)
        self._dispatcher = dispatcher

    def _validate_nesting(self, value: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ToolInputError("sitemap arguments exceed the nesting bound")
        if isinstance(value, Mapping):
            for key, item in value.items():
                self._validate_nesting(key, depth + 1)
                self._validate_nesting(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self._validate_nesting(item, depth + 1)

    async def _invoke(
        self,
        arguments: dict,
        operation: Callable[[], Awaitable[dict]],
        *,
        session_read: bool = False,
    ) -> dict:
        with self._recorded({}) as call:
            self._validate_nesting(arguments)
            try:
                # Session reads bound flush, lock acquisition and reading
                # independently; mutations retain their existing allowance.
                async with asyncio.timeout(None if session_read else 15):
                    result = await operation()
            except ArtifactAPIError as error:
                result = {
                    "status": "failed",
                    "ok": False,
                    "error": {"code": error.code, "retryable": error.retryable},
                }
            result.setdefault("status", "completed")
            call.succeed({"ok": result.get("ok", True)})
            return result

    async def _mutate(self, command: dict, key: str | None) -> dict:
        if "expectedRevision" in command:
            return await self._session.mutate(command, key or str(uuid4()))
        return await self._session.edit(command, key)


class ListSitemapEntries(_SitemapTool):
    name = "list_sitemap_entries"
    description = """List authorized Project entries or map metadata from one immutable view.

    Follow nextCursor with identical filters until it is null; filtered pages
    may be empty and still have a continuation. Refresh replaces the view and
    invalidates all older cursors. Later Project updates do not alter an existing
    sequence. This research map does not establish current coverage or proof.
    Use scope="metadata" to read complete Source preconditions, grouping rules
    and omission gaps, including when no surface entries remain. Metadata rows
    have kind and value; entry filters apply only to scope="entries".

    Args:
        cursor: Continuation from this operation, with identical filters.
        refresh: Replace the cached view; cannot be combined with cursor.
        limit: Maximum returned entries, from 1 to 100.
        method: Exact HTTP method filter.
        origin: Exact origin filter.
        path_prefix: Literal route prefix filter.
        include_suppressed: Include entries whose concrete members were removed.
        scope: entries for surface previews, or metadata for sources, rules and gaps.

    Returns:
        Authorized rows, original revision and nextCursor.
    """

    async def __call__(
        self,
        cursor: str | None = None,
        refresh: bool = False,
        limit: int = 100,
        method: str | None = None,
        origin: str | None = None,
        path_prefix: str | None = None,
        include_suppressed: bool = False,
        scope: Literal["entries", "metadata"] = "entries",
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        return await self._invoke(
            arguments, lambda: self._session.list(**arguments), session_read=True
        )


class ReadSitemapEntry(_SitemapTool):
    name = "read_sitemap_entry"
    description = """Read one authorized sitemap entry in bounded parts.

    Summary, annotation, example, evidence, declaration and removal records are
    separate parts. Follow nextCursor with identical arguments until null.
    Read map-wide sources, rules and gaps with list_sitemap_entries(scope="metadata").
    Member IDs remain the concrete selection for edits after regrouping.
    Provenance mentions do not grant access to foreign evidence bytes.

    Args:
        entry_id: Entry ID returned by list_sitemap_entries.
        cursor: Continuation from this operation.
        refresh: Replace the cached view and invalidate older cursors.
        limit: Maximum returned parts, from 1 to 100.
        include_suppressed: Include explicitly removed concrete members.

    Returns:
        Authorized parts, original revision and nextCursor.
    """

    async def __call__(
        self,
        entry_id: str,
        cursor: str | None = None,
        refresh: bool = False,
        limit: int = 100,
        include_suppressed: bool = False,
    ) -> dict:
        arguments = {
            "entry_id": entry_id,
            "cursor": cursor,
            "refresh": refresh,
            "limit": limit,
            "include_suppressed": include_suppressed,
        }
        return await self._invoke(
            arguments,
            lambda: self._session.read(
                entry_id,
                cursor=cursor,
                refresh=refresh,
                limit=limit,
                include_suppressed=include_suppressed,
            ),
            session_read=True,
        )


class UpsertSitemapEntry(_SitemapTool):
    name = "upsert_sitemap_entry"
    description = """Declare a concrete operation or annotate a reviewed concrete member selection.

    A declaration is an assertion, not an observed successful target request.
    Supply either url/method for a declaration or member_ids with note/tags for
    an annotation. Runtime uses the reviewed map version. Read/list first; after a successful
    edit or conflict, explicitly refresh before another change. Omit absent fields.

    Args:
        url: Concrete URL for a declaration.
        method: HTTP method for a declaration.
        member_ids: Fixed concrete member IDs for an annotation.
        note: Annotation text, at most 4096 UTF-8 bytes.
        tags: Up to 16 annotation tags.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Exact accepted artifact reference and replay/change information, or a classified error.
    """

    async def __call__(
        self,
        url: str | None = None,
        method: str | None = None,
        member_ids: list[str] | None = None,
        note: str | None = None,
        tags: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        command = {
            "operation": "annotate" if member_ids is not None else "declare",
        }
        for key, value in (
            ("url", url),
            ("method", method),
            ("memberIds", member_ids),
            ("note", note),
            ("tags", tags),
        ):
            if value is not None:
                command[key] = value
        return await self._invoke(arguments, lambda: self._mutate(command, idempotency_key))

    async def contractor_trusted_call(
        self,
        expected_revision: str,
        url: str | None = None,
        method: str | None = None,
        member_ids: list[str] | None = None,
        note: str | None = None,
        tags: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        command = {
            "operation": "annotate" if member_ids is not None else "declare",
            "expectedRevision": expected_revision,
        }
        for key, value in (
            ("url", url),
            ("method", method),
            ("memberIds", member_ids),
            ("note", note),
            ("tags", tags),
        ):
            if value is not None:
                command[key] = value
        return await self._invoke(arguments, lambda: self._mutate(command, idempotency_key))


class RemoveSitemapEntry(_SitemapTool):
    name = "remove_sitemap_entry"
    description = """Suppress a fixed concrete member selection without deleting research history.

    Future traffic never restores this selection. Keep the marker ID to restore
    it explicitly. Later newly discovered members are outside this selection.

    Args:
        member_ids: Up to 64 concrete member IDs returned by read_sitemap_entry.
        marker_id: Stable marker identity for this selection.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Exact accepted artifact reference and replay/change information, or a classified error.
    """

    async def __call__(
        self,
        member_ids: list[str],
        marker_id: str,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        command = {
            "operation": "remove",
            "memberIds": member_ids,
            "markerId": marker_id,
        }
        return await self._invoke(arguments, lambda: self._mutate(command, idempotency_key))

    async def contractor_trusted_call(
        self,
        expected_revision: str,
        member_ids: list[str],
        marker_id: str,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        command = {
            "operation": "remove",
            "expectedRevision": expected_revision,
            "memberIds": member_ids,
            "markerId": marker_id,
        }
        return await self._invoke(arguments, lambda: self._mutate(command, idempotency_key))


class RestoreSitemapEntry(_SitemapTool):
    name = "restore_sitemap_entry"
    description = """Restore one explicitly suppressed concrete selection.

    Args:
        marker_id: The reviewed removal marker to restore.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Exact accepted artifact reference and replay/change information, or a classified error.
    """

    async def __call__(self, marker_id: str, idempotency_key: str | None = None) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        return await self._invoke(
            arguments,
            lambda: self._mutate(
                {
                    "operation": "restore",
                    "markerId": marker_id,
                },
                idempotency_key,
            ),
        )

    async def contractor_trusted_call(
        self, expected_revision: str, marker_id: str, idempotency_key: str | None = None
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        return await self._invoke(
            arguments,
            lambda: self._mutate(
                {
                    "operation": "restore",
                    "expectedRevision": expected_revision,
                    "markerId": marker_id,
                },
                idempotency_key,
            ),
        )


class ConfigureSitemapGrouping(_SitemapTool):
    name = "configure_sitemap_grouping"
    description = """Replace this editor's grouping source using reviewed exact preconditions.

    The Server supplies source identity and attribution. Runtime derives the
    prior source precondition from the reviewed map. Read/list first, then
    refresh explicitly after a successful edit or conflict.
    A stale source/map precondition is a permanent conflict, never refreshed for you.
    Structural query parameters require an explicit finite allowed set in rules.

    Args:
        rules: Bounded rules with origin, method, path, pointer, parameters
            and structural selectors.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Exact accepted artifact reference and replay/change information, or a classified error.
    """

    async def __call__(
        self,
        rules: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]

        async def operation() -> dict:
            source = {"rules": rules}
            return await self._mutate(
                {
                    "operation": "replace-source",
                    "source": source,
                },
                idempotency_key,
            )

        return await self._invoke(arguments, operation)

    async def contractor_trusted_call(
        self,
        expected_revision: str,
        rules: list[dict[str, Any]],
        expected_source: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]

        async def operation() -> dict:
            source = {"rules": rules}
            if expected_source is not None:
                source["expected"] = expected_source
            return await self._mutate(
                {
                    "operation": "replace-source",
                    "expectedRevision": expected_revision,
                    "source": source,
                },
                idempotency_key,
            )

        return await self._invoke(arguments, operation)


class SnapshotSitemapContext(_SitemapTool):
    name = "snapshot_sitemap_context"
    description = """Retain selected Project context from the exact cached view in this Run.

    List/read first. The Server materializes permitted bytes and atomically
    protects the original exact map revision. An evicted source requires explicit
    refresh and reselection; cached bytes cannot revive it. This does not change
    a started Audit baseline or declare proof of current target behavior.

    Args:
        group_ids: Whole entry selections from the reviewed view.
        member_ids: Concrete member selections from the reviewed view.
        require_referenced_inputs: Fail when a mentioned needed input is unavailable.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Exact frozen Run context artifact metadata and replay information, or a classified error.
    """

    async def __call__(
        self,
        group_ids: list[str] | None = None,
        member_ids: list[str] | None = None,
        require_referenced_inputs: bool = False,
        idempotency_key: str | None = None,
    ) -> dict:
        arguments = locals().copy()
        del arguments["self"]
        return await self._invoke(
            arguments,
            lambda: self._session.snapshot(
                group_ids or [],
                member_ids or [],
                require_referenced_inputs,
                idempotency_key or str(uuid4()),
            ),
        )


class ImportSitemapSource(_SitemapTool):
    name = "import_sitemap_source"
    description = """Import an explicitly selected exact source through an installed format adapter.

    Each invocation reads the exact selected revision. Transport retries reuse
    the prepared observations; a new invocation fails if its source was removed.

    Args:
        format: ffuf-report@1 or katana-report@1.
        artifact_ref: Selected current-Run material with namespace and name.
        idempotency_key: Reuse only for an identical ambiguous operation.

    Returns:
        Import outcome and accepted map reference.
    """

    async def __call__(
        self, format: str, artifact_ref: MaterialSelector, idempotency_key: str | None = None
    ) -> dict:
        exact = self._client.versions.resolve(artifact_ref)
        return await self.contractor_trusted_call(
            format, exact.model_dump(exclude_none=True), idempotency_key
        )

    async def contractor_trusted_call(
        self, format: str, artifact_ref: dict[str, Any], idempotency_key: str | None = None
    ) -> dict:
        selection = {"format": format, "artifact": artifact_ref}

        async def operation() -> dict:
            if self._dispatcher is None:
                raise RuntimeError("sitemap import adapter is not installed")
            return await self._session.dispatch_import(
                lambda: self._dispatcher(self._client, selection, idempotency_key or str(uuid4()))
            )

        return await self._invoke(
            {"format": format, "artifact_ref": artifact_ref, "idempotency_key": idempotency_key},
            operation,
        )
