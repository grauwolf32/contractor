"""Model-visible invocation checkpoint inspection for overlay workspaces."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import jcs

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.observations import (
    WorkspaceToolObservation,
    workspace_changes_observation,
)
from contractor_runtime.projectfs.paths import ProjectPathError
from contractor_runtime.projectfs.storage import (
    WorkspaceChange,
    WorkspaceChanges,
    WorkspaceStorageError,
)
from contractor_runtime.toolsets.filesystem import FilesystemToolError
from contractor_runtime.toolsets.run_artifacts import ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_PAGE_ITEMS = 100
MAX_DIFF_BYTES = 1 << 20
MAX_CURSOR_BYTES = 2048


class WorkspaceChangesToolsetFactory:
    ref = "workspace-changes@1"
    exported_tools = frozenset({"changed_paths", "diff", "rollback_changes"})
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True
    workspace_access = "changes"

    async def probe(self) -> frozenset[str]:
        return self.exported_tools

    async def create_selected(
        self,
        *,
        selected: Sequence[str],
        allocation_id: str,
        run_id: str,
        namespace: str,
        runtime_settings: RuntimeSettings,
        workspace: AllocationWorkspace,
        state: Any,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: WorkspaceChanges | None = None,
    ) -> Mapping[str, Any]:
        del allocation_id, run_id, namespace, runtime_settings, workspace, adapter_handles
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        if project_workspace is None:
            raise FilesystemToolError("workspace_required")
        for method in ("change_entries", "diff", "rollback_changes"):
            if not callable(getattr(project_workspace, method, None)):
                raise FilesystemToolError("workspace_mode_unsupported")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("workspace-changes@1 requires State.metrics")
        session = _ChangesSession(project_workspace)
        builders: dict[str, Callable[[], Any]] = {
            "changed_paths": lambda: ChangedPathsTool(session, metrics),
            "diff": lambda: WorkspaceDiffTool(session, metrics),
            "rollback_changes": lambda: RollbackChangesTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


class _ChangesSession:
    def __init__(self, changes: WorkspaceChanges) -> None:
        self._changes = changes
        self._key = bytearray(secrets.token_bytes(32))
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        self._key[:] = b"\x00" * len(self._key)

    async def changed_paths(self, cursor: str, limit: int) -> dict[str, Any]:
        self._require_open()
        limit = _limit(limit)
        try:
            entries = await self._changes.change_entries()
        except (ProjectPathError, WorkspaceStorageError) as error:
            raise _mapped(error) from None
        fingerprint = _change_fingerprint(entries)
        offset = self._decode_cursor(cursor, "changed_paths", fingerprint) if cursor else 0
        if offset > len(entries):
            raise FilesystemToolError("workspace_cursor_invalid")
        page = entries[offset : offset + limit]
        next_offset = offset + len(page)
        next_cursor = (
            self._encode_cursor("changed_paths", fingerprint, next_offset)
            if next_offset < len(entries)
            else None
        )
        return {
            "changes": [{"path": entry.path, "change": entry.change} for entry in page],
            "nextCursor": next_cursor,
            "truncated": next_cursor is not None,
        }

    async def diff(self, path: str, cursor: str, max_bytes: int) -> dict[str, Any]:
        self._require_open()
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes <= 0
            or max_bytes > MAX_DIFF_BYTES
        ):
            raise FilesystemToolError("workspace_limit_exceeded")
        try:
            entries = await self._changes.change_entries(path)
            fingerprint = _change_fingerprint(entries)
            query = _query_digest({"tool": "diff", "path": path})
            offset = self._decode_cursor(cursor, query, fingerprint) if cursor else 0
            result = await self._changes.diff(path, max_bytes=max_bytes, offset_bytes=offset)
        except (ProjectPathError, WorkspaceStorageError) as error:
            raise _mapped(error) from None
        next_cursor = (
            self._encode_cursor(query, fingerprint, result.next_offset)
            if result.truncated and result.next_offset is not None
            else None
        )
        return {
            "text": result.text,
            "returnedBytes": result.returned_bytes,
            "nextCursor": next_cursor,
            "truncated": result.truncated,
            "authoritative": False,
        }

    async def rollback(self, path: str) -> dict[str, bool]:
        self._require_open()
        try:
            await self._changes.rollback_changes(path)
        except (ProjectPathError, WorkspaceStorageError) as error:
            raise _mapped(error) from None
        return {"changed": True}

    def _encode_cursor(self, query: str, fingerprint: str, offset: int) -> str:
        body = jcs.canonicalize({"query": query, "fingerprint": fingerprint, "offset": offset})
        signature = hmac.digest(bytes(self._key), body, "sha256")
        return f"{_b64(body)}.{_b64(signature)}"

    def _decode_cursor(self, value: str, query: str, fingerprint: str) -> int:
        if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_BYTES:
            raise FilesystemToolError("workspace_cursor_invalid")
        try:
            encoded_body, encoded_signature = value.split(".", 1)
            body = _unb64(encoded_body)
            signature = _unb64(encoded_signature)
            if not hmac.compare_digest(signature, hmac.digest(bytes(self._key), body, "sha256")):
                raise ValueError
            document = json.loads(body)
            if (
                jcs.canonicalize(document) != body
                or set(document) != {"query", "fingerprint", "offset"}
                or document["query"] != query
                or document["fingerprint"] != fingerprint
                or not isinstance(document["offset"], int)
                or isinstance(document["offset"], bool)
                or document["offset"] < 0
            ):
                raise ValueError
            return document["offset"]
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise FilesystemToolError("workspace_cursor_invalid") from None

    def _require_open(self) -> None:
        if self._closed:
            raise FilesystemToolError("workspace_not_found")


class _BaseChangesTool:
    name: str
    description: str

    def __init__(self, session: _ChangesSession, metrics: ToolMetrics) -> None:
        self._session = session
        self._metrics = metrics
        self.__name__ = self.name
        self.__doc__ = self.description

    async def close(self) -> None:
        await self._session.close()

    def contractor_observation(
        self,
        tool_args: Mapping[str, Any],
        result: Any,
    ) -> WorkspaceToolObservation | None:
        return workspace_changes_observation(self.name, tool_args, result)

    async def _invoke(self, operation: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            result = await operation()
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                result={
                    "count": len(result.get("changes", [])),
                    "truncated": bool(result.get("truncated", False)),
                },
                duration_ms=_elapsed_ms(started),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments={},
                error=error,
                duration_ms=_elapsed_ms(started),
            )
            raise


class ChangedPathsTool(_BaseChangesTool):
    name = "changed_paths"
    description = """List workspace changes since the current checkpoint.

    Args:
        cursor: Opaque nextCursor from this listing; empty starts a new page set.
            Restart the listing after further workspace changes.
        limit: Maximum entries per page, from 1 to 100; defaults to 100.

    Returns:
        Sorted created, modified, deleted and type-changed paths with pagination
        and truncation metadata.
    """

    async def __call__(self, cursor: str = "", limit: int = MAX_PAGE_ITEMS) -> dict[str, Any]:
        return await self._invoke(lambda: self._session.changed_paths(cursor, limit))


class WorkspaceDiffTool(_BaseChangesTool):
    name = "diff"
    description = """Read a bounded unified diff of workspace changes since the checkpoint.

    The diff is a review preview; use read_file for exact current content.

    Args:
        path: Project-relative file or subtree; empty selects the whole workspace.
        cursor: Opaque nextCursor from the same query; empty starts a new diff.
            Restart after further workspace changes.
        max_bytes: Maximum UTF-8 diff bytes, from 1 to 1048576; defaults to 65536.

    Returns:
        A unified diff preview with pagination and truncation metadata.
    """

    async def __call__(
        self,
        path: str = "",
        cursor: str = "",
        max_bytes: int = 65536,
    ) -> dict[str, Any]:
        return await self._invoke(lambda: self._session.diff(path, cursor, max_bytes))


class RollbackChangesTool(_BaseChangesTool):
    name = "rollback_changes"
    description = """Restore workspace content to the current checkpoint.

    Discards changes under the selected path, including newly created content.

    Args:
        path: Project-relative file or subtree; empty restores the whole workspace.

    Returns:
        {"changed": true} after restoration succeeds.
    """

    async def __call__(self, path: str = "") -> dict[str, Any]:
        return await self._invoke(lambda: self._session.rollback(path))


def _change_fingerprint(entries: Sequence[WorkspaceChange]) -> str:
    document = [
        {"path": entry.path, "change": entry.change, "token": entry.token} for entry in entries
    ]
    return "sha256:" + hashlib.sha256(jcs.canonicalize(document)).hexdigest()


def _query_digest(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(jcs.canonicalize(dict(document))).hexdigest()


def _limit(value: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > MAX_PAGE_ITEMS
    ):
        raise FilesystemToolError("workspace_limit_exceeded")
    return value


def _mapped(error: Exception) -> FilesystemToolError:
    if isinstance(error, ProjectPathError):
        return FilesystemToolError("workspace_path_invalid")
    code = error.args[0] if error.args else "workspace_unavailable"
    if code not in {
        "workspace_cursor_invalid",
        "workspace_limit_exceeded",
        "workspace_not_found",
        "workspace_path_invalid",
        "workspace_type_conflict",
    }:
        code = "workspace_unavailable"
    return FilesystemToolError(code)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not value or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in value
    ):
        raise ValueError
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64(decoded) != value:
        raise ValueError
    return decoded


def _elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)
