"""Bounded backend-independent read tools for one project workspace."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jcs
import regex as bounded_regex

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.contracts import RuntimeSettings
from contractor_runtime.observations import (
    WorkspaceToolObservation,
    filesystem_tool_observation,
)
from contractor_runtime.projectfs.paths import (
    ProjectPathError,
    normalize_project_glob,
    normalize_project_path,
    project_glob_matches,
)
from contractor_runtime.projectfs.storage import (
    WorkspaceReader,
    WorkspaceSnapshot,
    WorkspaceStorageError,
    WorkspaceTextFile,
)
from contractor_runtime.toolsets.run_artifacts import ToolMetrics
from contractor_runtime.workspace import AllocationWorkspace

MAX_PAGE_ITEMS = 100
MAX_SCAN_PATHS = 20_000
MAX_GREP_BYTES = 16 * 1024 * 1024
MAX_GREP_SECONDS = 1.0
MAX_GREP_PATTERN_CHARS = 512
MAX_GREP_LINE_CHARS = 65_536
MAX_EXCERPT_CHARS = 500
MAX_READ_LINES = 400
DEFAULT_READ_LINES = 200
MAX_READ_BYTES = 128 * 1024
MAX_READ_SCAN_BYTES = 16 * 1024 * 1024
MAX_CURSOR_BYTES = 2048
REGEX_LINE_TIMEOUT_SECONDS = 0.01


class FilesystemToolError(RuntimeError):
    """Stable model-facing failure without content or physical backend detail."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Filesystem operation failed ({code})")


class FilesystemToolsetFactory:
    ref = "filesystem@1"
    exported_tools = frozenset({"ls", "glob", "read_file", "grep"})
    infrastructure_channels = MappingProxyType({})
    requires_workspace = True
    workspace_access = "read"

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
        project_workspace: WorkspaceReader | None = None,
    ) -> Mapping[str, Any]:
        del allocation_id, run_id, namespace, runtime_settings, workspace, adapter_handles
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        if project_workspace is None:
            raise FilesystemToolError("workspace_required")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("filesystem@1 requires State.metrics")
        session = _FilesystemSession(project_workspace)
        builders: dict[str, Callable[[], Any]] = {
            "ls": lambda: ListWorkspaceTool(session, metrics),
            "glob": lambda: GlobWorkspaceTool(session, metrics),
            "read_file": lambda: ReadWorkspaceFileTool(session, metrics),
            "grep": lambda: GrepWorkspaceTool(session, metrics),
        }
        return {name: builders[name]() for name in selected}


@dataclass(frozen=True, slots=True)
class _Cursor:
    snapshot: str
    query: str
    offset: int
    line: int = 0


class _FilesystemSession:
    def __init__(self, reader: WorkspaceReader) -> None:
        self._reader = reader
        self._cursor_key = bytearray(secrets.token_bytes(32))
        self._closed = False

    async def close(self) -> None:
        self._closed = True
        self._cursor_key[:] = b"\x00" * len(self._cursor_key)

    async def ls(self, path: str, cursor: str, limit: int) -> dict[str, Any]:
        root = _path(path, allow_root=True)
        limit = _limit(limit)
        snapshot = await self._snapshot()
        _require_directory(snapshot, root)
        query = _query_digest({"tool": "ls", "path": root})
        offset = self._cursor_offset(cursor, snapshot, query).offset if cursor else 0
        paths = _all_paths(snapshot)
        page, next_offset, scanned = _scan_page(
            paths,
            offset,
            limit,
            lambda candidate: _immediate_child(candidate, root),
        )
        files = {file.path: file for file in snapshot.files}
        entries = [_entry(snapshot, files, candidate) for candidate in page]
        next_cursor = self._next_cursor(snapshot, query, next_offset, len(paths))
        return {
            "path": root,
            "entries": entries,
            "nextCursor": next_cursor,
            "truncated": next_cursor is not None,
            "scanned": scanned,
        }

    async def glob(self, pattern: str, cursor: str, limit: int) -> dict[str, Any]:
        try:
            normalized = normalize_project_glob(pattern)
        except ProjectPathError:
            raise FilesystemToolError("workspace_path_invalid") from None
        limit = _limit(limit)
        snapshot = await self._snapshot()
        query = _query_digest({"tool": "glob", "pattern": normalized})
        offset = self._cursor_offset(cursor, snapshot, query).offset if cursor else 0
        paths = _all_paths(snapshot)
        page, next_offset, scanned = _scan_page(
            paths,
            offset,
            limit,
            lambda candidate: project_glob_matches(candidate, normalized),
        )
        next_cursor = self._next_cursor(snapshot, query, next_offset, len(paths))
        files = {file.path: file for file in snapshot.files}
        return {
            "matches": [_entry(snapshot, files, candidate) for candidate in page],
            "nextCursor": next_cursor,
            "truncated": next_cursor is not None,
            "scanned": scanned,
        }

    async def read_file(self, path: str, start_line: int, max_lines: int) -> dict[str, Any]:
        normalized = _path(path, allow_root=False)
        if (
            not isinstance(start_line, int)
            or isinstance(start_line, bool)
            or start_line <= 0
            or not isinstance(max_lines, int)
            or isinstance(max_lines, bool)
            or max_lines <= 0
            or max_lines > MAX_READ_LINES
        ):
            raise FilesystemToolError("workspace_limit_exceeded")
        snapshot = await self._snapshot()
        file = _text_file(snapshot, normalized)
        if file.size > MAX_READ_SCAN_BYTES:
            raise FilesystemToolError("workspace_limit_exceeded")
        selected, total_lines, used, line_truncated = _read_line_window(
            file.text, start_line, max_lines
        )
        if start_line > total_lines + 1:
            raise FilesystemToolError("workspace_not_found")
        next_line = start_line + len(selected)
        truncated = next_line <= total_lines or line_truncated
        return {
            "path": normalized,
            "startLine": start_line,
            "lines": selected,
            "totalLines": total_lines,
            "nextLine": next_line if next_line <= total_lines else None,
            "returnedBytes": used,
            "truncated": truncated,
        }

    async def grep(
        self,
        pattern: str,
        path: str,
        glob: str,
        regex: bool,
        case_sensitive: bool,
        cursor: str,
        limit: int,
    ) -> dict[str, Any]:
        if (
            not isinstance(pattern, str)
            or not pattern
            or len(pattern) > MAX_GREP_PATTERN_CHARS
            or "\x00" in pattern
            or not isinstance(regex, bool)
            or not isinstance(case_sensitive, bool)
        ):
            raise FilesystemToolError("workspace_search_invalid")
        root = _path(path, allow_root=True)
        try:
            normalized_glob = normalize_project_glob(glob)
        except ProjectPathError:
            raise FilesystemToolError("workspace_path_invalid") from None
        limit = _limit(limit)
        matcher = _grep_matcher(pattern, regex, case_sensitive)
        snapshot = await self._snapshot()
        _require_grep_root(snapshot, root)
        query = _query_digest(
            {
                "tool": "grep",
                "pattern": pattern,
                "path": root,
                "glob": normalized_glob,
                "regex": regex,
                "caseSensitive": case_sensitive,
            }
        )
        position = (
            self._cursor_offset(cursor, snapshot, query)
            if cursor
            else _Cursor(snapshot="", query="", offset=0)
        )
        files = snapshot.files
        file_index = position.offset
        line_index = position.line
        results: list[dict[str, Any]] = []
        scanned_paths = 0
        scanned_bytes = 0
        incomplete = False
        deadline = time.monotonic() + MAX_GREP_SECONDS
        stopped = False
        while file_index < len(files):
            if time.monotonic() >= deadline or scanned_paths >= MAX_SCAN_PATHS:
                stopped = True
                break
            file = files[file_index]
            scanned_paths += 1
            relative = (
                file.path.rsplit("/", 1)[-1] if file.path == root else _relative(file.path, root)
            )
            if relative is None or not project_glob_matches(relative, normalized_glob):
                file_index += 1
                line_index = 0
                continue
            visible_text = file.text
            if file.size > MAX_GREP_BYTES:
                visible_text = _utf8_prefix(
                    file.text[: MAX_GREP_BYTES // 4].encode("utf-8"), MAX_GREP_BYTES
                )
                incomplete = True
            lines = visible_text.splitlines()
            while line_index < len(lines):
                if time.monotonic() >= deadline or scanned_bytes >= MAX_GREP_BYTES:
                    stopped = True
                    break
                raw_line = lines[line_index]
                encoded = raw_line.encode("utf-8")
                scanned_bytes += min(len(encoded), MAX_GREP_BYTES)
                candidate = raw_line
                if len(candidate) > MAX_GREP_LINE_CHARS:
                    candidate = candidate[:MAX_GREP_LINE_CHARS]
                    incomplete = True
                matched = matcher(candidate)
                if matched:
                    excerpt = candidate[:MAX_EXCERPT_CHARS]
                    results.append(
                        {
                            "path": file.path,
                            "line": line_index + 1,
                            "excerpt": excerpt,
                            "excerptTruncated": len(candidate) > len(excerpt),
                        }
                    )
                line_index += 1
                if len(results) >= limit:
                    stopped = True
                    break
            if stopped:
                if line_index >= len(lines):
                    file_index += 1
                    line_index = 0
                break
            file_index += 1
            line_index = 0
        next_cursor = (
            self._encode_cursor(snapshot, query, file_index, line_index)
            if stopped and file_index < len(files)
            else None
        )
        return {
            "matches": results,
            "nextCursor": next_cursor,
            "truncated": next_cursor is not None,
            "incomplete": incomplete,
            "scannedPaths": scanned_paths,
            "scannedBytes": min(scanned_bytes, MAX_GREP_BYTES),
        }

    async def _snapshot(self) -> WorkspaceSnapshot:
        if self._closed:
            raise FilesystemToolError("workspace_not_found")
        try:
            return await self._reader.snapshot()
        except WorkspaceStorageError as error:
            raise FilesystemToolError(
                error.args[0] if error.args else "workspace_not_found"
            ) from None
        except Exception:
            raise FilesystemToolError("workspace_unavailable") from None

    def _cursor_offset(self, value: str, snapshot: WorkspaceSnapshot, query: str) -> _Cursor:
        cursor = self._decode_cursor(value)
        if cursor.snapshot != _snapshot_token(snapshot) or cursor.query != query:
            raise FilesystemToolError("workspace_cursor_invalid")
        return cursor

    def _next_cursor(
        self, snapshot: WorkspaceSnapshot, query: str, offset: int, total: int
    ) -> str | None:
        return self._encode_cursor(snapshot, query, offset, 0) if offset < total else None

    def _encode_cursor(
        self, snapshot: WorkspaceSnapshot, query: str, offset: int, line: int
    ) -> str:
        body = jcs.canonicalize(
            {
                "snapshot": _snapshot_token(snapshot),
                "query": query,
                "offset": offset,
                "line": line,
            }
        )
        signature = hmac.digest(bytes(self._cursor_key), body, "sha256")
        return f"{_b64(body)}.{_b64(signature)}"

    def _decode_cursor(self, value: str) -> _Cursor:
        if not isinstance(value, str) or not value or len(value) > MAX_CURSOR_BYTES:
            raise FilesystemToolError("workspace_cursor_invalid")
        try:
            encoded_body, encoded_signature = value.split(".", 1)
            body = _unb64(encoded_body)
            signature = _unb64(encoded_signature)
            expected = hmac.digest(bytes(self._cursor_key), body, "sha256")
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            document = json.loads(body)
            if jcs.canonicalize(document) != body or set(document) != {
                "snapshot",
                "query",
                "offset",
                "line",
            }:
                raise ValueError
            if (
                not isinstance(document["snapshot"], str)
                or not isinstance(document["query"], str)
                or not isinstance(document["offset"], int)
                or isinstance(document["offset"], bool)
                or document["offset"] < 0
                or not isinstance(document["line"], int)
                or isinstance(document["line"], bool)
                or document["line"] < 0
            ):
                raise ValueError
            return _Cursor(**document)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            raise FilesystemToolError("workspace_cursor_invalid") from None


class _BaseFilesystemTool:
    name: str
    description: str

    def __init__(self, session: _FilesystemSession, metrics: ToolMetrics) -> None:
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
        return filesystem_tool_observation(self.name, tool_args, result)

    def _success(self, name: str, started: int, result: Mapping[str, Any]) -> None:
        self._metrics.record_tool_call(
            name,
            arguments={},
            result={
                "count": len(result.get("entries", result.get("matches", result.get("lines", [])))),
                "truncated": bool(result.get("truncated", False)),
            },
            duration_ms=_elapsed_ms(started),
        )

    def _failure(self, name: str, started: int, error: Exception) -> None:
        self._metrics.record_tool_call(
            name,
            arguments={},
            error=error,
            duration_ms=_elapsed_ms(started),
        )


class ListWorkspaceTool(_BaseFilesystemTool):
    name = "ls"
    description = "List immediate entries below one relative project workspace directory."

    async def __call__(
        self, path: str = "", cursor: str = "", limit: int = MAX_PAGE_ITEMS
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.ls(path, cursor, limit)
            self._success(self.name, started, result)
            return result
        except Exception as error:
            self._failure(self.name, started, error)
            raise


class GlobWorkspaceTool(_BaseFilesystemTool):
    name = "glob"
    description = "Find sorted project-relative paths with path-aware glob syntax."

    async def __call__(
        self, pattern: str, cursor: str = "", limit: int = MAX_PAGE_ITEMS
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.glob(pattern, cursor, limit)
            self._success(self.name, started, result)
            return result
        except Exception as error:
            self._failure(self.name, started, error)
            raise


class ReadWorkspaceFileTool(_BaseFilesystemTool):
    name = "read_file"
    description = "Read bounded numbered UTF-8 lines from one project-relative file."

    async def __call__(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = DEFAULT_READ_LINES,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.read_file(path, start_line, max_lines)
            self._success(self.name, started, result)
            return result
        except Exception as error:
            self._failure(self.name, started, error)
            raise


class GrepWorkspaceTool(_BaseFilesystemTool):
    name = "grep"
    description = "Search bounded project text with literal or timeout-bounded regex matching."

    async def __call__(
        self,
        pattern: str,
        path: str = "",
        glob: str = "**/*",
        regex: bool = False,
        case_sensitive: bool = True,
        cursor: str = "",
        limit: int = MAX_PAGE_ITEMS,
    ) -> dict[str, Any]:
        started = time.perf_counter_ns()
        try:
            result = await self._session.grep(
                pattern, path, glob, regex, case_sensitive, cursor, limit
            )
            self._success(self.name, started, result)
            return result
        except Exception as error:
            self._failure(self.name, started, error)
            raise


def _path(value: str, *, allow_root: bool) -> str:
    try:
        return normalize_project_path(value, allow_root=allow_root)
    except ProjectPathError:
        raise FilesystemToolError("workspace_path_invalid") from None


def _limit(value: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > MAX_PAGE_ITEMS
    ):
        raise FilesystemToolError("workspace_limit_exceeded")
    return value


def _all_paths(snapshot: WorkspaceSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(snapshot.directories)
            | {file.path for file in snapshot.files}
            | set(snapshot.binary_paths)
        )
    )


def _entry(
    snapshot: WorkspaceSnapshot, files: Mapping[str, WorkspaceTextFile], path: str
) -> dict[str, Any]:
    if path in snapshot.directories:
        return {"path": path, "type": "directory", "size": None}
    if path in files:
        return {"path": path, "type": "file", "size": files[path].size}
    return {"path": path, "type": "binary", "size": None}


def _scan_page(
    paths: tuple[str, ...],
    offset: int,
    limit: int,
    predicate: Callable[[str], bool],
) -> tuple[list[str], int, int]:
    if offset > len(paths):
        raise FilesystemToolError("workspace_cursor_invalid")
    result: list[str] = []
    scanned = 0
    index = offset
    while index < len(paths) and scanned < MAX_SCAN_PATHS and len(result) < limit:
        candidate = paths[index]
        index += 1
        scanned += 1
        if predicate(candidate):
            result.append(candidate)
    return result, index, scanned


def _immediate_child(path: str, root: str) -> bool:
    relative = _relative(path, root)
    return relative is not None and relative != "" and "/" not in relative


def _relative(path: str, root: str) -> str | None:
    if root == "":
        return path
    if path == root:
        return ""
    prefix = f"{root}/"
    return path[len(prefix) :] if path.startswith(prefix) else None


def _require_directory(snapshot: WorkspaceSnapshot, path: str) -> None:
    if path == "":
        return
    if path in snapshot.directories:
        return
    if path in snapshot.binary_paths or any(file.path == path for file in snapshot.files):
        raise FilesystemToolError("workspace_type_conflict")
    raise FilesystemToolError("workspace_not_found")


def _require_grep_root(snapshot: WorkspaceSnapshot, path: str) -> None:
    if path == "" or path in snapshot.directories:
        return
    if path in snapshot.binary_paths:
        raise FilesystemToolError("binary_file_unsupported")
    if any(file.path == path for file in snapshot.files):
        return
    raise FilesystemToolError("workspace_not_found")


def _text_file(snapshot: WorkspaceSnapshot, path: str) -> WorkspaceTextFile:
    for file in snapshot.files:
        if file.path == path:
            return file
    if path in snapshot.binary_paths:
        raise FilesystemToolError("binary_file_unsupported")
    if path in snapshot.directories:
        raise FilesystemToolError("workspace_type_conflict")
    raise FilesystemToolError("workspace_not_found")


def _read_line_window(
    text: str, start_line: int, max_lines: int
) -> tuple[list[dict[str, Any]], int, int, bool]:
    selected: list[dict[str, Any]] = []
    total = 0
    used = 0
    line_truncated = False
    start = 0
    index = 0
    length = len(text)
    while index < length:
        character = text[index]
        if character not in "\r\n":
            index += 1
            continue
        newline = "lf" if character == "\n" else "cr"
        end = index
        index += 1
        if character == "\r" and index < length and text[index] == "\n":
            newline = "crlf"
            index += 1
        total += 1
        if _should_collect_line(selected, total, start_line, max_lines, used, line_truncated):
            used, line_truncated = _collect_line(
                selected,
                total,
                text[start:end],
                newline,
                used,
                line_truncated,
            )
        start = index
    if start < length:
        total += 1
        if _should_collect_line(selected, total, start_line, max_lines, used, line_truncated):
            used, line_truncated = _collect_line(
                selected,
                total,
                text[start:],
                "none",
                used,
                line_truncated,
            )
    return selected, total, used, line_truncated


def _collect_line(
    selected: list[dict[str, Any]],
    number: int,
    text: str,
    newline: str,
    used: int,
    already_truncated: bool,
) -> tuple[int, bool]:
    encoded = text.encode("utf-8")
    remaining = MAX_READ_BYTES - used
    truncated = len(encoded) > remaining
    visible = _utf8_prefix(encoded, remaining) if truncated else text
    selected.append(
        {
            "number": number,
            "text": visible,
            "newline": newline,
            "truncated": truncated,
        }
    )
    return used + len(visible.encode("utf-8")), already_truncated or truncated


def _should_collect_line(
    selected: list[dict[str, Any]],
    number: int,
    start_line: int,
    max_lines: int,
    used: int,
    truncated: bool,
) -> bool:
    return (
        number >= start_line
        and len(selected) < max_lines
        and used < MAX_READ_BYTES
        and not truncated
    )


def _grep_matcher(pattern: str, regex: bool, case_sensitive: bool) -> Callable[[str], bool]:
    if regex:
        try:
            compiled = bounded_regex.compile(
                pattern, 0 if case_sensitive else bounded_regex.IGNORECASE
            )
        except bounded_regex.error:
            raise FilesystemToolError("workspace_search_invalid") from None

        def matches(value: str) -> bool:
            try:
                return compiled.search(value, timeout=REGEX_LINE_TIMEOUT_SECONDS) is not None
            except TimeoutError:
                raise FilesystemToolError("workspace_search_timeout") from None

        return matches
    needle = pattern if case_sensitive else pattern.casefold()
    return lambda value: needle in (value if case_sensitive else value.casefold())


def _snapshot_token(snapshot: WorkspaceSnapshot) -> str:
    document = {"managedDigest": snapshot.digest, "binaryPaths": list(snapshot.binary_paths)}
    return "sha256:" + hashlib.sha256(jcs.canonicalize(document)).hexdigest()


def _query_digest(document: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(jcs.canonicalize(dict(document))).hexdigest()


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


def _utf8_prefix(value: bytes, maximum: int) -> str:
    return value[:maximum].decode("utf-8", errors="ignore")


def _elapsed_ms(started: int) -> int:
    return max(0, (time.perf_counter_ns() - started) // 1_000_000)
