"""Safe, bounded read-only source analysis over an exact ZIP artifact."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import regex as bounded_regex

from contractor_runtime.adapters import AdapterHandles
from contractor_runtime.adapters.host import EMPTY_ADAPTER_HANDLES
from contractor_runtime.artifacts import ArtifactClient
from contractor_runtime.contracts import ArtifactRef, RuntimeSettings
from contractor_runtime.contracts.settings import (
    SourceArchiveLimits,
    default_source_archive_limits,
)
from contractor_runtime.core.input_errors import ToolInputError
from contractor_runtime.core.lines import line_spans, split_lines
from contractor_runtime.projectfs.operation_guard import WorkspaceOperationGuard
from contractor_runtime.projectfs.paths import (
    ProjectPathError,
    compile_project_glob,
    normalize_project_glob,
)
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.toolsets.common.artifact_visibility import (
    require_model_visible_binding,
)
from contractor_runtime.toolsets.common.artifacts import (
    ArtifactClientFactory,
    _reject_unconfigured_client,
)
from contractor_runtime.toolsets.common.factory import require_metrics, require_selected_tools
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_BYTES as MAX_VISIBLE_READ_BYTES,
)
from contractor_runtime.toolsets.common.line_window import (
    MAX_READ_LINES,
    bounded_line_window,
)
from contractor_runtime.toolsets.common.resources import ToolResources
from contractor_runtime.toolsets.common.source_archive import (
    _extract_archive,
    _SourceFile,
    _validated_member_path,
)
from contractor_runtime.toolsets.common.tool_base import SessionArtifactTool
from contractor_runtime.toolsets.common.versions import ModelArtifacts
from contractor_runtime.workspace import AllocationWorkspace

MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_QUERY_CHARS = 512
MAX_SEARCH_SCANNED_BYTES = 32 * 1024 * 1024
MAX_SEARCH_SNIPPET_CHARS = 500
MAX_SEARCH_SECONDS = 2.0
REGEX_LINE_TIMEOUT_SECONDS = 0.01
DEFAULT_READ_LINES = 200


class SourceAnalysisToolsetFactory:
    ref = "source-analysis@1"
    exported_tools = frozenset(
        {"open_source_archive", "list_source_files", "search_source", "read_source"}
    )
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _reject_unconfigured_client

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
        resources: ToolResources,
        adapter_handles: AdapterHandles = EMPTY_ADAPTER_HANDLES,
        project_workspace: Any = None,
    ) -> Mapping[str, Any]:
        del run_id, namespace, adapter_handles
        require_selected_tools(selected, self.exported_tools)
        metrics = require_metrics(resources, "source-analysis@1")
        client = ModelArtifacts(
            self._client_factory(allocation_id, runtime_settings), resources.selected_versions
        )
        session = _SourceArchiveSession(
            client,
            workspace,
            operation_timeout_seconds=runtime_settings.request_timeout_seconds,
            limits=runtime_settings.source_archive_limits,
        )
        resources.source_evidence = session.capture_evidence
        secrets = runtime_secrets(runtime_settings)
        builders: dict[str, Callable[[], Any]] = {
            "open_source_archive": lambda: OpenSourceArchiveTool(session, client, metrics, secrets),
            "list_source_files": lambda: ListSourceFilesTool(session, client, metrics, secrets),
            "search_source": lambda: SearchSourceTool(session, client, metrics, secrets),
            "read_source": lambda: ReadSourceTool(session, client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


@dataclass(frozen=True, slots=True)
class _OpenSummary:
    artifact: ArtifactRef
    files: tuple[_SourceFile, ...]
    total_uncompressed_bytes: int
    ignored_count: int

    def payload(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.model_dump(by_alias=True),
            "fileCount": len(self.files),
            "totalUncompressedBytes": self.total_uncompressed_bytes,
            "ignoredCount": self.ignored_count,
        }


class _SourceArchiveSession:
    def __init__(
        self,
        client: ArtifactClient,
        workspace: AllocationWorkspace,
        *,
        operation_timeout_seconds: float = 120,
        limits: SourceArchiveLimits | None = None,
    ) -> None:
        self._limits = limits or default_source_archive_limits()
        self._client = client
        self._workspace = workspace
        self._source_path = workspace.path / "source"
        self._lock = asyncio.Lock()
        self._guard = WorkspaceOperationGuard()
        self._operation_timeout_seconds = operation_timeout_seconds
        self._closed = False
        self._summary: _OpenSummary | None = None
        self._files: dict[str, _SourceFile] = {}

    async def open(self, ref: ArtifactRef) -> _OpenSummary:
        require_model_visible_binding(ref.namespace, ref.name)
        exact = ref.require_exact()
        async with self._lock:
            self._require_available()
            if (
                self._summary is not None
                and self._summary.artifact == exact
                and self._source_path.is_dir()
                and not self._source_path.is_symlink()
            ):
                return self._summary
            value = await self._client.read_artifact(exact)
            if value.artifact != exact:
                raise ValueError("Artifact API did not preserve the requested exact revision")
            if value.media_type != "application/zip":
                raise ValueError("source artifact media type must be application/zip")
            staging = self._workspace.path / f".source-staging-{uuid.uuid4().hex}"
            if staging.parent != self._workspace.path:
                raise RuntimeError("source staging path escaped allocation workspace")

            def extract_and_install() -> tuple[list[_SourceFile], int, int]:
                try:
                    extracted = _extract_archive(value.data, staging, self._limits)
                    # Cancellation retains this operation's ownership but must
                    # not start another publication after extraction completes.
                    self._require_available()
                    self._install_staging(staging)
                    return extracted
                finally:
                    _remove_path(staging)

            files, total_bytes, ignored_count = await self._guard.run(
                extract_and_install, deadline=self._deadline()
            )
            summary = _OpenSummary(
                artifact=value.artifact,
                files=tuple(files),
                total_uncompressed_bytes=total_bytes,
                ignored_count=ignored_count,
            )
            self._summary = summary
            self._files = {item.path: item for item in files}
            return summary

    async def list_files(self, pattern: str, offset: int, limit: int) -> dict[str, Any]:
        _validate_pattern(pattern)
        _validate_page(offset, limit)
        async with self._lock:
            # The summary's file tuple is immutable; matching it in the guard's
            # thread keeps up to maxEntries paths off the event loop (WSP-R179).
            files = self._require_open().files
            return await self._guard.run(
                lambda: _list_page(files, pattern, offset, limit), deadline=self._deadline()
            )

    async def search(
        self,
        *,
        query: str,
        path_pattern: str,
        regex: bool,
        case_sensitive: bool,
        max_results: int,
    ) -> dict[str, Any]:
        _validate_search(query, path_pattern, regex, case_sensitive, max_results)
        async with self._lock:
            files = self._require_open().files

            def select_and_scan() -> dict[str, Any]:
                return self._search_files(
                    _select_files(files, path_pattern), query, regex, case_sensitive, max_results
                )

            return await self._guard.run(select_and_scan, deadline=self._deadline())

    async def capture_evidence(self, path: str, start_line: int, end_line: int) -> dict:
        normalized = _validate_requested_path(path)
        _validate_read_window(start_line, end_line - start_line + 1)
        async with self._lock:
            summary = self._require_open()
            source = self._files.get(normalized)
            if source is None:
                raise ToolInputError("selected code source is unavailable")
            window = await self._guard.run(
                lambda: self._read_window(source, start_line, end_line - start_line + 1),
                deadline=self._deadline(),
            )
            if window["partialLine"] or window["endLine"] != end_line:
                raise ToolInputError("selected code range exceeds the source window")
            return {
                "schema": "contractor.code-evidence.v1",
                "source": {
                    "artifact": summary.artifact.model_dump(exclude_none=True),
                    "file": normalized,
                    "startLine": start_line,
                    "endLine": end_line,
                },
                "text": window["text"],
                "origin": "pinned-source",
            }

    async def read(self, path: str, start_line: int, max_lines: int) -> dict[str, Any]:
        normalized = _validate_requested_path(path)
        _validate_read_window(start_line, max_lines)
        async with self._lock:
            self._require_open()
            source = self._files.get(normalized)
            if source is None:
                raise ToolInputError(
                    "source path is absent or not a readable text file",
                    code="source_path_not_found",
                )
            return await self._guard.run(
                lambda: self._read_window(source, start_line, max_lines),
                deadline=self._deadline(),
            )

    def _read_window(self, source: _SourceFile, start_line: int, max_lines: int) -> dict[str, Any]:
        content = self._read_file(source)
        window: list[str] = []
        total_lines = 0
        for total_lines, (start, end) in enumerate(line_spans(content), start=1):
            if start_line <= total_lines < start_line + max_lines:
                window.append(content[start:end])
        if total_lines and start_line > total_lines:
            raise ToolInputError("start_line exceeds source file line count")
        selected, relative_end, partial_line = bounded_line_window(
            window, start_line=1, max_lines=max_lines, max_bytes=MAX_VISIBLE_READ_BYTES
        )
        end_line = start_line + relative_end - 1 if relative_end else 0
        return {
            "path": source.path,
            "size": source.size,
            "totalLines": total_lines,
            "startLine": start_line,
            "endLine": end_line,
            "text": selected,
            "truncated": partial_line or end_line < total_lines,
            "partialLine": partial_line,
        }

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            await self._guard.close(
                lambda: _remove_path(self._source_path), deadline=self._deadline()
            )
            self._files.clear()
            self._summary = None

    def _deadline(self) -> float:
        return time.monotonic() + self._operation_timeout_seconds

    def _require_available(self) -> None:
        if self._closed or self._guard.fenced:
            raise ValueError("source archive session is unavailable")

    def _install_staging(self, staging: Path) -> None:
        if self._source_path.is_symlink() or (
            self._source_path.exists() and not self._source_path.is_dir()
        ):
            raise ValueError("allocation source path is not a managed directory")
        backup = self._workspace.path / f".source-backup-{uuid.uuid4().hex}"
        had_current = self._source_path.exists()
        if had_current:
            self._source_path.rename(backup)
        try:
            staging.rename(self._source_path)
        except BaseException:
            if had_current and backup.exists():
                backup.rename(self._source_path)
            raise
        if had_current:
            _remove_path(backup)

    def _search_files(
        self,
        files: tuple[_SourceFile, ...],
        query: str,
        use_regex: bool,
        case_sensitive: bool,
        max_results: int,
    ) -> dict[str, Any]:
        compiled: Any | None = None
        needle = query if case_sensitive else query.casefold()
        if use_regex:
            flags = 0 if case_sensitive else bounded_regex.IGNORECASE
            try:
                compiled = bounded_regex.compile(query, flags)
            except bounded_regex.error as error:
                raise ToolInputError("query is not a valid regular expression") from error
        matches: list[dict[str, Any]] = []
        scanned_bytes = 0
        scanned_files = 0
        truncated = False
        deadline = time.monotonic() + MAX_SEARCH_SECONDS
        for source in files:
            if scanned_bytes + source.size > MAX_SEARCH_SCANNED_BYTES:
                truncated = True
                break
            if time.monotonic() >= deadline:
                truncated = True
                break
            content = self._read_file(source)
            scanned_bytes += source.size
            scanned_files += 1
            for line_number, line in enumerate(split_lines(content), start=1):
                if time.monotonic() >= deadline:
                    truncated = True
                    break
                try:
                    found = (
                        compiled.search(line, timeout=REGEX_LINE_TIMEOUT_SECONDS) is not None
                        if compiled is not None
                        else needle in (line if case_sensitive else line.casefold())
                    )
                except TimeoutError as error:
                    raise ToolInputError("regular expression search timed out") from error
                if not found:
                    continue
                matches.append(
                    {
                        "path": source.path,
                        "line": line_number,
                        "text": line[:MAX_SEARCH_SNIPPET_CHARS],
                        "textTruncated": len(line) > MAX_SEARCH_SNIPPET_CHARS,
                    }
                )
                if len(matches) >= max_results:
                    truncated = True
                    return {
                        "matches": matches,
                        "scannedFiles": scanned_files,
                        "scannedBytes": scanned_bytes,
                        "truncated": truncated,
                    }
        return {
            "matches": matches,
            "scannedFiles": scanned_files,
            "scannedBytes": scanned_bytes,
            "truncated": truncated,
        }

    def _read_file(self, source: _SourceFile) -> str:
        path = self._source_path.joinpath(*PurePosixPath(source.path).parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("managed source file is absent or no longer regular")
        data = path.read_bytes()
        if len(data) != source.size or hashlib.sha256(data).hexdigest() != source.digest:
            raise ValueError("managed source file changed after archive materialization")
        try:
            return data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("managed source file is no longer valid UTF-8") from error

    def _require_open(self) -> _OpenSummary:
        self._require_available()
        if self._summary is None:
            raise ToolInputError(
                "open_source_archive must be called first", code="source_archive_not_open"
            )
        return self._summary


_BaseSourceTool = SessionArtifactTool[_SourceArchiveSession]


class OpenSourceArchiveTool(_BaseSourceTool):
    name = "open_source_archive"
    description = """Open an exact ZIP artifact as a bounded read-only source tree.

    Call this before list_source_files, search_source or read_source. The archive
    is separate from the editable project workspace.

    Args:
        namespace: Source artifact namespace.
        name: Artifact binding name with media type application/zip.

    Returns:
        The exact artifact reference, fileCount, totalUncompressedBytes and
        ignoredCount for the opened archive.
    """

    async def __call__(self, namespace: str, name: str) -> dict[str, Any]:
        arguments = {"namespace": namespace, "name": name}
        return await self._call(
            arguments,
            self._open(namespace, name),
            lambda result: dict(result),
        )

    async def _open(self, namespace: str, name: str) -> dict[str, Any]:
        ref = await self._client.select(namespace, name)
        summary = await self._session.open(ref)
        return summary.payload()


class ListSourceFilesTool(_BaseSourceTool):
    name = "list_source_files"
    description = """List files in the archive opened by open_source_archive.

    Args:
        pattern: Archive-relative path glob where ** spans directories; defaults to "**/*".
        offset: Zero-based pagination offset; defaults to 0.
        limit: Maximum files per page, from 1 to 200; defaults to 200.

    Returns:
        Sorted relative file paths and sizes, offset, limit, total and truncated.
    """

    async def __call__(
        self, pattern: str = "**/*", offset: int = 0, limit: int = MAX_LIST_RESULTS
    ) -> dict[str, Any]:
        arguments = {"pattern": pattern, "offset": offset, "limit": limit}
        return await self._call(
            arguments,
            self._session.list_files(pattern, offset, limit),
            lambda result: {
                "offset": result["offset"],
                "limit": result["limit"],
                "total": result["total"],
                "returned": len(result["files"]),
                "truncated": result["truncated"],
            },
        )


class SearchSourceTool(_BaseSourceTool):
    name = "search_source"
    description = """Search UTF-8 files in the archive opened by open_source_archive.

    Search has time, scan and output limits. Narrow query or path_pattern if the
    result is truncated.

    Args:
        query: Non-empty literal string or regular expression, at most 512 characters.
        path_pattern: Archive-relative file glob where ** spans directories; defaults
            to "**/*".
        regex: Interpret query as a regular expression; defaults to false.
        case_sensitive: Match letter case; defaults to false.
        max_results: Maximum matches to return, from 1 to 100; defaults to 100.

    Returns:
        File/line matches and snippets, scannedFiles, scannedBytes and truncated.
    """

    async def __call__(
        self,
        query: str,
        path_pattern: str = "**/*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> dict[str, Any]:
        arguments = {
            "query_sha256": hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest()
            if isinstance(query, str)
            else "invalid",
            "query_chars": len(query) if isinstance(query, str) else 0,
            "path_pattern": path_pattern,
            "regex": regex,
            "case_sensitive": case_sensitive,
            "max_results": max_results,
        }
        return await self._call(
            arguments,
            self._session.search(
                query=query,
                path_pattern=path_pattern,
                regex=regex,
                case_sensitive=case_sensitive,
                max_results=max_results,
            ),
            lambda result: {
                "matches": len(result["matches"]),
                "scannedFiles": result["scannedFiles"],
                "scannedBytes": result["scannedBytes"],
                "truncated": result["truncated"],
            },
        )


class ReadSourceTool(_BaseSourceTool):
    name = "read_source"
    description = """Read a UTF-8 line window from the archive opened by open_source_archive.

    Output is limited to 128 KiB. Continue at endLine + 1; partialLine means one
    line alone exceeds the limit and only its prefix is returned.

    Args:
        path: Exact archive-relative file path from list_source_files or search_source.
        start_line: First line to read, 1-based and inclusive; defaults to 1.
        max_lines: Maximum lines to return, from 1 to 400; defaults to 200.

    Returns:
        File metadata, text, startLine, endLine, totalLines, truncated and partialLine.
    """

    async def __call__(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = DEFAULT_READ_LINES,
    ) -> dict[str, Any]:
        arguments = {"path": path, "start_line": start_line, "max_lines": max_lines}
        return await self._call(
            arguments,
            self._session.read(path, start_line, max_lines),
            lambda result: {
                "path": result["path"],
                "size": result["size"],
                "totalLines": result["totalLines"],
                "startLine": result["startLine"],
                "endLine": result["endLine"],
                "visibleUtf8Bytes": len(result["text"].encode("utf-8")),
                "truncated": result["truncated"],
                "partialLine": result["partialLine"],
            },
        )


def _validate_pattern(pattern: str) -> None:
    if (
        not isinstance(pattern, str)
        or not pattern
        or len(pattern) > 256
        or "\x00" in pattern
        or "\\" in pattern
        or pattern.startswith("/")
        or ".." in PurePosixPath(pattern).parts
    ):
        raise ToolInputError("path_pattern must be a relative glob of at most 256 characters")
    try:
        normalize_project_glob(pattern)
    except ProjectPathError:
        raise ToolInputError(
            "path_pattern must use / separators with ** only as a whole component"
        ) from None


def _path_matcher(pattern: str) -> Callable[[str], bool]:
    """Compile ``pattern`` once; a path that is no project path never matches."""

    if pattern in {"**", "**/*"}:
        return lambda _path: True
    try:
        glob = compile_project_glob(pattern)
    except ProjectPathError:
        return lambda _path: False

    def matches(path: str) -> bool:
        try:
            return glob.matches(path)
        except ProjectPathError:
            return False

    return matches


def _select_files(files: Sequence[_SourceFile], pattern: str) -> tuple[_SourceFile, ...]:
    """Return the files whose path matches ``pattern``, in path order.

    The work grows with the admitted entry count (LIM-GIT-16), so callers run it
    in the session guard's worker thread, never on the event loop (WSP-R179).
    """

    matches = _path_matcher(pattern)
    return tuple(sorted((item for item in files if matches(item.path)), key=_path_order))


def _path_order(item: _SourceFile) -> str:
    return item.path


def _list_page(
    files: Sequence[_SourceFile], pattern: str, offset: int, limit: int
) -> dict[str, Any]:
    """Select one validated page of ``files``; runs in the guard's worker thread."""

    matched = _select_files(files, pattern)
    page = matched[offset : offset + limit]
    return {
        "files": [{"path": item.path, "size": item.size} for item in page],
        "offset": offset,
        "limit": limit,
        "total": len(matched),
        "truncated": offset + len(page) < len(matched),
    }


def _validate_page(offset: int, limit: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ToolInputError("offset must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_LIST_RESULTS
    ):
        raise ToolInputError(f"limit must be between 1 and {MAX_LIST_RESULTS}")


def _validate_search(
    query: str,
    path_pattern: str,
    regex: bool,
    case_sensitive: bool,
    max_results: int,
) -> None:
    if not isinstance(query, str) or not query or len(query) > MAX_SEARCH_QUERY_CHARS:
        raise ToolInputError(
            f"query must contain between 1 and {MAX_SEARCH_QUERY_CHARS} characters"
        )
    try:
        query.encode("utf-8")
    except UnicodeError:
        raise ToolInputError("query must be valid UTF-8") from None
    _validate_pattern(path_pattern)
    if not isinstance(regex, bool) or not isinstance(case_sensitive, bool):
        raise ToolInputError("regex and case_sensitive must be booleans")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or max_results < 1
        or max_results > MAX_SEARCH_RESULTS
    ):
        raise ToolInputError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")


def _validate_requested_path(raw: str) -> str:
    if not isinstance(raw, str):
        raise ToolInputError("source path is invalid")
    try:
        normalized = _validated_member_path(raw)
    except ValueError:
        raise ToolInputError("source path must be normalized and relative") from None
    if raw != normalized:
        raise ToolInputError("source path must be normalized and relative")
    return normalized


def _validate_read_window(start_line: int, max_lines: int) -> None:
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ToolInputError("start_line must be a positive integer")
    if (
        isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or max_lines < 1
        or max_lines > MAX_READ_LINES
    ):
        raise ToolInputError(f"max_lines must be between 1 and {MAX_READ_LINES}")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)
