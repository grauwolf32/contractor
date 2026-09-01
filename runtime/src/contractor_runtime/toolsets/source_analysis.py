"""Safe, bounded read-only source analysis over an exact ZIP artifact."""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import io
import shutil
import stat
import time
import uuid
import zipfile
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
from contractor_runtime.toolsets.run_artifacts import (
    ArtifactClientFactory,
    ToolMetrics,
    gateway_secrets,
)
from contractor_runtime.workspace import AllocationWorkspace

MAX_ARCHIVE_ENTRIES = 10_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_FILE_UNCOMPRESSED_BYTES = 4 * 1024 * 1024
MAX_PATH_UTF8_BYTES = 512
MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_QUERY_CHARS = 512
MAX_SEARCH_SCANNED_BYTES = 32 * 1024 * 1024
MAX_SEARCH_SNIPPET_CHARS = 500
MAX_SEARCH_SECONDS = 2.0
REGEX_LINE_TIMEOUT_SECONDS = 0.01
MAX_READ_LINES = 400
DEFAULT_READ_LINES = 200
MAX_VISIBLE_READ_BYTES = 128 * 1024

IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
)

BINARY_EXTENSIONS = frozenset(
    {
        ".7z",
        ".a",
        ".avi",
        ".bin",
        ".bmp",
        ".class",
        ".db",
        ".dll",
        ".dylib",
        ".eot",
        ".exe",
        ".gif",
        ".gz",
        ".ico",
        ".jar",
        ".jpeg",
        ".jpg",
        ".lockb",
        ".mov",
        ".mp3",
        ".mp4",
        ".o",
        ".otf",
        ".pdf",
        ".png",
        ".pyc",
        ".so",
        ".sqlite",
        ".tar",
        ".tiff",
        ".ttf",
        ".wav",
        ".webp",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)


class SourceAnalysisToolsetFactory:
    ref = "source-analysis@1"
    exported_tools = frozenset(
        {"open_source_archive", "list_source_files", "search_source", "read_source"}
    )
    infrastructure_channels = MappingProxyType({})

    def __init__(self, client_factory: ArtifactClientFactory | None = None) -> None:
        self._client_factory = client_factory or _unconfigured_client

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
    ) -> Mapping[str, Any]:
        del run_id, namespace, adapter_handles
        unknown = sorted(set(selected) - self.exported_tools)
        if unknown:
            raise ValueError(f"unknown selected tools: {', '.join(unknown)}")
        metrics = getattr(state, "metrics", None)
        if metrics is None or not callable(getattr(metrics, "record_tool_call", None)):
            raise TypeError("source-analysis@1 requires State.metrics")
        client = self._client_factory(allocation_id, runtime_settings)
        session = _SourceArchiveSession(client, workspace)
        secrets = gateway_secrets(runtime_settings)
        builders: dict[str, Callable[[], Any]] = {
            "open_source_archive": lambda: OpenSourceArchiveTool(session, client, metrics, secrets),
            "list_source_files": lambda: ListSourceFilesTool(session, client, metrics, secrets),
            "search_source": lambda: SearchSourceTool(session, client, metrics, secrets),
            "read_source": lambda: ReadSourceTool(session, client, metrics, secrets),
        }
        return {name: builders[name]() for name in selected}


@dataclass(frozen=True, slots=True)
class _SourceFile:
    path: str
    size: int


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
    def __init__(self, client: ArtifactClient, workspace: AllocationWorkspace) -> None:
        self._client = client
        self._workspace = workspace
        self._source_path = workspace.path / "source"
        self._lock = asyncio.Lock()
        self._summary: _OpenSummary | None = None
        self._files: dict[str, _SourceFile] = {}

    async def open(self, ref: ArtifactRef) -> _OpenSummary:
        exact = ref.require_exact()
        async with self._lock:
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
            try:
                files, total_bytes, ignored_count = await asyncio.to_thread(
                    _extract_archive, value.data, staging
                )
                await asyncio.to_thread(self._install_staging, staging)
            except BaseException:
                await asyncio.to_thread(_remove_path, staging)
                raise
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
            self._require_open()
            matched = [item for item in self._files.values() if _path_matches(item.path, pattern)]
            matched.sort(key=lambda item: item.path)
            page = matched[offset : offset + limit]
            return {
                "files": [{"path": item.path, "size": item.size} for item in page],
                "offset": offset,
                "limit": limit,
                "total": len(matched),
                "truncated": offset + len(page) < len(matched),
            }

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
            self._require_open()
            files = tuple(
                item
                for item in sorted(self._files.values(), key=lambda item: item.path)
                if _path_matches(item.path, path_pattern)
            )
            return await asyncio.to_thread(
                self._search_files,
                files,
                query,
                regex,
                case_sensitive,
                max_results,
            )

    async def read(self, path: str, start_line: int, max_lines: int) -> dict[str, Any]:
        normalized = _validate_requested_path(path)
        _validate_read_window(start_line, max_lines)
        async with self._lock:
            self._require_open()
            source = self._files.get(normalized)
            if source is None:
                raise ValueError("source path is absent or not a readable text file")
            content = self._read_file(source)
            lines = content.splitlines(keepends=True)
            if lines and start_line > len(lines):
                raise ValueError("start_line exceeds source file line count")
            selected, end_line, partial_line = _bounded_lines(
                lines, start_line=start_line, max_lines=max_lines
            )
            return {
                "path": normalized,
                "size": source.size,
                "totalLines": len(lines),
                "startLine": start_line,
                "endLine": end_line,
                "text": selected,
                "truncated": partial_line or end_line < len(lines),
                "partialLine": partial_line,
            }

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(_remove_path, self._source_path)
            self._files.clear()
            self._summary = None

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
                raise ValueError("query is not a valid regular expression") from error
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
            for line_number, line in enumerate(content.splitlines(), start=1):
                try:
                    found = (
                        compiled.search(line, timeout=REGEX_LINE_TIMEOUT_SECONDS) is not None
                        if compiled is not None
                        else needle in (line if case_sensitive else line.casefold())
                    )
                except TimeoutError as error:
                    raise ValueError("regular expression search timed out") from error
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
        if len(data) != source.size:
            raise ValueError("managed source file changed after archive materialization")
        try:
            return data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("managed source file is no longer valid UTF-8") from error

    def _require_open(self) -> None:
        if self._summary is None:
            raise ValueError("open_source_archive must be called first")


class _BaseSourceTool:
    name: str
    description: str

    def __init__(
        self,
        session: _SourceArchiveSession,
        client: ArtifactClient,
        metrics: ToolMetrics,
        secrets: tuple[str, ...],
    ) -> None:
        self._session = session
        self._client = client
        self._metrics = metrics
        self._secrets = secrets
        self.__name__ = self.name
        self.__doc__ = self.description

    @property
    def known_exact_refs(self) -> tuple[ArtifactRef, ...]:
        return tuple(getattr(self._client, "known_exact_refs", ()))

    async def close(self) -> None:
        await self._session.close()
        self._secrets = ()

    async def _call(
        self,
        arguments: Mapping[str, Any],
        operation: Any,
        metric_result: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        started_ns = time.perf_counter_ns()
        try:
            result = await operation
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                result=metric_result(result),
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            return result
        except Exception as error:
            self._metrics.record_tool_call(
                self.name,
                arguments=arguments,
                error=error,
                secrets=self._secrets,
                duration_ms=_elapsed_ms(started_ns),
            )
            raise


class OpenSourceArchiveTool(_BaseSourceTool):
    name = "open_source_archive"
    description = (
        "Open an exact application/zip Run artifact as this Worker's bounded read-only "
        "source tree. Call this before other source tools."
    )

    async def __call__(self, namespace: str, name: str, revision: str) -> dict[str, Any]:
        arguments = {"namespace": namespace, "name": name, "revision": revision}
        return await self._call(
            arguments,
            self._open(namespace, name, revision),
            lambda result: dict(result),
        )

    async def _open(self, namespace: str, name: str, revision: str) -> dict[str, Any]:
        summary = await self._session.open(
            ArtifactRef(namespace=namespace, name=name, revision=revision)
        )
        return summary.payload()


class ListSourceFilesTool(_BaseSourceTool):
    name = "list_source_files"
    description = "List bounded POSIX-relative paths in the opened source archive."

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
    description = (
        "Search opened UTF-8 source files with a fixed string or timeout-bounded regular "
        "expression and return file/line evidence."
    )

    async def __call__(
        self,
        query: str,
        path_pattern: str = "**/*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = MAX_SEARCH_RESULTS,
    ) -> dict[str, Any]:
        arguments = {
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "query_chars": len(query),
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
    description = "Read a bounded line window from one opened UTF-8 source file."

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


def _extract_archive(data: bytes, staging: Path) -> tuple[list[_SourceFile], int, int]:
    staging.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            entries = archive.infolist()
            _validate_entries(entries)
            files: list[_SourceFile] = []
            ignored_count = 0
            total_bytes = 0
            for info in entries:
                normalized = _validated_member_path(info.filename)
                if info.is_dir():
                    if _is_ignored_path(normalized):
                        ignored_count += 1
                    continue
                if (
                    _is_ignored_path(normalized)
                    or PurePosixPath(normalized).suffix.lower() in BINARY_EXTENSIONS
                ):
                    ignored_count += 1
                    continue
                try:
                    payload = _read_member_bounded(archive, info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                    raise ValueError("source ZIP member could not be read safely") from error
                try:
                    payload.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    ignored_count += 1
                    continue
                total_bytes += len(payload)
                target = staging.joinpath(*PurePosixPath(normalized).parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                target.write_bytes(payload)
                files.append(_SourceFile(path=normalized, size=len(payload)))
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError("source artifact is not a valid bounded ZIP") from error
    if not files:
        raise ValueError("source ZIP contains no readable UTF-8 files")
    files.sort(key=lambda item: item.path)
    return files, total_bytes, ignored_count


def _validate_entries(entries: list[zipfile.ZipInfo]) -> None:
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ValueError("source ZIP exceeds the 10000 entry limit")
    seen: set[str] = set()
    regular_files: set[str] = set()
    total_declared = 0
    normalized_entries: list[tuple[zipfile.ZipInfo, str]] = []
    for info in entries:
        normalized = _validated_member_path(info.filename)
        if normalized in seen:
            raise ValueError("source ZIP contains duplicate normalized paths")
        seen.add(normalized)
        if info.flag_bits & 0x1:
            raise ValueError("encrypted source ZIP members are not supported")
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        allowed_kind = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
        if kind not in {0, allowed_kind}:
            raise ValueError("source ZIP contains a link or special file")
        if info.file_size < 0 or info.file_size > MAX_FILE_UNCOMPRESSED_BYTES:
            raise ValueError("source ZIP member exceeds the 4 MiB file limit")
        total_declared += info.file_size
        if total_declared > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("source ZIP exceeds the 64 MiB uncompressed limit")
        normalized_entries.append((info, normalized))
        if not info.is_dir():
            regular_files.add(normalized)
    for _info, normalized in normalized_entries:
        parts = PurePosixPath(normalized).parts
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in regular_files:
                raise ValueError("source ZIP nests an entry below a regular file")


def _validated_member_path(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("source ZIP contains an unsafe member path")
    if len(raw.encode("utf-8")) > MAX_PATH_UTF8_BYTES:
        raise ValueError("source ZIP member path exceeds the 512-byte limit")
    candidate = raw[:-1] if raw.endswith("/") else raw
    raw_parts = candidate.split("/")
    if not candidate or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("source ZIP contains an unsafe member path")
    path = PurePosixPath(candidate)
    if path.is_absolute():
        raise ValueError("source ZIP contains an unsafe member path")
    if any(":" in part for part in path.parts):
        raise ValueError("source ZIP contains an unsafe member path")
    normalized = path.as_posix()
    if not normalized:
        raise ValueError("source ZIP contains an unsafe member path")
    return normalized


def _read_member_bounded(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with archive.open(info, mode="r") as source:
        payload = source.read(MAX_FILE_UNCOMPRESSED_BYTES + 1)
        if len(payload) > MAX_FILE_UNCOMPRESSED_BYTES or source.read(1):
            raise ValueError("source ZIP member exceeds the 4 MiB streamed limit")
    if len(payload) != info.file_size:
        raise ValueError("source ZIP member size differs from its declaration")
    return payload


def _is_ignored_path(path: str) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in PurePosixPath(path).parts)


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
        raise ValueError("path pattern is invalid")


def _path_matches(path: str, pattern: str) -> bool:
    if pattern in {"*", "**", "**/*"}:
        return True
    return fnmatch.fnmatchcase(path, pattern) or PurePosixPath(path).match(pattern)


def _validate_page(offset: int, limit: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > MAX_LIST_RESULTS
    ):
        raise ValueError(f"limit must be between 1 and {MAX_LIST_RESULTS}")


def _validate_search(
    query: str,
    path_pattern: str,
    regex: bool,
    case_sensitive: bool,
    max_results: int,
) -> None:
    if not isinstance(query, str) or not query or len(query) > MAX_SEARCH_QUERY_CHARS:
        raise ValueError(f"query must contain between 1 and {MAX_SEARCH_QUERY_CHARS} characters")
    _validate_pattern(path_pattern)
    if not isinstance(regex, bool) or not isinstance(case_sensitive, bool):
        raise ValueError("regex and case_sensitive must be booleans")
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or max_results < 1
        or max_results > MAX_SEARCH_RESULTS
    ):
        raise ValueError(f"max_results must be between 1 and {MAX_SEARCH_RESULTS}")


def _validate_requested_path(raw: str) -> str:
    if not isinstance(raw, str):
        raise ValueError("source path is invalid")
    normalized = _validated_member_path(raw)
    if raw != normalized:
        raise ValueError("source path must be normalized and relative")
    return normalized


def _validate_read_window(start_line: int, max_lines: int) -> None:
    if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line < 1:
        raise ValueError("start_line must be a positive integer")
    if (
        isinstance(max_lines, bool)
        or not isinstance(max_lines, int)
        or max_lines < 1
        or max_lines > MAX_READ_LINES
    ):
        raise ValueError(f"max_lines must be between 1 and {MAX_READ_LINES}")


def _bounded_lines(lines: list[str], *, start_line: int, max_lines: int) -> tuple[str, int, bool]:
    if not lines:
        return "", 0, False
    window = lines[start_line - 1 : start_line - 1 + max_lines]
    result: list[str] = []
    size = 0
    partial_line = False
    for line in window:
        encoded = line.encode("utf-8")
        remaining = MAX_VISIBLE_READ_BYTES - size
        if len(encoded) <= remaining:
            result.append(line)
            size += len(encoded)
            continue
        if not result and remaining > 0:
            result.append(encoded[:remaining].decode("utf-8", errors="ignore"))
            partial_line = True
        break
    complete_lines = len(result) - (1 if partial_line else 0)
    end_line = start_line + complete_lines - 1
    if partial_line:
        end_line = start_line
    return "".join(result), end_line, partial_line


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _unconfigured_client(allocation_id: str, runtime_settings: RuntimeSettings) -> ArtifactClient:
    del allocation_id, runtime_settings
    raise RuntimeError("Artifact transport is not configured")


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
