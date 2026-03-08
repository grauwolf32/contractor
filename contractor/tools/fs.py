from __future__ import annotations

import fnmatch
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Final, Iterable, Literal, Optional, Union

import fsspec
from fsspec.implementations.local import LocalFileSystem, stringify_path
from magika import ContentTypeInfo, Magika


_IGNORE_DEFAULTS: Final[list[str]] = [
    "*.pyc",
    "*/__pycache__/*",
    "__pycache__/*",
    "*.so",
    "*.dll",
    "*.bin",
    "*.o",
    "*.dylib",
    "*.jpg",
    "*.jpeg",
    "*.webp",
    "*.png",
    "*.svg",
    "*.heic",
    "*.mov",
    "*.mp4",
    "*.avi",
    "*.zip",
    "*.rar",
    "*.tar",
    "*.tar.gz",
    "*.DS_Store",
]

INCORRECT_REGEXP_ERROR: Final[str] = "regex {regex} is incorrect:\n{err}"
PATH_NOT_FOUND_ERROR: Final[str] = "path {path} is not exists"
PATH_IS_NOT_A_FILE_ERROR: Final[str] = "{path} is not a file"


def _norm_unicode(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return unicodedata.normalize("NFC", value)


def _normalize_slashes(path: str) -> str:
    return path.replace("\\", "/")


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _is_ignored(path: str, patterns: list[str]) -> bool:
    normalized = _normalize_slashes(path)
    basename = normalized.split("/")[-1]
    return any(
        fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern)
        for pattern in patterns
    )


@dataclass(slots=True)
class FileLoc:
    """
    Location inside a file.

    - line_start, line_end: 0-based inclusive line indexes
    - byte_start, byte_end: 0-based half-open byte range [start, end)
    - content: excerpt around the match
    """

    line_start: Optional[int] = None
    line_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    content: Optional[str] = None


@dataclass(slots=True)
class FsEntry:
    name: str
    path: str
    size: int
    is_dir: bool = False
    filetype: Optional[ContentTypeInfo] = None
    loc: Optional[FileLoc] = None

    _magika: Magika = Magika()

    @staticmethod
    @lru_cache(maxsize=2048)
    def identify_type(
        file_path: str,
        fs: fsspec.AbstractFileSystem,
    ) -> Optional[ContentTypeInfo]:
        if not fs.exists(file_path) or not fs.isfile(file_path):
            return None

        try:
            with fs.open(file_path, mode="rb") as f:
                return FsEntry._magika.identify_stream(f).output
        except Exception:
            return None

    @classmethod
    def from_path(
        cls,
        path: str,
        fs: fsspec.AbstractFileSystem,
        *,
        with_types: bool = True,
    ) -> Optional["FsEntry"]:
        normalized_path = _norm_unicode(path)
        if normalized_path is None or not fs.exists(normalized_path):
            return None

        name = _norm_unicode(normalized_path.rstrip("/").split("/")[-1]) or ""

        if fs.isdir(normalized_path):
            return cls(
                name=name,
                path=normalized_path,
                size=0,
                is_dir=True,
            )

        if fs.isfile(normalized_path):
            filetype = cls.identify_type(normalized_path, fs) if with_types else None
            return cls(
                name=name,
                path=normalized_path,
                size=int(fs.size(normalized_path)),
                is_dir=False,
                filetype=filetype,
            )

        return None

    @staticmethod
    def _compute_line_starts(text: str) -> list[int]:
        starts = [0]
        for match in re.finditer(r"\n", text):
            starts.append(match.end())
        return starts

    @staticmethod
    def _char_to_line(line_starts: list[int], char_pos: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= char_pos:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(0, hi)

    @classmethod
    def from_matches(
        cls,
        matches: list[re.Match[str]],
        file_path: str,
        fs: fsspec.AbstractFileSystem,
        *,
        content: Optional[str] = None,
        with_types: bool = True,
        excerpt_max_chars: int = 500,
        context_lines: int = 0,
    ) -> Optional[list["FsEntry"]]:
        if not fs.exists(file_path) or not fs.isfile(file_path):
            return None

        if not matches:
            return []

        if content is None:
            try:
                content = fs.read_text(file_path, encoding="utf-8", errors="ignore")
            except Exception:
                return []

        proto = cls.from_path(file_path, fs, with_types=with_types)
        if proto is None:
            return None

        line_starts = cls._compute_line_starts(content)
        lines = content.splitlines()

        entries: list[FsEntry] = []
        for match in matches:
            begin_char, end_char = match.span()
            line_idx = cls._char_to_line(line_starts, begin_char)

            line_start = max(0, line_idx - context_lines)
            line_end = min(len(lines) - 1, line_idx + context_lines)

            try:
                byte_start = len(content[:begin_char].encode("utf-8", errors="ignore"))
                byte_end = len(content[:end_char].encode("utf-8", errors="ignore"))
            except Exception:
                byte_start, byte_end = None, None

            excerpt = "\n".join(lines[line_start : line_end + 1])
            if len(excerpt) > excerpt_max_chars:
                excerpt = excerpt[:excerpt_max_chars] + "…"

            entries.append(
                cls(
                    name=proto.name,
                    path=proto.path,
                    size=proto.size,
                    is_dir=False,
                    filetype=proto.filetype,
                    loc=FileLoc(
                        line_start=line_start,
                        line_end=line_end,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        content=excerpt,
                    ),
                )
            )

        return sorted(
            entries, key=lambda entry: (entry.path, entry.loc.line_start or 0)
        )


@dataclass(slots=True)
class FileFormat:
    with_types: bool = True
    with_file_info: bool = True
    _format: Literal["str", "json", "xml"] = "json"
    loc: Literal["lines", "bytes"] = "lines"

    def _format_loc(self, loc: FileLoc) -> Union[str, dict[str, Any]]:
        if self.loc == "bytes":
            payload: dict[str, Any] = {
                "byte_start": loc.byte_start,
                "byte_end": loc.byte_end,
            }
        else:
            payload = {
                "line_start": loc.line_start,
                "line_end": loc.line_end,
            }

        if loc.content is not None:
            payload["content"] = loc.content

        if self._format == "str":
            return json.dumps(payload, ensure_ascii=False)

        if self._format == "xml":
            parts = ["<loc>"]
            for key, value in payload.items():
                parts.append(f"<{key}>{_xml_escape(str(value))}</{key}>")
            parts.append("</loc>")
            return "".join(parts)

        return payload

    def format_fs_entry(self, entry: FsEntry) -> Union[str, dict[str, Any]]:
        kind = "dir" if entry.is_dir else "file"
        payload: dict[str, Any] = {}

        if self.with_file_info:
            payload.update(
                {
                    "kind": kind,
                    "name": entry.name,
                    "path": entry.path,
                    "size": entry.size,
                }
            )

        if self.with_types and entry.filetype is not None:
            try:
                payload["filetype"] = asdict(entry.filetype)
            except Exception:
                payload["filetype"] = str(entry.filetype)

        if entry.loc is not None:
            payload["loc"] = self._format_loc(entry.loc)

        if self._format == "str":
            return json.dumps(payload, ensure_ascii=False)

        if self._format == "xml":
            parts = [f"<{kind}>"]
            xml_payload = dict(payload)
            xml_payload.pop("kind", None)

            for key, value in xml_payload.items():
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value, ensure_ascii=False)
                    parts.append(f"<{key}>{_xml_escape(serialized)}</{key}>")
                else:
                    parts.append(f"<{key}>{_xml_escape(str(value))}</{key}>")

            parts.append(f"</{kind}>")
            return "".join(parts)

        return payload

    def format_file_list(
        self,
        files: list[Optional[FsEntry]],
    ) -> Union[str, list[dict[str, Any]]]:
        cleaned = [file for file in files if file is not None]

        if self._format == "str":
            return "\n".join(str(self.format_fs_entry(file)) for file in cleaned)

        if self._format == "xml":
            inner = "".join(str(self.format_fs_entry(file)) for file in cleaned)
            return f"<files>{inner}</files>"

        return [self.format_fs_entry(file) for file in cleaned]  # type: ignore[return-value]

    @staticmethod
    def format_output(content: str, max_output: int) -> str:
        lines = content.splitlines(True)
        out_parts: list[str] = []
        out_bytes = 0
        cut_at_line: Optional[int] = None

        for index, line in enumerate(lines):
            line_bytes = len(line.encode("utf-8", errors="ignore"))
            if out_bytes + line_bytes > max_output:
                cut_at_line = index
                break

            out_parts.append(line)
            out_bytes += line_bytes

        if cut_at_line is None:
            return "".join(out_parts)

        remaining = max(0, len(lines) - cut_at_line)
        footer = (
            f"\n\n### truncated at line: {cut_at_line} ### "
            f"lines left in the file: {remaining} ###"
        )
        footer_bytes = len(footer.encode("utf-8", errors="ignore"))

        if footer_bytes > max_output:
            return footer[:max_output]

        while out_parts and (out_bytes + footer_bytes) > max_output:
            removed = out_parts.pop()
            out_bytes -= len(removed.encode("utf-8", errors="ignore"))

        return "".join(out_parts) + footer


class InteractionKind(str, Enum):
    READ = "read"
    MATCH = "match"


class CoverageFilter(str, Enum):
    ANY = "any"
    READ_ONLY = "read_only"
    MATCH_ONLY = "match_only"
    READ_AND_MATCH = "read_and_match"


@dataclass(slots=True)
class FileCoverageEntry:
    path: str
    read_count: int = 0
    match_count: int = 0
    operations: dict[str, int] = field(default_factory=dict)

    def touch(self, operation: str, *, interaction: InteractionKind) -> None:
        self.operations[operation] = self.operations.get(operation, 0) + 1

        if interaction == InteractionKind.READ:
            self.read_count += 1
        elif interaction == InteractionKind.MATCH:
            self.match_count += 1

    @property
    def has_read(self) -> bool:
        return self.read_count > 0

    @property
    def has_match(self) -> bool:
        return self.match_count > 0

    @property
    def is_covered(self) -> bool:
        return self.has_read or self.has_match

    def matches_filter(self, flt: CoverageFilter) -> bool:
        if flt == CoverageFilter.ANY:
            return self.is_covered
        if flt == CoverageFilter.READ_ONLY:
            return self.has_read and not self.has_match
        if flt == CoverageFilter.MATCH_ONLY:
            return self.has_match and not self.has_read
        if flt == CoverageFilter.READ_AND_MATCH:
            return self.has_read and self.has_match
        return False


class FsspecCoverageFileTools:
    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        fmt: FileFormat,
        *,
        max_output: int = 8 * 10**4,
        max_items: int = 300,
        ignored_patterns: Optional[list[str]] = None,
        with_types: bool = True,
        with_file_info: bool = True,
    ) -> None:
        self.fs = fs
        self.fmt = fmt
        self.max_output = max_output
        self.max_items = max_items
        self.with_types = with_types
        self.with_file_info = with_file_info

        self.fmt.with_types = with_types
        self.fmt.with_file_info = with_file_info

        self._coverage: dict[str, FileCoverageEntry] = {}

        patterns: list[str] = []
        for pattern in _IGNORE_DEFAULTS + (ignored_patterns or []):
            if pattern and pattern not in patterns:
                patterns.append(pattern)
        self.patterns = patterns

    def _norm(self, path: Optional[str]) -> Optional[str]:
        return _norm_unicode(path)

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _paginate(
        self,
        items: list[str],
        *,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> tuple[list[str], int, int]:
        offset = max(0, offset)
        resolved_limit = self.max_items if limit is None else max(1, limit)
        return items[offset : offset + resolved_limit], offset, resolved_limit

    def _read_text(
        self,
        file_path: str,
        *,
        operation: str,
        interaction: InteractionKind,
    ) -> str:
        content = self.fs.read_text(file_path, encoding="utf-8", errors="ignore")
        self.mark_interaction(file_path, operation, interaction=interaction)
        return content

    def _iter_all_files(self, root: str) -> list[str]:
        files: list[str] = []

        if not self.fs.exists(root):
            return files

        if self.fs.isfile(root):
            if not self._is_ignored(root):
                files.append(root)
            return sorted(set(files))

        for current_path, _dirs, filenames in self.fs.walk(root):
            for filename in filenames:
                full_path = (
                    str(current_path).rstrip("/") + "/" + str(filename)
                ).replace("\\", "/")
                if self._is_ignored(full_path):
                    continue
                if self.fs.isfile(full_path):
                    files.append(full_path)

        return sorted(set(files))

    def _match_glob(self, file_path: str, root: str, pattern: str) -> bool:
        file_path = _normalize_slashes(file_path)
        root = _normalize_slashes(root).rstrip("/") or "/"
        pattern = _normalize_slashes(pattern)

        if pattern == "**/*":
            return True

        if root == "/":
            relative = file_path.lstrip("/")
        else:
            prefix = root + "/"
            relative = (
                file_path[len(prefix) :] if file_path.startswith(prefix) else file_path
            )

        return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(file_path, pattern)

    def _matched_files(self, path: str, pattern: str) -> list[str]:
        return [
            file_path
            for file_path in self._iter_all_files(path)
            if self._match_glob(file_path, path, pattern)
        ]

    def _coverage_entry(self, path: str) -> Optional[FileCoverageEntry]:
        return self._coverage.get(path)

    def _is_covered_path(self, path: str) -> bool:
        entry = self._coverage_entry(path)
        return bool(entry and entry.is_covered)

    def _covered_files(
        self,
        files: Iterable[str],
        *,
        interaction: CoverageFilter = CoverageFilter.ANY,
    ) -> list[str]:
        selected: list[str] = []
        for path in files:
            entry = self._coverage_entry(path)
            if entry is not None and entry.matches_filter(interaction):
                selected.append(path)
        return sorted(selected)

    def _uncovered_files(self, files: Iterable[str]) -> list[str]:
        return sorted(path for path in files if not self._is_covered_path(path))

    def _serialize_coverage_entry(self, path: str) -> dict[str, Any]:
        entry = self._coverage_entry(path)
        if entry is None:
            return {
                "path": path,
                "has_read": False,
                "has_match": False,
                "read_count": 0,
                "match_count": 0,
                "operations": {},
            }

        return {
            "path": path,
            "has_read": entry.has_read,
            "has_match": entry.has_match,
            "read_count": entry.read_count,
            "match_count": entry.match_count,
            "operations": dict(entry.operations),
        }

    def mark_interaction(
        self,
        path: str,
        operation: str,
        *,
        interaction: InteractionKind,
    ) -> None:
        entry = self._coverage.get(path)
        if entry is None:
            entry = FileCoverageEntry(path=path)
            self._coverage[path] = entry

        entry.touch(operation, interaction=interaction)

    def reset_coverage(self) -> None:
        self._coverage.clear()

    def get_coverage(self) -> dict[str, Any]:
        return {
            "files_seen": len(self._coverage),
            "files": {
                path: {
                    "read_count": entry.read_count,
                    "match_count": entry.match_count,
                    "has_read": entry.has_read,
                    "has_match": entry.has_match,
                    "operations": dict(entry.operations),
                }
                for path, entry in sorted(self._coverage.items())
            },
        }

    def ls(self, path: str) -> dict[str, Any]:
        normalized_path = self._norm(path)
        if normalized_path is None or not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        try:
            items = self.fs.ls(normalized_path, detail=False)
        except TypeError:
            items = self.fs.ls(normalized_path)

        entries = [
            FsEntry.from_path(str(item), self.fs, with_types=self.with_types)
            for item in items
            if not self._is_ignored(str(item))
        ]
        return {"result": self.fmt.format_file_list(entries)}

    def glob(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern)

        if normalized_pattern is None:
            return {
                "result": [],
                "offset": offset,
                "total_items": 0,
                "limit": self.max_items,
            }

        if normalized_path and not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        matches = [str(match) for match in self.fs.glob(normalized_pattern)]

        prefix = normalized_path.rstrip("/").replace("\\", "/") + "/"
        if normalized_path != "/":
            matches = [
                match
                for match in matches
                if match.replace("\\", "/").startswith(prefix)
            ]

        entries = [
            FsEntry.from_path(match, self.fs, with_types=self.with_types)
            for match in matches
            if not self._is_ignored(match)
        ]
        entries = [entry for entry in entries if entry is not None]
        entries.sort(key=lambda entry: entry.path)

        total = len(entries)
        paged = entries[offset : offset + self.max_items]

        return {
            "result": self.fmt.format_file_list(paged),
            "offset": offset,
            "total_items": total,
            "limit": self.max_items,
        }

    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        normalized_file = self._norm(file_path)
        if (
            normalized_file is None
            or not self.fs.exists(normalized_file)
            or self._is_ignored(normalized_file)
        ):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_file)}
        if not self.fs.isfile(normalized_file):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized_file)}

        try:
            content = self._read_text(
                normalized_file,
                operation="read_file",
                interaction=InteractionKind.READ,
            )
        except Exception:
            return {"result": ""}

        lines = content.splitlines()

        if offset is not None:
            offset = max(0, offset)
            if offset >= len(lines):
                return {"result": ""}
            lines = lines[offset:]

        if limit is not None:
            lines = lines[: max(1, limit)]

        sliced = "\n".join(lines)
        return {"result": self.fmt.format_output(sliced, self.max_output)}

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern)

        try:
            regex = re.compile(normalized_pattern or "")
        except re.error as err:
            return {
                "error": INCORRECT_REGEXP_ERROR.format(
                    regex=normalized_pattern,
                    err=str(err),
                )
            }

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        def build_entries_for_file(file_path: str) -> list[FsEntry]:
            if self._is_ignored(file_path):
                return []

            try:
                content = self.fs.read_text(
                    file_path, encoding="utf-8", errors="ignore"
                )
            except Exception:
                return []

            matches = list(regex.finditer(content))
            if matches:
                self.mark_interaction(
                    file_path,
                    "grep",
                    interaction=InteractionKind.MATCH,
                )

            return (
                FsEntry.from_matches(
                    matches=matches,
                    file_path=file_path,
                    fs=self.fs,
                    content=content,
                    with_types=self.with_types,
                )
                or []
            )

        if self.fs.isfile(normalized_path):
            entries = build_entries_for_file(normalized_path)
            total = len(entries)
            paged = entries[offset : offset + self.max_items]

            return {
                "result": self.fmt.format_file_list(paged),
                "offset": offset,
                "total_items": total,
                "limit": self.max_items,
            }

        results: list[FsEntry] = []
        for current_path, _dirs, filenames in self.fs.walk(normalized_path):
            for filename in filenames:
                full_path = (
                    str(current_path).rstrip("/") + "/" + str(filename)
                ).replace("\\", "/")
                results.extend(build_entries_for_file(full_path))

        results.sort(key=lambda entry: (entry.path, entry.loc.line_start or 0))
        total = len(results)
        paged = results[offset : offset + self.max_items]

        return {
            "result": self.fmt.format_file_list(paged),
            "offset": offset,
            "total_items": total,
            "limit": self.max_items,
        }

    def coverage_stats(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        covered_files = self._covered_files(files)
        uncovered_files = self._uncovered_files(files)

        total = len(files)
        covered = len(covered_files)
        uncovered = len(uncovered_files)
        coverage_percent = round((covered / total) * 100, 2) if total else 100.0

        return {
            "result": {
                "path": normalized_path,
                "pattern": normalized_pattern,
                "total_files": total,
                "covered_files_count": covered,
                "uncovered_files_count": uncovered,
                "coverage_percent": coverage_percent,
            }
        }

    def covered(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        interaction: CoverageFilter = CoverageFilter.ANY,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if isinstance(interaction, str):
            interaction = CoverageFilter(interaction)

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        selected = self._covered_files(files, interaction=interaction)
        page, resolved_offset, resolved_limit = self._paginate(
            selected,
            offset=offset,
            limit=limit,
        )

        return {
            "result": [self._serialize_coverage_entry(path) for path in page],
            "offset": resolved_offset,
            "total_items": len(selected),
            "limit": resolved_limit,
            "interaction": interaction.value,
        }

    def uncovered(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        selected = self._uncovered_files(files)
        page, resolved_offset, resolved_limit = self._paginate(
            selected,
            offset=offset,
            limit=limit,
        )

        return {
            "result": [{"path": path} for path in page],
            "offset": resolved_offset,
            "total_items": len(selected),
            "limit": resolved_limit,
        }


def file_tools(
    fs: fsspec.AbstractFileSystem,
    fmt: FileFormat,
    *,
    max_output: int = 8 * 10**4,
    max_items: int = 300,
    ignored_patterns: Optional[list[str]] = None,
    with_types: bool = True,
    with_file_info: bool = True,
    with_coverage_tools: bool = True,
) -> list[Callable[..., dict[str, Any]]]:
    """
    Return a registry of filesystem tools:
      - ls(path)
      - glob(pattern, path=None, offset=0)
      - read_file(file, offset=None, limit=None)
      - grep(pattern, path=None, offset=0)
    """

    tools = FsspecCoverageFileTools(
        fs=fs,
        fmt=fmt,
        max_output=max_output,
        max_items=max_items,
        ignored_patterns=ignored_patterns,
        with_types=with_types,
        with_file_info=with_file_info,
    )

    def ls(path: str) -> dict[str, Any]:
        """
        List immediate children of a directory.

        Use this tool when you need to inspect the structure of a specific folder
        before deciding which files to open or search.
        Args:
            path:
                Absolute or backend-specific path to a directory or filesystem location.

        Returns:
            A dict with:
            - "result": formatted list of filesystem entries
        Behavior:
            - Only direct children are returned; this tool is not recursive.

        Prefer this tool when:
            - You are exploring an unfamiliar directory
            - You need to discover candidate files to inspect

        Prefer glob() instead when:
            - You already know a filename mask or extension pattern
        """

        return tools.ls(path=path)

    def glob(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Find files or directories by glob pattern.

        Use this tool when you know a filename/path pattern, such as:
        - "*.py"
        - "**/*.md"
        - "/project/src/**/*.ts"

        Args:
            pattern:
                Glob pattern to match against filesystem paths.
            path:
                Optional root path used as an additional filter.
                If provided, only matches inside this path are returned.
            offset:
                Pagination offset for large result sets.

        Returns:
            A dict with:
            - "result": formatted list of matching filesystem entries
            - "offset": starting offset used for pagination
            - "total_items": total number of matches before pagination
            - "limit": page size

            On error returns:
            - {"error": "..."} if the provided path does not exist

        Behavior:
            - Results are sorted by path.
            - Pagination is applied using offset and internal max_items.

        Prefer this tool when:
            - You want files by extension or path mask
            - You need recursive discovery via patterns like "**/*.py"

        Prefer ls() instead when:
            - You want a direct non-recursive listing of one folder
        """

        return tools.glob(pattern=pattern, path=path, offset=offset)

    def read_file(
        file: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Read text content from a file.

        Use this tool to inspect file contents after locating a relevant file
        through ls() or glob(), or after identifying an interesting match via grep().

        Args:
            file:
                Path to a text file.
            offset:
                Optional 0-based line offset. If provided, reading starts from this line.
            limit:
                Optional maximum number of lines to return after applying offset.

        Returns:
            A dict with:
            - "result": text content, possibly truncated to fit max_output

            On error returns:
            - {"error": "..."} if the path does not exist
            - {"error": "..."} if the path is not a file

        Behavior:
            - Reads text as UTF-8 with errors ignored.
            - Applies line-based slicing, not byte-based slicing.
            - Large output is truncated safely by bytes while preserving line boundaries.
            - Hidden/ignored paths are treated as unavailable.

        Prefer this tool when:
            - You already know which file should be inspected
            - You need exact file contents, not just search hits

        Prefer grep() instead when:
            - You first need to locate relevant text across many files
        """

        return tools.read_file(file_path=file, offset=offset, limit=limit)

    def grep(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search file contents using a regular expression.

        Use this tool to locate relevant text in a single file or recursively
        across a directory tree before reading full files.

        Args:
            pattern:
                Python regular expression pattern.
            path:
                File or directory path to search in.
                - If a file is provided, only that file is searched.
                - If a directory is provided, files are searched recursively.
                - If omitted, the filesystem root is searched.
            offset:
                Pagination offset over individual matches.

        Returns:
            A dict with:
            - "result": one formatted FsEntry per regex match
            - "offset": starting offset used for pagination
            - "total_items": total number of matches before pagination
            - "limit": page size

            Each match entry may include:
            - file metadata (path, name, size)
            - loc with line range and excerpt
            - optional filetype metadata

            On error returns:
            - {"error": "..."} if the regex is invalid
            - {"error": "..."} if the path does not exist

        Behavior:
            - Uses Python regex syntax.
            - Searches recursively when path is a directory.
            - Returns matches, not whole-file contents.
            - Ignore patterns are respected.
            - Binary or unreadable files may be skipped silently.
            - Output format depends on FileFormat.

        Prefer this tool when:
            - You need to find where a symbol, error, string, or pattern appears
            - You want to narrow down which files to inspect further
        """
        return tools.grep(pattern=pattern, path=path, offset=offset)

    def coverage_stats(
        path: str = "/",
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        """
        Summarize repository exploration progress.
        Files are categorized using agent interaction history:
            - covered:
                files that were explicitly read using read_file()
            - uncovered:
                files that produced a grep() match but were not read afterward
            - untracked:
                files that were neither read nor matched by grep()

        This tool helps to understand which parts of the filesystem
        have already been inspected and where further exploration may be needed.

        Args:
            path:
                Root path used to compute coverage.
            pattern:
                Glob pattern used to select files inside the path.

        Returns:
            {
                "result": {
                    "path": str,
                    "pattern": str,
                    "total_files": int,
                    "covered_files_count": int,
                    "uncovered_files_count": int,
                    "untracked_files_count": int,
                    "coverage_percent": float
                }
            }

        Coverage percent reflects the fraction of files that were explicitly read.
        """

        path = tools._norm(path) or "/"
        pattern = tools._norm(pattern) or "**/*"

        if not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        files = tools._matched_files(path, pattern)

        covered = []
        uncovered = []
        untracked = []

        for f in files:
            entry = tools._coverage_entry(f)

            if entry is None:
                untracked.append(f)
                continue

            if entry.has_read:
                covered.append(f)
            elif entry.has_match:
                uncovered.append(f)
            else:
                untracked.append(f)

        total = len(files)
        covered_count = len(covered)

        percent = round((covered_count / total) * 100, 2) if total else 100.0

        return {
            "result": {
                "path": path,
                "pattern": pattern,
                "total_files": total,
                "covered_files_count": len(covered),
                "uncovered_files_count": len(uncovered),
                "untracked_files_count": len(untracked),
                "coverage_percent": percent,
            }
        }

    def covered(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that were already read by the agent.
        A file is considered *covered* if read_file() was executed on it.

        Args:
            path:
                Root directory to inspect.
            pattern:
                Glob pattern used to select candidate files.
            offset:
                Pagination offset.
            limit:
                Maximum number of results to return.

        Returns:
            {
                "result": [coverage_entry, ...],
                "offset": int,
                "total_items": int,
                "limit": int
            }

        Each coverage entry contains:
            - path
            - has_read
            - has_match
            - read_count
            - match_count
            - operations
        """

        path = tools._norm(path) or "/"
        pattern = tools._norm(pattern) or "**/*"

        if not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        files = tools._matched_files(path, pattern)

        selected = [
            f for f in files if (entry := tools._coverage_entry(f)) and entry.has_read
        ]

        page, offset, limit = tools._paginate(selected, offset=offset, limit=limit)

        return {
            "result": [tools._serialize_coverage_entry(p) for p in page],
            "offset": offset,
            "total_items": len(selected),
            "limit": limit,
        }

    def uncovered(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that matched a grep() search but were never read.

        These files are important follow-up candidates:
        You had detected relevant content but did not inspect the file.

        Args:
            path:
                Root directory to inspect.
            pattern:
                Glob pattern used to select candidate files.
            offset:
                Pagination offset.
            limit:
                Maximum number of results to return.

        Returns:
            {
                "result": [coverage_entry, ...],
                "offset": int,
                "total_items": int,
                "limit": int
            }
        """

        path = tools._norm(path) or "/"
        pattern = tools._norm(pattern) or "**/*"

        if not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        files = tools._matched_files(path, pattern)

        selected = []

        for f in files:
            entry = tools._coverage_entry(f)

            if entry and entry.has_match and not entry.has_read:
                selected.append(f)

        page, offset, limit = tools._paginate(selected, offset=offset, limit=limit)

        return {
            "result": [tools._serialize_coverage_entry(p) for p in page],
            "offset": offset,
            "total_items": len(selected),
            "limit": limit,
        }

    def untracked(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that were neither read nor matched by search.

        These files represent unexplored areas of the repository.

        Use this tool when the agent wants to systematically explore
        files that have not yet been inspected in any way.

        Args:
            path:
                Root directory to inspect.
            pattern:
                Glob pattern used to select candidate files.
            offset:
                Pagination offset.
            limit:
                Maximum number of results to return.

        Returns:
            {
                "result": [{"path": ...}, ...],
                "offset": int,
                "total_items": int,
                "limit": int
            }
        """

        path = tools._norm(path) or "/"
        pattern = tools._norm(pattern) or "**/*"

        if not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        files = tools._matched_files(path, pattern)

        selected = []

        for f in files:
            entry = tools._coverage_entry(f)

            if entry is None:
                selected.append(f)
            elif not entry.has_read and not entry.has_match:
                selected.append(f)

        page, offset, limit = tools._paginate(selected, offset=offset, limit=limit)

        return {
            "result": [{"path": p} for p in page],
            "offset": offset,
            "total_items": len(selected),
            "limit": limit,
        }

    def reset_coverage() -> dict[str, Any]:
        """
        Reset coverage tracking.

        Clears all stored information about files that were read
        or matched by grep().

        This is useful when starting a new independent analysis task.
        Call this tool only if you are absolutely sure, that you need this.

        Returns:
            {"result": "ok"}
        """

        tools.reset_coverage()
        return {"result": "ok"}

    registry = [ls, glob, read_file, grep]

    if with_coverage_tools:
        registry.extend(
            [
                coverage_stats,
                covered,
                uncovered,
                untracked,
                reset_coverage,
            ]
        )

    return registry


class RootedLocalFileSystem(LocalFileSystem):
    """
    Local filesystem sandboxed to root_path.
    Forbidden paths are treated as non-existent.
    """

    def __init__(self, root_path: str, *args: Any, **kwargs: Any) -> None:
        self.root_path = os.path.realpath(stringify_path(root_path))
        if not os.path.isdir(self.root_path):
            raise ValueError(f"root_path is not a directory: {root_path}")

        self._blocked_path = os.path.join(self.root_path, ".__blocked__")
        super().__init__(*args, **kwargs)

    def walk(self, path: str = "", **kwargs: Any):
        path = "" if path in (None, "/", "") else path
        host_root = self._strip_protocol(path)

        if host_root == self._blocked_path or not os.path.exists(host_root):
            return

        for current_root, dirs, files in os.walk(host_root, followlinks=False):
            real_root = os.path.realpath(current_root)

            if not (
                real_root == self.root_path
                or real_root.startswith(self.root_path + os.sep)
            ):
                continue

            dirs[:] = [
                d for d in dirs if not os.path.islink(os.path.join(current_root, d))
            ]

            rel_root = os.path.relpath(real_root, self.root_path)
            virtual_root = (
                "/" if rel_root == "." else "/" + rel_root.replace(os.sep, "/")
            )

            yield virtual_root, dirs, files

    def ls(self, path: str = "", detail: bool = False, **kwargs: Any):
        path = "" if path in (None, "/", "") else path
        host_path = self._strip_protocol(path)

        if host_path == self._blocked_path:
            return []

        try:
            entries = super().ls(host_path, detail=True, **kwargs)
        except FileNotFoundError:
            return []

        result = []
        for entry in entries:
            host_name = entry["name"]
            real = os.path.realpath(host_name)

            if not (real == self.root_path or real.startswith(self.root_path + os.sep)):
                continue

            relative = os.path.relpath(real, self.root_path)
            virtual = "/" if relative == "." else "/" + relative.replace(os.sep, "/")

            if detail:
                normalized_entry = entry.copy()
                normalized_entry["name"] = virtual
                result.append(normalized_entry)
            else:
                result.append(virtual)

        return result

    def glob(self, pattern: str, **kwargs: Any):
        """
        Sandbox-safe glob with Python-like semantics.
        Returns virtual paths like /file.txt, /dir/inner.txt.
        """
        if not pattern:
            return []

        pattern = _norm_unicode(pattern.lstrip("/")) or ""

        if ".." in pattern.split("/"):
            return []

        recursive = "**" in pattern
        matches: list[str] = []

        if recursive:
            walker = os.walk(self.root_path, followlinks=False)
        else:
            try:
                files = os.listdir(self.root_path)
            except FileNotFoundError:
                return []
            walker = [(self.root_path, [], files)]

        for host_root, dirs, files in walker:
            dirs[:] = [
                directory
                for directory in dirs
                if not os.path.islink(os.path.join(host_root, directory))
            ]

            rel_root = os.path.relpath(host_root, self.root_path)
            if rel_root == ".":
                rel_root = ""

            for name in files:
                normalized_name = _norm_unicode(name) or name
                host_path = os.path.join(host_root, normalized_name)

                if os.path.islink(host_path):
                    continue

                rel_path = (
                    os.path.join(rel_root, normalized_name)
                    if rel_root
                    else normalized_name
                )
                rel_path = _norm_unicode(rel_path.replace(os.sep, "/")) or rel_path

                if fnmatch.fnmatch(rel_path, pattern):
                    matches.append("/" + rel_path)
                    continue

                if recursive and "/" not in rel_path:
                    tail = pattern.split("/")[-1]
                    if fnmatch.fnmatch(normalized_name, tail):
                        matches.append("/" + normalized_name)

        return sorted(set(matches))

    def _strip_protocol(self, path):
        path = _norm_unicode(stringify_path(path))

        if path.startswith("file://"):
            path = path[7:]

        # If path already points inside root_path → accept as-is
        real = os.path.realpath(path)
        if real == self.root_path or real.startswith(self.root_path + os.sep):
            return real

        # Virtual FS paths: "/", "/file.txt", "dir/a.txt"
        if path in ("", "/"):
            candidate = self.root_path
        else:
            candidate = os.path.join(self.root_path, path.lstrip("/"))

        candidate = os.path.abspath(os.path.normpath(candidate))
        resolved = os.path.realpath(candidate)

        if resolved == self.root_path or resolved.startswith(self.root_path + os.sep):
            return candidate

        # Escape attempt → silent block
        return self._blocked_path
