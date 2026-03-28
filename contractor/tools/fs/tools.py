from __future__ import annotations

import fnmatch
import json
import re

from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

import fsspec
from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem
from contractor.utils.fs import norm_unicode, normalize_slashes, xml_escape
from magika import ContentTypeInfo, Magika


ToolResult: TypeAlias = dict[str, Any]
BackendTool: TypeAlias = Callable[..., ToolResult]
PatchPayload: TypeAlias = dict[str, Any] | str

_IGNORE_DEFAULTS: Final[list[str]] = [
    "*.pyc",
    "*/__pycache__/*",
    "__pycache__/*",
    "target/*",
    "build/*",
    "dist/*",
    "out/*",
    "bin/*",
    "obj/*",
    "Debug/*",
    "Release/*",
    "cmake-build-*/*",
    "DerivedData/*",
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
    ".git/*",
]

_COMMENT_PREFIX_BY_EXT: Final[dict[str, str]] = {
    ".py": "#",
    ".sh": "#",
    ".bash": "#",
    ".zsh": "#",
    ".yaml": "#",
    ".yml": "#",
    ".toml": "#",
    ".ini": "#",
    ".conf": "#",
    ".properties": "#",
    ".rb": "#",
    ".pl": "#",
    ".ps1": "#",
    ".r": "#",
    ".dockerfile": "#",
    ".mk": "#",
    ".make": "#",
    ".sql": "--",
    ".js": "//",
    ".jsx": "//",
    ".ts": "//",
    ".tsx": "//",
    ".java": "//",
    ".kt": "//",
    ".kts": "//",
    ".go": "//",
    ".rs": "//",
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".cxx": "//",
    ".h": "//",
    ".hh": "//",
    ".hpp": "//",
    ".swift": "//",
    ".scala": "//",
    ".dart": "//",
    ".php": "//",
}

INCORRECT_REGEXP_ERROR: Final[str] = "regex {regex} is incorrect:\n{err}"
PATH_NOT_FOUND_ERROR: Final[str] = "path {path} does not exist"
PATH_IS_NOT_A_FILE_ERROR: Final[str] = "{path} is not a file"


def _is_ignored(path: str, patterns: list[str]) -> bool:
    normalized = normalize_slashes(path)
    basename = normalized.split("/")[-1]
    return any(
        fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern)
        for pattern in patterns
    )


def _ensure_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _split_lines_keepends(text: str) -> list[str]:
    if not text:
        return []
    return text.splitlines(keepends=True)


def _line_ending_for_text(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _comment_prefix_for_path(path: str, comment_style: Optional[str] = None) -> str:
    if comment_style:
        return comment_style

    normalized = normalize_slashes(path)
    basename = normalized.split("/")[-1].lower()

    if basename in {"dockerfile", "makefile"}:
        return "#"

    for ext, prefix in _COMMENT_PREFIX_BY_EXT.items():
        if basename.endswith(ext):
            return prefix

    return "#"


def _format_comment_line(
    *,
    comment: str,
    indent: str,
    prefix: str,
    newline: str,
) -> str:
    stripped = comment.strip()
    if prefix in {"<!--", "<!-- -->"}:
        return f"{indent}<!-- {stripped} -->{newline}"
    return f"{indent}{prefix} {stripped}{newline}"


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

    _magika: ClassVar[Magika] = Magika()

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
        normalized_path = norm_unicode(path)
        if normalized_path is None or not fs.exists(normalized_path):
            return None

        name = norm_unicode(normalized_path.rstrip("/").split("/")[-1]) or ""

        if fs.isdir(normalized_path):
            return cls(name=name, path=normalized_path, size=0, is_dir=True)

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
                parts.append(f"<{key}>{xml_escape(str(value))}</{key}>")
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
                    parts.append(f"<{key}>{xml_escape(serialized)}</{key}>")
                else:
                    parts.append(f"<{key}>{xml_escape(str(value))}</{key}>")

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


class InteractionFilter(str, Enum):
    ANY = "any"
    READ_ONLY = "read_only"
    MATCH_ONLY = "match_only"
    READ_AND_MATCH = "read_and_match"


@dataclass(slots=True)
class FileInteractionEntry:
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
    def has_any_interaction(self) -> bool:
        return self.has_read or self.has_match

    def matches_filter(self, flt: InteractionFilter) -> bool:
        if flt == InteractionFilter.ANY:
            return self.has_any_interaction
        if flt == InteractionFilter.READ_ONLY:
            return self.has_read and not self.has_match
        if flt == InteractionFilter.MATCH_ONLY:
            return self.has_match and not self.has_read
        if flt == InteractionFilter.READ_AND_MATCH:
            return self.has_read and self.has_match
        return False


class FsspecInteractionFileTools:
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

        self._interactions: dict[str, FileInteractionEntry] = {}

        patterns: list[str] = []
        for pattern in _IGNORE_DEFAULTS + (ignored_patterns or []):
            if pattern and pattern not in patterns:
                patterns.append(pattern)
        self.patterns = patterns

    def _norm(self, path: Optional[str]) -> Optional[str]:
        return norm_unicode(path)

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
        self.record_interaction(file_path, operation, interaction=interaction)
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
        file_path = normalize_slashes(file_path)
        root = normalize_slashes(root).rstrip("/") or "/"
        pattern = normalize_slashes(pattern)

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

    def _interaction_entry(self, path: str) -> Optional[FileInteractionEntry]:
        return self._interactions.get(path)

    def _has_any_interaction(self, path: str) -> bool:
        entry = self._interaction_entry(path)
        return bool(entry and entry.has_any_interaction)

    def _files_with_interactions(
        self,
        files: Iterable[str],
        *,
        interaction: InteractionFilter = InteractionFilter.ANY,
    ) -> list[str]:
        selected: list[str] = []
        for path in files:
            entry = self._interaction_entry(path)
            if entry is not None and entry.matches_filter(interaction):
                selected.append(path)
        return sorted(selected)

    def _untouched_files(self, files: Iterable[str]) -> list[str]:
        return sorted(path for path in files if not self._has_any_interaction(path))

    def _serialize_interaction_entry(self, path: str) -> dict[str, Any]:
        entry = self._interaction_entry(path)
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

    def record_interaction(
        self,
        path: str,
        operation: str,
        *,
        interaction: InteractionKind,
    ) -> None:
        entry = self._interactions.get(path)
        if entry is None:
            entry = FileInteractionEntry(path=path)
            self._interactions[path] = entry

        entry.touch(operation, interaction=interaction)

    def reset_interactions(self) -> None:
        self._interactions.clear()

    def get_interactions(self) -> dict[str, Any]:
        return {
            "files_seen": len(self._interactions),
            "files": {
                path: {
                    "read_count": entry.read_count,
                    "match_count": entry.match_count,
                    "has_read": entry.has_read,
                    "has_match": entry.has_match,
                    "operations": dict(entry.operations),
                }
                for path, entry in sorted(self._interactions.items())
            },
        }

    def ls(self, path: str) -> ToolResult:
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
    ) -> ToolResult:
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
    ) -> ToolResult:
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
    ) -> ToolResult:
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
                self.record_interaction(
                    file_path, "grep", interaction=InteractionKind.MATCH
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

    def interaction_stats(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
    ) -> ToolResult:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        touched_files = self._files_with_interactions(files)
        untouched_files = self._untouched_files(files)

        total = len(files)
        touched = len(touched_files)
        untouched = len(untouched_files)
        interaction_percent = round((touched / total) * 100, 2) if total else 100.0

        return {
            "result": {
                "path": normalized_path,
                "pattern": normalized_pattern,
                "total_files": total,
                "touched_files_count": touched,
                "untouched_files_count": untouched,
                "interaction_percent": interaction_percent,
            }
        }

    def files_with_interactions(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        interaction: InteractionFilter = InteractionFilter.ANY,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> ToolResult:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if isinstance(interaction, str):
            interaction = InteractionFilter(interaction)

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        selected = self._files_with_interactions(files, interaction=interaction)
        page, resolved_offset, resolved_limit = self._paginate(
            selected, offset=offset, limit=limit
        )

        return {
            "result": [self._serialize_interaction_entry(path) for path in page],
            "offset": resolved_offset,
            "total_items": len(selected),
            "limit": resolved_limit,
            "interaction": interaction.value,
        }

    def touched_files(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> ToolResult:
        return self.files_with_interactions(
            path=path,
            pattern=pattern,
            interaction=InteractionFilter.ANY,
            offset=offset,
            limit=limit,
        )

    def untouched_files(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> ToolResult:
        normalized_path = self._norm(path) or "/"
        normalized_pattern = self._norm(pattern) or "**/*"

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        selected = self._untouched_files(files)
        page, resolved_offset, resolved_limit = self._paginate(
            selected, offset=offset, limit=limit
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
    with_interaction_tools: bool = True,
) -> list[Callable[..., dict[str, Any]]]:
    tools = FsspecInteractionFileTools(
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
        """
        return tools.ls(path=path)

    def glob(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Find files or directories by glob pattern.
        """
        offset = _ensure_int_or_none(offset) or 0
        return tools.glob(pattern=pattern, path=path, offset=offset)

    def read_file(
        file: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Read text content from a file.
        """
        offset = _ensure_int_or_none(offset)
        limit = _ensure_int_or_none(limit)
        return tools.read_file(file_path=file, offset=offset, limit=limit)

    def grep(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search file contents using a regular expression.
        """
        offset = _ensure_int_or_none(offset) or 0
        return tools.grep(pattern=pattern, path=path, offset=offset)

    def interaction_stats(
        path: str = "/",
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        """
        Summarize filesystem interaction progress.

        Notes:
            - "touched" means the file was either read or matched by grep()
            - "untouched" means the file had no recorded interaction
        """
        return tools.interaction_stats(path=path, pattern=pattern)

    def list_touched_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that had at least one recorded interaction.
        """
        offset = _ensure_int_or_none(offset) or 0
        limit = _ensure_int_or_none(limit)
        return tools.touched_files(
            path=path, pattern=pattern, offset=offset, limit=limit
        )

    def list_untouched_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that have not been read and did not match grep().
        """
        offset = _ensure_int_or_none(offset) or 0
        limit = _ensure_int_or_none(limit)
        return tools.untouched_files(
            path=path, pattern=pattern, offset=offset, limit=limit
        )

    def list_match_only_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that were matched (e.g. via grep) but never read.

        Use this tool to identify files where a search hit occurred,
        but the file content has not yet been inspected.

        This is useful for:
        - following up on grep results
        - ensuring all relevant matches are reviewed
        - detecting incomplete exploration of code/data

        Args:
            path:
                Root directory to search within.
            pattern:
                Glob pattern to filter files (applied after traversal).
            offset:
                Pagination offset (0-based).
            limit:
                Maximum number of items to return.

        Returns:
            A dict containing:
            - "result": list of file interaction entries
            - "offset": pagination offset
            - "total_items": total matching files
            - "limit": page size

            On failure:
            - {"error": "..."} if path is invalid

        Notes:
            - A file is included if:
                - it has at least one match interaction (e.g. grep)
                - it has zero read interactions
            - Equivalent to InteractionFilter.MATCH_ONLY
            - Files never seen (no grep, no read) are NOT included
        """
        offset = _ensure_int_or_none(offset) or 0
        limit = _ensure_int_or_none(limit)

        return tools.files_with_interactions(
            path=path,
            pattern=pattern,
            interaction=InteractionFilter.MATCH_ONLY,
            offset=offset,
            limit=limit,
        )

    def reset_interaction_tracking() -> dict[str, Any]:
        """
        Reset interaction tracking.
        """
        tools.reset_interactions()
        return {"result": "ok"}

    registry = [ls, glob, read_file, grep]

    if with_interaction_tools:
        registry.extend(
            [
                interaction_stats,
                list_touched_files,
                list_untouched_files,
                list_match_only_files,
                reset_interaction_tracking,
            ]
        )

    return registry


class FsspecWriteTools:
    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        *,
        ignored_patterns: Optional[list[str]] = None,
        wrap_overlay: bool = True,
        fmt: Optional[FileFormat] = None,
        max_output: int = 8 * 10**4,
        max_items: int = 300,
        with_types: bool = True,
        with_file_info: bool = True,
    ) -> None:
        self.base_fs = fs
        self.fs = (
            fs
            if isinstance(fs, MemoryOverlayFileSystem) or not wrap_overlay
            else MemoryOverlayFileSystem(fs)
        )

        patterns: list[str] = []
        for pattern in _IGNORE_DEFAULTS + (ignored_patterns or []):
            if pattern and pattern not in patterns:
                patterns.append(pattern)
        self.patterns = patterns

        self.fmt = fmt or FileFormat(
            with_types=with_types,
            with_file_info=with_file_info,
        )
        self.max_output = max_output
        self.max_items = max_items
        self.with_types = with_types
        self.with_file_info = with_file_info

        self._reader = FsspecInteractionFileTools(
            fs=self.fs,
            fmt=self.fmt,
            max_output=max_output,
            max_items=max_items,
            ignored_patterns=self.patterns,
            with_types=with_types,
            with_file_info=with_file_info,
        )

    def _norm(self, path: Optional[str]) -> Optional[str]:
        return norm_unicode(path)

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _parse_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        return default

    def _parse_patch_payload(self, patch: Any) -> PatchPayload:
        if isinstance(patch, dict):
            return patch
        if isinstance(patch, str):
            return json.loads(patch)
        raise ValueError("patch must be a dict or JSON string")

    # ---- read-like tools on overlay ----

    def ls(self, path: str) -> ToolResult:
        return self._reader.ls(path=path)

    def glob(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> ToolResult:
        return self._reader.glob(pattern=pattern, path=path, offset=offset)

    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> ToolResult:
        return self._reader.read_file(file_path=file_path, offset=offset, limit=limit)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> ToolResult:
        return self._reader.grep(pattern=pattern, path=path, offset=offset)

    # ---- write tools ----

    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}

        try:
            self.fs.write_text(
                normalized_path,
                value=content,
                encoding=encoding,
                errors="strict",
            )
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "op": "write_file",
                "path": normalized_path,
                "size": len(content.encode(encoding, errors="ignore")),
            }
        }

    def append_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}

        try:
            with self.fs.open(normalized_path, mode="a", encoding=encoding) as f:
                f.write(content)
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "op": "append_file",
                "path": normalized_path,
                "size": len(content.encode(encoding, errors="ignore")),
            }
        }

    def mkdir(
        self,
        path: str,
        create_parents: bool = True,
        exist_ok: bool = True,
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}

        try:
            if exist_ok:
                self.fs.makedirs(normalized_path, exist_ok=True)
            else:
                self.fs.mkdir(normalized_path, create_parents=create_parents)
        except Exception as err:
            return {"error": str(err)}

        return {"result": {"ok": True, "op": "mkdir", "path": normalized_path}}

    def rm(
        self,
        path: str,
        recursive: bool = False,
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None or not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}

        try:
            self.fs.rm(normalized_path, recursive=recursive)
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "op": "rm",
                "path": normalized_path,
                "recursive": recursive,
            }
        }

    def cp(
        self,
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> ToolResult:
        normalized_src = self._norm(src)
        normalized_dst = self._norm(dst)

        if normalized_src is None or not self.fs.exists(normalized_src):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_src)}
        if normalized_dst is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_dst)}
        if self._is_ignored(normalized_src) or self._is_ignored(normalized_dst):
            return {"error": "source or destination path is ignored"}

        try:
            self.fs.copy(normalized_src, normalized_dst, recursive=recursive)
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "op": "cp",
                "src": normalized_src,
                "dst": normalized_dst,
                "recursive": recursive,
            }
        }

    def mv(
        self,
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> ToolResult:
        normalized_src = self._norm(src)
        normalized_dst = self._norm(dst)

        if normalized_src is None or not self.fs.exists(normalized_src):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_src)}
        if normalized_dst is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_dst)}
        if self._is_ignored(normalized_src) or self._is_ignored(normalized_dst):
            return {"error": "source or destination path is ignored"}

        try:
            self.fs.mv(normalized_src, normalized_dst, recursive=recursive)
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "op": "mv",
                "src": normalized_src,
                "dst": normalized_dst,
                "recursive": recursive,
            }
        }

    def insert_comment(
        self,
        path: str,
        comment: str,
        *,
        anchor: str,
        where: Literal["before", "after"] = "before",
        occurrence: int = 1,
        comment_style: Optional[str] = None,
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}
        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if not self.fs.isfile(normalized_path):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized_path)}
        if not anchor:
            return {"error": "anchor is required"}
        if occurrence < 1:
            return {"error": "occurrence must be >= 1"}

        try:
            text = self.fs.read_text(normalized_path, encoding="utf-8")
        except Exception as err:
            return {"error": str(err)}

        lines = _split_lines_keepends(text)
        newline = _line_ending_for_text(text)
        prefix = _comment_prefix_for_path(normalized_path, comment_style=comment_style)

        matched_indexes = [i for i, line in enumerate(lines) if anchor in line]
        if not matched_indexes:
            return {
                "error": f"anchor not found in {normalized_path}",
                "path": normalized_path,
                "anchor": anchor,
            }

        if occurrence > len(matched_indexes):
            return {
                "error": (
                    f"anchor occurrence {occurrence} not found in {normalized_path}; "
                    f"found {len(matched_indexes)} matches"
                ),
                "path": normalized_path,
                "anchor": anchor,
                "matches": len(matched_indexes),
            }

        anchor_index = matched_indexes[occurrence - 1]
        anchor_line = lines[anchor_index]
        indent = _leading_ws(anchor_line)
        comment_line = _format_comment_line(
            comment=comment,
            indent=indent,
            prefix=prefix,
            newline=newline,
        )

        insert_at = anchor_index if where == "before" else anchor_index + 1

        prev_line = lines[insert_at - 1] if insert_at - 1 >= 0 else None
        next_line = lines[insert_at] if insert_at < len(lines) else None
        if prev_line == comment_line or next_line == comment_line:
            return {
                "result": {
                    "ok": True,
                    "changed": False,
                    "reason": "comment already present",
                    "path": normalized_path,
                    "anchor": anchor,
                    "where": where,
                    "occurrence": occurrence,
                    "comment_line": comment_line.rstrip("\r\n"),
                }
            }

        new_lines = list(lines)
        new_lines.insert(insert_at, comment_line)
        new_text = "".join(new_lines)

        try:
            self.fs.write_text(
                normalized_path,
                value=new_text,
                encoding="utf-8",
                errors="strict",
            )
        except Exception as err:
            return {"error": str(err)}

        return {
            "result": {
                "ok": True,
                "changed": True,
                "op": "insert_comment",
                "path": normalized_path,
                "anchor": anchor,
                "where": where,
                "occurrence": occurrence,
                "line": insert_at + 1,
                "comment_line": comment_line.rstrip("\r\n"),
            }
        }

    def replace_range(
        self,
        path: str,
        start_line: int,
        end_line: int,
        content: str,
        *,
        preserve_trailing_newline: bool = True,
    ) -> ToolResult:
        normalized_path = self._norm(path)
        if normalized_path is None:
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if self._is_ignored(normalized_path):
            return {"error": f"path {normalized_path} is ignored"}
        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}
        if not self.fs.isfile(normalized_path):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized_path)}

        try:
            start_line = int(start_line)
            end_line = int(end_line)
        except (TypeError, ValueError):
            return {"error": "start_line and end_line must be integers"}

        if start_line < 1:
            return {"error": "start_line must be >= 1"}
        if end_line < 0:
            return {"error": "end_line must be >= 0"}
        if end_line < start_line - 1:
            return {"error": "invalid range: end_line must be >= start_line - 1"}

        try:
            text = self.fs.read_text(normalized_path, encoding="utf-8")
        except Exception as err:
            return {"error": str(err)}

        lines = _split_lines_keepends(text)
        newline = _line_ending_for_text(text)
        line_count = len(lines)

        max_start = line_count + 1
        max_end = line_count

        if start_line > max_start:
            return {
                "error": f"start_line {start_line} is out of range; file has {line_count} lines"
            }
        if end_line > max_end:
            return {
                "error": f"end_line {end_line} is out of range; file has {line_count} lines"
            }

        start_idx = start_line - 1
        end_idx = end_line

        if content == "":
            replacement_lines: list[str] = []
        else:
            replacement_lines = _split_lines_keepends(content)
            if preserve_trailing_newline and replacement_lines:
                if not replacement_lines[-1].endswith(("\n", "\r")):
                    replacement_lines[-1] += newline

        old_segment = "".join(lines[start_idx:end_idx])
        new_segment = "".join(replacement_lines)

        if old_segment == new_segment:
            return {
                "result": {
                    "ok": True,
                    "changed": False,
                    "reason": "range already matches requested content",
                    "op": "replace_range",
                    "path": normalized_path,
                    "start_line": start_line,
                    "end_line": end_line,
                }
            }

        new_lines = list(lines)
        new_lines[start_idx:end_idx] = replacement_lines
        new_text = "".join(new_lines)

        try:
            self.fs.write_text(
                normalized_path,
                value=new_text,
                encoding="utf-8",
                errors="strict",
            )
        except Exception as err:
            return {"error": str(err)}

        inserted_line_count = len(replacement_lines)
        new_end_line = (
            start_line + inserted_line_count - 1
            if inserted_line_count
            else start_line - 1
        )

        return {
            "result": {
                "ok": True,
                "changed": True,
                "op": "replace_range",
                "path": normalized_path,
                "start_line": start_line,
                "end_line": end_line,
                "new_start_line": start_line,
                "new_end_line": new_end_line,
                "removed_line_count": max(0, end_line - start_line + 1),
                "inserted_line_count": inserted_line_count,
            }
        }


def write_tools(
    fs: fsspec.AbstractFileSystem,
    *,
    ignored_patterns: Optional[list[str]] = None,
    wrap_overlay: bool = True,
    fmt: Optional[FileFormat] = None,
    max_output: int = 8 * 10**4,
    max_items: int = 300,
    with_types: bool = True,
    with_file_info: bool = True,
) -> list[Callable[..., dict[str, Any]]]:
    tools = FsspecWriteTools(
        fs=fs,
        ignored_patterns=ignored_patterns,
        wrap_overlay=wrap_overlay,
        fmt=fmt,
        max_output=max_output,
        max_items=max_items,
        with_types=with_types,
        with_file_info=with_file_info,
    )

    def ls(path: str) -> dict[str, Any]:
        """
        List immediate children of a directory from the overlay-visible tree.
        """
        return tools.ls(path=path)

    def glob(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Find files or directories by glob pattern in the overlay-visible tree.
        """
        offset = _ensure_int_or_none(offset) or 0
        return tools.glob(pattern=pattern, path=path, offset=offset)

    def read_file(
        file: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Read text content from a file in the overlay-visible tree.
        """
        offset = _ensure_int_or_none(offset)
        limit = _ensure_int_or_none(limit)
        return tools.read_file(file_path=file, offset=offset, limit=limit)

    def grep(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Search file contents using a regular expression in the overlay-visible tree.
        """
        offset = _ensure_int_or_none(offset) or 0
        return tools.grep(pattern=pattern, path=path, offset=offset)

    def write_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        return tools.write_file(path=path, content=content, encoding=encoding)

    def append_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        return tools.append_file(path=path, content=content, encoding=encoding)

    def mkdir(
        path: str,
        create_parents: bool = True,
        exist_ok: bool = True,
    ) -> dict[str, Any]:
        return tools.mkdir(
            path=path,
            create_parents=tools._parse_bool(create_parents, True),
            exist_ok=tools._parse_bool(exist_ok, True),
        )

    def rm(
        path: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.rm(path=path, recursive=tools._parse_bool(recursive, False))

    def cp(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.cp(src=src, dst=dst, recursive=tools._parse_bool(recursive, False))

    def mv(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.mv(src=src, dst=dst, recursive=tools._parse_bool(recursive, False))

    def insert_comment(
        path: str,
        comment: str,
        anchor: str,
        where: Literal["before", "after"] = "before",
        occurrence: int = 1,
        comment_style: Optional[str] = None,
    ) -> dict[str, Any]:
        return tools.insert_comment(
            path=path,
            comment=comment,
            anchor=anchor,
            where=where,
            occurrence=int(occurrence),
            comment_style=comment_style,
        )

    def replace_range(
        path: str,
        start_line: int,
        end_line: int,
        content: str,
        preserve_trailing_newline: bool = True,
    ) -> dict[str, Any]:
        return tools.replace_range(
            path=path,
            start_line=int(start_line),
            end_line=int(end_line),
            content=content,
            preserve_trailing_newline=tools._parse_bool(
                preserve_trailing_newline, True
            ),
        )

    registry = [
        ls,
        glob,
        read_file,
        grep,
        write_file,
        append_file,
        insert_comment,
        replace_range,
        mkdir,
        rm,
        cp,
        mv,
    ]

    return registry
