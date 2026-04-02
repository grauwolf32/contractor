from __future__ import annotations

import fnmatch
import re

from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Optional,
    TypeAlias,
    Iterable,
)

import fsspec
from contractor.tools.fs.const import (
    INCORRECT_REGEXP_ERROR,
    PATH_NOT_FOUND_ERROR,
    PATH_IS_NOT_A_FILE_ERROR,
    _IGNORE_DEFAULTS,
)
from contractor.tools.fs.models import (
    FileInteractionEntry,
    FsEntry,
    InteractionFilter,
    InteractionKind,
)
from contractor.tools.fs.utils import (
    _is_ignored,
    _ensure_int_or_none,
    _split_lines_keepends,
    _line_ending_for_text,
    _parse_bool,
)
from contractor.utils.formatting import (
    norm_unicode,
    normalize_slashes,
)
from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem
from contractor.tools.fs.format import FileFormat

ToolResult: TypeAlias = dict[str, Any]
BackendTool: TypeAlias = Callable[..., ToolResult]
PatchPayload: TypeAlias = dict[str, Any] | str


def _build_ignore_patterns(ignored_patterns: Optional[list[str]] = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pattern in _IGNORE_DEFAULTS + (ignored_patterns or []):
        if pattern and pattern not in seen:
            seen.add(pattern)
            result.append(pattern)
    return result


def _join_path(directory: str, filename: str) -> str:
    return f"{str(directory).rstrip('/')}/{filename}".replace("\\", "/")


class FsspecInteractionFileTools:
    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        fmt: FileFormat,
        *,
        max_output: int = 80_000,
        max_items: int = 100,
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
        self.patterns = _build_ignore_patterns(ignored_patterns)

    def _norm(self, path: str) -> str:
        result = norm_unicode(path)
        if result is None:
            raise ValueError(f"Cannot normalize path: {path!r}")
        return result

    def _norm_optional(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return norm_unicode(path)

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _validate_path(
        self,
        path: str,
        *,
        must_exist: bool = False,
        must_be_file: bool = False,
        check_ignored: bool = True,
    ) -> tuple[Optional[str], Optional[ToolResult]]:
        try:
            normalized = self._norm(path)
        except ValueError:
            return None, {"error": PATH_NOT_FOUND_ERROR.format(path=path)}
        if check_ignored and self._is_ignored(normalized):
            return None, {"error": f"path {normalized} is ignored"}
        if must_exist and not self.fs.exists(normalized):
            return None, {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
        if must_be_file:
            if not self.fs.exists(normalized):
                return None, {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
            if not self.fs.isfile(normalized):
                return None, {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized)}
        return normalized, None

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

    def _iter_all_files(self, root: str) -> Iterator[str]:
        if not self.fs.exists(root):
            return

        if self.fs.isfile(root):
            if not self._is_ignored(root):
                yield root
            return

        seen: set[str] = set()
        for current_path, _dirs, filenames in self.fs.walk(root):
            for filename in filenames:
                full_path = _join_path(current_path, filename)
                if full_path not in seen and not self._is_ignored(full_path):
                    seen.add(full_path)
                    yield full_path

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
        return sorted(
            file_path
            for file_path in self._iter_all_files(path)
            if self._match_glob(file_path, path, pattern)
        )

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
        normalized_path, err = self._validate_path(
            path, must_exist=True, check_ignored=False
        )
        if err:
            return err

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
        normalized_path = self._norm_optional(path) or "/"
        normalized_pattern = self._norm_optional(pattern)

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
        normalized_file, err = self._validate_path(
            file_path,
            must_be_file=True,
            check_ignored=True,
        )
        if err:
            return err

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
        normalized_path = self._norm_optional(path) or "/"
        normalized_pattern = self._norm_optional(pattern)

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
                full_path = _join_path(current_path, filename)
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
        normalized_path = self._norm_optional(path) or "/"
        normalized_pattern = self._norm_optional(pattern) or "**/*"

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
        normalized_path = self._norm_optional(path) or "/"
        normalized_pattern = self._norm_optional(pattern) or "**/*"

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
            "result": [self._serialize_interaction_entry(p) for p in page],
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
        normalized_path = self._norm_optional(path) or "/"
        normalized_pattern = self._norm_optional(pattern) or "**/*"

        if not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        files = self._matched_files(normalized_path, normalized_pattern)
        selected = self._untouched_files(files)
        page, resolved_offset, resolved_limit = self._paginate(
            selected, offset=offset, limit=limit
        )

        return {
            "result": [{"path": p} for p in page],
            "offset": resolved_offset,
            "total_items": len(selected),
            "limit": resolved_limit,
        }


def ro_file_tools(
    fs: fsspec.AbstractFileSystem,
    fmt: FileFormat,
    *,
    max_output: int = 80_000,
    max_items: int = 100,
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
        max_output: int = 80_000,
        max_items: int = 100,
        with_types: bool = True,
        with_file_info: bool = True,
        with_interaction_tools: bool = False,
    ) -> None:
        self.base_fs = fs
        self.fs = (
            fs
            if isinstance(fs, MemoryOverlayFileSystem) or not wrap_overlay
            else MemoryOverlayFileSystem(fs)
        )

        self.patterns = _build_ignore_patterns(ignored_patterns)

        self.fmt = fmt or FileFormat(
            with_types=with_types,
            with_file_info=with_file_info,
        )
        self.max_output = max_output
        self.max_items = max_items
        self.with_types = with_types
        self.with_file_info = with_file_info
        self.with_interaction_tools = with_interaction_tools

        self._reader = FsspecInteractionFileTools(
            fs=self.fs,
            fmt=self.fmt,
            max_output=max_output,
            max_items=max_items,
            ignored_patterns=self.patterns,
            with_types=with_types,
            with_file_info=with_file_info,
        )

    def _normalize_edit_strings(
        self,
        current_content: Optional[str],
        old_string: str,
        new_string: str,
    ) -> tuple[str, str]:
        """
        If the file uses CRLF but the agent sent LF,
        adapt old/new to the file's actual line endings.
        """
        if current_content is None:
            return old_string, new_string

        # If old_string already matches verbatim, no adaptation needed
        if old_string in current_content:
            return old_string, new_string

        file_ending = _line_ending_for_text(current_content)

        def adapt(s: str) -> str:
            canonical = s.replace("\r\n", "\n")
            if file_ending == "\r\n":
                return canonical.replace("\n", "\r\n")
            return canonical

        return adapt(old_string), adapt(new_string)

    def _norm(self, path: str) -> str:
        result = norm_unicode(path)
        if result is None:
            raise ValueError(f"Cannot normalize path: {path!r}")
        return result

    def _norm_optional(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return norm_unicode(path)

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _validate_path(
        self,
        path: str,
        *,
        must_exist: bool = False,
        must_be_file: bool = False,
        check_ignored: bool = True,
    ) -> tuple[Optional[str], Optional[ToolResult]]:
        try:
            normalized = self._norm(path)
        except ValueError:
            return None, {"error": PATH_NOT_FOUND_ERROR.format(path=path)}
        if check_ignored and self._is_ignored(normalized):
            return None, {"error": f"path {normalized} is ignored"}
        if must_exist and not self.fs.exists(normalized):
            return None, {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
        if must_be_file:
            if not self.fs.exists(normalized):
                return None, {"error": PATH_NOT_FOUND_ERROR.format(path=normalized)}
            if not self.fs.isfile(normalized):
                return None, {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized)}
        return normalized, None

    def _ensure_interactions_enabled(self) -> Optional[ToolResult]:
        if not self.with_interaction_tools:
            return {
                "error": (
                    "interaction tools are disabled for this write toolset; "
                    "enable with_interaction_tools=True"
                )
            }
        return None

    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        with_line_numbers: bool = False,
    ) -> ToolResult:
        result = self._reader.read_file(
            file_path=file_path,
            offset=offset,
            limit=limit,
        )

        if with_line_numbers is not True:
            return result

        if "error" in result:
            return result

        content = result.get("result", "")
        if not isinstance(content, str) or content == "":
            return result

        start_line = 1 if offset is None else max(0, offset) + 1
        numbered = "\n".join(
            f"{line_no} | {line}"
            for line_no, line in enumerate(content.splitlines(), start=start_line)
        )
        return {"result": numbered}

    # ---- write tools ----

    def write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> ToolResult:
        normalized_path, err = self._validate_path(path)
        if err:
            return err

        try:
            self.fs.write_text(
                normalized_path,
                value=content,
                encoding=encoding,
                errors="strict",
            )
        except Exception as exc:
            return {"error": str(exc)}

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
        normalized_path, err = self._validate_path(path)
        if err:
            return err

        try:
            with self.fs.open(normalized_path, mode="a", encoding=encoding) as f:
                f.write(content)
        except Exception as exc:
            return {"error": str(exc)}

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
        normalized_path, err = self._validate_path(path)
        if err:
            return err

        try:
            if exist_ok:
                self.fs.makedirs(normalized_path, exist_ok=True)
            else:
                self.fs.mkdir(normalized_path, create_parents=create_parents)
        except Exception as exc:
            return {"error": str(exc)}

        return {"result": {"ok": True, "op": "mkdir", "path": normalized_path}}

    def rm(
        self,
        path: str,
        recursive: bool = False,
    ) -> ToolResult:
        normalized_path, err = self._validate_path(path, must_exist=True)
        if err:
            return err

        try:
            self.fs.rm(normalized_path, recursive=recursive)
        except Exception as exc:
            return {"error": str(exc)}

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
        normalized_src, err = self._validate_path(src, must_exist=True)
        if err:
            return err
        normalized_dst, err = self._validate_path(dst)
        if err:
            return err

        try:
            self.fs.copy(normalized_src, normalized_dst, recursive=recursive)
        except Exception as exc:
            return {"error": str(exc)}

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
        normalized_src, err = self._validate_path(src, must_exist=True)
        if err:
            return err
        normalized_dst, err = self._validate_path(dst)
        if err:
            return err

        try:
            self.fs.mv(normalized_src, normalized_dst, recursive=recursive)
        except Exception as exc:
            return {"error": str(exc)}

        return {
            "result": {
                "ok": True,
                "op": "mv",
                "src": normalized_src,
                "dst": normalized_dst,
                "recursive": recursive,
            }
        }

    def interaction_stats(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
    ) -> ToolResult:
        disabled = self._ensure_interactions_enabled()
        if disabled is not None:
            return disabled
        return self._reader.interaction_stats(path=path, pattern=pattern)

    def files_with_interactions(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        interaction: InteractionFilter = InteractionFilter.ANY,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> ToolResult:
        disabled = self._ensure_interactions_enabled()
        if disabled is not None:
            return disabled
        return self._reader.files_with_interactions(
            path=path,
            pattern=pattern,
            interaction=interaction,
            offset=offset,
            limit=limit,
        )

    def touched_files(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> ToolResult:
        disabled = self._ensure_interactions_enabled()
        if disabled is not None:
            return disabled
        return self._reader.touched_files(
            path=path,
            pattern=pattern,
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
        disabled = self._ensure_interactions_enabled()
        if disabled is not None:
            return disabled
        return self._reader.untouched_files(
            path=path,
            pattern=pattern,
            offset=offset,
            limit=limit,
        )

    def reset_interactions(self) -> ToolResult:
        disabled = self._ensure_interactions_enabled()
        if disabled is not None:
            return disabled
        self._reader.reset_interactions()
        return {"result": "ok"}

    def insert_line(
        self,
        path: str,
        content: str,
        *,
        anchor: str,
        where: Literal["before", "after"] = "before",
        occurrence: int = 1,
    ) -> ToolResult:
        normalized_path, err = self._validate_path(path, must_be_file=True)
        if err:
            return err

        if not anchor:
            return {"error": "anchor is required"}
        if occurrence < 1:
            return {"error": "occurrence must be >= 1"}

        try:
            text = self.fs.read_text(normalized_path, encoding="utf-8")
        except Exception as exc:
            return {"error": str(exc)}

        lines = _split_lines_keepends(text)

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
        insert_at = anchor_index if where == "before" else anchor_index + 1

        newline = _line_ending_for_text(text)
        new_line = content
        if not new_line.endswith(("\n", "\r")):
            new_line += newline

        prev_line = lines[insert_at - 1] if insert_at - 1 >= 0 else None
        next_line = lines[insert_at] if insert_at < len(lines) else None
        if prev_line == new_line or next_line == new_line:
            return {
                "result": {
                    "ok": True,
                    "changed": False,
                    "reason": "already present",
                    "path": normalized_path,
                    "anchor": anchor,
                    "where": where,
                    "occurrence": occurrence,
                    "insert_line": content,
                }
            }

        new_lines = list(lines)
        new_lines.insert(insert_at, new_line)
        new_text = "".join(new_lines)

        try:
            self.fs.write_text(
                normalized_path,
                value=new_text,
                encoding="utf-8",
                errors="strict",
            )
        except Exception as exc:
            return {"error": str(exc)}

        return {
            "result": {
                "ok": True,
                "changed": True,
                "op": "insert_line",
                "path": normalized_path,
                "anchor": anchor,
                "where": where,
                "occurrence": occurrence,
                "line": insert_at + 1,
                "insert_line": content,
            }
        }

    def edit(
        self,
        path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
        encoding: str = "utf-8",
    ) -> ToolResult:
        normalized_path, err = self._validate_path(path)
        if err:
            return err

        file_exists = self.fs.exists(normalized_path)

        # Empty old_string + missing file => create
        if not file_exists:
            if old_string != "":
                return {
                    "error": (
                        f"file not found: {normalized_path}; "
                        "use empty old_string to create a new file"
                    ),
                    "path": normalized_path,
                }

            try:
                self.fs.write_text(
                    normalized_path,
                    value=new_string,
                    encoding=encoding,
                    errors="strict",
                )
            except Exception as exc:
                return {"error": str(exc)}

            return {
                "result": {
                    "ok": True,
                    "changed": True,
                    "created": True,
                    "op": "edit",
                    "path": normalized_path,
                    "occurrences": 0,
                    "replaced_occurrences": 0,
                    "size": len(new_string.encode(encoding, errors="ignore")),
                }
            }

        if not self.fs.isfile(normalized_path):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=normalized_path)}

        try:
            current_content = self.fs.read_text(
                normalized_path,
                encoding=encoding,
                errors="ignore",
            )
        except Exception as exc:
            return {"error": str(exc)}

        if old_string == "":
            return {
                "error": (
                    "failed to edit: attempted to create a file that already exists"
                ),
                "path": normalized_path,
            }

        old_string, new_string = self._normalize_edit_strings(
            current_content,
            old_string,
            new_string,
        )

        occurrences = current_content.count(old_string)
        if occurrences == 0:
            return {
                "error": (
                    "failed to edit, could not find the string to replace; "
                    "check whitespace, indentation, and surrounding context"
                ),
                "path": normalized_path,
            }

        if not replace_all and occurrences > 1:
            return {
                "error": (
                    "failed to edit because the text matches multiple locations; "
                    "provide more context or set replace_all=True"
                ),
                "path": normalized_path,
                "occurrences": occurrences,
            }

        if old_string == new_string:
            return {
                "result": {
                    "ok": True,
                    "changed": False,
                    "reason": "old_string and new_string are identical",
                    "op": "edit",
                    "path": normalized_path,
                    "occurrences": occurrences,
                    "replaced_occurrences": 0,
                }
            }

        new_content = (
            current_content.replace(old_string, new_string)
            if replace_all
            else current_content.replace(old_string, new_string, 1)
        )

        if new_content == current_content:
            return {
                "result": {
                    "ok": True,
                    "changed": False,
                    "reason": "new content is identical to current content",
                    "op": "edit",
                    "path": normalized_path,
                    "occurrences": occurrences,
                    "replaced_occurrences": 0,
                }
            }

        try:
            self.fs.write_text(
                normalized_path,
                value=new_content,
                encoding=encoding,
                errors="strict",
            )
        except Exception as exc:
            return {"error": str(exc)}

        return {
            "result": {
                "ok": True,
                "changed": True,
                "created": False,
                "op": "edit",
                "path": normalized_path,
                "occurrences": occurrences,
                "replaced_occurrences": occurrences if replace_all else 1,
                "size": len(new_content.encode(encoding, errors="ignore")),
            }
        }

    def restore(self, path: str, recursive: bool = True) -> ToolResult:
        if not isinstance(self.fs, MemoryOverlayFileSystem):
            return {"error": "restore is only available with overlay filesystem"}

        normalized_path, err = self._validate_path(path)
        if err:
            return err

        if not self.fs.base_fs.exists(normalized_path):
            return {
                "error": (
                    f"path {normalized_path} does not exist in the base filesystem; "
                    "nothing can be restored"
                )
            }

        try:
            self.fs.restore(normalized_path, recursive=recursive)
        except Exception as exc:
            return {"error": str(exc)}

        return {
            "result": {
                "ok": True,
                "op": "restore",
                "path": normalized_path,
                "recursive": recursive,
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
        normalized_path, err = self._validate_path(path, must_be_file=True)
        if err:
            return err

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
        except Exception as exc:
            return {"error": str(exc)}

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
        except Exception as exc:
            return {"error": str(exc)}

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


def rw_file_tools(
    fs: fsspec.AbstractFileSystem,
    *,
    ignored_patterns: Optional[list[str]] = None,
    wrap_overlay: bool = True,
    fmt: Optional[FileFormat] = None,
    max_output: int = 80_000,
    max_items: int = 300,
    with_types: bool = True,
    with_file_info: bool = True,
    with_interaction_tools: bool = False,
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
        with_interaction_tools=with_interaction_tools,
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
        with_line_numbers: bool = False,
    ) -> dict[str, Any]:
        """
        Read text content from a file.

        This is the preferred tool to inspect a file before editing it.

        Args:
            file (str):
                Path to the file to read.

            offset (int, optional):
                Zero-based line offset. If provided, reading starts from this line.

            limit (int, optional):
                Maximum number of lines to return after applying offset.

            with_line_numbers (bool, optional):
                If True, each returned line is prefixed with its 1-based line number.

                Example:
                    10 | def hello():
                    11 |     return "world"

                Line numbers are NOT part of the file content.
                Do NOT include them in edit, insert_line, or replace_range content.

        How to use:
            - Call read_file before editing
            - Use with_line_numbers=True for insert_line and replace_range
            - Use raw text (without numbers) for edit.old_string
        """

        offset = _ensure_int_or_none(offset)
        limit = _ensure_int_or_none(limit)
        return tools.read_file(
            file_path=file,
            offset=offset,
            limit=limit,
            with_line_numbers=bool(with_line_numbers),
        )

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
            create_parents=_parse_bool(create_parents, True),
            exist_ok=_parse_bool(exist_ok, True),
        )

    def rm(
        path: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.rm(path=path, recursive=_parse_bool(recursive, False))

    def cp(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.cp(src=src, dst=dst, recursive=_parse_bool(recursive, False))

    def mv(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        return tools.mv(src=src, dst=dst, recursive=_parse_bool(recursive, False))

    def insert_line(
        path: str,
        content: str,
        anchor: str,
        where: Literal["before", "after"] = "before",
        occurrence: int = 1,
    ) -> dict[str, Any]:
        return tools.insert_line(
            path=path,
            content=content,
            anchor=anchor,
            where=where,
            occurrence=int(occurrence),
        )

    def replace_range(
        path: str,
        start_line: int,
        end_line: int,
        content: str,
        preserve_trailing_newline: bool = True,
    ) -> dict[str, Any]:
        """
        Replace a specific line range in a file with new content.

        This tool replaces lines by their positions (line numbers),
        NOT by matching text.

        Args:
            path (str):
                Path to the file.

            start_line (int):
                The starting line number (1-based, inclusive).

            end_line (int):
                The ending line number (1-based, inclusive).

            content (str):
                The content that will replace the specified line range.

            preserve_trailing_newline (bool, optional):
                If True (default), ensures the last replacement line
                ends with a newline.

        Behavior:
            - Replaces all lines from start_line to end_line (inclusive)
            - Line numbers are based on the current file content
            - If line numbers are incorrect, the wrong region will be modified

        How to use:
            1. ALWAYS call read_file first.
            2. Carefully count line numbers.
            3. Ensure the range exactly matches what you intend to replace.
            4. Use for contiguous blocks only.

        Good usage:
            - Replacing a full function or class when you know exact boundaries
            - Large block rewrites
            - Structured edits where line numbers are known

        Bad usage:
            - Small or precise edits (use edit instead)
            - When the file may change between steps
            - When line numbers are uncertain

        Warnings:
            - This tool is FRAGILE: line numbers can shift after edits
            - A small mistake in line numbers can corrupt the file

        Prefer using `edit` instead unless:
            - the target text is not unique, OR
            - you need to replace a large continuous block

        This tool is best used as a fallback when text-based matching is not reliable.
        """
        return tools.replace_range(
            path=path,
            start_line=int(start_line),
            end_line=int(end_line),
            content=content,
            preserve_trailing_newline=_parse_bool(preserve_trailing_newline, True),
        )

    def edit(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Edit a file by replacing a specific text fragment with new content.

        This tool performs an exact (literal) string replacement:
        it finds `old_string` in the file and replaces it with `new_string`.

        This is the PRIMARY tool for modifying files.
        Prefer using `edit` over other editing tools whenever possible.

        Args:
            path (str):
                Path to the file to edit.

            old_string (str):
                The exact text to find in the file.
                Must match the file content exactly, including whitespace,
                indentation, and line breaks.

            new_string (str):
                The text that will replace `old_string`.

            replace_all (bool, optional):
                If False (default):
                    - exactly ONE occurrence must exist in the file
                    - otherwise the tool fails

                If True:
                    - ALL occurrences will be replaced

            encoding (str, optional):
                File encoding (default: "utf-8").

        Behavior:
            - If the file does not exist:
                - it will be created ONLY if old_string == ""
            - If old_string is not found:
                - the tool fails
            - If multiple matches are found and replace_all=False:
                - the tool fails
            - If old_string == new_string:
                - no changes are made

        How to use:
            1. ALWAYS call read_file first to inspect the current content.
            2. Copy the exact fragment you want to modify.
            3. Include enough surrounding context to make it UNIQUE.
            4. Do NOT guess formatting — match it exactly.

        Good usage:
            - Modifying a function body
            - Renaming variables in a specific block
            - Updating a config value with context

        Bad usage:
            - Using partial or ambiguous snippets
            - Ignoring indentation or whitespace
            - Trying to modify multiple locations without replace_all=True

        Tips:
            - If you get "multiple locations" error:
                add more surrounding context to old_string
            - If you get "not found":
                re-check whitespace, indentation, and line endings
            - Prefer unique, multi-line snippets over short strings

        This tool is robust to file changes and should be used for most edits.
        """
        return tools.edit(
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=_parse_bool(replace_all, False),
            encoding=encoding,
        )

    def interaction_stats(
        path: str = "/",
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        """
        Summarize overlay-visible filesystem interaction progress.
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
            path=path,
            pattern=pattern,
            offset=offset,
            limit=limit,
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
            path=path,
            pattern=pattern,
            offset=offset,
            limit=limit,
        )

    def restore(
        path: str,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """
        Revert all changes and restore original file.
        """
        return tools.restore(path=path, recursive=_parse_bool(recursive, True))

    def list_match_only_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that were matched but never read.
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
        Reset interaction tracking for the overlay-visible read tools.
        """
        return tools.reset_interactions()

    registry = [
        ls,
        glob,
        read_file,
        grep,
        write_file,
        append_file,
        insert_line,
        replace_range,
        edit,
        restore,
        mkdir,
        rm,
        cp,
        mv,
    ]

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
