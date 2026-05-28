from __future__ import annotations

import fnmatch
import re
from typing import Any, Callable, Iterable, Iterator, Optional, TypeAlias

import fsspec
from google.adk.tools.tool_context import ToolContext

from contractor.tools.fs.const import (_IGNORE_DEFAULTS, FS_COVERAGE_STATE_KEY,
                                       INCORRECT_REGEXP_ERROR,
                                       PATH_NOT_FOUND_ERROR)
from contractor.tools.fs.format import FileFormat
from contractor.tools.fs.models import (FileInteractionEntry, FsEntry,
                                        InteractionFilter, InteractionKind)
from contractor.tools.fs.utils import _ensure_int_or_none, _is_ignored
from contractor.tools.fs.validation import PathValidationMixin
from contractor.tools.result import guard
from contractor.utils.formatting import norm_unicode, normalize_slashes
from contractor.utils.fs import join_path

ToolResult: TypeAlias = dict[str, Any]
BackendTool: TypeAlias = Callable[..., ToolResult]
PatchPayload: TypeAlias = dict[str, Any] | str


def _push_fs_coverage(
    tool_context: ToolContext | None, snapshot: dict[str, int]
) -> None:
    if tool_context is None:
        return
    state = getattr(tool_context, "state", None)
    if state is None:
        return
    try:
        state[FS_COVERAGE_STATE_KEY] = snapshot
    except Exception:
        pass


def _build_ignore_patterns(ignored_patterns: Optional[list[str]] = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for pattern in _IGNORE_DEFAULTS + (ignored_patterns or []):
        if pattern and pattern not in seen:
            seen.add(pattern)
            result.append(pattern)
    return result


class FsspecInteractionFileTools(PathValidationMixin):
    def __init__(
        self,
        fs: fsspec.AbstractFileSystem,
        fmt: FileFormat,
        *,
        root: str = "/",
        max_output: int = 80_000,
        max_items: int = 100,
        ignored_patterns: Optional[list[str]] = None,
        with_types: bool = True,
        with_file_info: bool = True,
    ) -> None:
        self.fs = fs
        self.fmt = fmt
        self.root = root
        self.max_output = max_output
        self.max_items = max_items
        self.with_types = with_types
        self.with_file_info = with_file_info

        self.fmt.with_types = with_types
        self.fmt.with_file_info = with_file_info

        self._interactions: dict[str, FileInteractionEntry] = {}
        self.patterns = _build_ignore_patterns(ignored_patterns)

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
                full_path = join_path(current_path, filename)
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

    def _resolve_root(self, path: str | None = None) -> str:
        raw = self.root if not path else path
        raw = normalize_slashes(raw)

        if not raw.startswith("/"):
            raw = f"{self.root.rstrip('/')}/{raw.lstrip('/')}"

        cleaned = "/" + "/".join(part for part in raw.split("/") if part)
        return cleaned or "/"

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

    def coverage_stats(self) -> dict[str, int]:
        files_read = 0
        files_matched = 0
        total_reads = 0
        total_matches = 0
        for entry in self._interactions.values():
            if entry.has_read:
                files_read += 1
            if entry.has_match:
                files_matched += 1
            total_reads += entry.read_count
            total_matches += entry.match_count
        return {
            "files_seen": len(self._interactions),
            "files_read": files_read,
            "files_matched": files_matched,
            "total_reads": total_reads,
            "total_matches": total_matches,
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
        normalized_path = norm_unicode(path) or "/"
        normalized_path = self._resolve_root(normalized_path)

        normalized_pattern = norm_unicode(pattern)

        if normalized_pattern is None:
            return {
                "result": [],
                "offset": offset,
                "total_items": 0,
                "limit": self.max_items,
            }

        if normalized_path and not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        # Root the pattern at *path* when the pattern is relative, so that
        # callers like `glob("*.py", path="/src")` actually search /src.
        # Absolute patterns retain their original behaviour and the post-filter
        # below still scopes the result.
        pattern_for_fs = normalized_pattern.replace("\\", "/")
        if not pattern_for_fs.startswith("/") and normalized_path != "/":
            pattern_for_fs = f"{normalized_path.rstrip('/')}/{pattern_for_fs}"

        matches = [str(match) for match in self.fs.glob(pattern_for_fs)]

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
        with_line_numbers: bool = False,
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
        except Exception as exc:
            return {"error": f"failed to read '{normalized_file}': {exc}"}

        lines = content.splitlines()

        start_line = 1
        if offset is not None:
            offset = max(0, offset)
            if offset >= len(lines):
                return {"result": ""}
            lines = lines[offset:]
            start_line = offset + 1

        if limit is not None:
            lines = lines[: max(1, limit)]

        if with_line_numbers:
            sliced = "\n".join(
                f"{line_no} | {line}"
                for line_no, line in enumerate(lines, start=start_line)
            )
        else:
            sliced = "\n".join(lines)
        return {"result": self.fmt.format_output(sliced, self.max_output)}

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> ToolResult:
        normalized_path = norm_unicode(path) or "/"
        normalized_pattern = norm_unicode(pattern)

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
                full_path = join_path(current_path, filename)
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
        normalized_path = norm_unicode(path) or "/"
        normalized_pattern = norm_unicode(pattern) or "**/*"

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
        normalized_path = norm_unicode(path) or "/"
        normalized_pattern = norm_unicode(pattern) or "**/*"

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
        normalized_path = norm_unicode(path) or "/"
        normalized_pattern = norm_unicode(pattern) or "**/*"

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
        List the immediate children of a directory.

        Args:
            path: Directory whose contents to list.

        Returns the directory entries, or an error if the path does not exist
        or is not a directory.
        """
        return guard(lambda: tools.ls(path=path))

    def glob(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Find files or directories by glob pattern.

        Relative patterns (e.g. "*.py", "**/*.py") are searched under ``path``.
        Absolute patterns are matched as-is and post-filtered by ``path``.
        """
        off = _ensure_int_or_none(offset) or 0
        return guard(lambda: tools.glob(pattern=pattern, path=path, offset=off))

    def read_file(
        file: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        with_line_numbers: bool = False,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """
        Read text content from a file.

        Args:
            file: Path to the file to read.
            offset: Zero-based line offset. If provided, reading starts from
                this line.
            limit: Maximum number of lines to return after applying offset.
            with_line_numbers: If True, each returned line is prefixed with
                its 1-based line number (e.g. ``10 | def hello():``). Line
                numbers are NOT part of the file content.
        """

        def _impl() -> dict[str, Any]:
            result = tools.read_file(
                file_path=file,
                offset=_ensure_int_or_none(offset),
                limit=_ensure_int_or_none(limit),
                with_line_numbers=bool(with_line_numbers),
            )
            _push_fs_coverage(tool_context, tools.coverage_stats())
            return result

        return guard(_impl)

    def grep(
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """
        Search file contents using a regular expression.

        Usage:
         - be more specific, avoid too general patterns like .*
        """

        def _impl() -> dict[str, Any]:
            off = _ensure_int_or_none(offset) or 0
            result = tools.grep(pattern=pattern, path=path, offset=off)
            _push_fs_coverage(tool_context, tools.coverage_stats())
            return result

        return guard(_impl)

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
        return guard(lambda: tools.interaction_stats(path=path, pattern=pattern))

    def list_touched_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that had at least one recorded interaction (read or grep
        match).

        Args:
            path: Root directory to search within.
            pattern: Glob pattern to filter files.
            offset: Pagination offset (0-based).
            limit: Maximum number of items to return.

        Returns the matching files (paginated), or an error if the path is
        invalid.
        """
        off = _ensure_int_or_none(offset) or 0
        lim = _ensure_int_or_none(limit)
        return guard(
            lambda: tools.touched_files(
                path=path, pattern=pattern, offset=off, limit=lim
            )
        )

    def list_untouched_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that have not been read and did not match grep() — the
        unexplored remainder.

        Args:
            path: Root directory to search within.
            pattern: Glob pattern to filter files.
            offset: Pagination offset (0-based).
            limit: Maximum number of items to return.

        Returns the matching files (paginated), or an error if the path is
        invalid.
        """
        off = _ensure_int_or_none(offset) or 0
        lim = _ensure_int_or_none(limit)
        return guard(
            lambda: tools.untouched_files(
                path=path, pattern=pattern, offset=off, limit=lim
            )
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
        off = _ensure_int_or_none(offset) or 0
        lim = _ensure_int_or_none(limit)

        return guard(
            lambda: tools.files_with_interactions(
                path=path,
                pattern=pattern,
                interaction=InteractionFilter.MATCH_ONLY,
                offset=off,
                limit=lim,
            )
        )

    def reset_interaction_tracking() -> dict[str, Any]:
        """
        Reset interaction tracking.
        """

        def _impl() -> dict[str, Any]:
            tools.reset_interactions()
            return {"result": "ok"}

        return guard(_impl)

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
