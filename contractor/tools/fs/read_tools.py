from __future__ import annotations

import contextlib
import fnmatch
import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Final, TypeAlias

import fsspec
from google.adk.tools.tool_context import ToolContext

from contractor.tools.fs.const import (
    _IGNORE_DEFAULTS,
    FS_COVERAGE_STATE_KEY,
    INCORRECT_REGEXP_ERROR,
    PATH_NOT_FOUND_ERROR,
)
from contractor.tools.fs.format import FileFormat
from contractor.tools.fs.models import (
    FileInteractionEntry,
    FsEntry,
    InteractionFilter,
    InteractionKind,
)
from contractor.tools.fs.utils import _ensure_int_or_none, _is_ignored
from contractor.tools.fs.validation import PathValidationMixin
from contractor.tools.observations import FILE_PATHS_STATE_KEY
from contractor.tools.result import guard, ok_page
from contractor.utils.formatting import norm_unicode, normalize_slashes
from contractor.utils.fs import join_path
from contractor.utils.settings import get_settings

ToolResult: TypeAlias = dict[str, Any]
BackendTool: TypeAlias = Callable[..., ToolResult]
PatchPayload: TypeAlias = dict[str, Any] | str

# Hard cap on the in-scope file walk (``in_scope_paths``). This walk feeds the
# always-on coverage-gap capture, so it must never run away on a huge tree; the
# coverage-gap projection caps the *surfaced* list far lower (25).
_IN_SCOPE_WALK_LIMIT: Final[int] = 2000

# Truncation notice attached to glob/grep output when the tree walk hit the
# ``fs_max_files_per_walk`` ceiling (style mirrors ``format_output``'s footer).
_WALK_TRUNCATION_NOTICE: Final[str] = (
    "### file walk truncated after scanning {max_files} files: results may be "
    "incomplete — narrow `path` or `pattern` ###"
)


def _push_fs_coverage(
    tool_context: ToolContext | None, snapshot: dict[str, int]
) -> None:
    if tool_context is None:
        return
    state = getattr(tool_context, "state", None)
    if state is None:
        return
    with contextlib.suppress(Exception):
        state[FS_COVERAGE_STATE_KEY] = snapshot


def _push_fs_paths(
    tool_context: ToolContext | None,
    read_paths: list[str],
    matched_paths: list[str],
    in_scope_paths: list[str],
) -> None:
    """Surface the concrete file paths read/matched/in-scope (names, not counts).

    Parallel to ``_push_fs_coverage`` but carries paths so observations can show
    *which* files the worker inspected (``read``/``matched``) and which ones it
    could still reach (``in_scope`` — drives the coverage-gap projection). Gated
    downstream by ``ObservationConfig.track_file_paths`` /
    ``track_coverage_gap``; the write is cheap/always-on (the in-scope walk is
    memoized by the tools instance, so it costs one walk per worker run).
    """
    if tool_context is None:
        return
    state = getattr(tool_context, "state", None)
    if state is None:
        return
    with contextlib.suppress(Exception):
        state[FILE_PATHS_STATE_KEY] = {
            "read": read_paths,
            "matched": matched_paths,
            "in_scope": in_scope_paths,
        }


def _build_ignore_patterns(ignored_patterns: list[str] | None = None) -> list[str]:
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
        max_output: int | None = None,
        max_items: int | None = None,
        max_lines: int | None = None,
        max_files_per_walk: int | None = None,
        ignored_patterns: list[str] | None = None,
        with_types: bool = True,
        with_file_info: bool = True,
        capture_in_scope: bool = False,
    ) -> None:
        s = get_settings()
        self.fs = fs
        self.fmt = fmt
        self.root = root
        # Opt-in: only walk the tree for the coverage-gap observation when a
        # caller asks for it. Off by default so the always-on capture path stays
        # cheap (no traversal) and the feature is a true no-op when disabled.
        self.capture_in_scope = capture_in_scope
        self.max_output = s.fs_max_output if max_output is None else max_output
        self.max_items = s.fs_max_items if max_items is None else max_items
        # Default line cap when read_file gets no explicit `limit`. Falls back
        # to the (possibly None) settings value, i.e. "no cap" unless configured.
        self.max_lines = s.fs_max_read_lines if max_lines is None else max_lines
        # Hard ceiling on files scanned by a single glob/grep tree walk so the
        # tools cannot run away on a huge repo (mirrors code_max_files_per_walk
        # for the code-tools walker). When hit, the output carries a notice.
        self.max_files_per_walk = (
            s.fs_max_files_per_walk
            if max_files_per_walk is None
            else max_files_per_walk
        )
        self.with_types = with_types
        self.with_file_info = with_file_info

        self.fmt.with_types = with_types
        self.fmt.with_file_info = with_file_info

        self._interactions: dict[str, FileInteractionEntry] = {}
        self._interaction_invocation_id: str | None = None
        # Memoized full in-scope file walk (see ``in_scope_paths``); invalidated
        # on a per-invocation reset so it can't go stale across worker runs.
        self._in_scope_cache: list[str] | None = None
        self.patterns = _build_ignore_patterns(ignored_patterns)

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _paginate(
        self,
        items: list[str],
        *,
        offset: int = 0,
        limit: int | None = None,
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

    def _interaction_entry(self, path: str) -> FileInteractionEntry | None:
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
        raw = path if path else self.root
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
        self._in_scope_cache = None

    def reset_interactions_for_invocation(self, invocation_id: str | None) -> None:
        """Clear accumulated interactions when entering a new ADK invocation.

        The streamline worker (and this single file-tools instance) is reused
        across every subtask in a planner iteration, while ``coverage_stats`` /
        ``read_paths`` report cumulatively. Without a per-run reset the
        deterministic observations projected back to the planner would carry
        forward files touched in earlier subtasks (or earlier retry attempts).
        Each worker ``run_async`` gets a fresh ``invocation_id`` (``AgentTool``
        spins up a new Runner + session), so resetting on a changed id scopes
        coverage/paths to exactly the current worker run. A ``None`` id (no
        tool_context) is a no-op so direct/test callers keep cumulative state.
        """
        if invocation_id is None or invocation_id == self._interaction_invocation_id:
            return
        self._interaction_invocation_id = invocation_id
        self._interactions.clear()
        self._in_scope_cache = None

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

    def read_paths(self) -> list[str]:
        """Paths the worker actually opened (read), sorted."""
        return sorted(p for p, e in self._interactions.items() if e.has_read)

    def matched_paths(self) -> list[str]:
        """Paths with a grep match, sorted."""
        return sorted(p for p, e in self._interactions.items() if e.has_match)

    def in_scope_paths(self) -> list[str]:
        """In-scope source files under the sandbox root, sorted (bounded).

        The (non-ignored) file set the worker *could* read — used by the
        coverage-gap observation to compute "in-scope minus read". Memoized for
        the lifetime of this tools instance (reset alongside interactions on a
        new invocation), so the walk runs at most once per worker run rather
        than on every read/grep.

        Returns ``[]`` (and skips the walk entirely) unless ``capture_in_scope``
        was requested at construction — keeping the disabled default free of any
        traversal. The walk is **hard-bounded** at ``_IN_SCOPE_WALK_LIMIT``
        files so it can never run away on a huge tree (or an unexpectedly broad
        root); the coverage-gap projection caps the surfaced list far lower.
        """
        if not self.capture_in_scope:
            return []
        if self._in_scope_cache is None:
            collected: list[str] = []
            # Collect up to a generous ceiling, then sort and cap — so the
            # retained subset is the deterministic lexicographically-first
            # _IN_SCOPE_WALK_LIMIT files, not an arbitrary walk-order slice.
            _ceiling = _IN_SCOPE_WALK_LIMIT * 10
            for path in self._iter_all_files(self.root):
                collected.append(path)
                if len(collected) >= _ceiling:
                    break
            self._in_scope_cache = sorted(collected)[:_IN_SCOPE_WALK_LIMIT]
        return self._in_scope_cache

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
        path: str | None = None,
        offset: int = 0,
    ) -> ToolResult:
        normalized_path = norm_unicode(path) or "/"
        normalized_path = self._resolve_root(normalized_path)

        normalized_pattern = norm_unicode(pattern)

        if normalized_pattern is None:
            return ok_page(
                [], 0, returned=0, offset=offset, limit=self.max_items
            )

        if normalized_path and not self.fs.exists(normalized_path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=normalized_path)}

        # Root the pattern at *path* when the pattern is relative, so that
        # callers like `glob("*.py", path="/src")` actually search /src.
        # Absolute patterns retain their original behaviour and the post-filter
        # below still scopes the result.
        pattern_for_fs = normalized_pattern.replace("\\", "/")
        if not pattern_for_fs.startswith("/") and normalized_path != "/":
            pattern_for_fs = f"{normalized_path.rstrip('/')}/{pattern_for_fs}"

        # Sandbox filesystems expose ``glob_scanned`` — a bounded glob whose
        # tree walk stops at the max-files ceiling and reports truncation.
        # Other backends fall back to the plain (unbounded) fsspec glob.
        glob_scanned = getattr(self.fs, "glob_scanned", None)
        if callable(glob_scanned):
            raw_matches, walk_truncated = glob_scanned(
                pattern_for_fs, max_files=self.max_files_per_walk
            )
        else:
            raw_matches, walk_truncated = self.fs.glob(pattern_for_fs), False

        matches = [str(match) for match in raw_matches]

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

        meta: dict[str, Any] = {}
        if walk_truncated:
            meta["walk_truncated"] = True
            meta["notice"] = _WALK_TRUNCATION_NOTICE.format(
                max_files=self.max_files_per_walk
            )

        return ok_page(
            self.fmt.format_file_list(paged),
            total,
            returned=len(paged),
            offset=offset,
            limit=self.max_items,
            **meta,
        )

    def read_file(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
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

        # No explicit limit → apply the configured default line cap (if any).
        if limit is None:
            limit = self.max_lines

        start_line = 1
        if offset is not None:
            offset = max(0, offset)
            if offset >= len(lines):
                return {"result": ""}
            lines = lines[offset:]
            start_line = offset + 1

        # Hand the full post-offset slice (not a pre-trimmed one) plus the line
        # cap to format_output, so the truncation footer fires whether the byte
        # OR the line cap binds — and the "lines left" / resume offset reflect
        # the true remaining count rather than the capped slice length.
        effective_limit = max(1, limit) if limit is not None else None

        if with_line_numbers:
            sliced = "\n".join(
                f"{line_no} | {line}"
                for line_no, line in enumerate(lines, start=start_line)
            )
        else:
            sliced = "\n".join(lines)
        return {
            "result": self.fmt.format_output(
                sliced,
                self.max_output,
                base_offset=start_line - 1,
                max_lines=effective_limit,
            )
        }

    def grep(
        self,
        pattern: str,
        path: str | None = None,
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

            return ok_page(
                self.fmt.format_file_list(paged),
                total,
                returned=len(paged),
                offset=offset,
                limit=self.max_items,
            )

        results: list[FsEntry] = []
        scanned = 0
        walk_truncated = False
        # Bound the tree walk so grep over a huge repo cannot run away; when
        # the ceiling is hit the (partial) results carry a truncation notice.
        for current_path, _dirs, filenames in self.fs.walk(normalized_path):
            for filename in filenames:
                if scanned >= self.max_files_per_walk:
                    walk_truncated = True
                    break
                scanned += 1
                full_path = join_path(current_path, filename)
                results.extend(build_entries_for_file(full_path))
            if walk_truncated:
                break

        results.sort(key=lambda entry: (entry.path, entry.loc.line_start or 0))
        total = len(results)
        paged = results[offset : offset + self.max_items]

        meta: dict[str, Any] = {}
        if walk_truncated:
            meta["walk_truncated"] = True
            meta["notice"] = _WALK_TRUNCATION_NOTICE.format(
                max_files=self.max_files_per_walk
            )

        return ok_page(
            self.fmt.format_file_list(paged),
            total,
            returned=len(paged),
            offset=offset,
            limit=self.max_items,
            **meta,
        )

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
        limit: int | None = None,
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

        return ok_page(
            [self._serialize_interaction_entry(p) for p in page],
            len(selected),
            returned=len(page),
            offset=resolved_offset,
            limit=resolved_limit,
            interaction=interaction.value,
        )

    def touched_files(
        self,
        path: str = "/",
        *,
        pattern: str = "**/*",
        offset: int = 0,
        limit: int | None = None,
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
        limit: int | None = None,
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

        return ok_page(
            [{"path": p} for p in page],
            len(selected),
            returned=len(page),
            offset=resolved_offset,
            limit=resolved_limit,
        )


def ro_file_tools(
    fs: fsspec.AbstractFileSystem,
    fmt: FileFormat,
    *,
    max_output: int | None = None,
    max_items: int = 100,
    ignored_patterns: list[str] | None = None,
    with_types: bool = True,
    with_file_info: bool = True,
    with_interaction_tools: bool = True,
    capture_in_scope: bool = False,
) -> list[Callable[..., dict[str, Any]]]:
    tools = FsspecInteractionFileTools(
        fs=fs,
        fmt=fmt,
        max_output=max_output,
        max_items=max_items,
        ignored_patterns=ignored_patterns,
        with_types=with_types,
        with_file_info=with_file_info,
        capture_in_scope=capture_in_scope,
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
        path: str | None = None,
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
        offset: int | None = None,
        limit: int | None = None,
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
            tools.reset_interactions_for_invocation(
                getattr(tool_context, "invocation_id", None)
            )
            result = tools.read_file(
                file_path=file,
                offset=_ensure_int_or_none(offset),
                limit=_ensure_int_or_none(limit),
                with_line_numbers=bool(with_line_numbers),
            )
            _push_fs_coverage(tool_context, tools.coverage_stats())
            _push_fs_paths(
                tool_context,
                tools.read_paths(),
                tools.matched_paths(),
                tools.in_scope_paths(),
            )
            return result

        return guard(_impl)

    def grep(
        pattern: str,
        path: str | None = None,
        offset: int = 0,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """
        Search file contents using a regular expression.

        Usage:
         - be more specific, avoid too general patterns like .*
        """

        def _impl() -> dict[str, Any]:
            tools.reset_interactions_for_invocation(
                getattr(tool_context, "invocation_id", None)
            )
            off = _ensure_int_or_none(offset) or 0
            result = tools.grep(pattern=pattern, path=path, offset=off)
            _push_fs_coverage(tool_context, tools.coverage_stats())
            _push_fs_paths(
                tool_context,
                tools.read_paths(),
                tools.matched_paths(),
                tools.in_scope_paths(),
            )
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
        limit: int | None = None,
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
        limit: int | None = None,
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
        limit: int | None = None,
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

    registry: list[BackendTool] = [ls, glob, read_file, grep]

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
