from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import fsspec
from google.adk.tools.tool_context import ToolContext

from contractor.tools.fs.const import PATH_IS_NOT_A_FILE_ERROR
from contractor.tools.fs.format import FileFormat
from contractor.tools.fs.models import InteractionFilter
from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem
from contractor.tools.fs.read_tools import (FsspecInteractionFileTools,
                                            ToolResult, _build_ignore_patterns,
                                            _push_fs_coverage)
from contractor.tools.fs.utils import (_ensure_int_or_none,
                                       _line_ending_for_text, _is_ignored,
                                       _parse_bool, _split_lines_keepends)
from contractor.tools.fs.validation import PathValidationMixin
from contractor.tools.result import guard


class FsspecWriteTools(PathValidationMixin):
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

    def _is_ignored(self, path: str) -> bool:
        return _is_ignored(path, self.patterns)

    def _ensure_interactions_enabled(self) -> Optional[ToolResult]:
        if not self.with_interaction_tools:
            return {
                "error": (
                    "interaction tools are disabled for this write toolset; "
                    "enable with_interaction_tools=True"
                )
            }
        return None

    # ---- read-through delegates ----

    def ls(self, path: str) -> ToolResult:
        """Delegate to the underlying reader's ls."""
        return self._reader.ls(path=path)

    def glob(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> ToolResult:
        """Delegate to the underlying reader's glob."""
        return self._reader.glob(pattern=pattern, path=path, offset=offset)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        offset: int = 0,
    ) -> ToolResult:
        """Delegate to the underlying reader's grep."""
        return self._reader.grep(pattern=pattern, path=path, offset=offset)

    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        with_line_numbers: bool = False,
    ) -> ToolResult:
        return self._reader.read_file(
            file_path=file_path,
            offset=offset,
            limit=limit,
            with_line_numbers=with_line_numbers,
        )

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

    def coverage_stats(self) -> dict[str, int]:
        return self._reader.coverage_stats()

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

    def changed_paths(self) -> ToolResult:
        if not isinstance(self.fs, MemoryOverlayFileSystem):
            return {
                "error": "changed_paths is only available with overlay filesystem"
            }
        return {"result": self.fs.changed_paths()}

    def diff(self, root: str = "/", context_lines: int = 3) -> ToolResult:
        if not isinstance(self.fs, MemoryOverlayFileSystem):
            return {"error": "diff is only available with overlay filesystem"}

        normalized_root, err = self._validate_path(
            root, must_exist=True, check_ignored=False
        )
        if err:
            return err

        try:
            context = max(0, int(context_lines))
        except (TypeError, ValueError):
            return {"error": "context_lines must be a non-negative integer"}

        try:
            diff_text = self.fs.diff(root=normalized_root, context_lines=context)
        except Exception as exc:
            return {"error": str(exc)}

        if not diff_text:
            return {"result": ""}

        return {"result": FileFormat.format_output(diff_text, self.max_output)}

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
        List the immediate children of a directory in the overlay-visible tree.

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
        Find files or directories by glob pattern in the overlay-visible tree.

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
        Search file contents using a regular expression in the overlay-visible tree.

        Usage:
         - be more specific, avoid too general patterns like .*
        """

        def _impl() -> dict[str, Any]:
            off = _ensure_int_or_none(offset) or 0
            result = tools.grep(pattern=pattern, path=path, offset=off)
            _push_fs_coverage(tool_context, tools.coverage_stats())
            return result

        return guard(_impl)

    def write_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Create a file, or overwrite it entirely with the given text.

        The existing content (if any) is fully replaced. To change part of a
        file, prefer `edit`; to add lines without rewriting, use `append_file`
        or `insert_line`.

        Args:
            path: Path to the file to write.
            content: Full new contents of the file.
            encoding: Text encoding (default "utf-8").

        Returns confirmation of the write, or an error if the path is invalid
        or escapes the sandbox.
        """
        return guard(
            lambda: tools.write_file(path=path, content=content, encoding=encoding)
        )

    def append_file(
        path: str,
        content: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        """
        Append text to the end of a file, creating it if it does not exist.

        Existing content is preserved; `content` is added after it.

        Args:
            path: Path to the file to append to.
            content: Text to add at the end of the file.
            encoding: Text encoding (default "utf-8").

        Returns confirmation of the append, or an error if the path is invalid.
        """
        return guard(
            lambda: tools.append_file(path=path, content=content, encoding=encoding)
        )

    def mkdir(
        path: str,
        create_parents: bool = True,
        exist_ok: bool = True,
    ) -> dict[str, Any]:
        """
        Create a directory.

        Args:
            path: Directory path to create.
            create_parents: If True (default), create missing parent
                directories as well.
            exist_ok: If True (default), succeed quietly when the directory
                already exists instead of failing.

        Returns confirmation of the created directory, or an error if the
        path is invalid.
        """
        return guard(
            lambda: tools.mkdir(
                path=path,
                create_parents=_parse_bool(create_parents, True),
                exist_ok=_parse_bool(exist_ok, True),
            )
        )

    def rm(
        path: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        Delete a file or directory.

        Args:
            path: Path to remove.
            recursive: Must be True to delete a non-empty directory; for a
                single file leave it False (default).

        Returns confirmation of the removal, or an error if the path does not
        exist or a non-empty directory is removed without recursive=True.
        """
        return guard(
            lambda: tools.rm(path=path, recursive=_parse_bool(recursive, False))
        )

    def cp(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        Copy a file or directory from one path to another.

        Args:
            src: Source path to copy from.
            dst: Destination path to copy to.
            recursive: Must be True to copy a directory and its contents.

        Returns confirmation of the copy, or an error if the source is missing
        or a directory is copied without recursive=True.
        """
        return guard(
            lambda: tools.cp(src=src, dst=dst, recursive=_parse_bool(recursive, False))
        )

    def mv(
        src: str,
        dst: str,
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        Move or rename a file or directory.

        Args:
            src: Source path to move from.
            dst: Destination path to move to.
            recursive: Must be True to move a directory and its contents.

        Returns confirmation of the move, or an error if the source is missing
        or a directory is moved without recursive=True.
        """
        return guard(
            lambda: tools.mv(src=src, dst=dst, recursive=_parse_bool(recursive, False))
        )

    def insert_line(
        path: str,
        content: str,
        anchor: str,
        where: Literal["before", "after"] = "before",
        occurrence: int = 1,
    ) -> dict[str, Any]:
        """
        Insert text before or after an existing anchor line in a file.

        The anchor is located by matching `anchor` against existing lines; the
        new `content` is inserted relative to it without rewriting the rest of
        the file. Use this for additive edits where you can name a nearby line;
        prefer `edit` for replacing text.

        Args:
            path: Path to the file to modify.
            content: Text to insert.
            anchor: Existing line text used to locate the insertion point.
            where: Insert "before" (default) or "after" the anchor.
            occurrence: Which matching anchor to use when several match
                (1-based, default the first).

        Returns confirmation of the insert (including whether the file
        changed), or an error if the anchor is not found or the path is
        invalid.
        """
        return guard(
            lambda: tools.insert_line(
                path=path,
                content=content,
                anchor=anchor,
                where=where,
                occurrence=int(occurrence),
            )
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
        return guard(
            lambda: tools.replace_range(
                path=path,
                start_line=int(start_line),
                end_line=int(end_line),
                content=content,
                preserve_trailing_newline=_parse_bool(preserve_trailing_newline, True),
            )
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
        return guard(
            lambda: tools.edit(
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=_parse_bool(replace_all, False),
                encoding=encoding,
            )
        )

    def interaction_stats(
        path: str = "/",
        pattern: str = "**/*",
    ) -> dict[str, Any]:
        """
        Summarize overlay-visible filesystem interaction progress.
        """
        return guard(lambda: tools.interaction_stats(path=path, pattern=pattern))

    def list_touched_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that had at least one recorded interaction.
        """
        off = _ensure_int_or_none(offset) or 0
        lim = _ensure_int_or_none(limit)
        return guard(
            lambda: tools.touched_files(
                path=path,
                pattern=pattern,
                offset=off,
                limit=lim,
            )
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
        off = _ensure_int_or_none(offset) or 0
        lim = _ensure_int_or_none(limit)
        return guard(
            lambda: tools.untouched_files(
                path=path,
                pattern=pattern,
                offset=off,
                limit=lim,
            )
        )

    def restore(
        path: str,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """
        Discard overlay edits and restore a path to its original base content.

        Use this to undo writes/edits made during the session and return the
        file (or subtree) to how it looked on the base filesystem.

        Args:
            path: File or directory to restore.
            recursive: If True (default), restore an entire directory subtree.

        Returns confirmation of the restore, or an error if the path is
        invalid.
        """
        return guard(
            lambda: tools.restore(path=path, recursive=_parse_bool(recursive, True))
        )

    def changed_paths() -> dict[str, Any]:
        """
        List paths added, modified, or deleted in the overlay vs the base
        filesystem.

        Returns:
            {"result": {"added": [...], "modified": [...], "deleted": [...]}}.
            Files whose overlay content matches the base content are not
            reported as modified.
        """
        return guard(lambda: tools.changed_paths())

    def diff(
        root: str = "/",
        context_lines: int = 3,
    ) -> dict[str, Any]:
        """
        Return a unified diff of all overlay changes vs the base filesystem.

        Args:
            root: Subtree to diff. Defaults to "/".
            context_lines: Number of unchanged context lines around each hunk.

        Output may be truncated when very large; the truncation marker
        notes how many lines were dropped.
        """
        return guard(
            lambda: tools.diff(
                root=root,
                context_lines=_ensure_int_or_none(context_lines) or 0,
            )
        )

    def list_match_only_files(
        path: str = "/",
        pattern: str = "**/*",
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        List files that were matched but never read.
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
        Reset interaction tracking for the overlay-visible read tools.
        """
        return guard(lambda: tools.reset_interactions())

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
        changed_paths,
        diff,
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
