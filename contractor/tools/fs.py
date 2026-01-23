from __future__ import annotations

import re
import fsspec
from typing import Final, Literal, Optional, Any
from magika import Magika, ContentTypeInfo
from dataclasses import dataclass, field, asdict

_IGNORE_DEFAULTS: Final[list[str]] = [
    "*.pyc",
    "*/__pycache__/*",
    "__pycache__/*",
    "*.so",
    ".dll",
    "*.bin",
    "*.o",
    "*.dylib*.jpg",
    "*.jpeg",
    "*.webp",
    "*.png",
    "*.svg",
    "*.heic*.mov",
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


@dataclass
class FileLoc:
    """
    Marks location in file.
    start, end: lines in file
    content: content within those lines
    """

    line_start: Optional[int] = None
    line_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    content: Optional[str] = None


@dataclass
class FileEntry:
    filename: str
    path: str
    size: int
    filetype: Optional[ContentTypeInfo] = None
    loc: Optional[FileLoc] = None

    @staticmethod
    @lru_cache()
    def identify_type(
        file: str, fs: fsspec.AbstractFileSystem
    ) -> Optional[ContentTypeInfo]:
        if not fs.exists(file):
            return
        with fs.open(file) as f:
            res = self._magika.identify_stream(f).output
            return res

    @classmethod
    def from_file(
        cls, file: str, fs: fsspec.AbstractFileSystem, with_types: bool
    ) -> Optional[FileEntry]:
        if not fs.exists(file):
            return

        filetype: Optional[ContentTypeInfo] = None
        if with_types:
            filetype = FileEntry.identify_type(file, fs)

        return cls(
            filename=file.split("/")[-1],
            path=file,
            size=fs.size(file),
            filetype=filetype,
            locs=None,
        )

    @classmethod
    def from_matches(
        cls,
        matches: re.Match,
        file: str,
        fs: fsspec.AbstractFileSystem,
        *,
        content: Optional[str],
        with_types: bool = True,
    ) -> Optional[list[FileEntry]]:
        if not fs.exists(file):
            return

        if not match:
            return []

        if not content:
            content = fs.read_text(file, encoding="utf-8", errors="ignore")

        lines = content.split("\n")
        proto = cls.from_file(file, fs, with_types)

        entries: list[FileEntry] = []
        for m in maches:
            begin_pos, end_pos = m.span
            entry = copy.deepcopy(proto)
            ...  # define corresponding lines for matches and fill loc of the entry
            entries.append(entry)

        return entry


@dataclass
class FileAnalysisTools:
    fs: fsspec.AbstractFileSystem
    max_output: int  # max output in bytes
    ignored_patterns: list[str] = field(default_factory=list)
    with_types: bool = True
    _magika: Magika = Magika()

    def _is_ignored(self, filename) -> bool:
        return False  # TODO: Implement

    def _process_match(
        self, match: re.Match, filename: str, content: str
    ) -> FileEntry: ...  # TODO: Implement

    def search_pattern(
        self, pattern: str, path: str = "."
    ) -> Optional[list[FileEntry]]:
        pattern = re.compile(pattern)
        if not fs.exists(path):
            return None

        results: list[FileEntry] = []
        for current_path, dirs, files in self.fs.walk(path):
            for file in files:
                full_path = "/".join((current_path, file))
                if self._is_ignored(full_path):
                    continue
                with fs.open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    matches = list(re.finditer(pattern, content))
                    results += FileEntry.from_matches(
                        matches=matches,
                        file=file,
                        fs=fs,
                        content=content,
                        with_types=self.with_types,
                    )

        return results


@dataclass
class FileFormat:
    with_types: bool = True
    with_file_info: bool = True
    _format: Literal["str", "json", "xml"] = "json"
    loc: Literal["lines", "bytes"] = "lines"

    def format_file_entry(f: FileEntry) -> Union[str, dict]: ...

    def _format_loc(loc: FileLoc) -> Union[str, dict]: ...

    def format_file_list(
        files: list[FileEntry],
    ) -> Union[str, list[dict[str, Any]]]: ...

    def format_output(content: str, max_output: int) -> str:
        lines: list[str] = content.split("\n")
        output: str = ""
        count: int | None = None

        for idx, line in enumerate(lines):
            if not len(line) + len(output) <= max_output:
                count = max(0, idx - 1)
                break
            output += line

        if count is None:
            return output

        footer: str = f"### current line: {count} ### lines left in the file: {len(lines) - count}"
        output += footer
        return output


def file_tools(
    fs: fsspec.AbstractFileSystem,
    fmt: FileFormat,
    max_output: int = 8 * 10**4,
    ignored_patterns: Optional[list[str]] = _IGNORE_DEFAULTS,
    with_types: bool = True,
    with_file_info: bool = True,
):
    tools = FileAnalysisTools(fs, max_output, ignored_patterns, with_types)

    def ls(path: str) -> dict[str, Any]:
        """
        Lists the names of files and subdirectories directly within a specified directory path.
        Args:
          path: The absolute path to the directory to list (must be absolute, not relative)
        """

        if not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        res = [
            FileEntry.from_file(f, fs, with_types=with_types)
            for f in tools.fs.ls(path)
            if not tools._is_ignored(f)
        ]

        return {"result": fmt.format_file_list(res)}

    def glob(pattern: str, path: Optional[str]) -> dict[str, Any]:
        """
        Fast file pattern matching tool that works with any codebase size
        - Supports glob patterns like "**/*.js" or "src/**/*.ts"
        - Returns matching file paths sorted by modification time
        - Use this tool when you need to find files by name patterns
        - When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
        - You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.',
        Args:
          pattern: The glob pattern to match files against
          path: The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior. Must be a valid directory path if provided.
        """

        if path and not tools.fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        res = [
            FileEntry.from_file(f)
            for f in tools.fs.glob(pattern)
            if not tools._is_ignored(f)
        ]

        if path:
            res = [entry for entry in res if entry.path.startswith(path)]

        return {"result": fmt.format_file_list(res)}

    def read_file(
        file: str, offset: Optional[int], limit: Optional[int]
    ) -> dict[str, Any]:
        """
        Reads and returns the content of a specified file.
        If the file is large, the content will be truncated.
        The tool's response will clearly indicate if truncation has occurred and will provide details on how to read more of the file using the 'offset' and 'limit' parameters.
        For text files, it can read specific line ranges.
        Args:
          file: The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path.
          offset: (Optional) For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.
          limit: (Optional) For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit).
        """

        if not tools.fs.exists(file) or tools._is_ignored(file):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=file)}

        if not tools.fs.isfile(file):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=file)}

        content = fs.read_text(file, encoding="utf-8", errors="ignore")
        lines = content.split("\n")
        if offset and offset < len(lines):
            lines = lines[offset:]
        if limit:
            lines = lines[:limit]
        content = "\n".join(lines)

        return fmt.format_output(content, tools.max_output)

    def grep(pattern: str, path: Optional[str]):
        """
        A powerful search tool for finding patterns in files
        Args:
          pattern: The regular expression pattern to search for in file contents
          path: (Optional) Absolute path to file or directory to search in. Defaults to current working directory.

        Usage:
          - ALWAYS use Grep for search tasks. NEVER invoke `grep` or `rg` as a Bash command. The Grep tool has been optimized for correct permissions and access.
          - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+")
        """

        if not tools.fs.exists(file) or tools._is_ignored(file):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=file)}

        res = tools.search_pattern(pattern, path=path or ".")
        return {"result": fmt.format_file_list(res)}
