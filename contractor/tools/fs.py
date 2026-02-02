from __future__ import annotations

import unicodedata
import fnmatch
import json
import re
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Final, Literal, Optional, Union

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


def _norm_unicode(s: str) -> str:
    if s is None:
        return None
    # NFC — стандарт де-факто для файловых путей
    return unicodedata.normalize("NFC", s)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _is_ignored(path: str, patterns: list[str]) -> bool:
    """
    Match both full-path and basename against ignore patterns.
    Uses forward-slash normalization for consistency across FS backends.
    """
    p = path.replace("\\", "/")
    name = p.split("/")[-1]
    for pat in patterns:
        if fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


@dataclass
class FileLoc:
    """
    Marks a location in file.

    - line_start, line_end: 0-based line indices (inclusive)
    - byte_start, byte_end: 0-based byte offsets (half-open: [start, end))
    - content: a small excerpt around the match or the matched line(s)
    """

    line_start: Optional[int] = None
    line_end: Optional[int] = None
    byte_start: Optional[int] = None
    byte_end: Optional[int] = None
    content: Optional[str] = None


@dataclass
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
        file: str, fs: fsspec.AbstractFileSystem
    ) -> Optional[ContentTypeInfo]:
        if not fs.exists(file) or not fs.isfile(file):
            return None
        try:
            with fs.open(file, mode="rb") as f:
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
        if not fs.exists(path):
            return None

        path = _norm_unicode(path)
        name = _norm_unicode(path.rstrip("/").split("/")[-1])

        if fs.isdir(path):
            return cls(
                name=name,
                path=path,
                size=0,
                is_dir=True,
                filetype=None,
                loc=None,
            )

        if fs.isfile(path):
            filetype = None
            if with_types:
                filetype = cls.identify_type(path, fs)

            return cls(
                name=name,
                path=path,
                size=int(fs.size(path)),
                is_dir=False,
                filetype=filetype,
                loc=None,
            )

        return None

    @staticmethod
    def _compute_line_starts(text: str) -> list[int]:
        starts = [0]
        for m in re.finditer(r"\n", text):
            starts.append(m.end())
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
        file: str,
        fs: fsspec.AbstractFileSystem,
        *,
        content: Optional[str] = None,
        with_types: bool = True,
        excerpt_max_chars: int = 500,
        context_lines: int = 0,
    ) -> Optional[list["FsEntry"]]:
        if not fs.exists(file) or not fs.isfile(file):
            return None

        if not matches:
            return []

        if content is None:
            try:
                content = fs.read_text(file, encoding="utf-8", errors="ignore")
            except Exception:
                return []

        proto = cls.from_path(file, fs, with_types=with_types)
        if proto is None:
            return None

        line_starts = cls._compute_line_starts(content)
        lines = content.splitlines()

        entries: list[FsEntry] = []
        for m in matches:
            begin_char, end_char = m.span()
            line_idx = cls._char_to_line(line_starts, begin_char)

            ls = max(0, line_idx - context_lines)
            le = min(len(lines) - 1, line_idx + context_lines)

            try:
                byte_start = len(content[:begin_char].encode("utf-8", errors="ignore"))
                byte_end = len(content[:end_char].encode("utf-8", errors="ignore"))
            except Exception:
                byte_start, byte_end = None, None

            excerpt = "\n".join(lines[ls : le + 1])
            if len(excerpt) > excerpt_max_chars:
                excerpt = excerpt[:excerpt_max_chars] + "…"

            entries.append(
                cls(
                    name=proto.name,
                    path=proto.path,
                    size=proto.size,
                    filetype=proto.filetype,
                    loc=FileLoc(
                        line_start=ls,
                        line_end=le,
                        byte_start=byte_start,
                        byte_end=byte_end,
                        content=excerpt,
                    ),
                )
            )

        return sorted(entries, key=lambda e: (e.path, e.loc.line_start or 0))


@dataclass
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
            payload = {"line_start": loc.line_start, "line_end": loc.line_end}

        if loc.content is not None:
            payload["content"] = loc.content

        if self._format == "str":
            return json.dumps(payload, ensure_ascii=False)
        if self._format == "xml":
            parts = ["<loc>"]
            for k, v in payload.items():
                parts.append(f"<{k}>{_xml_escape(str(v))}</{k}>")
            parts.append("</loc>")
            return "".join(parts)
        return payload

    def format_fs_entry(self, f: FsEntry) -> Union[str, dict[str, Any]]:
        base: dict[str, Any] = {}
        kind: str = "dir" if f.is_dir else "file"

        if self.with_file_info:
            base.update({"kind": kind,"name": f.name, "path": f.path, "size": f.size,})

        if self.with_types and f.filetype is not None:
            try:
                base["filetype"] = asdict(f.filetype)
            except Exception:
                base["filetype"] = str(f.filetype)

        if f.loc is not None:
            base["loc"] = self._format_loc(f.loc)

        if self._format == "str":
            return json.dumps(base, ensure_ascii=False)
        
        if self._format == "xml":
            parts = [f"<{kind}>"]
            base.pop("kind", None)
            
            for k, v in base.items():
                if isinstance(v, (dict, list)):
                    parts.append(
                        f"<{k}>{_xml_escape(json.dumps(v, ensure_ascii=False))}</{k}>"
                    )
                else:
                    parts.append(f"<{k}>{_xml_escape(str(v))}</{k}>")
            parts.append(f"</{kind}>")
            return "".join(parts)
        
        return base

    def format_file_list(
        self, files: list[Optional[FsEntry]]
    ) -> Union[str, list[dict[str, Any]]]:
        cleaned = [f for f in files if f is not None]
        if self._format == "str":
            return "\n".join(str(self.format_fs_entry(f)) for f in cleaned)
        if self._format == "xml":
            inner = "".join(str(self.format_fs_entry(f)) for f in cleaned)
            return f"<files>{inner}</files>"
        return [self.format_fs_entry(f) for f in cleaned]  # type: ignore[return-value]

    @staticmethod
    def format_output(content: str, max_output: int) -> str:
        # Truncate by bytes (utf-8) while preserving line boundaries.
        lines = content.splitlines(True)  # keep line endings
        out_parts: list[str] = []
        out_bytes = 0
        cut_at_line: Optional[int] = None

        for i, line in enumerate(lines):
            line_bytes = len(line.encode("utf-8", errors="ignore"))
            if out_bytes + line_bytes > max_output:
                cut_at_line = i
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


def file_tools(
    fs: fsspec.AbstractFileSystem,
    fmt: FileFormat,
    *,
    max_output: int = 8 * 10**4,
    max_items: int = 300,
    ignored_patterns: Optional[list[str]] = None,
    with_types: bool = True,
    with_file_info: bool = True,
) -> dict[str, Any]:
    """
    Returns a registry of tools:
      - ls(path)
      - glob(pattern, path=None)
      - read_file(file, offset=None, limit=None)
      - grep(pattern, path=None)

    The implementation is filesystem-backend agnostic via fsspec.
    """
    fmt.with_types = with_types
    fmt.with_file_info = with_file_info

    patterns = []
    for pat in _IGNORE_DEFAULTS + (ignored_patterns or []):
        if pat and pat not in patterns:
            patterns.append(pat)

    def ls(path: str) -> dict[str, Any]:
        path = _norm_unicode(path)

        if not fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}
        try:
            items = fs.ls(path, detail=False)
        except TypeError:
            items = fs.ls(path)

        res = [
            FsEntry.from_path(str(p), fs, with_types=with_types)
            for p in items
            if not _is_ignored(str(p), patterns)
        ]
        return {"result": fmt.format_file_list(res)}

    def glob(pattern: str, path: Optional[str] = None, offset: int = 0) -> dict[str, Any]:
        if path is None:
            path = "/"

        path = _norm_unicode(path)
        pattern = _norm_unicode(pattern)

        if path and not fs.exists(path):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=path)}

        matches = [str(p) for p in fs.glob(pattern)]
        if path:
            prefix = path.rstrip("/").replace("\\", "/") + "/"
            matches = [m for m in matches if m.replace("\\", "/").startswith(prefix)]

        res = [
            FsEntry.from_path(p, fs, with_types=with_types)
            for p in matches
            if not _is_ignored(p, patterns)
        ]

        total = len(res)
        res = sorted(res, key=lambda e: e.path)
        res = res[offset : offset + max_items]

        return {"result": fmt.format_file_list(res),  "offset": offset, "total_items": total, "limit": max_items}

    def read_file(
        file: str, offset: Optional[int] = None, limit: Optional[int] = None
    ) -> dict[str, Any]:
        file = _norm_unicode(file)

        if not fs.exists(file) or _is_ignored(file, patterns):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=file)}
        if not fs.isfile(file):
            return {"error": PATH_IS_NOT_A_FILE_ERROR.format(path=file)}

        try:
            content = fs.read_text(file, encoding="utf-8", errors="ignore")
        except Exception:
            return {"result": ""}

        lines = content.splitlines()
        if offset is not None:
            offset = max(0, offset)
            if offset >= len(lines):
                return {"result": ""}
            lines = lines[offset:]
        if limit is not None:
            limit = max(1, limit)
            lines = lines[:limit]

        sliced = "\n".join(lines)
        return {"result": fmt.format_output(sliced, max_output)}

    def grep(pattern: str, path: Optional[str] = None, offset: int = 0) -> dict[str, Any]:
        """
        Regex search across a file or directory tree.
        Args:
            pattern: The regex pattern to search for.
            path: The path to search in. If None, search the current directory.
        Returns one FsEntry per match, with loc describing where it matched.
        """
        if path is None:
            path = "/"

        path = _norm_unicode(path)
        pattern = _norm_unicode(pattern)

        try:
            regex = re.compile(pattern)
        except re.error as err:
            return {"error": INCORRECT_REGEXP_ERROR.format(regex=pattern, err=str(err))}

        target = path or "."
        if not fs.exists(target):
            return {"error": PATH_NOT_FOUND_ERROR.format(path=target)}

        # Search a single file
        if fs.isfile(target):
            if _is_ignored(target, patterns):
                return {"result": []}
            try:
                content = fs.read_text(target, encoding="utf-8", errors="ignore")
            except Exception as exc:
                return {"error": str(exc)}

            matches = list(regex.finditer(content))
            entries = FsEntry.from_matches(
                matches=matches,
                file=target,
                fs=fs,
                content=content,
                with_types=with_types,
            )

            total = len(entries)
            if offset:
                entries = entries[offset:offset+max_items]

            return {"result": fmt.format_file_list(entries or []), "offset": offset, "total_items": total, "limit": max_items}

        # Walk a directory tree
        results: list[FsEntry] = []
        for current_path, _dirs, files in fs.walk(target):
            for fname in files:
                full_path = (str(current_path).rstrip("/") + "/" + str(fname)).replace(
                    "\\", "/"
                )
                if _is_ignored(full_path, patterns):
                    continue
                try:
                    content = fs.read_text(full_path, encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                matches = list(regex.finditer(content))
                entries = FsEntry.from_matches(
                    matches=matches,
                    file=full_path,
                    fs=fs,
                    content=content,
                    with_types=with_types,
                )
                if entries:
                    results.extend(entries)
        
        total = len(results)
        results = sorted(results, key=lambda x: (x.path, x.loc.line_start or 0))
        results = results[offset:offset+max_items]

        return {"result": fmt.format_file_list(results),  "offset": offset, "total_items": total, "limit": max_items}

    return [ls, glob, read_file, grep]


class RootedLocalFileSystem(LocalFileSystem):
    """
    Local filesystem sandboxed to root_path.
    Forbidden paths are treated as non-existent (silent sandbox).
    """

    def __init__(self, root_path: str, *args, **kwargs):
        self.root_path = os.path.realpath(stringify_path(root_path))
        if not os.path.isdir(self.root_path):
            raise ValueError(f"root_path is not a directory: {root_path}")

        # guaranteed non-existent path inside root
        self._blocked_path = os.path.join(self.root_path, ".__blocked__")

        super().__init__(*args, **kwargs)

    def ls(self, path: str = "", detail: bool = False, **kwargs):
        path = "" if path in (None, "/", "") else path
        host_path = self._strip_protocol(path)

        if host_path == self._blocked_path:
            return [] if not detail else []

        try:
            entries = super().ls(host_path, detail=True, **kwargs)
        except FileNotFoundError:
            return [] if not detail else []

        out = []
        for e in entries:
            host_name = e["name"]

            real = os.path.realpath(host_name)
            if not (real == self.root_path or real.startswith(self.root_path + os.sep)):
                continue

            rel = os.path.relpath(real, self.root_path)
            virt = "/" if rel == "." else "/" + rel.replace(os.sep, "/")

            if detail:
                e = e.copy()
                e["name"] = virt
                out.append(e)
            else:
                out.append(virt)

        return out

    def glob(self, pattern: str, **kwargs):
        """
        Sandbox-safe glob with Python-like semantics.
        Returns virtual paths: "/file.txt", "/dir/inner.txt"
        """
        if not pattern:
            return []

        # Normalize
        pattern = _norm_unicode(pattern.lstrip("/"))

        # Parent traversal is forbidden
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
                d for d in dirs if not os.path.islink(os.path.join(host_root, d))
            ]

            rel_root = os.path.relpath(host_root, self.root_path)
            if rel_root == ".":
                rel_root = ""

            for name in files:
                name = _norm_unicode(name)
                host_path = os.path.join(host_root, name)

                if os.path.islink(host_path):
                    continue

                rel_path = os.path.join(rel_root, name) if rel_root else name
                rel_path = _norm_unicode(rel_path.replace(os.sep, "/"))

                if fnmatch.fnmatch(rel_path, pattern):
                    matches.append("/" + rel_path)
                    continue

                if recursive and "/" not in rel_path:
                    tail = pattern.split("/")[-1]
                    if fnmatch.fnmatch(name, tail):
                        matches.append("/" + name)

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

        candidate = os.path.realpath(os.path.normpath(candidate))

        if candidate == self.root_path or candidate.startswith(self.root_path + os.sep):
            return candidate

        # Escape attempt → silent block
        return self._blocked_path
