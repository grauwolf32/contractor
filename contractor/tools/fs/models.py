import re
import fsspec

from magika import ContentTypeInfo, Magika
from typing import Optional, ClassVar
from enum import Enum
from weakref import WeakKeyDictionary

from dataclasses import dataclass, field

from contractor.utils.formatting import norm_unicode


class InteractionKind(str, Enum):
    READ = "read"
    MATCH = "match"


class InteractionFilter(str, Enum):
    ANY = "any"
    READ_ONLY = "read_only"
    MATCH_ONLY = "match_only"
    READ_AND_MATCH = "read_and_match"


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


@dataclass(slots=True)
class FsEntry:
    name: str
    path: str
    size: int
    is_dir: bool = False
    filetype: Optional[ContentTypeInfo] = None
    loc: Optional[FileLoc] = None

    _magika: ClassVar[Magika] = Magika()
    # Per-fs path-keyed cache. Held weakly so fs instances can be GC'd.
    # Each entry is the dict[path, ContentTypeInfo|None] for that fs.
    _filetype_cache: ClassVar[
        "WeakKeyDictionary[fsspec.AbstractFileSystem, dict[str, Optional[ContentTypeInfo]]]"
    ] = WeakKeyDictionary()

    @staticmethod
    def identify_type(
        file_path: str,
        fs: fsspec.AbstractFileSystem,
    ) -> Optional[ContentTypeInfo]:
        try:
            cache = FsEntry._filetype_cache.setdefault(fs, {})
        except TypeError:
            # fs not weakly referenceable (rare); fall through without caching.
            cache = None

        if cache is not None and file_path in cache:
            return cache[file_path]

        if not fs.exists(file_path) or not fs.isfile(file_path):
            result: Optional[ContentTypeInfo] = None
        else:
            try:
                with fs.open(file_path, mode="rb") as f:
                    result = FsEntry._magika.identify_stream(f).output
            except Exception:
                result = None

        if cache is not None:
            cache[file_path] = result
        return result

    @staticmethod
    def invalidate_filetype_cache(
        fs: fsspec.AbstractFileSystem,
        path: Optional[str] = None,
    ) -> None:
        """Invalidate cached file-type guesses for *fs*, optionally limited to *path*."""
        try:
            cache = FsEntry._filetype_cache.get(fs)
        except TypeError:
            return
        if cache is None:
            return
        if path is None:
            cache.clear()
            return
        cache.pop(path, None)
        # Also drop descendants when path is a directory.
        prefix = path.rstrip("/") + "/"
        for key in [k for k in cache if k.startswith(prefix)]:
            cache.pop(key, None)

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
