"""One bounded, allocation-local payload for paging immutable Artifact revisions."""

from __future__ import annotations

import asyncio
import codecs
from array import array
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from contractor_runtime.artifacts import ArtifactClient, ArtifactValue
from contractor_runtime.contracts import ArtifactRef
from contractor_runtime.core.lines import LINE_BREAK_BYTES
from contractor_runtime.shared_config import artifact_payload_max_bytes
from contractor_runtime.threads import to_thread_until_done

if TYPE_CHECKING:
    from contractor_runtime.toolsets.common.resources import ToolResources

# The Artifact API ceiling comes from startup policy. Retain at most one such
# payload across Run and text artifact readers in an allocation.
# A line index keeps one count of line starts per block of the payload, so its
# size follows from the payload size alone, whatever its lines or characters,
# and finding a line scans at most one block. Decoded text is never retained.
LINE_INDEX_BLOCK_BYTES = 4096
_LINE_COUNT_TYPE = "Q"
# UTF-8 is validated in chunks, so validation never holds the decoded payload.
_UTF8_CHUNK_BYTES = 1024 * 1024


def _line_index_bytes(payload_bytes: int) -> int:
    blocks = -(-payload_bytes // LINE_INDEX_BLOCK_BYTES)
    return (blocks + 1) * array(_LINE_COUNT_TYPE).itemsize


# Everything the cache retains: one payload and its line index (LIM-ART-7).
def artifact_read_cache_max_bytes() -> int:
    size = artifact_payload_max_bytes()
    return size + _line_index_bytes(size)


class LineIndex:
    """Line boundaries of one valid UTF-8 payload, numbered like ``split_lines``.

    Only ``\\r\\n``, ``\\r`` and ``\\n`` end a line, and every byte offset
    returned here is a character boundary of the payload.
    """

    __slots__ = ("_block_bytes", "_data", "_starts_before_block", "total_lines")

    def __init__(self, data: bytes) -> None:
        block_bytes = LINE_INDEX_BLOCK_BYTES
        size = len(data)
        blocks = -(-size // block_bytes)
        starts = array(_LINE_COUNT_TYPE, [0]) * (blocks + 1)
        has_cr = data.find(b"\r") >= 0
        total = 0
        for block in range(blocks):
            begin = block * block_bytes
            total += _line_starts_in(data, begin, min(begin + block_bytes, size), has_cr)
            starts[block + 1] = total
        self._block_bytes = block_bytes
        self._data = data
        # Entry k counts the line starts before byte k * block_bytes.
        self._starts_before_block = starts
        self.total_lines = total

    def line_span(self, line: int) -> tuple[int, int]:
        """Return the byte range of 1-based ``line``, including its line break."""

        if not 1 <= line <= self.total_lines:
            raise RuntimeError("line is outside the indexed payload")
        start = self._line_start(line - 1)
        end = self._line_start(line) if line < self.total_lines else len(self._data)
        return start, end

    def extend_whole_lines(self, end: int, *, limit: int, line: int, count: int) -> tuple[int, int]:
        """Extend a window that ends with ``line`` at byte ``end``.

        Adds up to ``count`` following whole lines while each ends at or
        before byte ``limit``, and returns the new end byte and last line.
        Scans no byte after ``limit``.
        """

        data = self._data
        size = len(data)
        if count <= 0 or end >= size:
            return end, line
        for match in LINE_BREAK_BYTES.finditer(data, end, min(size, limit + 1)):
            if match.end() > limit:
                return end, line
            end, line, count = match.end(), line + 1, count - 1
            if count == 0:
                return end, line
        if end < size <= limit:
            # The last line has no line break.
            return size, line + 1
        return end, line

    def text(self, start: int, end: int) -> str:
        """Decode one window whose ends are character boundaries."""

        return self._data[start:end].decode("utf-8")

    def _line_start(self, index: int) -> int:
        starts = self._starts_before_block
        block = bisect_right(starts, index) - 1
        begin = block * self._block_bytes
        rank = index - starts[block]
        if begin == 0:
            if rank == 0:
                return 0
            rank -= 1
        # A line that starts at ``begin`` follows a break one byte earlier.
        scan_end = min(begin + self._block_bytes, len(self._data))
        for match in LINE_BREAK_BYTES.finditer(self._data, max(begin - 1, 0), scan_end):
            if rank == 0:
                return match.end()
            rank -= 1
        raise RuntimeError("line index does not match its payload")


@dataclass(frozen=True, slots=True)
class ExactArtifactRead:
    """One exact revision and what the cache derived from its payload."""

    value: ArtifactValue
    utf8: bool
    # Present when the reader asked for lines and the payload is valid UTF-8.
    lines: LineIndex | None


@dataclass(slots=True)
class _Entry:
    value: ArtifactValue
    utf8: bool | None = None
    lines: LineIndex | None = None


class ExactArtifactReadCache:
    """Serve pages of one exact revision from one download (ART-R117).

    A read that the cache cannot serve first resolves the ref's exact revision,
    media type and size through the metadata read: ``accept`` refuses the
    media type before any payload is downloaded, and a versionless ref whose
    current revision is already cached is served without downloading again.
    UTF-8 validation and the line index are derived once per cached revision,
    off the event loop and under the cache lock, so concurrent first reads
    share one derivation, and replacement and ``clear`` wait for it.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._entry: _Entry | None = None

    async def read(
        self,
        client: ArtifactClient,
        ref: ArtifactRef,
        *,
        accept: Callable[[str], None],
        lines: bool = False,
    ) -> ExactArtifactRead:
        if ref.revision is not None:
            async with self._lock:
                entry = self._entry
                if entry is not None and entry.value.artifact == ref:
                    accept(entry.value.media_type)
                    _observe(client, ref)
                    return await _derive(entry, lines=lines)
        metadata = await client.read_artifact_metadata(ref)
        accept(metadata.media_type)
        exact = metadata.artifact
        async with self._lock:
            entry = self._entry
            if entry is not None and entry.value.artifact == exact:
                _observe(client, exact)
            else:
                self._entry = None
                value = await client.read_artifact(exact)
                entry = _Entry(value)
                if _cacheable(exact, value):
                    self._entry = entry
            return await _derive(entry, lines=lines)

    async def clear(self) -> None:
        async with self._lock:
            self._entry = None


def allocation_artifact_read_cache(resources: ToolResources) -> ExactArtifactReadCache:
    return resources.artifact_read_cache


async def _derive(entry: _Entry, *, lines: bool) -> ExactArtifactRead:
    # Runs under the cache lock: a cancelled derivation is joined before the
    # lock is released, and its result is discarded.
    utf8 = entry.utf8
    if utf8 is None or (lines and utf8 and entry.lines is None):
        utf8, index = await to_thread_until_done(
            _prepare, entry.value.data, utf8, lines, name="artifact-read-cache-prepare"
        )
        entry.utf8 = utf8
        if index is not None:
            entry.lines = index
    return ExactArtifactRead(entry.value, utf8, entry.lines if lines else None)


def _prepare(data: bytes, utf8: bool | None, lines: bool) -> tuple[bool, LineIndex | None]:
    valid = _is_utf8(data) if utf8 is None else utf8
    return valid, LineIndex(data) if valid and lines else None


def _is_utf8(data: bytes) -> bool:
    if data.isascii():
        return True
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        for start in range(0, len(data), _UTF8_CHUNK_BYTES):
            decoder.decode(data[start : start + _UTF8_CHUNK_BYTES])
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def _line_starts_in(data: bytes, begin: int, end: int, has_cr: bool) -> int:
    # Line starts in [begin, end): byte 0 of a non-empty payload and every
    # position after "\n" or after a "\r" that "\n" does not follow. The
    # payload end starts no line.
    count = 1 if begin == 0 else 0
    low, high = max(begin - 1, 0), end - 1
    count += data.count(b"\n", low, high)
    if has_cr:
        count += data.count(b"\r", low, high) - data.count(b"\r\n", low, high + 1)
    return count


def _observe(client: ArtifactClient, ref: ArtifactRef) -> None:
    observe = getattr(client, "observe_cached_read", None)
    if callable(observe):
        observe(ref)


def _cacheable(ref: ArtifactRef, value: ArtifactValue) -> bool:
    exact = value.artifact
    size = len(value.data)
    return (
        exact.revision is not None
        and exact.namespace == ref.namespace
        and exact.name == ref.name
        and (ref.revision is None or exact.revision == ref.revision)
        and size + _line_index_bytes(size) <= artifact_read_cache_max_bytes()
    )
