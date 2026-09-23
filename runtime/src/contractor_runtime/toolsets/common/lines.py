"""Line splitting shared by tools that address files by line number."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contractor_runtime.toolsets.common.input_errors import ToolInputError

_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_lines(text: str, keepends: bool = False) -> list[str]:
    """Split like ``str.splitlines`` but only on ``\\r\\n``, ``\\r`` and ``\\n``.

    These are the line boundaries read_file numbers, so line numbers taken from
    a read stay valid for edits, grep results and annotations. Other Unicode
    separators such as form feed or U+2028 remain part of their line.
    """

    lines: list[str] = []
    start = 0
    for match in _LINE_BREAK.finditer(text):
        lines.append(text[start : match.end() if keepends else match.start()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


def newline_style(text: str) -> str:
    """Return the first line break in ``text``, defaulting to ``\\n``."""

    for index, character in enumerate(text):
        if character == "\n":
            return "\n"
        if character == "\r":
            return "\r\n" if index + 1 < len(text) and text[index + 1] == "\n" else "\r"
    return "\n"


@dataclass(frozen=True)
class LineWindow:
    """A byte-bounded window of lines and where the next window starts."""

    text: str
    end_line: int
    partial_line: bool
    next_start_line: int
    next_line_offset: int

    def truncated(self, total_lines: int) -> bool:
        return self.partial_line or self.end_line < total_lines

    def continuation(self, total_lines: int) -> dict[str, int | None]:
        """Return the model-facing paging fields, null once the text is exhausted."""

        more = self.truncated(total_lines)
        return {
            "nextStartLine": self.next_start_line if more else None,
            "nextLineOffset": self.next_line_offset if more else None,
        }


def validate_line_offset(line_offset: int) -> None:
    if isinstance(line_offset, bool) or not isinstance(line_offset, int) or line_offset < 0:
        raise ToolInputError("line_offset must be a non-negative integer")


def bounded_window(
    lines: list[str],
    *,
    start_line: int,
    max_lines: int,
    max_bytes: int,
    line_offset: int = 0,
) -> LineWindow:
    """Select whole lines from ``start_line`` within ``max_bytes`` of UTF-8.

    The window ends before a later line that does not fit, so that line is
    returned whole by the next window. Only a first line longer than the budget
    is split, on a UTF-8 character boundary; ``line_offset`` continues it.
    """

    if not lines:
        if line_offset:
            raise ToolInputError("line_offset exceeds the line length")
        return LineWindow("", 0, False, 1, 0)
    first = lines[start_line - 1].encode("utf-8")
    if line_offset and line_offset >= len(first):
        raise ToolInputError("line_offset exceeds the line length")
    if line_offset and (first[line_offset] & 0xC0) == 0x80:
        raise ToolInputError("line_offset must start a UTF-8 character")
    window = [
        first[line_offset:],
        *(line.encode("utf-8") for line in lines[start_line : start_line - 1 + max_lines]),
    ]
    result: list[bytes] = []
    size = 0
    for encoded in window:
        remaining = max_bytes - size
        if len(encoded) <= remaining:
            result.append(encoded)
            size += len(encoded)
            continue
        if not result:
            cut = remaining
            while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
                cut -= 1
            return LineWindow(
                encoded[:cut].decode("utf-8"), start_line, True, start_line, line_offset + cut
            )
        break
    end_line = start_line + len(result) - 1
    return LineWindow(b"".join(result).decode("utf-8"), end_line, False, end_line + 1, 0)
