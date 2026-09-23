"""Line splitting shared by tools that address files by line number."""

from __future__ import annotations

import re

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
