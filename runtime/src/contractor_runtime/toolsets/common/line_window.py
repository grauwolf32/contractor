"""Byte-bounded line windows shared by tools that page text by line number."""

from __future__ import annotations

from collections.abc import Sequence


def bounded_line_window(
    lines: Sequence[str],
    *,
    start_line: int,
    max_lines: int,
    max_bytes: int,
) -> tuple[str, int, bool]:
    """Return ``(text, end_line, partial_line)`` for a 1-based line window.

    ``lines`` keep their line breaks, as ``split_lines(text, keepends=True)``
    returns them. Whole lines are returned while their UTF-8 encoding fits
    ``max_bytes``; a later line that does not fit ends the window before it, so
    a caller continuing at ``end_line + 1`` reads that line in full. Only a first
    line larger than ``max_bytes`` is returned as a UTF-8-safe prefix, with
    ``partial_line`` set and ``end_line`` naming that line. ``partial_line`` is
    never set without returned text.
    """

    selected: list[str] = []
    size = 0
    end_line = min(start_line - 1, len(lines))
    window = lines[start_line - 1 : start_line - 1 + max_lines]
    for line_number, line in enumerate(window, start=start_line):
        encoded = line.encode("utf-8")
        if size + len(encoded) <= max_bytes:
            selected.append(line)
            size += len(encoded)
            end_line = line_number
            continue
        if not selected:
            prefix = encoded[:max_bytes].decode("utf-8", errors="ignore")
            if prefix:
                return prefix, line_number, True
        break
    return "".join(selected), end_line, False
