from __future__ import annotations

import pytest

from contractor_runtime.toolsets.common.lines import bounded_window, split_lines

BUDGET = 16


def window(text: str, *, start_line: int = 1, max_lines: int = 10, line_offset: int = 0):
    return bounded_window(
        split_lines(text, keepends=True),
        start_line=start_line,
        max_lines=max_lines,
        max_bytes=BUDGET,
        line_offset=line_offset,
    )


def test_whole_lines_fit_and_report_the_next_line() -> None:
    result = window("a\nb\nc\n", max_lines=2)
    assert (result.text, result.end_line, result.partial_line) == ("a\nb\n", 2, False)
    assert result.continuation(3) == {"nextStartLine": 3, "nextLineOffset": 0}
    assert window("a\nb\n").continuation(2) == {"nextStartLine": None, "nextLineOffset": None}


def test_later_oversized_line_starts_the_next_window_whole() -> None:
    lines = "short\n" + "x" * 20 + "\n"
    result = window(lines)
    assert (result.text, result.end_line, result.partial_line) == ("short\n", 1, False)
    assert result.continuation(2) == {"nextStartLine": 2, "nextLineOffset": 0}


def test_first_oversized_line_is_paged_on_character_boundaries() -> None:
    long_line = "ж" * 13 + "\n"
    text = long_line + "tail\n"
    parts: list[str] = []
    start, offset = 1, 0
    while True:
        result = window(text, start_line=start, max_lines=1, line_offset=offset)
        assert len(result.text.encode()) <= BUDGET
        parts.append(result.text)
        if not result.partial_line:
            break
        start, offset = result.next_start_line, result.next_line_offset
    assert "".join(parts) == long_line
    assert result.continuation(2) == {"nextStartLine": 2, "nextLineOffset": 0}


def test_invalid_offsets_and_empty_text() -> None:
    assert window("").continuation(0) == {"nextStartLine": None, "nextLineOffset": None}
    with pytest.raises(ValueError, match="line length"):
        window("", line_offset=1)
    with pytest.raises(ValueError, match="line length"):
        window("ab\n", line_offset=3)
    with pytest.raises(ValueError, match="UTF-8 character"):
        window("жж\n", line_offset=1)
