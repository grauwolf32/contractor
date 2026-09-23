from __future__ import annotations

from contractor_runtime.toolsets.common.line_window import bounded_line_window


def test_whole_lines_within_the_byte_budget() -> None:
    lines = ["one\n", "two\n", "three\n"]

    assert bounded_line_window(lines, start_line=2, max_lines=5, max_bytes=100) == (
        "two\nthree\n",
        3,
        False,
    )
    assert bounded_line_window(lines, start_line=1, max_lines=1, max_bytes=100) == (
        "one\n",
        1,
        False,
    )


def test_later_oversized_line_ends_the_window_before_it() -> None:
    lines = ["a\n", "b" * 10 + "\n", "c\n"]

    text, end_line, partial = bounded_line_window(lines, start_line=1, max_lines=3, max_bytes=8)

    # Continuing at end_line + 1 must reread the whole second line.
    assert (text, end_line, partial) == ("a\n", 1, False)


def test_only_an_oversized_first_line_is_returned_as_a_utf8_safe_prefix() -> None:
    lines = ["\u00e9" * 5 + "\n", "next\n"]

    text, end_line, partial = bounded_line_window(lines, start_line=1, max_lines=2, max_bytes=5)

    assert (text, end_line, partial) == ("\u00e9\u00e9", 1, True)


def test_exactly_filled_budget_is_not_reported_as_a_partial_line() -> None:
    lines = ["abc\n", "d\n"]

    assert bounded_line_window(lines, start_line=1, max_lines=2, max_bytes=4) == (
        "abc\n",
        1,
        False,
    )


def test_partial_line_is_never_reported_without_text() -> None:
    lines = ["\u00e9\n"]

    assert bounded_line_window(lines, start_line=1, max_lines=1, max_bytes=1) == ("", 0, False)


def test_empty_text_has_no_lines() -> None:
    assert bounded_line_window([], start_line=1, max_lines=10, max_bytes=10) == ("", 0, False)
