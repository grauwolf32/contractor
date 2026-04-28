from contractor.tools.fs.utils import (
    _ensure_int_or_none,
    _format_comment_line,
    _is_ignored,
    _leading_ws,
    _line_ending_for_text,
    _parse_bool,
    _split_lines_keepends,
)


# ---------------------------------------------------------------------------
# _is_ignored
# ---------------------------------------------------------------------------


def test_is_ignored_matches_basename_pattern():
    assert _is_ignored("/a/b/foo.pyc", ["*.pyc"]) is True


def test_is_ignored_matches_path_pattern():
    assert _is_ignored("/a/__pycache__/x.txt", ["*/__pycache__/*"]) is True


def test_is_ignored_returns_false_when_no_match():
    assert _is_ignored("/a/b/c.py", ["*.pyc", "*/__pycache__/*"]) is False


def test_is_ignored_normalizes_backslashes():
    # The helper normalises slashes before matching.
    assert _is_ignored(r"a\b\foo.pyc", ["*.pyc"]) is True


def test_is_ignored_empty_patterns():
    assert _is_ignored("/a/b/c.py", []) is False


# ---------------------------------------------------------------------------
# _ensure_int_or_none
# ---------------------------------------------------------------------------


def test_ensure_int_or_none_passthrough_int():
    assert _ensure_int_or_none(42) == 42


def test_ensure_int_or_none_parses_str():
    assert _ensure_int_or_none("17") == 17


def test_ensure_int_or_none_returns_none_for_none():
    assert _ensure_int_or_none(None) is None


def test_ensure_int_or_none_returns_none_for_garbage():
    assert _ensure_int_or_none("not a number") is None
    assert _ensure_int_or_none(object()) is None


# ---------------------------------------------------------------------------
# _split_lines_keepends + _line_ending_for_text
# ---------------------------------------------------------------------------


def test_split_lines_keepends_empty_returns_empty():
    assert _split_lines_keepends("") == []


def test_split_lines_keepends_preserves_endings():
    assert _split_lines_keepends("a\nb\n") == ["a\n", "b\n"]


def test_split_lines_keepends_preserves_crlf():
    assert _split_lines_keepends("a\r\nb\r\n") == ["a\r\n", "b\r\n"]


def test_line_ending_for_text_detects_crlf():
    assert _line_ending_for_text("a\r\nb\r\n") == "\r\n"


def test_line_ending_for_text_defaults_to_lf():
    assert _line_ending_for_text("a\nb\n") == "\n"


def test_line_ending_for_text_defaults_to_lf_for_empty():
    assert _line_ending_for_text("") == "\n"


# ---------------------------------------------------------------------------
# _leading_ws
# ---------------------------------------------------------------------------


def test_leading_ws_spaces():
    assert _leading_ws("    indented") == "    "


def test_leading_ws_tabs():
    assert _leading_ws("\t\tindented") == "\t\t"


def test_leading_ws_mixed():
    assert _leading_ws(" \t mixed") == " \t "


def test_leading_ws_no_indent():
    assert _leading_ws("flush") == ""


# ---------------------------------------------------------------------------
# _format_comment_line
# ---------------------------------------------------------------------------


def test_format_comment_line_python_style():
    out = _format_comment_line(comment="hello", indent="    ", prefix="#", newline="\n")
    assert out == "    # hello\n"


def test_format_comment_line_strips_inner_whitespace():
    out = _format_comment_line(comment="  hello  ", indent="", prefix="//", newline="\n")
    assert out == "// hello\n"


def test_format_comment_line_html_block():
    out = _format_comment_line(
        comment="hello", indent="  ", prefix="<!--", newline="\n"
    )
    assert out == "  <!-- hello -->\n"


def test_format_comment_line_preserves_crlf():
    out = _format_comment_line(
        comment="hello", indent="", prefix="#", newline="\r\n"
    )
    assert out.endswith("\r\n")


# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------


def test_parse_bool_passthrough_bool():
    assert _parse_bool(True) is True
    assert _parse_bool(False) is False


def test_parse_bool_string_truthy_values():
    for val in ("1", "true", "TRUE", "yes", "y", "on"):
        assert _parse_bool(val) is True, val


def test_parse_bool_string_falsy_values():
    for val in ("0", "false", "FALSE", "no", "n", "off"):
        assert _parse_bool(val, default=True) is False, val


def test_parse_bool_unknown_string_returns_default():
    assert _parse_bool("maybe", default=True) is True
    assert _parse_bool("maybe", default=False) is False


def test_parse_bool_numeric():
    assert _parse_bool(1) is True
    assert _parse_bool(0) is False
    assert _parse_bool(2.5) is True


def test_parse_bool_none_returns_default():
    assert _parse_bool(None) is False
    assert _parse_bool(None, default=True) is True
