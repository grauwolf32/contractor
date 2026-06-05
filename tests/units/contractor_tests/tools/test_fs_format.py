import json

import pytest

from contractor.tools.fs.format import FileFormat
from contractor.tools.fs.models import FileLoc, FsEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def file_entry() -> FsEntry:
    return FsEntry(name="a.py", path="/src/a.py", size=42, is_dir=False)


@pytest.fixture()
def dir_entry() -> FsEntry:
    return FsEntry(name="src", path="/src", size=0, is_dir=True)


# ---------------------------------------------------------------------------
# format_fs_entry — JSON
# ---------------------------------------------------------------------------


def test_format_fs_entry_json_file(file_entry: FsEntry):
    fmt = FileFormat(_format="json", with_types=False)
    out = fmt.format_fs_entry(file_entry)
    assert isinstance(out, dict)
    assert out["kind"] == "file"
    assert out["name"] == "a.py"
    assert out["path"] == "/src/a.py"
    assert out["size"] == 42


def test_format_fs_entry_json_dir(dir_entry: FsEntry):
    fmt = FileFormat(_format="json", with_types=False)
    out = fmt.format_fs_entry(dir_entry)
    assert out["kind"] == "dir"
    assert out["size"] == 0


def test_format_fs_entry_json_with_loc(file_entry: FsEntry):
    file_entry.loc = FileLoc(line_start=3, line_end=5, content="hit")
    fmt = FileFormat(_format="json", with_types=False, loc="lines")
    out = fmt.format_fs_entry(file_entry)
    assert isinstance(out["loc"], dict)
    assert out["loc"]["line_start"] == 3
    assert out["loc"]["line_end"] == 5
    assert out["loc"]["content"] == "hit"


def test_format_fs_entry_json_loc_bytes_mode(file_entry: FsEntry):
    file_entry.loc = FileLoc(byte_start=10, byte_end=20)
    fmt = FileFormat(_format="json", with_types=False, loc="bytes")
    out = fmt.format_fs_entry(file_entry)
    assert "byte_start" in out["loc"]
    assert "line_start" not in out["loc"]


def test_format_fs_entry_omits_file_info_when_disabled(file_entry: FsEntry):
    file_entry.loc = FileLoc(line_start=1, line_end=1, content="x")
    fmt = FileFormat(_format="json", with_types=False, with_file_info=False)
    out = fmt.format_fs_entry(file_entry)
    assert "name" not in out
    assert "path" not in out
    assert "size" not in out
    assert "loc" in out  # loc still flows through


# ---------------------------------------------------------------------------
# format_fs_entry — XML
# ---------------------------------------------------------------------------


def test_format_fs_entry_xml_file(file_entry: FsEntry):
    fmt = FileFormat(_format="xml", with_types=False)
    out = fmt.format_fs_entry(file_entry)
    assert isinstance(out, str)
    assert out.startswith("<file>")
    assert out.endswith("</file>")
    assert "<name>a.py</name>" in out
    assert "<size>42</size>" in out


def test_format_fs_entry_xml_dir(dir_entry: FsEntry):
    fmt = FileFormat(_format="xml", with_types=False)
    out = fmt.format_fs_entry(dir_entry)
    assert out.startswith("<dir>")
    assert out.endswith("</dir>")


def test_format_fs_entry_xml_escapes_special_characters():
    entry = FsEntry(name="<weird>&", path="/x", size=0, is_dir=False)
    fmt = FileFormat(_format="xml", with_types=False)
    out = fmt.format_fs_entry(entry)
    assert "<name>&lt;weird&gt;&amp;</name>" in out


# ---------------------------------------------------------------------------
# format_fs_entry — str (json-encoded)
# ---------------------------------------------------------------------------


def test_format_fs_entry_str_emits_json_string(file_entry: FsEntry):
    fmt = FileFormat(_format="str", with_types=False)
    out = fmt.format_fs_entry(file_entry)
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["name"] == "a.py"


# ---------------------------------------------------------------------------
# format_file_list
# ---------------------------------------------------------------------------


def test_format_file_list_filters_none_entries(file_entry: FsEntry):
    fmt = FileFormat(_format="json", with_types=False)
    out = fmt.format_file_list([file_entry, None, file_entry])
    assert isinstance(out, list)
    assert len(out) == 2


def test_format_file_list_xml_wraps_in_files_root(
    file_entry: FsEntry, dir_entry: FsEntry
):
    fmt = FileFormat(_format="xml", with_types=False)
    out = fmt.format_file_list([file_entry, dir_entry])
    assert isinstance(out, str)
    assert out.startswith("<files>")
    assert out.endswith("</files>")
    assert "<file>" in out
    assert "<dir>" in out


def test_format_file_list_str_joins_with_newlines(
    file_entry: FsEntry, dir_entry: FsEntry
):
    fmt = FileFormat(_format="str", with_types=False)
    out = fmt.format_file_list([file_entry, dir_entry])
    assert isinstance(out, str)
    assert out.count("\n") == 1


# ---------------------------------------------------------------------------
# format_output (truncation)
# ---------------------------------------------------------------------------


def test_format_output_no_truncation_when_under_limit():
    text = "line1\nline2\nline3\n"
    out = FileFormat.format_output(text, max_output=1024)
    assert out == text


def test_format_output_truncates_when_over_limit_and_appends_footer():
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=80)

    assert "### truncated at line:" in out
    assert "lines left in the file:" in out
    # Body should be shorter than the original.
    assert len(out) <= len(text) + 200


def test_format_output_handles_max_smaller_than_footer():
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=20)
    # When the footer alone exceeds max_output we get a truncated banner.
    assert isinstance(out, str)
    assert len(out) <= 20


def test_format_output_empty_input_returns_empty():
    assert FileFormat.format_output("", max_output=100) == ""


# ---------------------------------------------------------------------------
# format_output — resume offset footer
# ---------------------------------------------------------------------------


def _parse_resume_offset(out: str) -> int:
    marker = "resume with read_file offset="
    assert marker in out
    tail = out.split(marker, 1)[1]
    return int(tail.split(" ", 1)[0].rstrip("#").strip())


def test_format_output_emits_resume_offset_when_base_offset_given():
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=250, base_offset=0)
    assert "### truncated at line:" in out  # existing info preserved
    assert "lines left in the file:" in out
    assert "resume with read_file offset=" in out


def test_format_output_resume_offset_round_trips_without_gap_or_overlap():
    # base_offset=0: the offset must point at the first dropped line so a
    # follow-up read_file(offset=...) continues seamlessly.
    lines = [f"line{i}\n" for i in range(50)]
    text = "".join(lines)
    out = FileFormat.format_output(text, max_output=250, base_offset=0)

    resume = _parse_resume_offset(out)
    body = out.split("\n\n###", 1)[0]
    emitted = body.count("\n")  # full lines kept (each ends with \n)
    assert resume == emitted  # no gap, no overlap with the emitted body
    # The line at `resume` is exactly the first one not shown.
    assert lines[resume] not in body
    assert lines[resume - 1] in body


def test_format_output_resume_offset_is_absolute_when_base_offset_nonzero():
    # Caller already skipped 100 lines; the resume offset must be absolute.
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=250, base_offset=100)
    assert _parse_resume_offset(out) >= 100


def test_format_output_omits_resume_offset_by_default():
    # Non-paginated callers (e.g. diff output) must not get a misleading offset.
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=80)
    assert "resume with read_file offset=" not in out
    assert "### truncated at line:" in out


def test_format_output_omits_resume_offset_when_no_line_fits():
    # A single line wider than the budget emits nothing; a resume offset would
    # equal the requested offset and loop, so it must be suppressed.
    text = "x" * 500 + "\n" + "more\n"
    out = FileFormat.format_output(text, max_output=50, base_offset=7)
    assert "resume with read_file offset=" not in out


# ---------------------------------------------------------------------------
# format_output — line cap (max_lines)
# ---------------------------------------------------------------------------


def test_format_output_line_cap_emits_footer_even_when_under_byte_budget():
    # Short-line file: 100 lines fit easily under the byte budget, but the line
    # cap binds — the footer (and resume offset) must still fire so the agent
    # learns the read was incomplete. Regression for silent line-cap truncation.
    text = "".join(f"l{i}\n" for i in range(100))
    out = FileFormat.format_output(text, max_output=1_000_000, base_offset=0, max_lines=30)
    assert "### truncated at line: 30 ###" in out
    assert "lines left in the file: 70 ###" in out
    assert "resume with read_file offset=30 ###" in out
    # Body holds exactly the first 30 lines.
    body = out.split("\n\n###", 1)[0]
    assert body.count("\n") == 30


def test_format_output_line_cap_under_count_returns_whole_content():
    text = "".join(f"l{i}\n" for i in range(10))
    out = FileFormat.format_output(text, max_output=1_000_000, base_offset=0, max_lines=50)
    assert out == text  # cap doesn't bind → no footer


def test_format_output_byte_cap_wins_when_more_restrictive_than_line_cap():
    # Byte budget cuts before the line cap; footer still consistent.
    text = "".join(f"line{i}\n" for i in range(50))
    out = FileFormat.format_output(text, max_output=80, base_offset=0, max_lines=40)
    assert "### truncated at line:" in out
    body = out.split("\n\n###", 1)[0]
    assert body.count("\n") < 40  # byte cap, not the 40-line cap, bound


def test_format_output_footer_counts_consistent_after_trim_pop():
    # When the footer itself forces lines to be popped, the 'truncated at line'
    # label, 'lines left' count and resume offset must all agree on the lines
    # actually emitted (regression: they used to read the pre-trim cut point).
    lines = [f"line{i}\n" for i in range(50)]
    out = FileFormat.format_output("".join(lines), max_output=250, base_offset=0)
    resume = _parse_resume_offset(out)
    body = out.split("\n\n###", 1)[0]
    emitted = body.count("\n")
    truncated_at = int(out.split("### truncated at line:", 1)[1].split("###", 1)[0].strip())
    left = int(out.split("lines left in the file:", 1)[1].split("###", 1)[0].strip())
    assert truncated_at == emitted == resume
    assert left == len(lines) - emitted
