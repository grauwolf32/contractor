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
