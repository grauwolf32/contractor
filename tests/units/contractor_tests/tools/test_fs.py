from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import fsspec
import pytest

from contractor.tools.fs import (
    FileFormat,
    FsspecInteractionFileTools,
    InteractionFilter,
    InteractionKind,
    RootedLocalFileSystem,
    ro_file_tools,
    rw_file_tools,
)



@pytest.fixture()
def tmpdir_path(tmp_path: Path) -> Path:
    # Create a small directory tree
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "data").mkdir()

    # Files
    (tmp_path / "src" / "a.py").write_text(
        "line0\nhello world\nERROR: boom\ntail\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "b.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(
        b"\x00\x01\x02"
    )
    (tmp_path / "data" / "image.png").write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00"
    )  # fake-ish header
    (tmp_path / "README.md").write_text("# readme\nhello\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def fs() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("file")


@pytest.fixture()
def interaction_tools(
    fs: fsspec.AbstractFileSystem, tmpdir_path: Path
) -> FsspecInteractionFileTools:
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    return FsspecInteractionFileTools(
        fs=fs,
        fmt=fmt,
        max_output=80_000,
        max_items=300,
        ignored_patterns=None,
        with_types=False,
        with_file_info=True,
    )


def test_interaction_stats_empty_for_unread_tree(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    res = interaction_tools.interaction_stats(path=abs_path(tmpdir_path))
    assert "error" not in res

    stats = res["result"]
    assert stats["path"] == abs_path(tmpdir_path)
    assert stats["total_files"] == 3  # README.md, src/a.py, src/b.txt ; png/pyc ignored
    assert stats["touched_files_count"] == 0
    assert stats["untouched_files_count"] == 3
    assert stats["interaction_percent"] == 0.0


def test_read_file_marks_file_as_read(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "a.py")

    read_res = interaction_tools.read_file(file_path)
    assert "error" not in read_res

    touched_res = interaction_tools.touched_files(path=abs_path(tmpdir_path))
    assert "error" not in touched_res

    out = touched_res["result"]
    assert len(out) == 1
    assert out[0]["path"] == file_path
    assert out[0]["has_read"] is True
    assert out[0]["has_match"] is False
    assert out[0]["read_count"] == 1
    assert out[0]["match_count"] == 0
    assert out[0]["operations"]["read_file"] == 1


def test_grep_marks_file_as_match_only_when_match_found(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "a.py")

    grep_res = interaction_tools.grep(r"ERROR:\s+\w+", path=file_path)
    assert "error" not in grep_res
    assert len(grep_res["result"]) == 1

    touched_res = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.MATCH_ONLY,
    )
    assert "error" not in touched_res

    out = touched_res["result"]
    assert len(out) == 1
    assert out[0]["path"] == file_path
    assert out[0]["has_read"] is False
    assert out[0]["has_match"] is True
    assert out[0]["read_count"] == 0
    assert out[0]["match_count"] == 1
    assert out[0]["operations"]["grep"] == 1


def test_grep_without_matches_does_not_mark_interaction(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "b.txt")

    grep_res = interaction_tools.grep(r"does-not-exist", path=file_path)
    assert "error" not in grep_res
    assert grep_res["result"] == []

    touched_res = interaction_tools.touched_files(path=abs_path(tmpdir_path))
    assert "error" not in touched_res
    assert touched_res["result"] == []


def test_read_and_grep_same_file_moves_it_to_read_and_match(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "a.py")

    read_res = interaction_tools.read_file(file_path)
    assert "error" not in read_res

    grep_res = interaction_tools.grep(r"ERROR:\s+\w+", path=file_path)
    assert "error" not in grep_res

    both_res = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.READ_AND_MATCH,
    )
    assert "error" not in both_res

    out = both_res["result"]
    assert len(out) == 1
    assert out[0]["path"] == file_path
    assert out[0]["has_read"] is True
    assert out[0]["has_match"] is True
    assert out[0]["read_count"] == 1
    assert out[0]["match_count"] == 1
    assert out[0]["operations"]["read_file"] == 1
    assert out[0]["operations"]["grep"] == 1


def test_files_with_interactions_filters_split_files_by_interaction_kind(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    read_only_file = abs_path(tmpdir_path / "README.md")
    match_only_file = abs_path(tmpdir_path / "src" / "b.txt")
    both_file = abs_path(tmpdir_path / "src" / "a.py")

    assert "error" not in interaction_tools.read_file(read_only_file)
    assert "error" not in interaction_tools.grep(r"beta", path=match_only_file)
    assert "error" not in interaction_tools.read_file(both_file)
    assert "error" not in interaction_tools.grep(r"ERROR:\s+\w+", path=both_file)

    read_only = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.READ_ONLY,
    )
    match_only = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.MATCH_ONLY,
    )
    both = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.READ_AND_MATCH,
    )
    any_interaction = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction=InteractionFilter.ANY,
    )

    assert {x["path"] for x in read_only["result"]} == {read_only_file}
    assert {x["path"] for x in match_only["result"]} == {match_only_file}
    assert {x["path"] for x in both["result"]} == {both_file}
    assert {x["path"] for x in any_interaction["result"]} == {
        read_only_file,
        match_only_file,
        both_file,
    }


def test_untouched_files_returns_only_not_touched_files(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    touched_file = abs_path(tmpdir_path / "src" / "a.py")
    untouched_files = {
        abs_path(tmpdir_path / "README.md"),
        abs_path(tmpdir_path / "src" / "b.txt"),
    }

    assert "error" not in interaction_tools.read_file(touched_file)

    res = interaction_tools.untouched_files(path=abs_path(tmpdir_path))
    assert "error" not in res

    assert {x["path"] for x in res["result"]} == untouched_files


def test_interaction_stats_after_mixed_operations(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "README.md")
    )
    assert "error" not in interaction_tools.grep(
        r"ERROR:\s+\w+", path=abs_path(tmpdir_path / "src" / "a.py")
    )

    res = interaction_tools.interaction_stats(path=abs_path(tmpdir_path))
    assert "error" not in res

    stats = res["result"]
    assert stats["total_files"] == 3
    assert stats["touched_files_count"] == 2
    assert stats["untouched_files_count"] == 1
    assert stats["interaction_percent"] == round((2 / 3) * 100, 2)


def test_touched_and_untouched_support_pattern_filter(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "README.md")
    )
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "src" / "a.py")
    )

    res = interaction_tools.touched_files(
        path=abs_path(tmpdir_path),
        pattern="src/*",
    )
    assert "error" not in res
    assert {x["path"] for x in res["result"]} == {
        abs_path(tmpdir_path / "src" / "a.py")
    }

    res = interaction_tools.untouched_files(
        path=abs_path(tmpdir_path),
        pattern="src/*",
    )
    assert "error" not in res
    assert {x["path"] for x in res["result"]} == {
        abs_path(tmpdir_path / "src" / "b.txt")
    }


def test_touched_files_pagination(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "README.md")
    )
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "src" / "a.py")
    )
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "src" / "b.txt")
    )

    page1 = interaction_tools.touched_files(
        path=abs_path(tmpdir_path),
        offset=0,
        limit=2,
    )
    page2 = interaction_tools.touched_files(
        path=abs_path(tmpdir_path),
        offset=2,
        limit=2,
    )

    assert "error" not in page1
    assert "error" not in page2

    assert page1["offset"] == 0
    assert page1["limit"] == 2
    assert page1["total_items"] == 3
    assert len(page1["result"]) == 2

    assert page2["offset"] == 2
    assert page2["limit"] == 2
    assert page2["total_items"] == 3
    assert len(page2["result"]) == 1


def test_get_interactions_returns_raw_state(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "a.py")

    assert "error" not in interaction_tools.read_file(file_path)
    assert "error" not in interaction_tools.grep(r"ERROR:\s+\w+", path=file_path)

    res = interaction_tools.get_interactions()

    assert res["files_seen"] == 1
    assert file_path in res["files"]

    entry = res["files"][file_path]
    assert entry["read_count"] == 1
    assert entry["match_count"] == 1
    assert entry["has_read"] is True
    assert entry["has_match"] is True
    assert entry["operations"]["read_file"] == 1
    assert entry["operations"]["grep"] == 1


def test_reset_interactions_clears_state(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    assert "error" not in interaction_tools.read_file(
        abs_path(tmpdir_path / "src" / "a.py")
    )
    assert interaction_tools.get_interactions()["files_seen"] == 1

    interaction_tools.reset_interactions()

    res = interaction_tools.get_interactions()
    assert res["files_seen"] == 0
    assert res["files"] == {}


def test_record_interaction_manual(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "b.txt")

    interaction_tools.record_interaction(
        file_path,
        "custom_read",
        interaction=InteractionKind.READ,
    )
    interaction_tools.record_interaction(
        file_path,
        "custom_match",
        interaction=InteractionKind.MATCH,
    )
    interaction_tools.record_interaction(
        file_path,
        "custom_match",
        interaction=InteractionKind.MATCH,
    )

    res = interaction_tools.get_interactions()
    entry = res["files"][file_path]

    assert entry["read_count"] == 1
    assert entry["match_count"] == 2
    assert entry["has_read"] is True
    assert entry["has_match"] is True
    assert entry["operations"]["custom_read"] == 1
    assert entry["operations"]["custom_match"] == 2


def test_interaction_stats_missing_path_returns_error(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    res = interaction_tools.interaction_stats(path=abs_path(tmpdir_path / "missing"))
    assert "error" in res


def test_files_with_interactions_missing_path_returns_error(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    res = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path / "missing")
    )
    assert "error" in res


def test_untouched_files_missing_path_returns_error(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    res = interaction_tools.untouched_files(path=abs_path(tmpdir_path / "missing"))
    assert "error" in res


def test_files_with_interactions_accepts_string_filter(
    interaction_tools: FsspecInteractionFileTools, tmpdir_path: Path
):
    file_path = abs_path(tmpdir_path / "src" / "a.py")

    assert "error" not in interaction_tools.read_file(file_path)

    res = interaction_tools.files_with_interactions(
        path=abs_path(tmpdir_path),
        interaction="read_only",
    )
    assert "error" not in res
    assert {x["path"] for x in res["result"]} == {file_path}


@pytest.fixture()
def tools_json(fs: fsspec.AbstractFileSystem, tmpdir_path: Path):
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    tools = ro_file_tools(
        fs=fs,
        fmt=fmt,
        max_output=80_000,
        ignored_patterns=None,
        with_types=False,
        with_file_info=True,
    )
    return {fn.__name__: fn for fn in tools}


@pytest.fixture()
def tools_xml(fs: fsspec.AbstractFileSystem, tmpdir_path: Path):
    fmt = FileFormat(_format="xml", loc="lines", with_types=False, with_file_info=True)
    tools = ro_file_tools(
        fs=fs,
        fmt=fmt,
        max_output=80_000,
        ignored_patterns=None,
        with_types=False,
        with_file_info=True,
    )
    return {fn.__name__: fn for fn in tools}


def abs_path(p: Path) -> str:
    return str(p.resolve())


def test_ls_lists_and_ignores_defaults(tools_json, tmpdir_path: Path):
    res = tools_json["ls"](abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    assert isinstance(out, list)

    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "src" / "a.py") in paths
    assert abs_path(tmpdir_path / "src" / "b.txt") in paths
    assert all(not e["path"].endswith(".pyc") for e in out)


def test_glob_respects_path_filter(tools_json, tmpdir_path: Path):
    pattern = abs_path(tmpdir_path) + "/**/*.py"
    res = tools_json["glob"](pattern, path=abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    paths = [e["path"] for e in out]
    assert abs_path(tmpdir_path / "src" / "a.py") in paths
    assert all(p.startswith(abs_path(tmpdir_path / "src")) for p in paths)


def test_read_file_full_and_paginated(tools_json, tmpdir_path: Path):
    f = abs_path(tmpdir_path / "src" / "a.py")

    full = tools_json["read_file"](f)
    assert "error" not in full
    assert "line0" in full["result"]
    assert "ERROR: boom" in full["result"]

    page = tools_json["read_file"](f, offset=1, limit=2)
    assert "error" not in page
    assert page["result"].splitlines() == ["hello world", "ERROR: boom"]


def test_read_file_errors(tools_json, tmpdir_path: Path):
    missing = tools_json["read_file"](abs_path(tmpdir_path / "nope.txt"))
    assert "error" in missing

    not_a_file = tools_json["read_file"](abs_path(tmpdir_path / "src"))
    assert "error" in not_a_file


def test_read_file_truncation_footer(fs, tmpdir_path: Path):
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    tools = ro_file_tools(fs=fs, fmt=fmt, max_output=25, with_types=False)
    tools = {fn.__name__: fn for fn in tools}

    f = abs_path(tmpdir_path / "src" / "a.py")
    res = tools["read_file"](f)
    assert "error" not in res
    assert "truncated" in res["result"]


def test_grep_invalid_regex(tools_json):
    res = tools_json["grep"]("[unclosed", path=".")
    assert "error" in res
    assert "incorrect" in res["error"].lower()


def test_grep_on_single_file_returns_matches_with_loc(tools_json, tmpdir_path: Path):
    f = abs_path(tmpdir_path / "src" / "a.py")
    res = tools_json["grep"](r"ERROR:\s+\w+", path=f)
    assert "error" not in res
    out = res["result"]
    assert isinstance(out, list)
    assert len(out) == 1

    entry = out[0]
    assert entry["path"] == f
    assert "loc" in entry
    loc = entry["loc"]
    assert "line_start" in loc and "line_end" in loc
    assert loc["line_start"] <= loc["line_end"]
    assert "ERROR: boom" in loc.get("content", "")


def test_grep_directory_walk_finds_across_files(tools_json, tmpdir_path: Path):
    res = tools_json["grep"](r"hello", path=abs_path(tmpdir_path))
    assert "error" not in res
    out = res["result"]
    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "README.md") in paths
    assert abs_path(tmpdir_path / "src" / "a.py") in paths


def test_grep_respects_ignored_patterns(tools_json, tmpdir_path: Path):
    res = tools_json["grep"](r".", path=abs_path(tmpdir_path / "src" / "__pycache__"))
    assert "error" not in res
    out = res["result"]
    assert out == []


def test_xml_formatting(tools_xml, tmpdir_path: Path):
    res = tools_xml["ls"](abs_path(tmpdir_path))
    assert "error" not in res
    out = res["result"]
    assert isinstance(out, str)
    assert out.startswith("<files>")
    assert "<file>" in out


def test_custom_ignored_patterns_override_additional(fs, tmpdir_path: Path):
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    tools = ro_file_tools(
        fs=fs,
        fmt=fmt,
        ignored_patterns=["*.txt"],
        with_types=False,
    )
    tools = {fn.__name__: fn for fn in tools}
    res = tools["ls"](abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "src" / "b.txt") not in paths
    assert abs_path(tmpdir_path / "src" / "a.py") in paths


def test_loc_byte_offsets_present_when_possible(tools_json, tmpdir_path: Path):
    f = abs_path(tmpdir_path / "src" / "a.py")
    res = tools_json["grep"](r"hello", path=f)
    assert "error" not in res
    out = res["result"]
    assert len(out) == 1
    loc = out[0]["loc"]
    assert isinstance(loc.get("line_start"), int)
    assert isinstance(loc.get("line_end"), int)
    if "byte_start" in loc:
        assert isinstance(loc["byte_start"], int)
    if "byte_end" in loc:
        assert isinstance(loc["byte_end"], int)


@pytest.fixture
def fs_root_fixture(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    (root / "file.txt").write_text("hello")
    (root / "dir").mkdir()
    (root / "dir" / "inner.txt").write_text("inner")

    return root


@pytest.fixture
def fs_root(fs_root_fixture):
    return RootedLocalFileSystem(fs_root_fixture)


def test_root_ls(fs_root):
    items = fs_root.ls("/")
    names = {os.path.basename(p) for p in items}
    assert names == {"file.txt", "dir"}


def test_read_file(fs_root):
    with fs_root.open("/file.txt") as f:
        assert f.read().decode("utf-8") == "hello"


def test_walk(fs_root):
    files = []
    for _, _, fnames in fs_root.walk("/"):
        for f in fnames:
            files.append(f)

    assert "file.txt" in files
    assert "inner.txt" in files


def test_escape_parent_denied(fs_root):
    assert not fs_root.exists("/../outside.txt")


def test_escape_relative_denied(fs_root):
    assert not fs_root.exists("../../etc/passwd")


def test_symlink_escape_denied(fs_root, fs_root_fixture, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    os.symlink(outside, fs_root_fixture / "link")

    assert not fs_root.exists("/link")

    with pytest.raises(FileNotFoundError):
        fs_root.open("/link")


def test_absolute_host_path_denied(fs_root):
    assert not fs_root.exists("/etc/passwd")

    with pytest.raises(FileNotFoundError):
        fs_root.open("/etc/passwd")


def test_exists_inside(fs_root):
    assert fs_root.exists("/file.txt")
    assert not fs_root.exists("/nope.txt")


def test_glob_basic_inside_root(fs_root):
    paths = fs_root.glob("*.txt")
    names = {os.path.basename(p) for p in paths}

    assert names == {"file.txt"}


def test_glob_recursive_inside_root(fs_root):
    paths = fs_root.glob("**/*.txt")
    names = {os.path.basename(p) for p in paths}

    assert names == {"file.txt", "inner.txt"}


def test_glob_from_root_slash(fs_root):
    paths = fs_root.glob("/**/*.txt")
    names = {os.path.basename(p) for p in paths}

    assert "file.txt" in names
    assert "inner.txt" in names


def test_glob_does_not_escape_root(fs_root):
    paths = fs_root.glob("/etc/**/*.conf")
    assert paths == []


def test_glob_with_parent_traversal_is_empty(fs_root):
    paths = fs_root.glob("../**/*")
    assert paths == []


def test_glob_symlink_escape_denied(fs_root, fs_root_fixture, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    os.symlink(outside, fs_root_fixture / "link")

    paths = fs_root.glob("**/*")
    names = {os.path.basename(p) for p in paths}

    assert "link" not in names


def test_glob_returns_virtual_paths(fs_root):
    paths = fs_root.glob("**/*.txt")

    for p in paths:
        assert p.startswith("/")
        assert not p.startswith(fs_root.root_path)


def test_glob_empty_result(fs_root):
    paths = fs_root.glob("**/*.doesnotexist")
    assert paths == []


def test_glob_respects_ignored_patterns_with_ro_file_tools(fs_root):
    fmt = FileFormat(_format="json", with_types=False)
    tools = ro_file_tools(
        fs=fs_root,
        fmt=fmt,
        ignored_patterns=["*.txt"],
        with_types=False,
    )
    tools = {fn.__name__: fn for fn in tools}

    res = tools["glob"]("**/*")
    out = res["result"]
    paths = {e["path"] for e in out}

    assert all(not p.endswith(".txt") for p in paths)


@pytest.fixture()
def cyrillic_fs(tmp_path: Path):
    fname = "Новая заметка 2 - о работе.md"
    p = tmp_path / fname
    p.write_text("привет мир\nстрока 2", encoding="utf-8")
    return tmp_path, fname


def test_ls_cyrillic_name(tools_json, cyrillic_fs):
    root, fname = cyrillic_fs
    res = tools_json["ls"](str(root))
    paths = {e.get("name") for e in res["result"]}
    assert fname in paths


def test_read_file_cyrillic(tools_json, cyrillic_fs):
    root, fname = cyrillic_fs
    res = tools_json["read_file"](str(root / fname))
    assert "привет мир" in res["result"]


def test_glob_cyrillic(tools_json, cyrillic_fs):
    root, fname = cyrillic_fs
    res = tools_json["glob"](str(root / "*.md"))
    paths = {e["name"] for e in res["result"]}
    assert fname in paths


def test_grep_cyrillic(tools_json, cyrillic_fs):
    root, fname = cyrillic_fs
    res = tools_json["grep"]("привет", path=str(root))
    paths = {e["name"] for e in res["result"]}
    assert fname in paths


def test_unicode_normalization_glob(tools_json, tmp_path):
    name_nfc = "о"
    name_nfd = unicodedata.normalize("NFD", name_nfc)

    fname = f"{name_nfc}.txt"
    (tmp_path / fname).write_text("ok", encoding="utf-8")

    pattern = f"{name_nfd}.txt"
    res = tools_json["glob"](str(tmp_path / pattern))

    assert len(res["result"]) == 1


@pytest.fixture()
def write_tmpdir(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    print('hi')\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def write_fs() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("file")


@pytest.fixture()
def write_tool_map(write_fs: fsspec.AbstractFileSystem):
    tools = rw_file_tools(
        fs=write_fs,
        ignored_patterns=None,
        wrap_overlay=True,
    )
    return {fn.__name__: fn for fn in tools}

def test_write_file_creates_new_file(write_tool_map, write_tmpdir: Path):
    target = abs_path(write_tmpdir / "new.txt")

    res = write_tool_map["write_file"](target, "hello\nworld\n")
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "write_file"
    assert res["result"]["path"] == target
    assert res["result"]["size"] == len("hello\nworld\n".encode("utf-8"))


def test_write_file_replaces_existing_content(write_tool_map, write_tmpdir: Path):
    target = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["write_file"](target, "replaced\n")
    assert "error" not in res
    assert res["result"]["ok"] is True


def test_append_file_appends_content(write_tool_map, write_tmpdir: Path):
    target = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["append_file"](target, "delta\n")
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "append_file"


def test_mkdir_creates_directory(write_tool_map, write_tmpdir: Path):
    target = abs_path(write_tmpdir / "nested" / "dir")

    res = write_tool_map["mkdir"](target)
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "mkdir"
    assert res["result"]["path"] == target


def test_rm_deletes_file(write_tool_map, write_tmpdir: Path):
    target = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["rm"](target)
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "rm"
    assert res["result"]["path"] == target
    assert res["result"]["recursive"] is False


def test_cp_copies_file(write_tool_map, write_tmpdir: Path):
    src = abs_path(write_tmpdir / "notes.txt")
    dst = abs_path(write_tmpdir / "notes-copy.txt")

    res = write_tool_map["cp"](src, dst)
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "cp"
    assert res["result"]["src"] == src
    assert res["result"]["dst"] == dst


def test_mv_moves_file(write_tool_map, write_tmpdir: Path):
    src = abs_path(write_tmpdir / "notes.txt")
    dst = abs_path(write_tmpdir / "notes-moved.txt")

    res = write_tool_map["mv"](src, dst)
    assert "error" not in res
    assert res["result"]["ok"] is True
    assert res["result"]["op"] == "mv"
    assert res["result"]["src"] == src
    assert res["result"]["dst"] == dst


def test_insert_line_before_anchor(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "src" / "main.py")

    res = write_tool_map["insert_line"](
        path=path,
        content="say hello",
        anchor="print('hi')",
        where="before",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["op"] == "insert_line"
    assert result["path"] == path
    assert result["anchor"] == "print('hi')"
    assert result["where"] == "before"
    assert result["occurrence"] == 1
    assert result["insert_line"] == "say hello"


def test_insert_line_after_anchor(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "src" / "main.py")

    res = write_tool_map["insert_line"](
        path=path,
        content="done",
        anchor="print('hi')",
        where="after",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["insert_line"] == "done"


def test_insert_line_is_noop_when_same_line_already_adjacent(
    write_tool_map, write_tmpdir: Path
):
    path = abs_path(write_tmpdir / "src" / "main.py")

    first = write_tool_map["insert_line"](
        path=path,
        content="say hello",
        anchor="print('hi')",
        where="before",
    )
    assert "error" not in first

    second = write_tool_map["insert_line"](
        path=path,
        content="say hello",
        anchor="print('hi')",
        where="before",
    )
    assert "error" not in second
    result = second["result"]
    assert result["ok"] is True
    assert result["changed"] is False
    assert result["reason"] == "already present"


def test_insert_line_missing_anchor_returns_error(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "src" / "main.py")

    res = write_tool_map["insert_line"](
        path=path,
        content="x",
        anchor="does not exist",
    )
    assert "error" in res
    assert res["path"] == path
    assert res["anchor"] == "does not exist"


def test_replace_range_replaces_single_line(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=2,
        end_line=2,
        content="BETA",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["op"] == "replace_range"
    assert result["path"] == path
    assert result["start_line"] == 2
    assert result["end_line"] == 2
    assert result["new_start_line"] == 2
    assert result["new_end_line"] == 2
    assert result["removed_line_count"] == 1
    assert result["inserted_line_count"] == 1


def test_replace_range_insert_before_line(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=2,
        end_line=1,
        content="inserted",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["start_line"] == 2
    assert result["end_line"] == 1
    assert result["inserted_line_count"] == 1
    assert result["removed_line_count"] == 0


def test_replace_range_append_at_eof(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=4,
        end_line=3,
        content="delta",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["inserted_line_count"] == 1
    assert result["removed_line_count"] == 0


def test_replace_range_delete_lines(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=2,
        end_line=3,
        content="",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["removed_line_count"] == 2
    assert result["inserted_line_count"] == 0
    assert result["new_end_line"] == 1


def test_replace_range_noop_when_content_matches(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=2,
        end_line=2,
        content="beta",
    )
    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is False
    assert result["reason"] == "range already matches requested content"


def test_replace_range_rejects_bad_range(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=3,
        end_line=1,
        content="x",
    )
    assert "error" in res
    assert "invalid range" in res["error"]


def test_replace_range_rejects_missing_file(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "missing.txt")

    res = write_tool_map["replace_range"](
        path=path,
        start_line=1,
        end_line=1,
        content="x",
    )
    assert "error" in res


def test_rw_file_tools_respect_ignored_patterns(
    write_fs: fsspec.AbstractFileSystem, write_tmpdir: Path
):
    tools = rw_file_tools(
        fs=write_fs,
        ignored_patterns=["*.txt"],
        wrap_overlay=True,
    )
    tool_map = {fn.__name__: fn for fn in tools}

    res = tool_map["write_file"](abs_path(write_tmpdir / "notes.txt"), "blocked")
    assert "error" in res
    assert "ignored" in res["error"]


def test_rm_missing_path_returns_error(write_tool_map, write_tmpdir: Path):
    res = write_tool_map["rm"](abs_path(write_tmpdir / "missing.txt"))
    assert "error" in res


def test_cp_missing_source_returns_error(write_tool_map, write_tmpdir: Path):
    res = write_tool_map["cp"](
        abs_path(write_tmpdir / "missing.txt"),
        abs_path(write_tmpdir / "copy.txt"),
    )
    assert "error" in res


def test_mv_missing_source_returns_error(write_tool_map, write_tmpdir: Path):
    res = write_tool_map["mv"](
        abs_path(write_tmpdir / "missing.txt"),
        abs_path(write_tmpdir / "moved.txt"),
    )
    assert "error" in res


def test_edit_replaces_single_occurrence(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "src" / "main.py")

    res = write_tool_map["edit"](
        path=path,
        old_string="print('hi')",
        new_string="print('hello')",
    )

    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["created"] is False
    assert result["op"] == "edit"
    assert result["path"] == path
    assert result["occurrences"] == 1
    assert result["replaced_occurrences"] == 1


def test_edit_rejects_multiple_occurrences_without_replace_all(
    write_tool_map, write_tmpdir: Path
):
    path = abs_path(write_tmpdir / "dups.txt")
    (write_tmpdir / "dups.txt").write_text("x\nx\nx\n", encoding="utf-8")

    res = write_tool_map["edit"](
        path=path,
        old_string="x",
        new_string="y",
    )

    assert "error" in res
    assert "multiple locations" in res["error"]
    assert res["occurrences"] == 3


def test_edit_replace_all(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "dups.txt")
    (write_tmpdir / "dups.txt").write_text("x\nx\nx\n", encoding="utf-8")

    res = write_tool_map["edit"](
        path=path,
        old_string="x",
        new_string="y",
        replace_all=True,
    )

    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["replaced_occurrences"] == 3


def test_edit_creates_new_file_when_old_string_empty(
    write_tool_map, write_tmpdir: Path
):
    path = abs_path(write_tmpdir / "created.txt")

    res = write_tool_map["edit"](
        path=path,
        old_string="",
        new_string="hello\n",
    )

    assert "error" not in res
    result = res["result"]
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["created"] is True
    assert result["op"] == "edit"


def test_edit_fails_when_old_string_not_found(write_tool_map, write_tmpdir: Path):
    path = abs_path(write_tmpdir / "notes.txt")

    res = write_tool_map["edit"](
        path=path,
        old_string="does-not-exist",
        new_string="beta",
    )

    assert "error" in res
    assert "could not find the string" in res["error"]
