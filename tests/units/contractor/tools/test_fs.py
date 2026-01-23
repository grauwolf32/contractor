from __future__ import annotations

import os
import pathlib
import re

import fsspec
import pytest

from contractor.tools.fs import file_tools, FileFormat


@pytest.fixture()
def tmpdir_path(tmp_path: pathlib.Path) -> pathlib.Path:
    # Create a small directory tree
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "data").mkdir()

    # Files
    (tmp_path / "src" / "a.py").write_text(
        "line0\n"
        "hello world\n"
        "ERROR: boom\n"
        "tail\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "b.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00\x01\x02")
    (tmp_path / "data" / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")  # fake-ish header
    (tmp_path / "README.md").write_text("# readme\nhello\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def fs() -> fsspec.AbstractFileSystem:
    # Local FS backend
    return fsspec.filesystem("file")


@pytest.fixture()
def tools_json(fs: fsspec.AbstractFileSystem, tmpdir_path: pathlib.Path):
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    return file_tools(
        fs=fs,
        fmt=fmt,
        max_output=80_000,
        ignored_patterns=None,
        with_types=False,  # keep tests deterministic + fast (no Magika dependency behavior)
        with_file_info=True,
    )


@pytest.fixture()
def tools_xml(fs: fsspec.AbstractFileSystem, tmpdir_path: pathlib.Path):
    fmt = FileFormat(_format="xml", loc="lines", with_types=False, with_file_info=True)
    return file_tools(
        fs=fs,
        fmt=fmt,
        max_output=80_000,
        ignored_patterns=None,
        with_types=False,
        with_file_info=True,
    )


def abs_path(p: pathlib.Path) -> str:
    return str(p.resolve())


def test_ls_lists_and_ignores_defaults(tools_json, tmpdir_path: pathlib.Path):
    # __pycache__ and *.pyc should be ignored by default patterns
    res = tools_json["ls"](abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    # JSON format => list[dict]
    assert isinstance(out, list)

    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "src" / "a.py") in paths
    assert abs_path(tmpdir_path / "src" / "b.txt") in paths

    # The __pycache__ directory might appear depending on backend ls results;
    # but .pyc file should be ignored.
    assert all(not e["path"].endswith(".pyc") for e in out)


def test_glob_respects_path_filter(tools_json, tmpdir_path: pathlib.Path):
    # Glob all *.py under tmpdir, then filter to src
    pattern = abs_path(tmpdir_path) + "/**/*.py"
    res = tools_json["glob"](pattern, path=abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    paths = [e["path"] for e in out]
    assert abs_path(tmpdir_path / "src" / "a.py") in paths
    assert all(p.startswith(abs_path(tmpdir_path / "src")) for p in paths)


def test_read_file_full_and_paginated(tools_json, tmpdir_path: pathlib.Path):
    f = abs_path(tmpdir_path / "src" / "a.py")

    full = tools_json["read_file"](f)
    assert "error" not in full
    assert "line0" in full["result"]
    assert "ERROR: boom" in full["result"]

    page = tools_json["read_file"](f, offset=1, limit=2)
    assert "error" not in page
    assert page["result"].splitlines() == ["hello world", "ERROR: boom"]


def test_read_file_errors(tools_json, tmpdir_path: pathlib.Path):
    missing = tools_json["read_file"](abs_path(tmpdir_path / "nope.txt"))
    assert "error" in missing

    # Directory => not a file
    not_a_file = tools_json["read_file"](abs_path(tmpdir_path / "src"))
    assert "error" in not_a_file


def test_read_file_truncation_footer(fs, tmpdir_path: pathlib.Path):
    # Force truncation with tiny max_output
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    tools = file_tools(fs=fs, fmt=fmt, max_output=25, with_types=False)

    f = abs_path(tmpdir_path / "src" / "a.py")
    res = tools["read_file"](f)
    assert "error" not in res
    assert "truncated" in res["result"]  # footer should be present


def test_grep_invalid_regex(tools_json):
    res = tools_json["grep"]("[unclosed", path=".")
    assert "error" in res
    assert "incorrect" in res["error"].lower()


def test_grep_on_single_file_returns_matches_with_loc(tools_json, tmpdir_path: pathlib.Path):
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


def test_grep_directory_walk_finds_across_files(tools_json, tmpdir_path: pathlib.Path):
    res = tools_json["grep"](r"hello", path=abs_path(tmpdir_path))
    assert "error" not in res
    out = res["result"]
    # should match README.md and src/a.py ("hello world")
    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "README.md") in paths
    assert abs_path(tmpdir_path / "src" / "a.py") in paths


def test_grep_respects_ignored_patterns(tools_json, tmpdir_path: pathlib.Path):
    # There is a .pyc file in __pycache__; searching for binary-ish bytes won't matter,
    # but ensure ignored file doesn't show up as a match path even if regex matches empty etc.
    res = tools_json["grep"](r".", path=abs_path(tmpdir_path / "src" / "__pycache__"))
    assert "error" not in res
    out = res["result"]
    # directory contains only ignored *.pyc; should return empty
    assert out == []


def test_xml_formatting(tools_xml, tmpdir_path: pathlib.Path):
    res = tools_xml["ls"](abs_path(tmpdir_path))
    assert "error" not in res
    out = res["result"]
    assert isinstance(out, str)
    assert out.startswith("<files>")
    assert "<file>" in out


def test_custom_ignored_patterns_override_additional(fs, tmpdir_path: pathlib.Path):
    # Add "*.txt" to ignore patterns and ensure b.txt disappears
    fmt = FileFormat(_format="json", loc="lines", with_types=False, with_file_info=True)
    tools = file_tools(
        fs=fs,
        fmt=fmt,
        ignored_patterns=["*.txt"],
        with_types=False,
    )
    res = tools["ls"](abs_path(tmpdir_path / "src"))
    assert "error" not in res
    out = res["result"]
    paths = {e["path"] for e in out}
    assert abs_path(tmpdir_path / "src" / "b.txt") not in paths
    assert abs_path(tmpdir_path / "src" / "a.py") in paths


def test_loc_byte_offsets_present_when_possible(tools_json, tmpdir_path: pathlib.Path):
    f = abs_path(tmpdir_path / "src" / "a.py")
    res = tools_json["grep"](r"hello", path=f)
    assert "error" not in res
    out = res["result"]
    assert len(out) == 1
    loc = out[0]["loc"]
    # In JSON/lines mode, bytes are not required, but your implementation includes both in loc object.
    # We check they exist and are ints if present.
    # (If you later decide to omit bytes entirely in lines mode, change this test accordingly.)
    assert isinstance(loc.get("line_start"), int)
    assert isinstance(loc.get("line_end"), int)
    # byte offsets might be absent in some weird cases; if present they should be ints
    if "byte_start" in loc:
        assert isinstance(loc["byte_start"], int)
    if "byte_end" in loc:
        assert isinstance(loc["byte_end"], int)
