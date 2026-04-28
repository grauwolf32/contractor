import re
from pathlib import Path

import fsspec
import pytest

from contractor.tools.fs.models import (
    FileInteractionEntry,
    FsEntry,
    InteractionFilter,
    InteractionKind,
)


# ---------------------------------------------------------------------------
# FileInteractionEntry
# ---------------------------------------------------------------------------


def test_interaction_entry_records_read():
    entry = FileInteractionEntry(path="/x")
    entry.touch("read_file", interaction=InteractionKind.READ)

    assert entry.read_count == 1
    assert entry.match_count == 0
    assert entry.has_read
    assert not entry.has_match
    assert entry.has_any_interaction
    assert entry.operations == {"read_file": 1}


def test_interaction_entry_records_match():
    entry = FileInteractionEntry(path="/x")
    entry.touch("grep", interaction=InteractionKind.MATCH)

    assert entry.match_count == 1
    assert entry.read_count == 0
    assert entry.has_match


def test_interaction_entry_accumulates_operations():
    entry = FileInteractionEntry(path="/x")
    entry.touch("read_file", interaction=InteractionKind.READ)
    entry.touch("read_file", interaction=InteractionKind.READ)
    entry.touch("grep", interaction=InteractionKind.MATCH)

    assert entry.operations == {"read_file": 2, "grep": 1}
    assert entry.read_count == 2
    assert entry.match_count == 1


def test_matches_filter_any():
    entry = FileInteractionEntry(path="/x")
    assert not entry.matches_filter(InteractionFilter.ANY)
    entry.touch("read_file", interaction=InteractionKind.READ)
    assert entry.matches_filter(InteractionFilter.ANY)


def test_matches_filter_read_only():
    entry = FileInteractionEntry(path="/x")
    entry.touch("read_file", interaction=InteractionKind.READ)
    assert entry.matches_filter(InteractionFilter.READ_ONLY)
    entry.touch("grep", interaction=InteractionKind.MATCH)
    assert not entry.matches_filter(InteractionFilter.READ_ONLY)


def test_matches_filter_match_only():
    entry = FileInteractionEntry(path="/x")
    entry.touch("grep", interaction=InteractionKind.MATCH)
    assert entry.matches_filter(InteractionFilter.MATCH_ONLY)
    entry.touch("read_file", interaction=InteractionKind.READ)
    assert not entry.matches_filter(InteractionFilter.MATCH_ONLY)


def test_matches_filter_read_and_match():
    entry = FileInteractionEntry(path="/x")
    entry.touch("read_file", interaction=InteractionKind.READ)
    assert not entry.matches_filter(InteractionFilter.READ_AND_MATCH)
    entry.touch("grep", interaction=InteractionKind.MATCH)
    assert entry.matches_filter(InteractionFilter.READ_AND_MATCH)


# ---------------------------------------------------------------------------
# FsEntry._compute_line_starts / _char_to_line
# ---------------------------------------------------------------------------


def test_compute_line_starts_simple():
    starts = FsEntry._compute_line_starts("a\nbb\nccc\n")
    # Lines start at: 0 (a), 2 (bb), 5 (ccc), 9 (after final \n)
    assert starts == [0, 2, 5, 9]


def test_compute_line_starts_no_newlines():
    assert FsEntry._compute_line_starts("plain text") == [0]


def test_compute_line_starts_empty_string():
    assert FsEntry._compute_line_starts("") == [0]


def test_char_to_line_maps_positions():
    text = "abc\nde\nfgh"
    starts = FsEntry._compute_line_starts(text)

    assert FsEntry._char_to_line(starts, 0) == 0   # 'a' on line 0
    assert FsEntry._char_to_line(starts, 2) == 0   # 'c' on line 0
    assert FsEntry._char_to_line(starts, 4) == 1   # 'd' on line 1
    assert FsEntry._char_to_line(starts, 7) == 2   # 'f' on line 2


# ---------------------------------------------------------------------------
# FsEntry.from_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def local_fs() -> fsspec.AbstractFileSystem:
    return fsspec.filesystem("file")


def test_from_path_returns_none_for_missing(local_fs, tmp_path: Path):
    entry = FsEntry.from_path(str(tmp_path / "nope.txt"), local_fs, with_types=False)
    assert entry is None


def test_from_path_returns_dir_entry(local_fs, tmp_path: Path):
    (tmp_path / "sub").mkdir()
    entry = FsEntry.from_path(str(tmp_path / "sub"), local_fs, with_types=False)
    assert entry is not None
    assert entry.is_dir
    assert entry.size == 0


def test_from_path_returns_file_entry_with_size(local_fs, tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello", encoding="utf-8")
    entry = FsEntry.from_path(str(tmp_path / "f.txt"), local_fs, with_types=False)
    assert entry is not None
    assert not entry.is_dir
    assert entry.size == 5
    assert entry.name == "f.txt"


def test_from_path_skips_filetype_when_disabled(local_fs, tmp_path: Path):
    (tmp_path / "f.py").write_text("def f(): return 1\n", encoding="utf-8")
    entry = FsEntry.from_path(str(tmp_path / "f.py"), local_fs, with_types=False)
    assert entry is not None
    assert entry.filetype is None


# ---------------------------------------------------------------------------
# FsEntry.from_matches
# ---------------------------------------------------------------------------


def test_from_matches_returns_empty_when_no_matches(local_fs, tmp_path: Path):
    (tmp_path / "f.txt").write_text("nothing to see\n", encoding="utf-8")
    entries = FsEntry.from_matches(
        matches=[], file_path=str(tmp_path / "f.txt"), fs=local_fs, with_types=False
    )
    assert entries == []


def test_from_matches_returns_none_for_missing(local_fs, tmp_path: Path):
    entries = FsEntry.from_matches(
        matches=[re.match("x", "x")],  # type: ignore[arg-type]
        file_path=str(tmp_path / "nope.txt"),
        fs=local_fs,
        with_types=False,
    )
    assert entries is None


def test_from_matches_records_line_indexes(local_fs, tmp_path: Path):
    content = "alpha\nbeta\nERROR: boom\ngamma\n"
    (tmp_path / "log.txt").write_text(content, encoding="utf-8")

    matches = list(re.compile(r"ERROR").finditer(content))
    entries = FsEntry.from_matches(
        matches=matches,
        file_path=str(tmp_path / "log.txt"),
        fs=local_fs,
        content=content,
        with_types=False,
    )

    assert entries is not None
    assert len(entries) == 1
    loc = entries[0].loc
    assert loc is not None
    assert loc.line_start == 2
    assert loc.line_end == 2
    assert loc.content == "ERROR: boom"


def test_from_matches_includes_context_lines(local_fs, tmp_path: Path):
    content = "a\nb\nMATCH\nd\ne\n"
    (tmp_path / "f.txt").write_text(content, encoding="utf-8")

    matches = list(re.compile(r"MATCH").finditer(content))
    entries = FsEntry.from_matches(
        matches=matches,
        file_path=str(tmp_path / "f.txt"),
        fs=local_fs,
        content=content,
        with_types=False,
        context_lines=1,
    )

    assert entries is not None
    loc = entries[0].loc
    assert loc is not None
    assert loc.line_start == 1
    assert loc.line_end == 3
    assert loc.content == "b\nMATCH\nd"


def test_from_matches_truncates_long_excerpt(local_fs, tmp_path: Path):
    long_line = "x" * 1000
    content = f"prefix\n{long_line}\nsuffix\n"
    (tmp_path / "f.txt").write_text(content, encoding="utf-8")

    matches = list(re.compile(r"x{10}").finditer(content))
    entries = FsEntry.from_matches(
        matches=matches,
        file_path=str(tmp_path / "f.txt"),
        fs=local_fs,
        content=content,
        with_types=False,
        excerpt_max_chars=50,
    )

    assert entries is not None
    excerpt = entries[0].loc.content
    assert excerpt is not None
    assert len(excerpt) <= 51  # 50 chars + the ellipsis
    assert excerpt.endswith("…")


def test_from_matches_sorts_by_line(local_fs, tmp_path: Path):
    content = "x\nx\nx\nx\n"
    (tmp_path / "f.txt").write_text(content, encoding="utf-8")
    matches = list(re.compile(r"x").finditer(content))

    entries = FsEntry.from_matches(
        matches=matches,
        file_path=str(tmp_path / "f.txt"),
        fs=local_fs,
        content=content,
        with_types=False,
    )

    assert entries is not None
    line_starts = [e.loc.line_start for e in entries]
    assert line_starts == sorted(line_starts)
