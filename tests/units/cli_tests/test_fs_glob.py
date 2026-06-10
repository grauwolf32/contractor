"""Regression tests for RootedLocalFileSystem.glob (bug H2).

Pre-fix, the non-recursive branch listed only the sandbox root (so
``glob('sub/*.py')`` returned nothing), and the recursive fallback matched only
the pattern's tail segment (so ``glob('sub/**/*.py')`` leaked top-level files).
The matcher is now path-aware: ``*``/``?``/``[...]`` stay within a single path
segment, and ``**`` spans any number of segments (including zero).
"""

import os

import pytest

from cli.fs import RootedLocalFileSystem


@pytest.fixture
def fs(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "sub", "deep"))
    for rel in ("top.py", "sub/b.py", "sub/note.txt", "sub/deep/c.py"):
        open(os.path.join(root, rel), "w").close()
    return RootedLocalFileSystem(root)


class TestGlobSubdirectories:
    def test_non_recursive_subdir_pattern(self, fs):
        # The core H2 regression: pre-fix this returned [].
        assert fs.glob("sub/*.py") == ["/sub/b.py"]

    def test_leading_slash_is_normalized(self, fs):
        assert fs.glob("/sub/*.py") == ["/sub/b.py"]

    def test_non_recursive_does_not_descend(self, fs):
        # '*' must not cross '/', so a single-level pattern ignores deeper files.
        assert fs.glob("sub/*.py") == ["/sub/b.py"]  # not /sub/deep/c.py

    def test_extension_filter_in_subdir(self, fs):
        assert fs.glob("sub/*.txt") == ["/sub/note.txt"]

    def test_nested_exact_dir(self, fs):
        assert fs.glob("sub/deep/*.py") == ["/sub/deep/c.py"]


class TestGlobRecursive:
    def test_top_level_star(self, fs):
        assert fs.glob("*.py") == ["/top.py"]

    def test_double_star_matches_all_depths(self, fs):
        assert fs.glob("**/*.py") == ["/sub/b.py", "/sub/deep/c.py", "/top.py"]

    def test_double_star_under_subdir_does_not_leak_top(self, fs):
        # Pre-fix this leaked '/top.py' (tail-only match) and missed '/sub/b.py'.
        assert fs.glob("sub/**/*.py") == ["/sub/b.py", "/sub/deep/c.py"]

    def test_trailing_double_star_matches_everything_under_dir(self, fs):
        assert fs.glob("sub/**") == [
            "/sub/b.py",
            "/sub/deep/c.py",
            "/sub/note.txt",
        ]


class TestGlobEdgeCases:
    def test_no_match_returns_empty(self, fs):
        assert fs.glob("nomatch/*.py") == []

    def test_empty_pattern_returns_empty(self, fs):
        assert fs.glob("") == []

    def test_traversal_pattern_rejected(self, fs):
        assert fs.glob("../*.py") == []
        assert fs.glob("sub/../*.py") == []

    def test_character_class(self, fs):
        # Character classes are honored and stay within the top-level segment.
        assert fs.glob("*.[pt]*") == ["/top.py"]


class TestGlobWalkCeiling:
    # The fixture tree has 4 files total.

    def test_truncates_when_ceiling_hit(self, fs):
        matches, truncated = fs.glob_scanned("**/*", max_files=2)
        assert truncated is True
        assert len(matches) <= 2

    def test_no_truncation_under_ceiling(self, fs):
        matches, truncated = fs.glob_scanned("**/*.py", max_files=100)
        assert truncated is False
        assert matches == ["/sub/b.py", "/sub/deep/c.py", "/top.py"]

    def test_default_ceiling_comes_from_settings(self, fs, monkeypatch):
        import cli.fs as cli_fs_module
        from contractor.utils.settings import Settings

        monkeypatch.setattr(
            cli_fs_module,
            "get_settings",
            lambda: Settings(fs_max_files_per_walk=1),
        )

        matches, truncated = fs.glob_scanned("**/*")
        assert truncated is True
        assert len(matches) <= 1
