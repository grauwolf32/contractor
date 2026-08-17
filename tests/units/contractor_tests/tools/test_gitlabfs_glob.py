"""Regression tests for path-aware glob/grep matching in gitlabfs.

Pre-fix, all four match sites used stdlib ``fnmatch``, which ignores ``/``:
``fnmatch("README.md", "**/*")`` is False, so a bare grep over the default
``**/*`` silently dropped every top-level file. The in-memory path (raw
fnmatch) and the API fallback (normalized fnmatch) also disagreed. Both now use
the project's path-aware ``glob_to_regex`` on a normalized pattern.
"""
from __future__ import annotations

from contractor.tools.fs.gitlabfs import GitlabFileSystem, _GitlabApiFallback

TREE_PATHS = ["README.md", "Dockerfile", "src/app.py", "src/sub/util.py"]


def _fallback() -> _GitlabApiFallback:
    return object.__new__(_GitlabApiFallback)


def _fs_with_entries(paths) -> GitlabFileSystem:
    fs = object.__new__(GitlabFileSystem)
    fs._entries = dict.fromkeys(paths)
    return fs


def test_match_tree_double_star_includes_top_level_files():
    tree = [{"path": p} for p in TREE_PATHS]
    # The core regression: "**/*" must include Dockerfile and README.md.
    assert _fallback()._match_tree(tree, "**/*") == [
        "/Dockerfile",
        "/README.md",
        "/src/app.py",
        "/src/sub/util.py",
    ]


def test_match_tree_extension_glob_is_path_aware():
    tree = [{"path": p} for p in ["a.py", "src/b.py", "src/c.txt"]]
    assert _fallback()._match_tree(tree, "**/*.py") == ["/a.py", "/src/b.py"]


def test_match_tree_single_star_stays_within_a_segment():
    tree = [{"path": p} for p in ["a.py", "src/b.py"]]
    # "*.py" must match only the top-level file, not cross "/".
    assert _fallback()._match_tree(tree, "*.py") == ["/a.py"]


def test_in_memory_glob_matches_api_fallback():
    # The two code paths must agree (they used to diverge).
    fs = _fs_with_entries(TREE_PATHS)
    tree = [{"path": p} for p in TREE_PATHS]
    fb = _fallback()
    for pattern in ("**/*", "**/*.py", "*.md", "src/**/*.py"):
        assert fs._glob_in_memory(pattern) == fb._match_tree(tree, pattern), pattern


def test_in_memory_glob_double_star_includes_top_level():
    fs = _fs_with_entries(TREE_PATHS)
    assert "/Dockerfile" in fs._glob_in_memory("**/*")
    assert "/README.md" in fs._glob_in_memory("**/*")
