from __future__ import annotations

import itertools
import unicodedata
from fnmatch import fnmatchcase
from functools import cache
from typing import Any

import pytest

from contractor_runtime.projectfs.paths import (
    MAX_PROJECT_PATH_BYTES,
    MAX_PROJECT_PATH_COMPONENTS,
    ProjectPathError,
    compile_project_glob,
    join_project_path,
    normalize_project_glob,
    normalize_project_path,
    parent_paths,
)


def test_project_paths_are_nfc_relative_and_have_stable_parents() -> None:
    assert normalize_project_path("") == ""
    assert normalize_project_path("cafe\u0301/api.py") == "café/api.py"
    assert join_project_path("backend", "src/api.py") == "backend/src/api.py"
    assert parent_paths("backend/src/api.py") == ("backend", "backend/src")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/etc/passwd",
        "../escape",
        "safe/../escape",
        "safe/./file",
        "safe//file",
        "safe\\file",
        "C:/windows",
        "file://source",
        "safe/\x00file",
        "safe/\x1ffile",
        "//server/share",
        "safe/\ud800",
    ],
)
def test_project_paths_reject_non_relative_or_ambiguous_syntax(value: str) -> None:
    with pytest.raises(ProjectPathError):
        normalize_project_path(value, allow_root=False)


@pytest.mark.parametrize("value", ["\u212a:docs", "\u212a:/x/y"])
def test_project_paths_and_globs_judge_a_drive_after_nfc(value: str) -> None:
    # NFC turns U+212A KELVIN SIGN into "K", so the normalized path would start
    # with the drive "K:", which Worker State and the Server reject (WSP-R13).
    with pytest.raises(ProjectPathError, match="Windows drive"):
        normalize_project_path(value, allow_root=False)
    with pytest.raises(ProjectPathError):
        normalize_project_glob(value)


def test_project_paths_enforce_depth_and_utf8_byte_bounds() -> None:
    with pytest.raises(ProjectPathError, match="depth"):
        normalize_project_path("/".join(["a"] * (MAX_PROJECT_PATH_COMPONENTS + 1)))
    with pytest.raises(ProjectPathError, match="byte"):
        normalize_project_path("x" * (MAX_PROJECT_PATH_BYTES + 1))


def test_project_globs_are_path_aware_and_double_star_matches_zero_segments() -> None:
    assert normalize_project_glob("src/**/test_*.py") == "src/**/test_*.py"
    glob = compile_project_glob("src/**/test_*.py")
    assert glob.pattern == "src/**/test_*.py"
    assert glob.matches("src/test_api.py")
    assert glob.matches("src/unit/test_api.py")
    assert not compile_project_glob("src/*.py").matches("src/unit/test_api.py")
    assert compile_project_glob("**/*.py").matches("root.py")
    for invalid in ("", "../*", "/**", "safe\\*", "a/***/b"):
        with pytest.raises(ProjectPathError):
            normalize_project_glob(invalid)
        with pytest.raises(ProjectPathError):
            compile_project_glob(invalid)
    with pytest.raises(ProjectPathError):
        glob.matches("../src/test_api.py")


def test_compiled_glob_matches_exactly_like_per_path_matching() -> None:
    components = ("**", "*", "src", "s?c", "[a-c]*", "[!s]*", "*.py", "café")
    patterns = [
        "/".join(parts)
        for length in range(1, 5)
        for parts in itertools.product(components, repeat=length)
    ]
    names = ("src", "a.py", ".b", unicodedata.normalize("NFD", "café"))
    paths = [
        "/".join(parts)
        for length in range(1, 5)
        for parts in itertools.product(names, repeat=length)
    ]
    checked = 0
    for index, pattern in enumerate(patterns):
        if not _valid_glob(pattern):
            continue
        glob = compile_project_glob(pattern)
        assert glob.pattern == normalize_project_glob(pattern)
        # Every pattern of up to three components meets every path; longer
        # ones meet a deterministic sample, which keeps the test fast.
        for path in paths if pattern.count("/") < 3 else paths[index % 13 :: 13]:
            assert glob.matches(path) is reference_glob_matches(path, pattern), (path, pattern)
            checked += 1
    assert checked > 250_000


@pytest.mark.parametrize(
    ("pattern", "components"),
    [
        ("/".join(["**"] * 128), 1),
        ("/".join(["**"] * 127 + ["zz"]), 2),
        ("a/**/**/b/**/**/**", 4),
        ("/".join(["**", "*a"] * 32), 64),
    ],
)
def test_double_star_runs_compile_once_and_bound_matching_work(
    pattern: str, components: int
) -> None:
    glob = compile_project_glob(pattern)
    # The normalized pattern, and so every cursor query digest, is unchanged.
    assert glob.pattern == normalize_project_glob(pattern)
    assert len(glob._components) == components
    assert max(len(closure) for closure in glob._closures) <= 2
    counting = _CountingClosures(glob._closures)
    object.__setattr__(glob, "_closures", counting)
    for depth in (20, 60, 120):
        middle = [f"d{index}a" for index in range(depth - 3)]
        for last in ("zz", "yy"):
            path = "/".join(["a", *middle, "b", last])
            counting.transitions = 0
            assert glob.matches(path) is reference_glob_matches(path, pattern)
            # A path component unions at most two indices for each live state.
            assert counting.transitions <= 2 + 2 * (components + 1) * depth


class _CountingClosures(tuple[frozenset[int], ...]):
    """A compiled closure table that counts the state indices matching unions."""

    transitions = 0

    def __getitem__(self, index: Any) -> Any:
        closure = super().__getitem__(index)
        self.transitions += len(closure)
        return closure


def _valid_glob(pattern: str) -> bool:
    try:
        normalize_project_glob(pattern)
    except ProjectPathError:
        return False
    return True


def reference_glob_matches(path: str, pattern: str) -> bool:
    """Per-path matching as the Runtime did before globs were compiled."""

    path_parts = normalize_project_path(path, allow_root=False).split("/")
    pattern_parts = normalize_project_glob(pattern).split("/")

    @cache
    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        component = pattern_parts[pattern_index]
        if component == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatchcase(path_parts[path_index], component)
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)
