from __future__ import annotations

import pytest

from contractor_runtime.projectfs.paths import (
    MAX_PROJECT_PATH_BYTES,
    MAX_PROJECT_PATH_COMPONENTS,
    ProjectPathError,
    join_project_path,
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


def test_project_paths_enforce_depth_and_utf8_byte_bounds() -> None:
    with pytest.raises(ProjectPathError, match="depth"):
        normalize_project_path("/".join(["a"] * (MAX_PROJECT_PATH_COMPONENTS + 1)))
    with pytest.raises(ProjectPathError, match="byte"):
        normalize_project_path("x" * (MAX_PROJECT_PATH_BYTES + 1))
