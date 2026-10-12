"""Canonical project-relative POSIX path handling."""

from __future__ import annotations

import re
import unicodedata
from fnmatch import fnmatchcase
from functools import cache

from contractor_runtime.contracts.base import (
    MAX_WORKER_STATE_PATH_BYTES,
    MAX_WORKER_STATE_PATH_COMPONENTS,
    relative_path_problem,
)

# Project paths are what the Worker State records (WSP-R13), so they share its
# bounds and the one relative path rule.
MAX_PROJECT_PATH_BYTES = MAX_WORKER_STATE_PATH_BYTES
MAX_PROJECT_PATH_COMPONENTS = MAX_WORKER_STATE_PATH_COMPONENTS
MAX_PROJECT_GLOB_BYTES = 1024
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class ProjectPathError(ValueError):
    """A model/input path is not a canonical project-relative path."""


def normalize_project_path(value: str, *, allow_root: bool = True) -> str:
    """Return the NFC form of a project-relative path, or raise ProjectPathError.

    The rule applies to the normalized form: NFC turns some characters into ASCII
    ones (U+212A KELVIN SIGN becomes "K"), so a drive or other syntax is judged
    on the path that is used and recorded.
    """

    if not isinstance(value, str):
        raise ProjectPathError("project path must be text")
    if value == "":
        if allow_root:
            return value
        raise ProjectPathError("project path must not be empty")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ProjectPathError("project path is not valid UTF-8 text") from None
    normalized = unicodedata.normalize("NFC", value)
    problem = relative_path_problem(
        normalized,
        max_bytes=MAX_PROJECT_PATH_BYTES,
        max_components=MAX_PROJECT_PATH_COMPONENTS,
    )
    if problem is not None:
        raise ProjectPathError(f"project path {problem}")
    return normalized


def join_project_path(prefix: str, child: str) -> str:
    normalized_prefix = normalize_project_path(prefix, allow_root=True)
    normalized_child = normalize_project_path(child, allow_root=False)
    return (
        normalized_child if normalized_prefix == "" else f"{normalized_prefix}/{normalized_child}"
    )


def parent_paths(path: str) -> tuple[str, ...]:
    normalized = normalize_project_path(path, allow_root=False)
    parts = normalized.split("/")
    return tuple("/".join(parts[:index]) for index in range(1, len(parts)))


def normalize_project_glob(value: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ProjectPathError("project glob must not be empty")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ProjectPathError("project glob is not valid UTF-8 text") from None
    # Like a path, a glob is judged in its NFC form (normalize_project_path).
    normalized = unicodedata.normalize("NFC", value)
    if (
        normalized.startswith("/")
        or "\\" in normalized
        or "\x00" in normalized
        or "://" in normalized
        or _WINDOWS_DRIVE.match(normalized)
    ):
        raise ProjectPathError("project glob must be relative POSIX syntax")
    encoded = normalized.encode("utf-8")
    parts = normalized.split("/")
    if len(encoded) > MAX_PROJECT_GLOB_BYTES or len(parts) > MAX_PROJECT_PATH_COMPONENTS:
        raise ProjectPathError("project glob exceeds its bound")
    for part in parts:
        if (
            part in {"", ".", ".."}
            or ("**" in part and part != "**")
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in part)
        ):
            raise ProjectPathError("project glob contains an invalid component")
    return "/".join(parts)


def project_glob_matches(path: str, pattern: str) -> bool:
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
