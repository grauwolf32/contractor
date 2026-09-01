"""Canonical project-relative POSIX path handling."""

from __future__ import annotations

import re
import unicodedata

MAX_PROJECT_PATH_BYTES = 4096
MAX_PROJECT_PATH_COMPONENTS = 128
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


class ProjectPathError(ValueError):
    """A model/input path is not a canonical project-relative path."""


def normalize_project_path(value: str, *, allow_root: bool = True) -> str:
    if not isinstance(value, str):
        raise ProjectPathError("project path must be text")
    if value == "":
        if allow_root:
            return value
        raise ProjectPathError("project path must not be empty")
    if (
        value.startswith("/")
        or value.startswith("//")
        or "\\" in value
        or "\x00" in value
        or "://" in value
        or _WINDOWS_DRIVE.match(value)
    ):
        raise ProjectPathError("project path must be relative POSIX syntax")
    normalized = unicodedata.normalize("NFC", value)
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeError:
        raise ProjectPathError("project path is not valid UTF-8 text") from None
    if len(encoded) > MAX_PROJECT_PATH_BYTES:
        raise ProjectPathError("project path exceeds its byte bound")
    parts = normalized.split("/")
    if len(parts) > MAX_PROJECT_PATH_COMPONENTS:
        raise ProjectPathError("project path exceeds its depth bound")
    for part in parts:
        if part in {"", ".", ".."} or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in part
        ):
            raise ProjectPathError("project path contains an invalid component")
    return "/".join(parts)


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
