"""Canonical project-relative POSIX path handling."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import translate as fnmatch_translate

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


@dataclass(frozen=True, slots=True)
class ProjectGlob:
    """A normalized project glob, compiled once to match many paths.

    A ``**`` component matches zero or more whole path components; any other
    component matches exactly one component as ``fnmatch.fnmatchcase`` does.
    Callers that test many paths compile the glob once instead of normalizing
    and translating it again for every path.
    """

    pattern: str
    # One fnmatchcase matcher per component, or None for "**". A run of "**"
    # components means the same as one and is compiled as one, so no two
    # adjacent entries are None.
    _components: tuple[Callable[[str], object] | None, ...] = field(repr=False)
    # The component indices reachable from each index by letting a "**"
    # component match nothing; index len(_components) means a full match.
    # Without adjacent "**" entries each closure holds at most two indices,
    # so one path component costs O(len(_components)) and a path
    # O(path components x glob components).
    _closures: tuple[frozenset[int], ...] = field(repr=False)

    def matches(self, path: str) -> bool:
        """Return whether project ``path`` matches; ProjectPathError if it is invalid."""

        parts = normalize_project_path(path, allow_root=False).split("/")
        components = self._components
        closures = self._closures
        accepted = len(components)
        states = closures[0]
        for part in parts:
            following: set[int] = set()
            for index in states:
                if index == accepted:
                    continue
                component = components[index]
                if component is None:
                    following |= closures[index]
                elif component(part) is not None:
                    following |= closures[index + 1]
            if not following:
                return False
            states = frozenset(following)
        return accepted in states


def compile_project_glob(value: str) -> ProjectGlob:
    """Normalize ``value`` once, as normalize_project_glob does, and compile it."""

    pattern = normalize_project_glob(value)
    compiled: list[Callable[[str], object] | None] = []
    for part in pattern.split("/"):
        if part != "**":
            compiled.append(re.compile(fnmatch_translate(part)).match)
        elif not compiled or compiled[-1] is not None:
            compiled.append(None)
    components = tuple(compiled)
    closures: list[frozenset[int]] = []
    for index in range(len(components) + 1):
        reachable = [index]
        while reachable[-1] < len(components) and components[reachable[-1]] is None:
            reachable.append(reachable[-1] + 1)
        closures.append(frozenset(reachable))
    return ProjectGlob(pattern=pattern, _components=components, _closures=tuple(closures))
