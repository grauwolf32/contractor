"""Content-free workspace observations that model-visible tools report.

Workspace toolsets build one WorkspaceToolObservation per successful call; the
Worker's observation reducer (worker/observations.py) accumulates them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path


@dataclass(frozen=True, slots=True)
class WorkspaceToolObservation:
    """One content-free fact extracted from a successful model-visible call."""

    discovered: tuple[str, ...] = ()
    read: tuple[str, ...] = ()
    matched: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()


class ObservationExtractor(Protocol):
    """Reviewed opt-in interface implemented by model-visible tool objects."""

    def contractor_observation(
        self,
        tool_args: Mapping[str, Any],
        result: Any,
    ) -> WorkspaceToolObservation | None: ...


def filesystem_tool_observation(
    tool_name: str,
    tool_args: Mapping[str, Any],
    result: Any,
) -> WorkspaceToolObservation | None:
    del tool_args
    document = _mapping(result)
    if tool_name == "ls":
        return WorkspaceToolObservation(discovered=_file_entry_paths(document, "entries"))
    if tool_name == "glob":
        return WorkspaceToolObservation(discovered=_file_entry_paths(document, "matches"))
    if tool_name == "read_file":
        return WorkspaceToolObservation(read=(_result_path(document),))
    if tool_name == "grep":
        return WorkspaceToolObservation(matched=_item_paths(document, "matches"))
    return None


def edit_tool_observation(
    tool_name: str,
    tool_args: Mapping[str, Any],
    result: Any,
) -> WorkspaceToolObservation | None:
    document = _mapping(result)
    if document.get("changed") is not True:
        return WorkspaceToolObservation()
    if tool_name in {"write_file", "append_file", "rm", "insert_line", "edit", "replace_range"}:
        return WorkspaceToolObservation(modified=(_argument_path(tool_args, "path"),))
    if tool_name == "cp":
        return WorkspaceToolObservation(modified=(_argument_path(tool_args, "destination"),))
    if tool_name == "mv":
        return WorkspaceToolObservation(
            modified=(
                _argument_path(tool_args, "source"),
                _argument_path(tool_args, "destination"),
            )
        )
    if tool_name == "mkdir":
        return WorkspaceToolObservation()
    return None


def annotation_tool_observation(
    tool_args: Mapping[str, Any],
    result: Any,
) -> WorkspaceToolObservation:
    del tool_args
    document = _mapping(result)
    if document.get("changed") is not True:
        return WorkspaceToolObservation()
    return WorkspaceToolObservation(modified=(_result_path(document),))


def workspace_changes_observation(
    tool_name: str,
    tool_args: Mapping[str, Any],
    result: Any,
) -> WorkspaceToolObservation | None:
    del tool_args
    if tool_name != "changed_paths":
        return None
    return WorkspaceToolObservation(modified=_item_paths(_mapping(result), "changes"))


def exact_observation_path(value: Any) -> str:
    """Return ``value`` if it is already a normalized project path, else raise ValueError."""

    if not isinstance(value, str):
        raise ValueError("workspace observation path is invalid")
    try:
        normalized = normalize_project_path(value, allow_root=False)
    except ProjectPathError:
        raise ValueError("workspace observation path is invalid") from None
    if normalized != value:
        raise ValueError("workspace observation path is not normalized")
    return normalized


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tool observation result is invalid")
    return value


def _file_entry_paths(document: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    items = document.get(field_name)
    if not isinstance(items, list):
        raise ValueError("tool observation entries are invalid")
    paths: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
            raise ValueError("tool observation entry is invalid")
        if item["type"] == "file":
            paths.append(_result_path(item))
    return tuple(paths)


def _item_paths(document: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    items = document.get(field_name)
    if not isinstance(items, list):
        raise ValueError("tool observation items are invalid")
    return tuple(_result_path(_mapping(item)) for item in items)


def _result_path(document: Mapping[str, Any]) -> str:
    return exact_observation_path(document.get("path"))


def _argument_path(arguments: Mapping[str, Any], field_name: str) -> str:
    if not isinstance(arguments, Mapping):
        raise ValueError("tool observation arguments are invalid")
    try:
        return normalize_project_path(arguments.get(field_name), allow_root=False)
    except ProjectPathError:
        raise ValueError("workspace observation path is invalid") from None
