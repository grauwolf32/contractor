"""Bounded deterministic observations for one Worker workspace invocation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from contractor_runtime.contracts import WorkspaceObservationSummary
from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path
from contractor_runtime.projectfs.storage import WorkspaceObservationMetadata

MAX_WORKSPACE_SCOPE_PATHS = 10_000
MAX_WORKSPACE_SCOPE_PATH_BYTES = 2 * 1024 * 1024
MAX_WORKSPACE_INTERACTIONS = 10_000
MAX_WORKSPACE_INTERACTION_PATH_BYTES = 2 * 1024 * 1024
MAX_LEAN_FILES_READ = 25
_MAX_UINT64 = 2**64 - 1
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


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


class WorkspaceObservationSource(Protocol):
    """Narrow content-free metadata surface supplied by a WorkspaceSession."""

    async def observation_metadata(self) -> WorkspaceObservationMetadata: ...


@dataclass(slots=True)
class _Interaction:
    path: str
    first_ordinal: int
    last_ordinal: int
    discovery_calls: int = 0
    read_calls: int = 0
    match_calls: int = 0
    mutation_calls: int = 0

    def wire(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "firstOrdinal": self.first_ordinal,
            "lastOrdinal": self.last_ordinal,
            "discoveryCalls": self.discovery_calls,
            "readCalls": self.read_calls,
            "matchCalls": self.match_calls,
            "mutationCalls": self.mutation_calls,
        }


@dataclass(slots=True)
class WorkspaceObservationReducer:
    """Accumulate bounded unique path interactions for one invocation."""

    workspace_digest: str
    scope_paths: tuple[str, ...]
    scope_complete: bool
    _interactions: dict[str, _Interaction] = field(default_factory=dict, repr=False)
    _interaction_path_bytes: int = field(default=2, repr=False)
    detail_complete: bool = True

    @classmethod
    def from_metadata(
        cls,
        metadata: WorkspaceObservationMetadata,
    ) -> WorkspaceObservationReducer:
        if (
            not isinstance(metadata, WorkspaceObservationMetadata)
            or not _DIGEST.fullmatch(metadata.digest)
            or not isinstance(metadata.managed_text_paths, tuple)
        ):
            raise ValueError("workspace observation metadata is invalid")
        normalized_paths: list[str] = []
        complete = True
        for raw_path in metadata.managed_text_paths:
            try:
                normalized_paths.append(_exact_path(raw_path))
            except ValueError:
                complete = False

        lexical_paths = sorted(set(normalized_paths))
        if normalized_paths != lexical_paths:
            complete = False

        admitted: list[str] = []
        encoded_bytes = 2
        for path in lexical_paths:
            addition = _json_string_bytes(path) + int(bool(admitted))
            if (
                len(admitted) >= MAX_WORKSPACE_SCOPE_PATHS
                or encoded_bytes + addition > MAX_WORKSPACE_SCOPE_PATH_BYTES
            ):
                complete = False
                break
            admitted.append(path)
            encoded_bytes += addition
        return cls(
            workspace_digest=metadata.digest,
            scope_paths=tuple(admitted),
            scope_complete=complete,
        )

    def record(self, observation: WorkspaceToolObservation, *, ordinal: int) -> None:
        if not isinstance(observation, WorkspaceToolObservation):
            raise TypeError("workspace tool observation is invalid")
        if type(ordinal) is not int or not 1 <= ordinal <= _MAX_UINT64:
            raise ValueError("workspace observation ordinal is invalid")
        categories = (
            ("discovery_calls", observation.discovered),
            ("read_calls", observation.read),
            ("match_calls", observation.matched),
            ("mutation_calls", observation.modified),
        )
        normalized: dict[str, set[str]] = {}
        for field_name, paths in categories:
            normalized[field_name] = {_exact_path(path) for path in paths}
        for path in sorted(set().union(*normalized.values())):
            interaction = self._interactions.get(path)
            if interaction is None:
                addition = _json_string_bytes(path) + int(bool(self._interactions))
                if (
                    len(self._interactions) >= MAX_WORKSPACE_INTERACTIONS
                    or self._interaction_path_bytes + addition
                    > MAX_WORKSPACE_INTERACTION_PATH_BYTES
                ):
                    self.detail_complete = False
                    continue
                interaction = _Interaction(
                    path=path,
                    first_ordinal=ordinal,
                    last_ordinal=ordinal,
                )
                self._interactions[path] = interaction
                self._interaction_path_bytes += addition
            interaction.last_ordinal = max(interaction.last_ordinal, ordinal)
            for field_name, paths in normalized.items():
                if path in paths:
                    current = getattr(interaction, field_name)
                    setattr(interaction, field_name, min(_MAX_UINT64, current + 1))

    def mark_incomplete(self) -> None:
        self.detail_complete = False

    def snapshot(self) -> dict[str, Any]:
        interactions = sorted(
            self._interactions.values(),
            key=lambda item: (item.first_ordinal, item.path),
        )
        return {
            "workspaceDigest": self.workspace_digest,
            "scopePaths": list(self.scope_paths),
            "scopeComplete": self.scope_complete,
            "interactions": [item.wire() for item in interactions],
            "detailComplete": self.detail_complete,
        }


def validate_workspace_observation(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Strict-copy one State workspace section without accepting arbitrary facts."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "workspaceDigest",
        "scopePaths",
        "scopeComplete",
        "interactions",
        "detailComplete",
    }:
        raise ValueError("Worker workspace observation fields are invalid")
    digest = value.get("workspaceDigest")
    scope_paths = value.get("scopePaths")
    scope_complete = value.get("scopeComplete")
    interactions = value.get("interactions")
    detail_complete = value.get("detailComplete")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise ValueError("Worker workspace observation digest is invalid")
    if (
        not isinstance(scope_paths, list)
        or len(scope_paths) > MAX_WORKSPACE_SCOPE_PATHS
        or type(scope_complete) is not bool
        or not isinstance(interactions, list)
        or len(interactions) > MAX_WORKSPACE_INTERACTIONS
        or type(detail_complete) is not bool
    ):
        raise ValueError("Worker workspace observation bounds are invalid")
    copied_scope = [_exact_path(path) for path in scope_paths]
    if copied_scope != sorted(set(copied_scope)) or _json_list_bytes(copied_scope) > (
        MAX_WORKSPACE_SCOPE_PATH_BYTES
    ):
        raise ValueError("Worker workspace observation scope is invalid")
    copied_interactions = [_validate_interaction(item) for item in interactions]
    interaction_order = [(item["firstOrdinal"], item["path"]) for item in copied_interactions]
    if interaction_order != sorted(interaction_order) or len(
        {item["path"] for item in copied_interactions}
    ) != len(copied_interactions):
        raise ValueError("Worker workspace interaction order is invalid")
    if _json_list_bytes([item["path"] for item in copied_interactions]) > (
        MAX_WORKSPACE_INTERACTION_PATH_BYTES
    ):
        raise ValueError("Worker workspace interaction paths exceed their bound")
    return {
        "workspaceDigest": digest,
        "scopePaths": copied_scope,
        "scopeComplete": scope_complete,
        "interactions": copied_interactions,
        "detailComplete": detail_complete,
    }


def lean_workspace_summary(
    value: Mapping[str, Any] | None,
) -> tuple[WorkspaceObservationSummary | None, bool]:
    """Project the fixed lean@1 workspace view and its truncation signal."""

    workspace = validate_workspace_observation(value)
    if workspace is None:
        return None, False
    interactions = workspace["interactions"]
    discovered = {item["path"] for item in interactions if item["discoveryCalls"] > 0}
    read = {item["path"] for item in interactions if item["readCalls"] > 0}
    matched = {item["path"] for item in interactions if item["matchCalls"] > 0}
    modified = {item["path"] for item in interactions if item["mutationCalls"] > 0}
    ordered_read = [item["path"] for item in interactions if item["readCalls"] > 0]
    files_read = ordered_read[:MAX_LEAN_FILES_READ]
    complete = workspace["scopeComplete"] and workspace["detailComplete"]
    unread_files = len(set(workspace["scopePaths"]) - read) if complete else None
    truncated = not complete or len(files_read) < len(read)
    return (
        WorkspaceObservationSummary(
            scopedFiles=len(workspace["scopePaths"]),
            scopeComplete=workspace["scopeComplete"],
            discoveredFiles=len(discovered),
            readFiles=len(read),
            matchedFiles=len(matched),
            modifiedFiles=len(modified),
            detailComplete=workspace["detailComplete"],
            unreadFiles=unread_files,
            filesRead=files_read,
            filesReadTruncated=len(files_read) < len(read),
        ),
        truncated,
    )


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


def _validate_interaction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "firstOrdinal",
        "lastOrdinal",
        "discoveryCalls",
        "readCalls",
        "matchCalls",
        "mutationCalls",
    }:
        raise ValueError("Worker workspace interaction fields are invalid")
    result = {"path": _exact_path(value.get("path"))}
    for field_name in (
        "firstOrdinal",
        "lastOrdinal",
        "discoveryCalls",
        "readCalls",
        "matchCalls",
        "mutationCalls",
    ):
        field_value = value.get(field_name)
        minimum = 1 if field_name in {"firstOrdinal", "lastOrdinal"} else 0
        if type(field_value) is not int or not minimum <= field_value <= _MAX_UINT64:
            raise ValueError("Worker workspace interaction counter is invalid")
        result[field_name] = field_value
    if result["lastOrdinal"] < result["firstOrdinal"] or not any(
        result[field_name] > 0
        for field_name in ("discoveryCalls", "readCalls", "matchCalls", "mutationCalls")
    ):
        raise ValueError("Worker workspace interaction is inconsistent")
    return result


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
    return _exact_path(document.get("path"))


def _argument_path(arguments: Mapping[str, Any], field_name: str) -> str:
    if not isinstance(arguments, Mapping):
        raise ValueError("tool observation arguments are invalid")
    return _exact_path(arguments.get(field_name))


def _exact_path(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("workspace observation path is invalid")
    try:
        normalized = normalize_project_path(value, allow_root=False)
    except ProjectPathError:
        raise ValueError("workspace observation path is invalid") from None
    if normalized != value:
        raise ValueError("workspace observation path is not normalized")
    return normalized


def _json_string_bytes(value: str) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _json_list_bytes(values: Sequence[str]) -> int:
    return len(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
