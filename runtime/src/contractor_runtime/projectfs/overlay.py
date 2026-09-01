"""Cumulative managed-text overlay state and invocation checkpoints."""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import jcs

from contractor_runtime.projectfs.paths import normalize_project_path, parent_paths
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage
from contractor_runtime.projectfs.storage import (
    DirectWorkspaceSession,
    ManagedWorkspaceTree,
    WorkspaceDiff,
    WorkspaceSnapshot,
    WorkspaceStorageError,
    workspace_digest,
)
from contractor_runtime.settings import WorkspaceLimits

WORKSPACE_OVERLAY_API_VERSION = "contractor.workspace/v1"
WORKSPACE_OVERLAY_KIND = "WorkspaceOverlay"
WORKSPACE_OVERLAY_MEDIA_TYPE = "application/vnd.contractor.workspace-overlay+json"
MAX_DIFF_BYTES = 1 << 20


class WorkspaceStateError(ValueError):
    """A cumulative state artifact is malformed or inconsistent with its source."""


@dataclass(frozen=True, slots=True)
class OverlayOperation:
    op: Literal["create_directory", "write_file", "delete_path"]
    path: str
    text: str | None = None

    def document(self) -> dict[str, str]:
        result = {"op": self.op, "path": self.path}
        if self.op == "write_file":
            assert self.text is not None
            result["text"] = self.text
        return result


class OverlayWorkspaceSession(DirectWorkspaceSession):
    """An allocation-private copy-on-write managed-text view over hydrated input."""

    def __init__(
        self,
        *,
        storage: ProjectWorkspaceStorage,
        content_root: str,
        limits: WorkspaceLimits,
        directories: set[str],
        text_files: dict[str, str],
        binary_paths: set[str],
        stored_binary_paths: set[str],
    ) -> None:
        super().__init__(
            mode="overlay",
            storage=storage,
            content_root=content_root,
            limits=limits,
            directories=directories,
            text_files=text_files,
            binary_paths=binary_paths,
            stored_binary_paths=stored_binary_paths,
        )
        self._source = self._tree.clone()
        self._checkpoint = self._tree.clone()

    async def write_text(self, path: str, text: str) -> None:
        normalized = normalize_project_path(path, allow_root=False)
        _validate_text(text, self._limits)
        async with self._lock:
            self._require_open()
            candidate = self._tree.clone()
            kind = candidate.kind(normalized)
            if kind == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if kind == "directory":
                raise WorkspaceStorageError("workspace_type_conflict")
            _require_parent_directory(candidate, normalized)
            candidate.text_files[normalized] = text
            _validate_tree(candidate, self._limits)
            self._tree = candidate

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        normalized = normalize_project_path(path, allow_root=False)
        async with self._lock:
            self._require_open()
            candidate = self._tree.clone()
            existing = candidate.kind(normalized)
            if existing == "directory":
                return
            if existing is not None:
                raise WorkspaceStorageError("workspace_type_conflict")
            missing = [
                parent for parent in parent_paths(normalized) if candidate.kind(parent) is None
            ]
            if missing and not parents:
                raise WorkspaceStorageError("workspace_not_found")
            for parent in parent_paths(normalized):
                kind = candidate.kind(parent)
                if kind not in {None, "directory"}:
                    raise WorkspaceStorageError("workspace_type_conflict")
                candidate.directories.add(parent)
            candidate.directories.add(normalized)
            _validate_tree(candidate, self._limits)
            self._tree = candidate

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        normalized = normalize_project_path(path, allow_root=False)
        async with self._lock:
            self._require_open()
            if self._tree.kind(normalized) is None:
                raise WorkspaceStorageError("workspace_not_found")
            descendants = _descendants(self._tree, normalized)
            selected = descendants | {normalized}
            if any(candidate in self._tree.binary_paths for candidate in selected):
                raise WorkspaceStorageError("binary_file_unsupported")
            if descendants and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            candidate = self._tree.clone()
            _remove_subtree(candidate, normalized)
            _validate_tree(candidate, self._limits)
            self._tree = candidate

    def _commit_candidate(self, candidate: ManagedWorkspaceTree) -> None:
        _validate_tree(candidate, self._limits)
        self._tree = candidate

    async def import_state(self, payload: bytes) -> None:
        async with self._lock:
            self._require_open()
            candidate = decode_workspace_state(payload, self._source, self._limits)
            self._tree = candidate
            self._checkpoint = candidate.clone()

    async def export_state(self) -> bytes:
        async with self._lock:
            self._require_open()
            return encode_workspace_state(self._source, self._tree)

    async def changed_paths(self, path: str = "") -> tuple[str, ...]:
        normalized = normalize_project_path(path, allow_root=True)
        async with self._lock:
            self._require_open()
            return tuple(
                candidate
                for candidate in sorted(self._checkpoint.paths() | self._tree.paths())
                if _within(candidate, normalized)
                and _path_value(self._checkpoint, candidate) != _path_value(self._tree, candidate)
            )

    async def diff(self, path: str = "", *, max_bytes: int = 65536) -> WorkspaceDiff:
        normalized = normalize_project_path(path, allow_root=True)
        if max_bytes <= 0 or max_bytes > MAX_DIFF_BYTES:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        async with self._lock:
            self._require_open()
            return _workspace_diff(self._checkpoint, self._tree, normalized, max_bytes)

    async def rollback_changes(self, path: str = "") -> None:
        normalized = normalize_project_path(path, allow_root=True)
        async with self._lock:
            self._require_open()
            if normalized == "":
                self._tree = self._checkpoint.clone()
                return
            if self._tree.kind(normalized) is None and self._checkpoint.kind(normalized) is None:
                raise WorkspaceStorageError("workspace_not_found")
            candidate = self._tree.clone()
            _remove_subtree(candidate, normalized)
            _copy_subtree(self._checkpoint, candidate, normalized)
            _validate_tree(candidate, self._limits)
            self._tree = candidate

    async def commit_checkpoint(self) -> WorkspaceSnapshot:
        async with self._lock:
            self._require_open()
            self._checkpoint = self._tree.clone()
            return self._checkpoint.snapshot()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            for tree in (self._tree, self._source, self._checkpoint):
                tree.directories.clear()
                tree.text_files.clear()
                tree.binary_paths.clear()
                tree.stored_binary_paths.clear()


def encode_workspace_state(
    source: ManagedWorkspaceTree,
    result: ManagedWorkspaceTree,
) -> bytes:
    operations = canonical_overlay_operations(source, result)
    document = {
        "apiVersion": WORKSPACE_OVERLAY_API_VERSION,
        "kind": WORKSPACE_OVERLAY_KIND,
        "baseWorkspaceDigest": workspace_digest(source.directories, source.text_files),
        "resultWorkspaceDigest": workspace_digest(result.directories, result.text_files),
        "operations": [operation.document() for operation in operations],
    }
    return jcs.canonicalize(document)


def decode_workspace_state(
    payload: bytes,
    source: ManagedWorkspaceTree,
    limits: WorkspaceLimits,
) -> ManagedWorkspaceTree:
    maximum = min(
        16 * 1024 * 1024, max(1 << 20, limits.max_managed_text_bytes + 512 * limits.max_files)
    )
    if not isinstance(payload, bytes) or len(payload) > maximum:
        raise WorkspaceStateError("workspace_state_invalid")
    document = _strict_json_document(payload)
    if set(document) != {
        "apiVersion",
        "kind",
        "baseWorkspaceDigest",
        "resultWorkspaceDigest",
        "operations",
    }:
        raise WorkspaceStateError("workspace_state_invalid")
    if (
        document["apiVersion"] != WORKSPACE_OVERLAY_API_VERSION
        or document["kind"] != WORKSPACE_OVERLAY_KIND
        or document["baseWorkspaceDigest"]
        != workspace_digest(source.directories, source.text_files)
        or not isinstance(document["resultWorkspaceDigest"], str)
        or not isinstance(document["operations"], list)
        or len(document["operations"]) > limits.max_files * 3
    ):
        raise WorkspaceStateError("workspace_state_invalid")
    try:
        operations = tuple(_parse_operation(item) for item in document["operations"])
        result = source.clone()
        for operation in operations:
            _apply_operation(result, operation)
            _validate_tree(result, limits)
    except WorkspaceStorageError:
        raise WorkspaceStateError("workspace_state_invalid") from None
    if document["resultWorkspaceDigest"] != workspace_digest(result.directories, result.text_files):
        raise WorkspaceStateError("workspace_state_invalid")
    if operations != canonical_overlay_operations(source, result):
        raise WorkspaceStateError("workspace_state_invalid")
    return result


def canonical_overlay_operations(
    source: ManagedWorkspaceTree,
    result: ManagedWorkspaceTree,
) -> tuple[OverlayOperation, ...]:
    source_paths = source.paths()
    deletion_candidates = {
        path
        for path in source_paths
        if result.kind(path) is None or result.kind(path) != source.kind(path)
    }
    deletions: list[str] = []
    for path in sorted(deletion_candidates, key=lambda value: (value.count("/"), value)):
        if not any(_within(path, parent) for parent in deletions):
            deletions.append(path)

    working = source.clone()
    operations: list[OverlayOperation] = []
    for path in deletions:
        _remove_subtree(working, path)
        operations.append(OverlayOperation(op="delete_path", path=path))

    for path in sorted(result.directories, key=lambda value: (value.count("/"), value)):
        if working.kind(path) is None:
            working.directories.add(path)
            operations.append(OverlayOperation(op="create_directory", path=path))

    for path in sorted(result.text_files):
        text = result.text_files[path]
        if working.text_files.get(path) != text or working.kind(path) != "text":
            working.text_files[path] = text
            operations.append(OverlayOperation(op="write_file", path=path, text=text))

    if _tree_projection(working) != _tree_projection(result):
        raise WorkspaceStateError("workspace_state_invalid")
    return tuple(operations)


def _strict_json_document(payload: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise WorkspaceStateError("workspace_state_invalid")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise WorkspaceStateError("workspace_state_invalid")

    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
        if not isinstance(document, dict) or jcs.canonicalize(document) != payload:
            raise WorkspaceStateError("workspace_state_invalid")
        return document
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        raise WorkspaceStateError("workspace_state_invalid") from None


def _parse_operation(value: Any) -> OverlayOperation:
    if not isinstance(value, dict) or not isinstance(value.get("op"), str):
        raise WorkspaceStateError("workspace_state_invalid")
    op = value["op"]
    expected = {"op", "path", "text"} if op == "write_file" else {"op", "path"}
    if op not in {"create_directory", "write_file", "delete_path"} or set(value) != expected:
        raise WorkspaceStateError("workspace_state_invalid")
    path = value.get("path")
    if not isinstance(path, str):
        raise WorkspaceStateError("workspace_state_invalid")
    try:
        if normalize_project_path(path, allow_root=False) != path:
            raise WorkspaceStateError("workspace_state_invalid")
    except (ValueError, UnicodeError):
        raise WorkspaceStateError("workspace_state_invalid") from None
    text = value.get("text")
    if op == "write_file":
        if not isinstance(text, str):
            raise WorkspaceStateError("workspace_state_invalid")
        try:
            _validate_text(text, None)
        except WorkspaceStorageError:
            raise WorkspaceStateError("workspace_state_invalid") from None
    return OverlayOperation(op=op, path=path, text=text)  # type: ignore[arg-type]


def _apply_operation(tree: ManagedWorkspaceTree, operation: OverlayOperation) -> None:
    if operation.op == "delete_path":
        if tree.kind(operation.path) is None:
            raise WorkspaceStateError("workspace_state_invalid")
        _remove_subtree(tree, operation.path)
        return
    _require_parent_directory(tree, operation.path, state=True)
    if operation.op == "create_directory":
        if tree.kind(operation.path) is not None:
            raise WorkspaceStateError("workspace_state_invalid")
        tree.directories.add(operation.path)
        return
    if tree.kind(operation.path) not in {None, "text"}:
        raise WorkspaceStateError("workspace_state_invalid")
    assert operation.text is not None
    tree.text_files[operation.path] = operation.text


def _validate_tree(tree: ManagedWorkspaceTree, limits: WorkspaceLimits) -> None:
    if (
        tree.directories & tree.text_files.keys()
        or tree.directories & tree.binary_paths
        or tree.text_files.keys() & tree.binary_paths
        or len(tree.paths()) > limits.max_files
    ):
        raise WorkspaceStorageError("workspace_limit_exceeded")
    total = 0
    for path in tree.paths():
        if normalize_project_path(path, allow_root=False) != path:
            raise WorkspaceStorageError("workspace_path_invalid")
        for parent in parent_paths(path):
            if parent not in tree.directories:
                raise WorkspaceStorageError("workspace_type_conflict")
    for text in tree.text_files.values():
        encoded = _validate_text(text, limits)
        total += len(encoded)
    if total > limits.max_managed_text_bytes or total > limits.max_expanded_bytes:
        raise WorkspaceStorageError("workspace_limit_exceeded")


def _validate_text(text: str, limits: WorkspaceLimits | None) -> bytes:
    if not isinstance(text, str) or "\x00" in text:
        raise WorkspaceStorageError("binary_file_unsupported")
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        raise WorkspaceStorageError("binary_file_unsupported") from None
    if limits is not None and len(encoded) > limits.max_file_bytes:
        raise WorkspaceStorageError("workspace_limit_exceeded")
    return encoded


def _require_parent_directory(
    tree: ManagedWorkspaceTree, path: str, *, state: bool = False
) -> None:
    parents = parent_paths(path)
    if parents and tree.kind(parents[-1]) != "directory":
        error = WorkspaceStateError if state else WorkspaceStorageError
        raise error("workspace_state_invalid" if state else "workspace_not_found")


def _remove_subtree(tree: ManagedWorkspaceTree, path: str) -> None:
    selected = {candidate for candidate in tree.paths() if _within(candidate, path)}
    tree.directories.difference_update(selected)
    for candidate in selected:
        tree.text_files.pop(candidate, None)
    tree.binary_paths.difference_update(selected)
    tree.stored_binary_paths.difference_update(selected)


def _copy_subtree(
    source: ManagedWorkspaceTree, destination: ManagedWorkspaceTree, path: str
) -> None:
    for candidate in source.directories:
        if _within(candidate, path):
            destination.directories.add(candidate)
    for candidate, text in source.text_files.items():
        if _within(candidate, path):
            destination.text_files[candidate] = text
    destination.binary_paths.update(
        candidate for candidate in source.binary_paths if _within(candidate, path)
    )
    destination.stored_binary_paths.update(
        candidate for candidate in source.stored_binary_paths if _within(candidate, path)
    )


def _descendants(tree: ManagedWorkspaceTree, path: str) -> set[str]:
    prefix = f"{path}/"
    return {candidate for candidate in tree.paths() if candidate.startswith(prefix)}


def _within(path: str, root: str) -> bool:
    return root == "" or path == root or path.startswith(f"{root}/")


def _path_value(tree: ManagedWorkspaceTree, path: str) -> tuple[str | None, str | None]:
    kind = tree.kind(path)
    return kind, tree.text_files.get(path) if kind == "text" else None


def _tree_projection(tree: ManagedWorkspaceTree) -> tuple[set[str], dict[str, str], set[str]]:
    return tree.directories, tree.text_files, tree.binary_paths


def _workspace_diff(
    before: ManagedWorkspaceTree,
    after: ManagedWorkspaceTree,
    root: str,
    maximum: int,
) -> WorkspaceDiff:
    changed = [
        path
        for path in sorted(before.paths() | after.paths())
        if _within(path, root) and _path_value(before, path) != _path_value(after, path)
    ]
    chunks: list[str] = []
    used = 0
    truncated = False
    for path in changed:
        lines = _path_diff(before, after, path)
        for line in lines:
            if not line.endswith("\n"):
                line += "\n"
            encoded = line.encode("utf-8")
            remaining = maximum - used
            if len(encoded) > remaining:
                chunks.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                used += len(chunks[-1].encode("utf-8"))
                truncated = True
                break
            chunks.append(line)
            used += len(encoded)
        if truncated:
            break
    return WorkspaceDiff(text="".join(chunks), returned_bytes=used, truncated=truncated)


def _path_diff(
    before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, path: str
) -> Iterable[str]:
    before_kind = before.kind(path)
    after_kind = after.kind(path)
    if before_kind == "directory" and after_kind == "directory":
        return ()
    if before_kind == "binary" or after_kind == "binary":
        return (f"Binary path changed: {path}",)
    before_text = before.text_files.get(path)
    after_text = after.text_files.get(path)
    if before_text is None and after_text is None:
        return ()
    return difflib.unified_diff(
        [] if before_text is None else before_text.splitlines(keepends=True),
        [] if after_text is None else after_text.splitlines(keepends=True),
        fromfile="/dev/null" if before_text is None else f"a/{path}",
        tofile="/dev/null" if after_text is None else f"b/{path}",
        lineterm="\n",
        n=3,
    )
