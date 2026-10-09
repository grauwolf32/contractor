"""Cumulative managed-text overlay state and invocation checkpoints."""

from __future__ import annotations

import asyncio
import difflib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import jcs

from contractor_runtime.digests import jcs_digest
from contractor_runtime.projectfs.paths import normalize_project_path, parent_paths
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage
from contractor_runtime.projectfs.storage import (
    DirectWorkspaceSession,
    ManagedWorkspaceTree,
    WorkspaceChange,
    WorkspaceChanges,
    WorkspaceChangesView,
    WorkspaceDiff,
    WorkspaceSnapshot,
    WorkspaceStorageError,
    _copy_tree,
    _normalized_path,
    _remove_tree,
    workspace_digest,
)
from contractor_runtime.settings import WorkspaceLimits
from contractor_runtime.strict_json import strict_json_loads
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.toolsets.common.lines import split_patch_lines

WORKSPACE_OVERLAY_API_VERSION = "contractor.workspace/v1"
WORKSPACE_OVERLAY_KIND = "WorkspaceOverlay"
WORKSPACE_OVERLAY_MEDIA_TYPE = "application/vnd.contractor.workspace-overlay+json"
MAX_DIFF_BYTES = 1 << 20
# Rendered diff bytes one session keeps between pages: the largest page plus the
# diff of a fully rewritten 16 MiB file. A larger window is not retained.
MAX_DIFF_CACHE_BYTES = 64 << 20
_NO_NEWLINE = "\\ No newline at end of file\n"
MAX_WORKSPACE_EXPORT_BYTES = 16 << 20


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
            if not (self.text is not None):
                raise RuntimeError("Expected self.text is not None")
            result["text"] = self.text
        return result


@dataclass(frozen=True, slots=True)
class WorkspaceExportBundle:
    """One immutable F snapshot and its two durable export payloads."""

    snapshot: WorkspaceSnapshot = field(repr=False)
    state: bytes = field(repr=False)
    diff: bytes = field(repr=False)
    result_workspace_digest: str
    generation: int = field(repr=False)


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
    ) -> None:
        super().__init__(
            mode="overlay",
            storage=storage,
            content_root=content_root,
            limits=limits,
            directories=directories,
            text_files=text_files,
            binary_paths=binary_paths,
        )
        self._source = self._tree.clone()
        self._checkpoint = self._tree.clone()
        # Managed text bytes of one effective tree. Writes adjust it instead
        # of re-encoding every text file; replacing the tree recounts it once.
        self._text_bytes: tuple[ManagedWorkspaceTree, int] | None = None
        self._generation = 0
        # The latest diff query and the generation it rendered. Diff worker
        # threads and close replace it, always under the session lock.
        self._diff_cache: tuple[int, _DiffPager] | None = None

    async def write_text(self, path: str, text: str) -> None:
        normalized = normalize_project_path(path, allow_root=False)
        encoded = _validate_text(text, self._limits)
        async with self._lock:
            self._require_open()
            tree = self._tree
            kind = tree.kind(normalized)
            if kind == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if kind == "directory":
                raise WorkspaceStorageError("workspace_type_conflict")
            _require_parent_directory(tree, normalized)
            # The effective tree is always valid, so only the written path's
            # count and byte delta can break an invariant.
            total = await self._managed_text_bytes_async() + len(encoded)
            if kind == "text":
                total -= len(tree.text_files[normalized].encode("utf-8"))
            elif (
                len(tree.directories) + len(tree.text_files) + len(tree.binary_paths)
                >= self._limits.max_files
            ):
                raise WorkspaceStorageError("workspace_limit_exceeded")
            if (
                total > self._limits.max_managed_text_bytes
                or total > self._limits.max_expanded_bytes
            ):
                raise WorkspaceStorageError("workspace_limit_exceeded")
            tree.text_files[normalized] = text
            self._text_bytes = (tree, total)
            self._generation += 1

    async def _managed_text_bytes_async(self) -> int:
        if self._text_bytes is not None and self._text_bytes[0] is self._tree:
            return self._text_bytes[1]
        return await to_thread_until_done(self._managed_text_bytes, name="workspace-text-account")

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None:
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            if self._tree.kind(normalized) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            try:
                current = self._tree.text_files[normalized]
            except KeyError:
                raise WorkspaceStorageError("workspace_not_found") from None
            updated, previous_size, updated_size = await to_thread_until_done(
                self._transform_text,
                current,
                transform,
                name="workspace-text-update",
            )
            total = await self._managed_text_bytes_async() - previous_size + updated_size
            if (
                total > self._limits.max_managed_text_bytes
                or total > self._limits.max_expanded_bytes
            ):
                raise WorkspaceStorageError("workspace_limit_exceeded")
            self._tree.text_files[normalized] = updated
            self._text_bytes = (self._tree, total)
            self._generation += 1

    def _transform_text(
        self, current: str, transform: Callable[[str], str]
    ) -> tuple[str, int, int]:
        updated = transform(current)
        return updated, len(current.encode("utf-8")), len(_validate_text(updated, self._limits))

    async def copy_path(self, source: str, destination: str, *, recursive: bool = False) -> None:
        await self._copy_or_move(source, destination, recursive=recursive, move=False)

    async def move_path(self, source: str, destination: str) -> None:
        await self._copy_or_move(source, destination, recursive=True, move=True)

    async def _copy_or_move(
        self, source: str, destination: str, *, recursive: bool, move: bool
    ) -> None:
        normalized_source = _normalized_path(source)
        normalized_destination = _normalized_path(destination)
        async with self._lock:
            self._require_open()
            candidate, total = await to_thread_until_done(
                self._copy_move_candidate,
                normalized_source,
                normalized_destination,
                recursive,
                move,
                name="workspace-path-edit",
            )
            self._tree = candidate
            self._text_bytes = (candidate, total)
            self._generation += 1

    def _copy_move_candidate(
        self, source: str, destination: str, recursive: bool, move: bool
    ) -> tuple[ManagedWorkspaceTree, int]:
        original = self._tree
        candidate = original.clone()
        _copy_tree(candidate, source, destination, recursive=recursive)
        if move:
            _remove_tree(candidate, source)
        if len(candidate.paths()) > self._limits.max_files:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        for path in candidate.paths() - original.paths():
            if _normalized_path(path) != path:
                raise WorkspaceStorageError("workspace_path_invalid")
            if any(parent not in candidate.directories for parent in parent_paths(path)):
                raise WorkspaceStorageError("workspace_type_conflict")
        total = self._managed_text_bytes()
        for path, text in original.text_files.items():
            if path not in candidate.text_files:
                total -= len(text.encode("utf-8"))
        for path, text in candidate.text_files.items():
            if path not in original.text_files or original.text_files[path] != text:
                total += len(_validate_text(text, self._limits))
                if path in original.text_files:
                    total -= len(original.text_files[path].encode("utf-8"))
        if total > self._limits.max_managed_text_bytes or total > self._limits.max_expanded_bytes:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        return candidate, total

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            edit = await to_thread_until_done(
                self._directory_candidate, normalized, parents, name="workspace-path-edit"
            )
            if edit is None:
                return
            candidate, total = edit
            self._tree = candidate
            self._text_bytes = (candidate, total)
            self._generation += 1

    def _directory_candidate(
        self, path: str, parents: bool
    ) -> tuple[ManagedWorkspaceTree, int] | None:
        original = self._tree
        existing = original.kind(path)
        if existing == "directory":
            return None
        if existing is not None:
            raise WorkspaceStorageError("workspace_type_conflict")
        ancestors = parent_paths(path)
        missing = [parent for parent in ancestors if original.kind(parent) is None]
        if missing and not parents:
            raise WorkspaceStorageError("workspace_not_found")
        if any(original.kind(parent) not in {None, "directory"} for parent in ancestors):
            raise WorkspaceStorageError("workspace_type_conflict")
        # The effective tree is always valid, so only the created directories
        # can break an invariant; managed text is unchanged.
        created = [*missing, path]
        if any(_normalized_path(item) != item for item in created):
            raise WorkspaceStorageError("workspace_path_invalid")
        count = len(original.directories) + len(original.text_files) + len(original.binary_paths)
        if count + len(created) > self._limits.max_files:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        candidate = original.clone()
        candidate.directories.update(created)
        return candidate, self._managed_text_bytes()

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            candidate, total = await to_thread_until_done(
                self._delete_candidate, normalized, recursive, name="workspace-path-edit"
            )
            self._tree = candidate
            self._text_bytes = (candidate, total)
            self._generation += 1

    def _delete_candidate(self, path: str, recursive: bool) -> tuple[ManagedWorkspaceTree, int]:
        original = self._tree
        kind = original.kind(path)
        if kind is None:
            raise WorkspaceStorageError("workspace_not_found")
        # Only a directory has descendants; avoid scanning the whole tree.
        selected = (
            {candidate for candidate in original.paths() if _within(candidate, path)}
            if kind == "directory"
            else {path}
        )
        if not selected.isdisjoint(original.binary_paths):
            raise WorkspaceStorageError("binary_file_unsupported")
        if len(selected) > 1 and not recursive:
            raise WorkspaceStorageError("workspace_type_conflict")
        # Removing a whole subtree cannot break an invariant; only the removed
        # text leaves the byte account.
        total = self._managed_text_bytes()
        candidate = original.clone()
        candidate.directories.difference_update(selected)
        for removed in selected:
            text = candidate.text_files.pop(removed, None)
            if text is not None:
                total -= len(text.encode("utf-8"))
        return candidate, total

    def _managed_text_bytes(self) -> int:
        if self._text_bytes is None or self._text_bytes[0] is not self._tree:
            # Hydration and every mutation validate the effective tree. This
            # recount is only for accounting, never for revalidating old text.
            total = sum(len(text.encode("utf-8")) for text in self._tree.text_files.values())
            self._text_bytes = (self._tree, total)
        return self._text_bytes[1]

    async def import_state(self, payload: bytes) -> None:
        async with self._lock:
            self._require_open()
            # Linear in the state size, but still too long for the event loop
            # that must keep sending lease heartbeats.
            candidate, checkpoint = await to_thread_until_done(
                self._import_candidate, payload, name="workspace-state-import"
            )
            self._tree = candidate
            self._checkpoint = checkpoint
            self._generation += 1

    def _import_candidate(
        self, payload: bytes
    ) -> tuple[ManagedWorkspaceTree, ManagedWorkspaceTree]:
        candidate = decode_workspace_state(payload, self._source, self._limits)
        return candidate, candidate.clone()

    async def export_state(self) -> bytes:
        async with self._lock:
            self._require_open()
            return await to_thread_until_done(
                encode_workspace_state,
                self._source,
                self._tree,
                name="workspace-state-export",
            )

    async def changed_paths(self, path: str = "") -> tuple[str, ...]:
        return tuple(entry.path for entry in await self.change_entries(path))

    async def change_entries(self, path: str = "") -> tuple[WorkspaceChange, ...]:
        normalized = normalize_project_path(path, allow_root=True)
        async with self._lock:
            self._require_open()
            # Tokens hash every changed text; every diff page fingerprints them.
            return await to_thread_until_done(
                self._change_entries, normalized, name="workspace-change-entries"
            )

    def _change_entries(self, root: str) -> tuple[WorkspaceChange, ...]:
        return tuple(
            WorkspaceChange(
                path=candidate,
                change=_change_kind(self._checkpoint, self._tree, candidate),
                token=_change_token(self._checkpoint, self._tree, candidate),
            )
            for candidate in _changed_paths(self._checkpoint, self._tree, root)
        )

    async def diff(
        self,
        path: str = "",
        *,
        max_bytes: int = 65536,
        offset_bytes: int = 0,
    ) -> WorkspaceDiff:
        normalized = normalize_project_path(path, allow_root=True)
        if (
            max_bytes <= 0
            or max_bytes > MAX_DIFF_BYTES
            or not isinstance(offset_bytes, int)
            or isinstance(offset_bytes, bool)
            or offset_bytes < 0
        ):
            raise WorkspaceStorageError("workspace_limit_exceeded")
        async with self._lock:
            self._require_open()
            # difflib is superlinear in file size. The lock keeps both trees
            # unchanged until the thread returns, even when this call is cancelled.
            return await to_thread_until_done(
                self._diff_page, normalized, max_bytes, offset_bytes, name="workspace-diff"
            )

    def _diff_page(self, root: str, maximum: int, offset: int) -> WorkspaceDiff:
        cached, self._diff_cache = self._diff_cache, None
        if (
            cached is not None
            and cached[0] == self._generation
            and cached[1].root == root
            and cached[1].start <= offset
        ):
            pager = cached[1]
        else:
            pager = _DiffPager(root, _changed_paths(self._checkpoint, self._tree, root))
        result = pager.page(self._checkpoint, self._tree, offset, maximum)
        # Every mutation advances the generation, so a retained pager is only
        # resumed over the same trees it has already rendered.
        if len(pager.window) <= MAX_DIFF_CACHE_BYTES:
            self._diff_cache = (self._generation, pager)
        return result

    def changes_view(self) -> WorkspaceChanges:
        self._require_open()
        return WorkspaceChangesView(self)

    async def rollback_changes(self, path: str = "") -> None:
        normalized = normalize_project_path(path, allow_root=True)
        async with self._lock:
            self._require_open()
            self._tree = await to_thread_until_done(
                self._rollback_candidate, normalized, name="workspace-rollback"
            )
            self._generation += 1

    def _rollback_candidate(self, normalized: str) -> ManagedWorkspaceTree:
        if normalized == "":
            return self._checkpoint.clone()
        if self._tree.kind(normalized) is None and self._checkpoint.kind(normalized) is None:
            raise WorkspaceStorageError("workspace_not_found")
        candidate = self._tree.clone()
        _remove_subtree(candidate, normalized)
        if self._checkpoint.kind(normalized) is not None:
            # Parents deleted after the checkpoint come back with the path;
            # one that has since become a file is not silently replaced.
            for parent in parent_paths(normalized):
                kind = candidate.kind(parent)
                if kind is None:
                    candidate.directories.add(parent)
                elif kind != "directory":
                    raise WorkspaceStorageError("workspace_type_conflict")
        _copy_subtree(self._checkpoint, candidate, normalized)
        _validate_tree(candidate, self._limits)
        return candidate

    async def commit_checkpoint(self) -> WorkspaceSnapshot:
        async with self._lock:
            self._require_open()
            self._checkpoint, snapshot = await to_thread_until_done(
                self._checkpoint_candidate, name="workspace-checkpoint"
            )
            self._generation += 1
            return snapshot

    def _checkpoint_candidate(self) -> tuple[ManagedWorkspaceTree, WorkspaceSnapshot]:
        candidate = self._tree.clone()
        return candidate, candidate.snapshot()

    async def prepare_export(
        self, *, max_payload_bytes: int = MAX_WORKSPACE_EXPORT_BYTES
    ) -> WorkspaceExportBundle:
        """Capture cumulative state and the checkpoint delta from one F."""

        if (
            not isinstance(max_payload_bytes, int)
            or isinstance(max_payload_bytes, bool)
            or max_payload_bytes <= 0
            or max_payload_bytes > MAX_WORKSPACE_EXPORT_BYTES
        ):
            raise WorkspaceStorageError("workspace_limit_exceeded")
        async with self._lock:
            self._require_open()
            return await to_thread_until_done(
                self._prepare_export_bundle,
                max_payload_bytes,
                name="workspace-export-prepare",
            )

    def _prepare_export_bundle(self, max_payload_bytes: int) -> WorkspaceExportBundle:
        snapshot = self._tree.snapshot()
        state = encode_workspace_state(self._source, self._tree)
        diff = _workspace_diff(self._checkpoint, self._tree, "", max_payload_bytes, 0)
        if len(state) > max_payload_bytes or diff.truncated:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        return WorkspaceExportBundle(
            snapshot=snapshot,
            state=state,
            diff=diff.text.encode("utf-8"),
            result_workspace_digest=snapshot.digest,
            generation=self._generation,
        )

    async def commit_export(self, bundle: WorkspaceExportBundle) -> WorkspaceSnapshot:
        """Advance B only if the exported F is still the effective tree."""

        async with self._lock:
            self._require_open()
            if self._generation != bundle.generation:
                raise WorkspaceStorageError("workspace_export_stale")
            self._checkpoint = await to_thread_until_done(
                self._tree.clone, name="workspace-export-commit"
            )
            self._generation += 1
            return bundle.snapshot

    async def close(self, *, deadline: float | None = None) -> None:
        async with asyncio.timeout_at(deadline), self._lock:
            self._closed = True
            self._diff_cache = None
            for tree in (self._tree, self._source, self._checkpoint):
                tree.directories.clear()
                tree.text_files.clear()
                tree.binary_paths.clear()


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
        MAX_WORKSPACE_EXPORT_BYTES,
        max(1 << 20, limits.max_managed_text_bytes + 512 * limits.max_files),
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
        # Operations check their own structure; limits bind only the result.
        # Validating the whole tree after each operation was quadratic.
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
    deleted: set[str] = set()
    for path in sorted(deletion_candidates, key=lambda value: (value.count("/"), value)):
        # Shallower paths come first, so a covering deletion is an ancestor.
        if not any(parent in deleted for parent in parent_paths(path)):
            deletions.append(path)
            deleted.add(path)

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
    try:
        document = strict_json_loads(payload.decode("utf-8"))
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
    return OverlayOperation(op=op, path=path, text=text)


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
    if not (operation.text is not None):
        raise RuntimeError("Expected operation.text is not None")
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
    if path and path not in tree.directories:
        # Only a directory has descendants; avoid scanning the whole tree.
        tree.text_files.pop(path, None)
        tree.binary_paths.discard(path)
        return
    selected = {candidate for candidate in tree.paths() if _within(candidate, path)}
    tree.directories.difference_update(selected)
    for candidate in selected:
        tree.text_files.pop(candidate, None)
    tree.binary_paths.difference_update(selected)


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


def _within(path: str, root: str) -> bool:
    return root == "" or path == root or path.startswith(f"{root}/")


def _path_value(tree: ManagedWorkspaceTree, path: str) -> tuple[str | None, str | None]:
    kind = tree.kind(path)
    return kind, tree.text_files.get(path) if kind == "text" else None


def _change_kind(before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, path: str) -> str:
    before_kind = before.kind(path)
    after_kind = after.kind(path)
    if before_kind is None:
        return "created"
    if after_kind is None:
        return "deleted"
    if before_kind != after_kind:
        return "type_changed"
    return "modified"


def _change_token(before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, path: str) -> str:
    document = {
        "before": _path_value(before, path),
        "after": _path_value(after, path),
    }
    return jcs_digest(document)


def _tree_projection(tree: ManagedWorkspaceTree) -> tuple[set[str], dict[str, str], set[str]]:
    return tree.directories, tree.text_files, tree.binary_paths


def _workspace_diff(
    before: ManagedWorkspaceTree,
    after: ManagedWorkspaceTree,
    root: str,
    maximum: int,
    offset: int,
) -> WorkspaceDiff:
    pager = _DiffPager(root, _changed_paths(before, after, root))
    return pager.page(before, after, offset, maximum)


def _changed_paths(
    before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, root: str
) -> tuple[str, ...]:
    return tuple(
        path
        for path in sorted(before.paths() | after.paths())
        if _within(path, root) and _path_value(before, path) != _path_value(after, path)
    )


@dataclass(slots=True)
class _DiffPager:
    """Byte pages of one rendered diff, kept from the latest requested page on.

    Paths render whole and in order, so paging forward renders each changed
    path once. Bytes before the latest requested offset are discarded; an
    earlier offset needs a new pager.
    """

    root: str
    paths: tuple[str, ...]
    rendered_paths: int = 0
    start: int = 0
    window: bytearray = field(default_factory=bytearray)

    def page(
        self,
        before: ManagedWorkspaceTree,
        after: ManagedWorkspaceTree,
        offset: int,
        maximum: int,
    ) -> WorkspaceDiff:
        if offset < self.start:
            raise RuntimeError("diff pager cannot return before its window")
        self._discard_before(offset)
        # One byte past the page tells whether the diff continues.
        while self.rendered_paths < len(self.paths) and (
            self.start + len(self.window) <= offset + maximum
        ):
            self.window += _rendered_path_diff(before, after, self.paths[self.rendered_paths])
            self.rendered_paths += 1
            self._discard_before(offset)
        relative = offset - self.start
        if relative > len(self.window) or (
            relative < len(self.window) and _utf8_continuation(self.window[relative])
        ):
            raise WorkspaceStorageError("workspace_cursor_invalid")
        end = relative + maximum
        truncated = end < len(self.window)
        if truncated:
            while end > relative and _utf8_continuation(self.window[end]):
                end -= 1
            if end == relative:
                raise WorkspaceStorageError("workspace_limit_exceeded")
        else:
            end = len(self.window)
        used = end - relative
        return WorkspaceDiff(
            text=self.window[relative:end].decode("utf-8"),
            returned_bytes=used,
            truncated=truncated,
            offset_bytes=offset,
            next_offset=offset + used if truncated else None,
        )

    def _discard_before(self, offset: int) -> None:
        count = min(offset - self.start, len(self.window))
        if count:
            del self.window[:count]
            self.start += count


def _utf8_continuation(value: int) -> bool:
    return value & 0xC0 == 0x80


def _rendered_path_diff(
    before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, path: str
) -> bytes:
    lines: list[str] = []
    for line in _path_diff(before, after, path):
        if not line.endswith("\n"):
            # Only a content line can lack LF: that side of the file ends
            # without a newline, which a valid patch must say explicitly.
            line += "\n" + _NO_NEWLINE
        lines.append(line)
    return "".join(lines).encode("utf-8")


def _path_diff(
    before: ManagedWorkspaceTree, after: ManagedWorkspaceTree, path: str
) -> Iterable[str]:
    before_kind = before.kind(path)
    after_kind = after.kind(path)
    if before_kind == "directory" and after_kind == "directory":
        return ()
    if before_kind == "binary" or after_kind == "binary":
        return (f"Binary path changed: {path}\n",)
    before_text = before.text_files.get(path)
    after_text = after.text_files.get(path)
    if before_text is None and after_text is None:
        return ()
    return difflib.unified_diff(
        [] if before_text is None else split_patch_lines(before_text),
        [] if after_text is None else split_patch_lines(after_text),
        fromfile="/dev/null" if before_text is None else f"a/{path}",
        tofile="/dev/null" if after_text is None else f"b/{path}",
        lineterm="\n",
        n=3,
    )
