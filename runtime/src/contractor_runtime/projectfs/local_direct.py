"""Private disk-authoritative implementation; no retained source-content tree."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.local_io import (
    LocalTree,
    RootedLocalFilesystem,
    _NoEffectMutation,
)
from contractor_runtime.projectfs.operation_guard import WorkspaceOperationGuard
from contractor_runtime.projectfs.paths import parent_paths
from contractor_runtime.projectfs.storage import (
    ManagedWorkspaceTree,
    WorkspaceObservationMetadata,
    WorkspaceSnapshot,
    _copy_tree,
    _normalized_path,
    _remove_tree,
    _require_parent,
    _subtree_paths,
    _validate_managed_text,
    _validate_managed_tree,
)
from contractor_runtime.settings import WorkspaceLimits

T = TypeVar("T")
# Internal ceiling; a caller's earlier timeout also fences and retains ownership.
_Mutation = Callable[[RootedLocalFilesystem, float], None]
# A plan sees the managed projection and the opaque leaves it may not touch.
_Plan = Callable[[ManagedWorkspaceTree, frozenset[str]], _Mutation]
_Inspect = Callable[[LocalTree], set[str]]


class LocalDirectWorkspace:
    def __init__(
        self, content_root: str, limits: WorkspaceLimits, *, operation_timeout_seconds: float
    ) -> None:
        if not math.isfinite(operation_timeout_seconds) or operation_timeout_seconds <= 0:
            raise ValueError("workspace operation timeout must be finite and positive")
        self._operation_timeout_seconds = operation_timeout_seconds
        self._root = Path(content_root)
        self._limits = limits
        self._filesystem: RootedLocalFilesystem | None = None
        self.guard = WorkspaceOperationGuard()

    async def _run(
        self,
        operation: Callable[[RootedLocalFilesystem, float], T],
        *,
        deadline: float | None = None,
    ) -> T:
        deadline = self._deadline(deadline)

        def owned() -> T:
            # Even initial root verification must not block the event loop.
            if self._filesystem is None:
                self._filesystem = RootedLocalFilesystem(self._root, self._limits)
            return operation(self._filesystem, deadline)

        return await self.guard.run(owned, deadline=deadline)

    async def initialize(self, *, deadline: float) -> None:
        await self._run(lambda fs, end: fs.scan(deadline=end), deadline=deadline)

    async def snapshot(self) -> WorkspaceSnapshot:
        return await self._run(lambda fs, deadline: _managed(fs.scan(deadline=deadline)).snapshot())

    async def observation_metadata(self) -> WorkspaceObservationMetadata:
        snapshot = await self.snapshot()
        return WorkspaceObservationMetadata(
            digest=snapshot.digest,
            managed_text_paths=tuple(file.path for file in snapshot.files),
        )

    async def read_text(self, path: str) -> str:
        path = _normalized_path(path)

        def read(fs: RootedLocalFilesystem, deadline: float) -> str:
            data = fs.read(path, deadline=deadline)
            try:
                text = data.decode("utf-8")
            except UnicodeError:
                raise WorkspaceStorageError("binary_file_unsupported") from None
            if "\x00" in text:
                raise WorkspaceStorageError("binary_file_unsupported")
            if len(data) > self._limits.max_managed_text_bytes:
                raise WorkspaceStorageError("workspace_limit_exceeded")
            return text

        return await self._run(read)

    async def _mutate(self, plan: _Plan, *, inspect: _Inspect = lambda _: set()) -> None:
        def owned(fs: RootedLocalFilesystem, deadline: float) -> None:
            acquired = fs.scan(deadline=deadline, contents=False)
            loaded: dict[str, str] | None = None
            if acquired.expanded_bytes > self._limits.max_managed_text_bytes:
                # Physical bytes no longer prove the text bound: classify the
                # whole tree rather than guessing which bytes are binary.
                acquired = fs.scan(deadline=deadline)
                candidate = _managed(acquired)
            else:
                loaded = {}
                for path in inspect(acquired):
                    item = acquired.entries[path]
                    text, unreadable = fs.read_classified(path, deadline=deadline, expected=item)
                    if unreadable:
                        acquired.entries[path] = replace(item, opaque=True)
                        acquired.opaque_paths.add(path)
                        acquired.expanded_bytes -= item.size
                    elif text is None:
                        acquired.binary_paths.add(path)
                    else:
                        acquired.texts[path] = loaded[path] = text
                candidate = _managed(acquired, metadata_only=True)
            opaque = frozenset(acquired.opaque_paths)
            mutation = plan(candidate, opaque)
            encoded_lengths: dict[str, int] | None = {} if loaded is not None else None
            managed_bytes = _validate_managed_tree(
                candidate, self._limits, encoded_lengths=encoded_lengths
            )
            if loaded is not None:
                assert encoded_lengths is not None
                expanded = _projected_bytes(candidate, acquired, loaded, encoded_lengths)
                if expanded > self._limits.max_managed_text_bytes:
                    actual = fs.scan(deadline=deadline)
                    _materialize_unselected(candidate, acquired, actual, loaded)
                    acquired = actual
                    managed_bytes = _validate_managed_tree(candidate, self._limits)
                    expanded = managed_bytes + sum(
                        acquired.entries[path].size
                        for path in candidate.binary_paths - acquired.opaque_paths
                    )
            else:
                expanded = managed_bytes + sum(
                    acquired.entries[path].size for path in candidate.binary_paths - opaque
                )
            # Managed tree validation deliberately excludes binary bytes for
            # memory/overlay. Local direct counts every non-opaque physical
            # file, including the ones whose contents were not needed.
            if expanded > self._limits.max_expanded_bytes:
                raise WorkspaceStorageError("workspace_limit_exceeded")
            try:
                mutation(fs, deadline)
            except _NoEffectMutation as error:
                # The primitive confirmed that no managed change survived and
                # temporary cleanup completed. Keep the shared execution guard.
                raise WorkspaceStorageError(error.args[0]) from None
            except Exception:
                # No stale reverse delta: a partially applied multi-file action
                # cannot prove recovery. Deny further work; close still joins.
                self.guard.fence()
                raise WorkspaceStorageError("workspace_unavailable") from None

        await self._run(owned)

    async def write_text(self, path: str, text: str) -> None:
        path = _normalized_path(path)
        data = _validate_managed_text(text, self._limits)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            _refuse_opaque(opaque, {path})
            if tree.kind(path) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if tree.kind(path) == "directory":
                raise WorkspaceStorageError("workspace_type_conflict")
            _require_parent(tree, path)
            tree.text_files[path] = text
            return lambda fs, deadline: fs.write(path, data, deadline=deadline)

        await self._mutate(plan, inspect=lambda tree: _file_at(tree, path))

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None:
        path = _normalized_path(path)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            _refuse_opaque(opaque, {path})
            if tree.kind(path) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if path not in tree.text_files:
                raise WorkspaceStorageError("workspace_not_found")
            updated = transform(tree.text_files[path])
            data = _validate_managed_text(updated, self._limits)
            tree.text_files[path] = updated
            return lambda fs, deadline: fs.write(path, data, deadline=deadline)

        await self._mutate(plan, inspect=lambda tree: _file_at(tree, path))

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        path = _normalized_path(path)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            if tree.kind(path) == "directory":
                return lambda fs, deadline: None
            if tree.kind(path) is not None:
                raise WorkspaceStorageError("workspace_type_conflict")
            for parent in parent_paths(path):
                if tree.kind(parent) is None and not parents:
                    raise WorkspaceStorageError("workspace_not_found")
                if tree.kind(parent) not in {None, "directory"}:
                    raise WorkspaceStorageError("workspace_type_conflict")
                tree.directories.add(parent)
            tree.directories.add(path)
            return lambda fs, deadline: fs.mkdir(path, parents=parents, deadline=deadline)

        await self._mutate(plan)

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        path = _normalized_path(path)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            if tree.kind(path) is None:
                raise WorkspaceStorageError("workspace_not_found")
            selected = _subtree_paths(tree, path)
            _refuse_opaque(opaque, selected)
            if selected & tree.binary_paths:
                raise WorkspaceStorageError("binary_file_unsupported")
            if len(selected) > 1 and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            _remove_tree(tree, path)
            return lambda fs, deadline: fs.remove(path, recursive=recursive, deadline=deadline)

        await self._mutate(plan, inspect=lambda tree: _files_under(tree, path))

    async def copy_path(self, source: str, destination: str, *, recursive: bool = False) -> None:
        source, destination = _normalized_path(source), _normalized_path(destination)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            _refuse_opaque(opaque, _subtree_paths(tree, source) | {destination})
            _copy_tree(tree, source, destination, recursive=recursive)
            return lambda fs, deadline: fs.copy(
                source, destination, recursive=recursive, deadline=deadline
            )

        await self._mutate(plan, inspect=lambda tree: _files_under(tree, source))

    async def move_path(self, source: str, destination: str) -> None:
        source, destination = _normalized_path(source), _normalized_path(destination)

        def plan(tree: ManagedWorkspaceTree, opaque: frozenset[str]) -> _Mutation:
            _refuse_opaque(opaque, _subtree_paths(tree, source) | {destination})
            _copy_tree(tree, source, destination, recursive=True)
            _remove_tree(tree, source)
            return lambda fs, deadline: fs.move(source, destination, deadline=deadline)

        await self._mutate(plan, inspect=lambda tree: _files_under(tree, source))

    def _deadline(self, enclosing: float | None) -> float:
        operation_deadline = time.monotonic() + self._operation_timeout_seconds
        if enclosing is None:
            return operation_deadline
        if not math.isfinite(enclosing):
            raise WorkspaceStorageError("workspace_unavailable")
        return min(operation_deadline, enclosing)

    async def close(self, *, deadline: float | None = None) -> None:
        # AllocationService cleans provider storage only after this barrier.
        await self.guard.close(lambda: None, deadline=self._deadline(deadline))


def _managed(tree: LocalTree, *, metadata_only: bool = False) -> ManagedWorkspaceTree:
    # Opaque leaves (links, special, oversize or unreadable entries) are listed
    # like binary files: visible and counted, but never read as text.
    binary_paths = tree.binary_paths | tree.opaque_paths
    return ManagedWorkspaceTree(
        directories={path for path, item in tree.entries.items() if item.directory},
        text_files=(
            {
                path: tree.texts.get(path, "")
                for path, item in tree.entries.items()
                if not item.directory and not item.opaque and path not in tree.binary_paths
            }
            if metadata_only
            else dict(tree.texts)
        ),
        binary_paths=set(binary_paths),
    )


def _refuse_opaque(opaque: frozenset[str], paths: set[str]) -> None:
    """An opaque leaf is never followed, so neither it nor paths below it resolve."""
    if opaque and any(
        path in opaque or not opaque.isdisjoint(parent_paths(path)) for path in paths
    ):
        raise WorkspaceStorageError("workspace_type_conflict")


def _file_at(tree: LocalTree, path: str) -> set[str]:
    item = tree.entries.get(path)
    return {path} if item is not None and not item.directory and not item.opaque else set()


def _files_under(tree: LocalTree, root: str) -> set[str]:
    return {
        path
        for path, item in tree.entries.items()
        if (path == root or path.startswith(f"{root}/")) and not item.directory and not item.opaque
    }


def _projected_bytes(
    candidate: ManagedWorkspaceTree,
    before: LocalTree,
    loaded: dict[str, str],
    encoded_lengths: dict[str, int],
) -> int:
    total = sum(before.entries[path].size for path in candidate.binary_paths - before.opaque_paths)
    for path, text in candidate.text_files.items():
        item = before.entries.get(path)
        if item is not None and (path not in loaded or loaded[path] == text):
            total += item.size
        else:
            total += encoded_lengths[path]
    return total


def _materialize_unselected(
    candidate: ManagedWorkspaceTree,
    before: LocalTree,
    actual: LocalTree,
    loaded: dict[str, str],
) -> None:
    # A fallback may happen after a transform has run. Fill only unchanged
    # placeholders from the fresh scan; never call a user transform twice.
    if before.entries.keys() != actual.entries.keys() or any(
        before.entries[path].directory != actual.entries[path].directory for path in before.entries
    ):
        raise WorkspaceStorageError("workspace_unavailable")
    if any(actual.texts.get(path) != text for path, text in loaded.items()):
        raise WorkspaceStorageError("workspace_unavailable")
    for path in list(candidate.text_files):
        if path not in before.entries or path in loaded:
            continue
        if path in actual.texts:
            candidate.text_files[path] = actual.texts[path]
        else:
            del candidate.text_files[path]
            candidate.binary_paths.add(path)
