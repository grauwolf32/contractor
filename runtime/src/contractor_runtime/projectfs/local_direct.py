"""Private disk-authoritative implementation; no retained source-content tree."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.local_io import LocalTree, RootedLocalFilesystem
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

    async def _mutate(self, plan: Callable[[ManagedWorkspaceTree], _Mutation]) -> None:
        def owned(fs: RootedLocalFilesystem, deadline: float) -> None:
            acquired = fs.scan(deadline=deadline)
            candidate = _managed(acquired)
            mutation = plan(candidate)
            _validate_managed_tree(candidate, self._limits)
            # Managed tree validation deliberately excludes binary bytes for
            # memory/overlay. Local direct must count all physical regular files.
            expanded = sum(len(text.encode("utf-8")) for text in candidate.text_files.values())
            expanded += sum(acquired.entries[path].size for path in candidate.binary_paths)
            if expanded > self._limits.max_expanded_bytes:
                raise WorkspaceStorageError("workspace_limit_exceeded")
            try:
                mutation(fs, deadline)
            except Exception:
                # No stale reverse delta: a partially applied multi-file action
                # cannot prove recovery. Deny further work; close still joins.
                self.guard.fence()
                raise WorkspaceStorageError("workspace_unavailable") from None

        await self._run(owned)

    async def write_text(self, path: str, text: str) -> None:
        path = _normalized_path(path)
        data = _validate_managed_text(text, self._limits)

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
            if tree.kind(path) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if tree.kind(path) == "directory":
                raise WorkspaceStorageError("workspace_type_conflict")
            _require_parent(tree, path)
            tree.text_files[path] = text
            return lambda fs, deadline: fs.write(path, data, deadline=deadline)

        await self._mutate(plan)

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None:
        path = _normalized_path(path)

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
            if tree.kind(path) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if path not in tree.text_files:
                raise WorkspaceStorageError("workspace_not_found")
            updated = transform(tree.text_files[path])
            data = _validate_managed_text(updated, self._limits)
            tree.text_files[path] = updated
            return lambda fs, deadline: fs.write(path, data, deadline=deadline)

        await self._mutate(plan)

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        path = _normalized_path(path)

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
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

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
            if tree.kind(path) is None:
                raise WorkspaceStorageError("workspace_not_found")
            selected = _subtree_paths(tree, path)
            if selected & tree.binary_paths:
                raise WorkspaceStorageError("binary_file_unsupported")
            if len(selected) > 1 and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            _remove_tree(tree, path)
            return lambda fs, deadline: fs.remove(path, recursive=recursive, deadline=deadline)

        await self._mutate(plan)

    async def copy_path(self, source: str, destination: str, *, recursive: bool = False) -> None:
        source, destination = _normalized_path(source), _normalized_path(destination)

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
            _copy_tree(tree, source, destination, recursive=recursive)
            return lambda fs, deadline: fs.copy(
                source, destination, recursive=recursive, deadline=deadline
            )

        await self._mutate(plan)

    async def move_path(self, source: str, destination: str) -> None:
        source, destination = _normalized_path(source), _normalized_path(destination)

        def plan(tree: ManagedWorkspaceTree) -> _Mutation:
            _copy_tree(tree, source, destination, recursive=True)
            _remove_tree(tree, source)
            return lambda fs, deadline: fs.move(source, destination, deadline=deadline)

        await self._mutate(plan)

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


def _managed(tree: LocalTree) -> ManagedWorkspaceTree:
    return ManagedWorkspaceTree(
        directories={path for path, item in tree.entries.items() if item.directory},
        text_files=dict(tree.texts),
        binary_paths=set(tree.binary_paths),
        stored_binary_paths=set(tree.binary_paths),
    )
