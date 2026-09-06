"""Narrow backend-independent project workspace snapshots and handles."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol

import jcs

from contractor_runtime.contracts import WorkspaceModeV2
from contractor_runtime.projectfs.errors import WorkspaceStorageError as WorkspaceStorageError
from contractor_runtime.projectfs.paths import (
    ProjectPathError,
    normalize_project_path,
    parent_paths,
)
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage
from contractor_runtime.settings import WorkspaceLimits


@dataclass(frozen=True, slots=True)
class WorkspaceTextFile:
    path: str
    text: str = field(repr=False)
    size: int


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    directories: tuple[str, ...]
    files: tuple[WorkspaceTextFile, ...]
    binary_paths: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class WorkspaceObservationMetadata:
    """Content-free metadata for one effective managed-text workspace tree."""

    digest: str
    managed_text_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkspaceDiff:
    text: str = field(repr=False)
    returned_bytes: int
    truncated: bool
    offset_bytes: int = 0
    next_offset: int | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceChange:
    path: str
    change: str
    token: str = field(repr=False)


class WorkspaceReader(Protocol):
    async def snapshot(self) -> WorkspaceSnapshot: ...

    async def read_text(self, path: str) -> str: ...

    async def observation_metadata(self) -> WorkspaceObservationMetadata: ...


class WorkspaceWriter(WorkspaceReader, Protocol):
    async def write_text(self, path: str, text: str) -> None: ...

    async def make_directory(self, path: str, *, parents: bool = False) -> None: ...

    async def delete_path(self, path: str, *, recursive: bool = False) -> None: ...

    async def copy_path(
        self, source: str, destination: str, *, recursive: bool = False
    ) -> None: ...

    async def move_path(self, source: str, destination: str) -> None: ...

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None: ...


class WorkspaceChanges(Protocol):
    async def change_entries(self, path: str = "") -> tuple[WorkspaceChange, ...]: ...

    async def diff(
        self, path: str = "", *, max_bytes: int = 65536, offset_bytes: int = 0
    ) -> WorkspaceDiff: ...

    async def rollback_changes(self, path: str = "") -> None: ...


class WorkspaceReaderView:
    """Narrow model-tool handle with no backend or provider properties."""

    __slots__ = ("__reader",)

    def __init__(self, reader: WorkspaceReader) -> None:
        self.__reader = reader

    async def snapshot(self) -> WorkspaceSnapshot:
        return await self.__reader.snapshot()

    async def read_text(self, path: str) -> str:
        return await self.__reader.read_text(path)

    async def observation_metadata(self) -> WorkspaceObservationMetadata:
        return await self.__reader.observation_metadata()


class WorkspaceWriterView:
    """Narrow text writer/reader with no concrete backend escape hatch."""

    __slots__ = ("__writer",)

    def __init__(self, writer: WorkspaceWriter) -> None:
        self.__writer = writer

    async def snapshot(self) -> WorkspaceSnapshot:
        return await self.__writer.snapshot()  # type: ignore[attr-defined]

    async def read_text(self, path: str) -> str:
        return await self.__writer.read_text(path)  # type: ignore[attr-defined]

    async def observation_metadata(self) -> WorkspaceObservationMetadata:
        return await self.__writer.observation_metadata()  # type: ignore[attr-defined]

    async def write_text(self, path: str, text: str) -> None:
        await self.__writer.write_text(path, text)

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        await self.__writer.make_directory(path, parents=parents)

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        await self.__writer.delete_path(path, recursive=recursive)

    async def copy_path(self, source: str, destination: str, *, recursive: bool = False) -> None:
        await self.__writer.copy_path(source, destination, recursive=recursive)

    async def move_path(self, source: str, destination: str) -> None:
        await self.__writer.move_path(source, destination)

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None:
        await self.__writer.update_text(path, transform)


class WorkspaceChangesView:
    """Narrow invocation-delta view without cumulative state or checkpoint commit."""

    __slots__ = ("__changes",)

    def __init__(self, changes: WorkspaceChanges) -> None:
        self.__changes = changes

    async def change_entries(self, path: str = "") -> tuple[WorkspaceChange, ...]:
        return await self.__changes.change_entries(path)

    async def diff(
        self, path: str = "", *, max_bytes: int = 65536, offset_bytes: int = 0
    ) -> WorkspaceDiff:
        return await self.__changes.diff(path, max_bytes=max_bytes, offset_bytes=offset_bytes)

    async def rollback_changes(self, path: str = "") -> None:
        await self.__changes.rollback_changes(path)


@dataclass(slots=True)
class ManagedWorkspaceTree:
    directories: set[str] = field(default_factory=set)
    text_files: dict[str, str] = field(default_factory=dict, repr=False)
    binary_paths: set[str] = field(default_factory=set)
    stored_binary_paths: set[str] = field(default_factory=set)

    def clone(self) -> ManagedWorkspaceTree:
        return ManagedWorkspaceTree(
            directories=set(self.directories),
            text_files=dict(self.text_files),
            binary_paths=set(self.binary_paths),
            stored_binary_paths=set(self.stored_binary_paths),
        )

    def snapshot(self) -> WorkspaceSnapshot:
        return _snapshot(self.directories, self.text_files, self.binary_paths)

    def kind(self, path: str) -> str | None:
        if path in self.directories:
            return "directory"
        if path in self.text_files:
            return "text"
        if path in self.binary_paths:
            return "binary"
        return None

    def paths(self) -> set[str]:
        return self.directories | self.text_files.keys() | self.binary_paths


class DirectWorkspaceSession:
    """Disk-authoritative local direct or managed memory/overlay session."""

    def __init__(
        self,
        *,
        mode: WorkspaceModeV2,
        storage: ProjectWorkspaceStorage,
        content_root: str,
        limits: WorkspaceLimits,
        directories: set[str],
        text_files: dict[str, str],
        binary_paths: set[str],
        stored_binary_paths: set[str],
    ) -> None:
        self._mode = mode
        self._storage = storage
        self._content_root = content_root
        self._limits = limits
        # Local imports keep the private disk implementation behind the narrow
        # interfaces declared in this module, without a module import cycle.
        from contractor_runtime.projectfs.local_direct import LocalDirectWorkspace

        self._local = (
            LocalDirectWorkspace(content_root, limits)
            if mode == "direct" and storage.storage == "local"
            else None
        )
        self._tree = (
            ManagedWorkspaceTree()
            if self._local is not None
            else ManagedWorkspaceTree(
                directories=set(directories),
                text_files=dict(text_files),
                binary_paths=set(binary_paths),
                stored_binary_paths=set(stored_binary_paths),
            )
        )
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def mode(self) -> WorkspaceModeV2:
        return self._mode

    @property
    def storage(self) -> ProjectWorkspaceStorage:
        return self._storage

    @property
    def execution_guard(self):
        """Private lifecycle seam; reader/writer tool views do not expose this."""
        self._require_open()
        if self._local is None:
            raise WorkspaceStorageError("workspace_unavailable")
        return self._local.guard

    @property
    def limits(self) -> WorkspaceLimits:
        return self._limits

    def reader_view(self) -> WorkspaceReader:
        self._require_open()
        return WorkspaceReaderView(self)

    def writer_view(self) -> WorkspaceWriter:
        self._require_open()
        return WorkspaceWriterView(self)

    def changes_view(self) -> WorkspaceChanges:
        raise WorkspaceStorageError("workspace_mode_unsupported")

    async def snapshot(self) -> WorkspaceSnapshot:
        if self._local is not None:
            self._require_open()
            return await self._local.snapshot()
        async with self._lock:
            self._require_open()
            return self._tree.snapshot()

    async def observation_metadata(self) -> WorkspaceObservationMetadata:
        if self._local is not None:
            self._require_open()
            return await self._local.observation_metadata()
        async with self._lock:
            self._require_open()
            return WorkspaceObservationMetadata(
                digest=workspace_digest(self._tree.directories, self._tree.text_files),
                managed_text_paths=tuple(sorted(self._tree.text_files)),
            )

    async def read_text(self, path: str) -> str:
        if self._local is not None:
            self._require_open()
            return await self._local.read_text(path)
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            if normalized in self._tree.binary_paths:
                raise WorkspaceStorageError("binary_file_unsupported")
            if normalized in self._tree.directories:
                raise WorkspaceStorageError("workspace_type_conflict")
            try:
                return self._tree.text_files[normalized]
            except KeyError:
                raise WorkspaceStorageError("workspace_not_found") from None

    async def write_text(self, path: str, text: str) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.write_text(path, text)
        normalized = _normalized_path(path)
        _validate_managed_text(text, self._limits)
        async with self._lock:
            self._require_open()
            candidate = self._tree.clone()
            kind = candidate.kind(normalized)
            if kind == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            if kind == "directory":
                raise WorkspaceStorageError("workspace_type_conflict")
            _require_parent(candidate, normalized)
            candidate.text_files[normalized] = text
            self._commit_candidate(candidate)

    async def update_text(self, path: str, transform: Callable[[str], str]) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.update_text(path, transform)
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            if self._tree.kind(normalized) == "binary":
                raise WorkspaceStorageError("binary_file_unsupported")
            try:
                current = self._tree.text_files[normalized]
            except KeyError:
                raise WorkspaceStorageError("workspace_not_found") from None
            updated = transform(current)
            _validate_managed_text(updated, self._limits)
            candidate = self._tree.clone()
            candidate.text_files[normalized] = updated
            self._commit_candidate(candidate)

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.make_directory(path, parents=parents)
        normalized = _normalized_path(path)
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
            self._commit_candidate(candidate)

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.delete_path(path, recursive=recursive)
        normalized = _normalized_path(path)
        async with self._lock:
            self._require_open()
            kind = self._tree.kind(normalized)
            if kind is None:
                raise WorkspaceStorageError("workspace_not_found")
            selected = _subtree_paths(self._tree, normalized)
            if any(path in self._tree.binary_paths for path in selected):
                raise WorkspaceStorageError("binary_file_unsupported")
            if len(selected) > 1 and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            candidate = self._tree.clone()
            _remove_tree(candidate, normalized)
            self._commit_candidate(candidate)

    async def copy_path(self, source: str, destination: str, *, recursive: bool = False) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.copy_path(source, destination, recursive=recursive)
        normalized_source = _normalized_path(source)
        normalized_destination = _normalized_path(destination)
        async with self._lock:
            self._require_open()
            candidate = self._tree.clone()
            _copy_tree(
                candidate,
                normalized_source,
                normalized_destination,
                recursive=recursive,
            )
            self._commit_candidate(candidate)

    async def move_path(self, source: str, destination: str) -> None:
        if self._local is not None:
            self._require_open()
            return await self._local.move_path(source, destination)
        normalized_source = _normalized_path(source)
        normalized_destination = _normalized_path(destination)
        async with self._lock:
            self._require_open()
            candidate = self._tree.clone()
            _copy_tree(
                candidate,
                normalized_source,
                normalized_destination,
                recursive=True,
            )
            _remove_tree(candidate, normalized_source)
            self._commit_candidate(candidate)

    async def close(self) -> None:
        if self._local is not None:
            await self._local.close()
            self._closed = True
            return
        async with self._lock:
            self._closed = True
            self._tree.directories.clear()
            self._tree.text_files.clear()
            self._tree.binary_paths.clear()
            self._tree.stored_binary_paths.clear()

    def _source_tree(self) -> ManagedWorkspaceTree:
        self._require_open()
        if self._local is not None:
            raise WorkspaceStorageError("workspace_mode_unsupported")
        return self._tree.clone()

    def _commit_candidate(self, candidate: ManagedWorkspaceTree) -> None:
        _validate_managed_tree(candidate, self._limits)
        current = self._tree
        try:
            self._apply_direct_delta(current, candidate)
        except Exception:
            try:
                self._apply_direct_delta(candidate, current)
            except Exception:
                # The caller still sees a stable failure and the allocation
                # retains its authoritative in-memory snapshot. Hardening owns
                # process fencing when physical rollback cannot be confirmed.
                return _raise_workspace_unavailable()
            raise WorkspaceStorageError("workspace_unavailable") from None
        self._tree = candidate

    def _apply_direct_delta(
        self, current: ManagedWorkspaceTree, candidate: ManagedWorkspaceTree
    ) -> None:
        filesystem = self._storage.filesystem
        removed = {
            path
            for path in current.paths()
            if candidate.kind(path) is None or candidate.kind(path) != current.kind(path)
        }
        roots: list[str] = []
        for path in sorted(removed, key=lambda value: (value.count("/"), value)):
            if not any(path == root or path.startswith(f"{root}/") for root in roots):
                roots.append(path)
        for path in roots:
            backend = self._backend_path(path)
            if filesystem.exists(backend):
                filesystem.rm(backend, recursive=True)
        for path in sorted(candidate.directories, key=lambda value: (value.count("/"), value)):
            if current.kind(path) != "directory":
                filesystem.makedirs(self._backend_path(path), exist_ok=True)
        for path, text in sorted(candidate.text_files.items()):
            if current.text_files.get(path) != text or current.kind(path) != "text":
                encoded = text.encode("utf-8")
                if self._storage.storage == "local":
                    _atomic_local_text_write(self._content_root, path, encoded)
                else:
                    filesystem.pipe(self._backend_path(path), encoded)

    def _backend_path(self, path: str) -> str:
        return f"{self._content_root.rstrip('/')}/{path}"

    def _require_open(self) -> None:
        if self._closed:
            raise WorkspaceStorageError("workspace_not_found")


def workspace_digest(directories: set[str], text_files: dict[str, str]) -> str:
    document = {
        "directories": sorted(directories),
        "files": [{"path": path, "text": text_files[path]} for path in sorted(text_files)],
    }
    encoded = jcs.canonicalize(document)
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _snapshot(
    directories: set[str], text_files: dict[str, str], binary_paths: set[str]
) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        directories=tuple(sorted(directories)),
        files=tuple(
            WorkspaceTextFile(
                path=path,
                text=text_files[path],
                size=len(text_files[path].encode("utf-8")),
            )
            for path in sorted(text_files)
        ),
        binary_paths=tuple(sorted(binary_paths)),
        digest=workspace_digest(directories, text_files),
    )


def _normalized_path(path: str) -> str:
    try:
        return normalize_project_path(path, allow_root=False)
    except ProjectPathError:
        raise WorkspaceStorageError("workspace_path_invalid") from None


def _validate_managed_text(text: str, limits: WorkspaceLimits) -> bytes:
    if not isinstance(text, str) or "\x00" in text:
        raise WorkspaceStorageError("binary_file_unsupported")
    try:
        encoded = text.encode("utf-8")
    except UnicodeError:
        raise WorkspaceStorageError("binary_file_unsupported") from None
    if len(encoded) > limits.max_file_bytes:
        raise WorkspaceStorageError("workspace_limit_exceeded")
    return encoded


def _validate_managed_tree(tree: ManagedWorkspaceTree, limits: WorkspaceLimits) -> None:
    if (
        tree.directories & tree.text_files.keys()
        or tree.directories & tree.binary_paths
        or tree.text_files.keys() & tree.binary_paths
        or len(tree.paths()) > limits.max_files
    ):
        raise WorkspaceStorageError("workspace_limit_exceeded")
    managed_bytes = 0
    for path in tree.paths():
        if _normalized_path(path) != path:
            raise WorkspaceStorageError("workspace_path_invalid")
        for parent in parent_paths(path):
            if parent not in tree.directories:
                raise WorkspaceStorageError("workspace_type_conflict")
    for text in tree.text_files.values():
        managed_bytes += len(_validate_managed_text(text, limits))
    if managed_bytes > limits.max_managed_text_bytes or managed_bytes > limits.max_expanded_bytes:
        raise WorkspaceStorageError("workspace_limit_exceeded")


def _require_parent(tree: ManagedWorkspaceTree, path: str) -> None:
    parents = parent_paths(path)
    if parents and tree.kind(parents[-1]) != "directory":
        raise WorkspaceStorageError("workspace_not_found")


def _subtree_paths(tree: ManagedWorkspaceTree, root: str) -> set[str]:
    return {path for path in tree.paths() if path == root or path.startswith(f"{root}/")}


def _remove_tree(tree: ManagedWorkspaceTree, root: str) -> None:
    selected = _subtree_paths(tree, root)
    tree.directories.difference_update(selected)
    for path in selected:
        tree.text_files.pop(path, None)
    tree.binary_paths.difference_update(selected)
    tree.stored_binary_paths.difference_update(selected)


def _copy_tree(
    tree: ManagedWorkspaceTree,
    source: str,
    destination: str,
    *,
    recursive: bool,
) -> None:
    kind = tree.kind(source)
    if kind is None:
        raise WorkspaceStorageError("workspace_not_found")
    selected = _subtree_paths(tree, source)
    if any(path in tree.binary_paths for path in selected):
        raise WorkspaceStorageError("binary_file_unsupported")
    if kind == "directory" and not recursive:
        raise WorkspaceStorageError("workspace_type_conflict")
    if tree.kind(destination) is not None:
        raise WorkspaceStorageError("workspace_type_conflict")
    if destination.startswith(f"{source}/"):
        raise WorkspaceStorageError("workspace_type_conflict")
    _require_parent(tree, destination)
    for path in sorted(selected, key=lambda value: (value.count("/"), value)):
        suffix = path[len(source) :]
        target = destination + suffix
        if tree.kind(target) is not None:
            raise WorkspaceStorageError("workspace_type_conflict")
        if path in tree.directories:
            tree.directories.add(target)
        elif path in tree.text_files:
            tree.text_files[target] = tree.text_files[path]


def _raise_workspace_unavailable() -> None:
    raise WorkspaceStorageError("workspace_unavailable")


def _atomic_local_text_write(content_root: str, relative: str, data: bytes) -> None:
    """Replace one local file without following a swapped final symlink.

    Each parent is opened relative to an already verified directory descriptor,
    so a path component changed into a symlink causes a safe failure instead of
    redirecting a direct-mode write outside the private workspace.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = []
    temporary = f".contractor-write-{secrets.token_hex(16)}"
    temporary_created = False
    try:
        current = os.open(content_root, flags)
        descriptors.append(current)
        parts = relative.split("/")
        for component in parts[:-1]:
            current = os.open(component, flags, dir_fd=current)
            descriptors.append(current)
        write_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        output = os.open(temporary, write_flags, 0o600, dir_fd=current)
        temporary_created = True
        try:
            view = memoryview(data)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    raise OSError("short workspace write")
                view = view[written:]
            os.fsync(output)
        finally:
            os.close(output)
        os.replace(temporary, parts[-1], src_dir_fd=current, dst_dir_fd=current)
        temporary_created = False
    finally:
        if descriptors and temporary_created:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=descriptors[-1])
        for descriptor in reversed(descriptors):
            os.close(descriptor)
