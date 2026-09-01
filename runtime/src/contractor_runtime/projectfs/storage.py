"""Narrow backend-independent project workspace snapshots and handles."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import jcs

from contractor_runtime.contracts import WorkspaceModeV2
from contractor_runtime.projectfs.paths import normalize_project_path
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage
from contractor_runtime.settings import WorkspaceLimits


class WorkspaceStorageError(RuntimeError):
    """Stable storage/path failure without backend details."""


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
class WorkspaceDiff:
    text: str = field(repr=False)
    returned_bytes: int
    truncated: bool


class WorkspaceReader(Protocol):
    async def snapshot(self) -> WorkspaceSnapshot: ...

    async def read_text(self, path: str) -> str: ...


class WorkspaceWriter(Protocol):
    async def write_text(self, path: str, text: str) -> None: ...

    async def make_directory(self, path: str, *, parents: bool = False) -> None: ...

    async def delete_path(self, path: str, *, recursive: bool = False) -> None: ...


class WorkspaceReaderView:
    """Narrow model-tool handle with no backend or provider properties."""

    __slots__ = ("__reader",)

    def __init__(self, reader: WorkspaceReader) -> None:
        self.__reader = reader

    async def snapshot(self) -> WorkspaceSnapshot:
        return await self.__reader.snapshot()

    async def read_text(self, path: str) -> str:
        return await self.__reader.read_text(path)


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
    """One allocation-private effective tree backed by local or memory fsspec."""

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
        self._tree = ManagedWorkspaceTree(
            directories=set(directories),
            text_files=dict(text_files),
            binary_paths=set(binary_paths),
            stored_binary_paths=set(stored_binary_paths),
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
    def limits(self) -> WorkspaceLimits:
        return self._limits

    def reader_view(self) -> WorkspaceReader:
        self._require_open()
        return WorkspaceReaderView(self)

    async def snapshot(self) -> WorkspaceSnapshot:
        async with self._lock:
            self._require_open()
            return self._tree.snapshot()

    async def read_text(self, path: str) -> str:
        normalized = normalize_project_path(path, allow_root=False)
        async with self._lock:
            self._require_open()
            if normalized in self._tree.binary_paths:
                raise WorkspaceStorageError("binary_file_unsupported")
            try:
                return self._tree.text_files[normalized]
            except KeyError:
                raise WorkspaceStorageError("workspace_not_found") from None

    async def write_text(self, path: str, text: str) -> None:
        del path, text
        raise WorkspaceStorageError("workspace_operation_unsupported")

    async def make_directory(self, path: str, *, parents: bool = False) -> None:
        del path, parents
        raise WorkspaceStorageError("workspace_operation_unsupported")

    async def delete_path(self, path: str, *, recursive: bool = False) -> None:
        del path, recursive
        raise WorkspaceStorageError("workspace_operation_unsupported")

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._tree.directories.clear()
            self._tree.text_files.clear()
            self._tree.binary_paths.clear()
            self._tree.stored_binary_paths.clear()

    def _source_tree(self) -> ManagedWorkspaceTree:
        self._require_open()
        return self._tree.clone()

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
