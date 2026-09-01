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


class WorkspaceReader(Protocol):
    async def snapshot(self) -> WorkspaceSnapshot: ...

    async def read_text(self, path: str) -> str: ...


class WorkspaceWriter(Protocol):
    async def write_text(self, path: str, text: str) -> None: ...


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
        self._directories = set(directories)
        self._text_files = dict(text_files)
        self._binary_paths = set(binary_paths)
        self._stored_binary_paths = set(stored_binary_paths)
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

    async def snapshot(self) -> WorkspaceSnapshot:
        async with self._lock:
            self._require_open()
            return _snapshot(self._directories, self._text_files, self._binary_paths)

    async def read_text(self, path: str) -> str:
        normalized = normalize_project_path(path, allow_root=False)
        async with self._lock:
            self._require_open()
            if normalized in self._binary_paths:
                raise WorkspaceStorageError("binary_file_unsupported")
            try:
                return self._text_files[normalized]
            except KeyError:
                raise WorkspaceStorageError("workspace_not_found") from None

    async def write_text(self, path: str, text: str) -> None:
        del path, text
        raise WorkspaceStorageError("workspace_operation_unsupported")

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            self._directories.clear()
            self._text_files.clear()
            self._binary_paths.clear()
            self._stored_binary_paths.clear()

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
