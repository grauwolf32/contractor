"""Immutable process-level providers for disposable project workspaces."""

from __future__ import annotations

import asyncio
import shutil
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from contractor_runtime.contracts import (
    WorkspaceCapabilitiesV2,
    WorkspaceLimitsV2,
    WorkspaceModeV2,
    WorkspaceStorageV2,
)
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings

LOCAL_DIRECTORY_PREFIX = "workspace-"
LOCAL_OWNER_MARKER = ".contractor-workspace-owner"
_LOCAL_OWNER_PREFIX = "contractor-runtime-workspace-v1\n"
_MEMORY_ROOT = "/contractor-workspaces"


@dataclass(frozen=True, slots=True)
class WorkspaceCapabilitySnapshot:
    storage: WorkspaceStorageV2
    modes: tuple[WorkspaceModeV2, ...]
    limits: WorkspaceLimits

    def wire(self) -> WorkspaceCapabilitiesV2:
        return WorkspaceCapabilitiesV2(
            storage=self.storage,
            modes=list(self.modes),
            limits=WorkspaceLimitsV2(
                maxFiles=self.limits.max_files,
                maxExpandedBytes=self.limits.max_expanded_bytes,
                maxManagedTextBytes=self.limits.max_managed_text_bytes,
                maxFileBytes=self.limits.max_file_bytes,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProjectWorkspaceStorage:
    """Opaque provider-owned storage handle; physical locations stay private."""

    storage: WorkspaceStorageV2
    filesystem: AbstractFileSystem = field(repr=False)
    root: str = field(repr=False)
    provider_id: str = field(repr=False)
    owner_token: str = field(repr=False)


class WorkspaceProvider(Protocol):
    @property
    def capability(self) -> WorkspaceCapabilitySnapshot: ...

    async def probe(self) -> bool: ...

    async def create(self, allocation_id: str) -> ProjectWorkspaceStorage: ...

    async def cleanup(self, storage: ProjectWorkspaceStorage) -> None: ...


class LocalWorkspaceProvider:
    """Own marker-authenticated immediate children below one dedicated root."""

    def __init__(self, settings: WorkspaceSettings) -> None:
        if settings.storage != "local" or settings.work_root is None:
            raise ValueError("local workspace provider requires a local work root")
        self._root = settings.work_root
        self._provider_id = uuid.uuid4().hex
        self._capability = _capability(settings)
        self._filesystem = LocalFileSystem(auto_mkdir=False)
        self._initialization_lock = asyncio.Lock()
        self._initialized = False

    @property
    def capability(self) -> WorkspaceCapabilitySnapshot:
        return self._capability

    async def probe(self) -> bool:
        await self._initialize()
        storage = await self.create("workspace-capability-probe")
        await self.cleanup(storage)
        return True

    async def create(self, allocation_id: str) -> ProjectWorkspaceStorage:
        if not allocation_id.strip():
            raise ValueError("allocation ID is required")
        await self._initialize()
        owner_token = uuid.uuid4().hex
        path = self._root / f"{LOCAL_DIRECTORY_PREFIX}{owner_token}"
        if path.parent != self._root:
            raise RuntimeError("generated project workspace escaped its root")
        path.mkdir(mode=0o700)
        try:
            marker = path / LOCAL_OWNER_MARKER
            marker.write_text(_marker_contents(path.name), encoding="ascii", newline="\n")
            marker.chmod(0o600)
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, path, ignore_errors=True)
            raise
        return ProjectWorkspaceStorage(
            storage="local",
            filesystem=self._filesystem,
            root=str(path),
            provider_id=self._provider_id,
            owner_token=owner_token,
        )

    async def cleanup(self, storage: ProjectWorkspaceStorage) -> None:
        if storage.storage != "local" or storage.provider_id != self._provider_id:
            raise ValueError("workspace storage belongs to another provider")
        path = Path(storage.root)
        expected = f"{LOCAL_DIRECTORY_PREFIX}{storage.owner_token}"
        if path.parent != self._root or path.name != expected:
            raise ValueError("refusing to remove an unowned project workspace")
        await asyncio.to_thread(_remove_owned_local_workspace, path)

    async def _initialize(self) -> None:
        async with self._initialization_lock:
            if self._initialized:
                return
            await asyncio.to_thread(_initialize_and_cleanup_local_root, self._root)
            self._initialized = True


class MemoryWorkspaceProvider:
    """Give each allocation a unique prefix in a private fsspec instance."""

    def __init__(self, settings: WorkspaceSettings) -> None:
        if settings.storage != "memory" or settings.work_root is not None:
            raise ValueError("memory workspace provider does not accept a local work root")
        self._provider_id = uuid.uuid4().hex
        self._capability = _capability(settings)

    @property
    def capability(self) -> WorkspaceCapabilitySnapshot:
        return self._capability

    async def probe(self) -> bool:
        storage = await self.create("workspace-capability-probe")
        await self.cleanup(storage)
        return True

    async def create(self, allocation_id: str) -> ProjectWorkspaceStorage:
        if not allocation_id.strip():
            raise ValueError("allocation ID is required")
        owner_token = uuid.uuid4().hex
        root = f"{_MEMORY_ROOT}/{self._provider_id}/{owner_token}"
        filesystem = _IsolatedMemoryFileSystem(skip_instance_cache=True)
        filesystem.makedirs(root, exist_ok=False)
        return ProjectWorkspaceStorage(
            storage="memory",
            filesystem=filesystem,
            root=root,
            provider_id=self._provider_id,
            owner_token=owner_token,
        )

    async def cleanup(self, storage: ProjectWorkspaceStorage) -> None:
        expected = f"{_MEMORY_ROOT}/{self._provider_id}/{storage.owner_token}"
        if (
            storage.storage != "memory"
            or storage.provider_id != self._provider_id
            or storage.root != expected
        ):
            raise ValueError("workspace storage belongs to another provider")
        await asyncio.to_thread(_remove_memory_workspace, storage)


class _IsolatedMemoryFileSystem(MemoryFileSystem):
    """Override fsspec's class-global store for one allocation handle."""

    def __init__(self, **storage_options: object) -> None:
        super().__init__(**storage_options)
        self.store = {}
        self.pseudo_dirs = [""]


def build_workspace_provider(settings: WorkspaceSettings | None) -> WorkspaceProvider | None:
    if settings is None:
        return None
    if settings.storage == "local":
        return LocalWorkspaceProvider(settings)
    if settings.storage == "memory":
        return MemoryWorkspaceProvider(settings)
    raise ValueError("unsupported workspace storage")


def cleanup_stale_local_workspaces(root: Path) -> None:
    """Remove only immediate, regular, marker-owned allocation directories."""

    _initialize_local_root(root)
    for candidate in root.iterdir():
        if _is_owned_directory(candidate):
            shutil.rmtree(candidate)


def _initialize_and_cleanup_local_root(root: Path) -> None:
    cleanup_stale_local_workspaces(root)


def _remove_owned_local_workspace(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if not _is_owned_directory(path):
        raise ValueError("refusing to remove an unowned project workspace")
    shutil.rmtree(path)


def _remove_memory_workspace(storage: ProjectWorkspaceStorage) -> None:
    if storage.filesystem.exists(storage.root):
        storage.filesystem.rm(storage.root, recursive=True)


def _initialize_local_root(root: Path) -> None:
    if not root.is_absolute() or root == Path(root.anchor):
        raise ValueError("workspace work root must be absolute and non-root")
    if root.is_symlink():
        raise ValueError("workspace work root must not be a symlink")
    if root.exists() and not root.is_dir():
        raise ValueError("workspace work root must be a directory")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)


def _is_owned_directory(path: Path) -> bool:
    if not path.name.startswith(LOCAL_DIRECTORY_PREFIX) or path.is_symlink():
        return False
    token = path.name.removeprefix(LOCAL_DIRECTORY_PREFIX)
    if len(token) != 32 or any(character not in "0123456789abcdef" for character in token):
        return False
    try:
        directory_status = path.lstat()
        marker = path / LOCAL_OWNER_MARKER
        marker_status = marker.lstat()
        if not stat.S_ISDIR(directory_status.st_mode) or not stat.S_ISREG(marker_status.st_mode):
            return False
        if marker_status.st_size > 128:
            return False
        return marker.read_text(encoding="ascii") == _marker_contents(path.name)
    except (FileNotFoundError, OSError, UnicodeError):
        return False


def _marker_contents(directory_name: str) -> str:
    return f"{_LOCAL_OWNER_PREFIX}{directory_name}\n"


def _capability(settings: WorkspaceSettings) -> WorkspaceCapabilitySnapshot:
    return WorkspaceCapabilitySnapshot(
        storage=settings.storage,
        modes=("direct", "overlay"),
        limits=settings.limits,
    )
