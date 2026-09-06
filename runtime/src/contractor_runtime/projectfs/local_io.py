"""Bounded descriptor-relative access to one private local content directory.

These synchronous primitives belong behind WorkspaceOperationGuard. They never
follow symlinks, accept special files or trust a previous hydration snapshot.
"""

from __future__ import annotations

import errno
import math
import os
import secrets
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path
from contractor_runtime.settings import WorkspaceLimits

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CHUNK = 64 * 1024


@dataclass(frozen=True, slots=True)
class LocalEntry:
    path: str
    directory: bool
    size: int
    mode: int
    identity: tuple[int, int]
    version: tuple[int, int, int]


@dataclass(slots=True)
class LocalTree:
    entries: dict[str, LocalEntry] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict, repr=False)
    binary_paths: set[str] = field(default_factory=set)
    expanded_bytes: int = 0


class RootedLocalFilesystem:
    def __init__(self, root: Path, limits: WorkspaceLimits) -> None:
        if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
            raise WorkspaceStorageError("workspace_path_invalid")
        if any(
            value <= 0
            for value in (
                limits.max_files,
                limits.max_file_bytes,
                limits.max_expanded_bytes,
                limits.max_managed_text_bytes,
            )
        ):
            raise ValueError("workspace limits must be positive")
        self._root = root
        self.limits = limits
        self._identity: tuple[int, int] | None = None
        with self._opened_root() as descriptor:
            self._identity = _identity(os.fstat(descriptor))

    def stat(self, path: str, *, deadline: float) -> LocalEntry:
        path = _path(path, allow_root=True)
        with _safe_errors(), self._opened_root() as root:
            _check_deadline(deadline)
            if not path:
                return _entry(path, os.fstat(root))
            with _parent(root, path) as (parent, name):
                return _entry(path, os.stat(name, dir_fd=parent, follow_symlinks=False))

    def read(self, path: str, *, deadline: float) -> bytes:
        path = _path(path)
        with _safe_errors(), self._opened_root() as root:
            return self._read(root, path, deadline)

    def scan(self, *, deadline: float, path: str = "", contents: bool = True) -> LocalTree:
        path = _path(path, allow_root=True)
        with _safe_errors(), self._opened_root() as root:
            return self._scan(root, path, deadline, contents=contents)

    def write(self, path: str, data: bytes, *, deadline: float) -> None:
        path = _path(path)
        if len(data) > self.limits.max_file_bytes:
            raise WorkspaceStorageError("workspace_limit_exceeded")
        with _safe_errors(), self._opened_root() as root:
            self._write(root, path, data, deadline)

    def mkdir(self, path: str, *, deadline: float, parents: bool = False) -> None:
        path = _path(path)
        with _safe_errors(), self._opened_root() as root:
            # Validate every existing component before creating any directory.
            missing = False
            parts = path.split("/")
            for index in range(len(parts)):
                _check_deadline(deadline)
                current = "/".join(parts[: index + 1])
                if not missing:
                    with _parent(root, current) as (parent, name):
                        entry = _optional_entry(parent, name, current)
                    if entry is not None and not entry.directory:
                        raise WorkspaceStorageError("workspace_type_conflict")
                    missing = entry is None
                    if missing and index < len(parts) - 1 and not parents:
                        raise WorkspaceStorageError("workspace_not_found")
            for index in range(len(parts)):
                _check_deadline(deadline)
                current = "/".join(parts[: index + 1])
                with _parent(root, current) as (parent, name):
                    entry = _optional_entry(parent, name, current)
                    if entry is None:
                        os.mkdir(name, mode=0o700, dir_fd=parent)
                    elif not entry.directory:
                        raise WorkspaceStorageError("workspace_type_conflict")

    def remove(self, path: str, *, deadline: float, recursive: bool = False) -> None:
        path = _path(path)
        with _safe_errors(), self._opened_root() as root:
            tree = self._scan(root, path, deadline, contents=False)
            if len(tree.entries) > 1 and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            for item in sorted(
                tree.entries.values(), key=lambda e: e.path.count("/"), reverse=True
            ):
                _check_deadline(deadline)
                with _parent(root, item.path) as (parent, name):
                    _verify(
                        item, _entry(item.path, os.stat(name, dir_fd=parent, follow_symlinks=False))
                    )
                    if item.directory:
                        os.rmdir(name, dir_fd=parent)
                    else:
                        os.unlink(name, dir_fd=parent)

    def copy(
        self, source: str, destination: str, *, deadline: float, recursive: bool = False
    ) -> None:
        source, destination = _copy_paths(source, destination)
        with _safe_errors(), self._opened_root() as root:
            tree = self._scan(root, source, deadline, contents=False)
            if tree.entries[source].directory and not recursive:
                raise WorkspaceStorageError("workspace_type_conflict")
            with _parent(root, destination) as (parent, name):
                if _optional_entry(parent, name, destination) is not None:
                    raise WorkspaceStorageError("workspace_type_conflict")
            for item in sorted(tree.entries.values(), key=lambda e: (e.path.count("/"), e.path)):
                _check_deadline(deadline)
                target = destination + item.path[len(source) :]
                if item.directory:
                    with _parent(root, target) as (parent, name):
                        os.mkdir(name, mode=0o700, dir_fd=parent)
                else:
                    data = self._read(root, item.path, deadline, expected=item)
                    self._write(root, target, data, deadline, must_create=True)

    def move(self, source: str, destination: str, *, deadline: float) -> None:
        source, destination = _copy_paths(source, destination)
        with _safe_errors(), self._opened_root() as root:
            tree = self._scan(root, source, deadline, contents=False)
            with (
                _parent(root, source) as (src_parent, src_name),
                _parent(root, destination) as (dst_parent, dst_name),
            ):
                if _optional_entry(dst_parent, dst_name, destination) is not None:
                    raise WorkspaceStorageError("workspace_type_conflict")
                _verify(
                    tree.entries[source],
                    _entry(
                        source,
                        os.stat(
                            src_name,
                            dir_fd=src_parent,
                            follow_symlinks=False,
                        ),
                    ),
                )
                _check_deadline(deadline)
                os.rename(src_name, dst_name, src_dir_fd=src_parent, dst_dir_fd=dst_parent)

    def _read(
        self,
        root: int,
        path: str,
        deadline: float,
        *,
        expected: LocalEntry | None = None,
    ) -> bytes:
        _check_deadline(deadline)
        with _parent(root, path) as (parent, name):
            before = _entry(path, os.stat(name, dir_fd=parent, follow_symlinks=False))
            if before.directory:
                raise WorkspaceStorageError("workspace_type_conflict")
            if expected is not None:
                _verify(expected, before, version=True)
            if before.size > self.limits.max_file_bytes:
                raise WorkspaceStorageError("workspace_limit_exceeded")
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent)
            try:
                _verify(before, _entry(path, os.fstat(descriptor)), version=True)
                data = bytearray()
                while True:
                    _check_deadline(deadline)
                    chunk = os.read(
                        descriptor, min(_CHUNK, self.limits.max_file_bytes + 1 - len(data))
                    )
                    if not chunk:
                        break
                    data.extend(chunk)
                    if len(data) > self.limits.max_file_bytes:
                        raise WorkspaceStorageError("workspace_limit_exceeded")
                _verify(before, _entry(path, os.fstat(descriptor)), version=True)
                _verify(
                    before,
                    _entry(path, os.stat(name, dir_fd=parent, follow_symlinks=False)),
                    version=True,
                )
                return bytes(data)
            finally:
                os.close(descriptor)

    def _scan(self, root: int, path: str, deadline: float, *, contents: bool) -> LocalTree:
        tree = LocalTree()
        managed_bytes = 0

        def add(item: LocalEntry) -> None:
            nonlocal managed_bytes
            _check_deadline(deadline)
            tree.entries[item.path] = item
            if len(tree.entries) > self.limits.max_files:
                raise WorkspaceStorageError("workspace_limit_exceeded")
            if item.directory:
                return
            tree.expanded_bytes += item.size
            if (
                item.size > self.limits.max_file_bytes
                or tree.expanded_bytes > self.limits.max_expanded_bytes
            ):
                raise WorkspaceStorageError("workspace_limit_exceeded")
            if contents:
                data = self._read(root, item.path, deadline, expected=item)
                try:
                    text = data.decode("utf-8")
                except UnicodeError:
                    tree.binary_paths.add(item.path)
                    return
                if "\x00" in text:
                    tree.binary_paths.add(item.path)
                    return
                managed_bytes += len(data)
                if managed_bytes > self.limits.max_managed_text_bytes:
                    raise WorkspaceStorageError("workspace_limit_exceeded")
                tree.texts[item.path] = text

        def walk(descriptor: int, prefix: str) -> None:
            _check_deadline(deadline)
            before = os.fstat(descriptor)
            # scandir is incremental; never allocate an unbounded listdir.
            with os.scandir(descriptor) as entries:
                for child in entries:
                    _check_deadline(deadline)
                    relative = f"{prefix}/{child.name}" if prefix else child.name
                    if _path(relative) != relative:
                        raise WorkspaceStorageError("workspace_path_invalid")
                    item = _entry(relative, child.stat(follow_symlinks=False))
                    add(item)
                    if item.directory:
                        nested = os.open(child.name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                        try:
                            _verify(item, _entry(relative, os.fstat(nested)))
                            walk(nested, relative)
                        finally:
                            os.close(nested)
            if (before.st_mtime_ns, before.st_ctime_ns) != (
                os.fstat(descriptor).st_mtime_ns,
                os.fstat(descriptor).st_ctime_ns,
            ):
                raise WorkspaceStorageError("workspace_unavailable")

        if path:
            with _parent(root, path) as (parent, name):
                item = _entry(path, os.stat(name, dir_fd=parent, follow_symlinks=False))
                add(item)
                if item.directory:
                    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent)
                    try:
                        _verify(item, _entry(path, os.fstat(descriptor)))
                        walk(descriptor, path)
                    finally:
                        os.close(descriptor)
        else:
            walk(root, "")
        return tree

    def _write(
        self, root: int, path: str, data: bytes, deadline: float, *, must_create: bool = False
    ) -> None:
        with _parent(root, path) as (parent, name):
            before = _optional_entry(parent, name, path)
            if before is not None and (before.directory or must_create):
                raise WorkspaceStorageError("workspace_type_conflict")
            _check_deadline(deadline)
            temporary = f".contractor-write-{secrets.token_hex(16)}"
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent,
            )
            try:
                view = memoryview(data)
                while view:
                    _check_deadline(deadline)
                    written = os.write(descriptor, view[:_CHUNK])
                    if written <= 0:
                        raise WorkspaceStorageError("workspace_unavailable")
                    view = view[written:]
                if before is not None:
                    os.fchmod(descriptor, stat.S_IMODE(before.mode) & 0o777)
                os.fsync(descriptor)
                _check_deadline(deadline)
                current = _optional_entry(parent, name, path)
                if before is None:
                    if current is not None:
                        raise WorkspaceStorageError("workspace_unavailable")
                elif current is None:
                    raise WorkspaceStorageError("workspace_unavailable")
                else:
                    _verify(before, current, version=True)
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                os.close(descriptor)
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=parent)

    @contextmanager
    def _opened_root(self) -> Iterator[int]:
        with _safe_errors():
            descriptor = os.open(self._root.anchor, _DIRECTORY_FLAGS)
            try:
                for component in self._root.parts[1:]:
                    nested = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = nested
                if self._identity is not None and _identity(os.fstat(descriptor)) != self._identity:
                    raise WorkspaceStorageError("workspace_unavailable")
                yield descriptor
            finally:
                os.close(descriptor)


@contextmanager
def _parent(root: int, path: str) -> Iterator[tuple[int, str]]:
    descriptor = os.dup(root)
    try:
        parts = path.split("/")
        for component in parts[:-1]:
            nested = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = nested
        yield descriptor, parts[-1]
    finally:
        os.close(descriptor)


def _entry(path: str, info: os.stat_result) -> LocalEntry:
    directory = stat.S_ISDIR(info.st_mode)
    if not directory and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
        raise WorkspaceStorageError("workspace_type_conflict")
    return LocalEntry(
        path,
        directory,
        info.st_size if not directory else 0,
        info.st_mode,
        _identity(info),
        (info.st_size, info.st_mtime_ns, info.st_ctime_ns),
    )


def _optional_entry(parent: int, name: str, path: str) -> LocalEntry | None:
    try:
        return _entry(path, os.stat(name, dir_fd=parent, follow_symlinks=False))
    except FileNotFoundError:
        return None


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _verify(expected: LocalEntry, actual: LocalEntry, *, version: bool = False) -> None:
    if (
        expected.identity != actual.identity
        or expected.directory != actual.directory
        or (version and expected.version != actual.version)
    ):
        raise WorkspaceStorageError("workspace_unavailable")


def _path(value: str, *, allow_root: bool = False) -> str:
    try:
        return normalize_project_path(value, allow_root=allow_root)
    except ProjectPathError:
        raise WorkspaceStorageError("workspace_path_invalid") from None


def _copy_paths(source: str, destination: str) -> tuple[str, str]:
    source, destination = _path(source), _path(destination)
    if destination == source or destination.startswith(source + "/"):
        raise WorkspaceStorageError("workspace_type_conflict")
    return source, destination


def _check_deadline(deadline: float) -> None:
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise WorkspaceStorageError("workspace_unavailable")


@contextmanager
def _safe_errors() -> Iterator[None]:
    try:
        yield
    except OSError as error:
        if error.errno == errno.ENOENT:
            code = "workspace_not_found"
        elif error.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
            errno.EISDIR,
            errno.EEXIST,
            errno.ENOTEMPTY,
        }:
            code = "workspace_type_conflict"
        else:
            code = "workspace_unavailable"
        raise WorkspaceStorageError(code) from None
