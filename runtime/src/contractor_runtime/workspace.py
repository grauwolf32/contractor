"""Safe lifecycle for allocation-local work directories."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

ALLOCATION_DIRECTORY_PREFIX = "allocation-"
# Sibling of the directory it authorizes, so allocation tools cannot see,
# change or delete it through their scratch directory.
ALLOCATION_OWNER_SUFFIX = ".contractor-owner"
_ALLOCATION_OWNER_PREFIX = "contractor-runtime-allocation-v1\n"


def _resolved_work_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("work root must not be a filesystem root")
    return resolved


@dataclass(frozen=True, slots=True)
class AllocationWorkspace:
    """One internally named directory below a dedicated Runtime Agent root."""

    root: Path
    path: Path


class LocalWorkdirFactory:
    """Implements the first-slice ``local-workdir@1`` sandbox profile."""

    ref = "local-workdir@1"

    def __init__(self, root: Path) -> None:
        self._root = _resolved_work_root(root)

    async def probe(self) -> bool:
        workspace = await self.prepare()
        await self.cleanup(workspace)
        return True

    async def prepare(self) -> AllocationWorkspace:
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._root / f"{ALLOCATION_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        if path.parent != self._root:
            raise RuntimeError("generated allocation workspace escaped its root")
        # Marker first: a crash in between leaves a removable marker, never an
        # unowned directory that startup cleanup must not touch.
        _write_owner_marker(self._root, path.name)
        try:
            path.mkdir(mode=0o700)
        except BaseException:
            _remove_owner_marker(self._root, path.name)
            raise
        return AllocationWorkspace(root=self._root, path=path)

    async def cleanup(self, workspace: AllocationWorkspace) -> None:
        root = _resolved_work_root(workspace.root)
        path = workspace.path
        if (
            root != self._root
            or path.parent != root
            or not path.name.startswith(ALLOCATION_DIRECTORY_PREFIX)
            or path == root
        ):
            raise ValueError("refusing to clean an unrecognized allocation workspace")
        await asyncio.to_thread(_remove_owned_workdir, root, path.name)


def cleanup_orphan_workdirs(root: Path) -> None:
    """Remove only allocation directories this Runtime marked as its own.

    ``allocation-*`` entries without a valid sibling owner marker may belong
    to an operator or another service and are never touched.
    """

    resolved = _resolved_work_root(root)
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    for entry in list(resolved.iterdir()):
        name = entry.name.removesuffix(ALLOCATION_OWNER_SUFFIX)
        if name != entry.name and _is_allocation_name(name) and _owns(resolved, name):
            _remove_owned_workdir(resolved, name)


def _is_allocation_name(name: str) -> bool:
    token = name.removeprefix(ALLOCATION_DIRECTORY_PREFIX)
    return (
        token != name
        and len(token) == 32
        and all(character in "0123456789abcdef" for character in token)
    )


def _owner_contents(name: str) -> bytes:
    return f"{_ALLOCATION_OWNER_PREFIX}{name}\n".encode("ascii")


def _write_owner_marker(root: Path, name: str) -> None:
    descriptor = os.open(
        root / f"{name}{ALLOCATION_OWNER_SUFFIX}",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        contents = _owner_contents(name)
        if os.write(descriptor, contents) != len(contents):
            raise OSError("incomplete allocation owner marker")
    except BaseException:
        os.close(descriptor)
        _remove_owner_marker(root, name)
        raise
    os.close(descriptor)


def _owns(root: Path, name: str) -> bool:
    try:
        descriptor = os.open(
            root / f"{name}{ALLOCATION_OWNER_SUFFIX}",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        )
    except OSError:
        return False
    try:
        info = os.fstat(descriptor)
        expected = _owner_contents(name)
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_size == len(expected)
            and os.read(descriptor, len(expected) + 1) == expected
        )
    except OSError:
        return False
    finally:
        os.close(descriptor)


def _remove_owner_marker(root: Path, name: str) -> None:
    with suppress(FileNotFoundError):
        (root / f"{name}{ALLOCATION_OWNER_SUFFIX}").unlink()


def _remove_owned_workdir(root: Path, name: str) -> None:
    # The marker goes last so that a failed removal stays recognizable.
    _remove_workspace_path(root / name)
    _remove_owner_marker(root, name)


def _remove_workspace_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    shutil.rmtree(path)
