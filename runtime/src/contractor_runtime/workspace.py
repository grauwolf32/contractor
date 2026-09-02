"""Safe lifecycle for allocation-local work directories."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

ALLOCATION_DIRECTORY_PREFIX = "allocation-"


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
        path.mkdir(mode=0o700)
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
        if not path.exists() and not path.is_symlink():
            return
        await asyncio.to_thread(_remove_workspace_path, path)


def cleanup_orphan_workdirs(root: Path) -> None:
    resolved = _resolved_work_root(root)
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    for entry in resolved.iterdir():
        if not entry.name.startswith(ALLOCATION_DIRECTORY_PREFIX):
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)


def _remove_workspace_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    shutil.rmtree(path)
