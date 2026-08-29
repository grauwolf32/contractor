"""Safe startup cleanup for allocation-local work directories."""

from __future__ import annotations

import shutil
from pathlib import Path

ALLOCATION_DIRECTORY_PREFIX = "allocation-"


def cleanup_orphan_workdirs(root: Path) -> None:
    resolved = root.expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("work root must not be a filesystem root")
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    for entry in resolved.iterdir():
        if not entry.name.startswith(ALLOCATION_DIRECTORY_PREFIX):
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)
