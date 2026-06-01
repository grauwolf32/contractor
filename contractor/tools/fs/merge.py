"""Merge multiple ``MemoryOverlayFileSystem`` forks that diverged from a
common snapshot.

Typical flow::

    patch = overlay.save()
    pre = dict(overlay._files)
    forks = [fork_overlay(base_fs, patch) for _ in range(N)]
    # … run agents on each fork concurrently …
    conflicts = merge_overlay_forks(overlay, forks, pre)
"""

from __future__ import annotations

import logging
from typing import Sequence

from fsspec import AbstractFileSystem

from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem

logger = logging.getLogger(__name__)


def fork_overlay(
    base_fs: AbstractFileSystem,
    patch: dict | None,
) -> MemoryOverlayFileSystem:
    """Create an independent overlay fork from a saved patch.

    The returned overlay wraps the same *read-only* ``base_fs`` and
    replays ``patch`` (the output of ``MemoryOverlayFileSystem.save()``)
    so it starts from the same logical state as the original overlay at
    snapshot time.
    """
    fork = MemoryOverlayFileSystem(base_fs, skip_instance_cache=True)
    if patch and patch.get("patches"):
        fork.load(patch)
    return fork


def merge_overlay_forks(
    target: MemoryOverlayFileSystem,
    forks: Sequence[MemoryOverlayFileSystem],
    pre_fork_files: dict[str, bytes],
) -> list[str]:
    """Merge writes produced by parallel forks back into *target*.

    Only changes made *after* the fork point are considered: a file is
    "new work" when it appears in ``fork._files`` but either didn't
    exist in ``pre_fork_files`` or has different content.

    Conflict resolution (same path written by >1 fork): the version
    with the most bytes wins. For trace annotations this is a good
    heuristic — the longest file has the most ``@trace`` comments.
    All conflicts are logged and returned.

    Parameters
    ----------
    target:
        The overlay to merge into (typically the workflow's main overlay).
    forks:
        The independently-run overlay forks.
    pre_fork_files:
        ``dict(target._files)`` captured **before** forking. Used to
        distinguish pre-existing content from new work.

    Returns
    -------
    list[str]
        Paths where more than one fork produced different content
        (conflicts). Empty when all forks touched disjoint files.
    """
    new_writes: dict[str, list[tuple[int, bytes]]] = {}

    for idx, fork in enumerate(forks):
        for path, content in fork._files.items():
            pre = pre_fork_files.get(path)
            if pre is None or content != pre:
                new_writes.setdefault(path, []).append((idx, content))

    conflicts: list[str] = []

    with target._lock:
        for path, versions in new_writes.items():
            if len(versions) == 1:
                target._files[path] = versions[0][1]
            else:
                unique = {v[1] for v in versions}
                if len(unique) == 1:
                    target._files[path] = versions[0][1]
                else:
                    conflicts.append(path)
                    best = max(versions, key=lambda v: len(v[1]))
                    target._files[path] = best[1]
                    logger.warning(
                        "Overlay merge conflict on %s: %d forks wrote different "
                        "content, taking version from fork %d (%d bytes)",
                        path,
                        len(versions),
                        best[0],
                        len(best[1]),
                    )

        for fork in forks:
            target._dirs.update(fork._dirs)
            new_deletes = fork._deleted - set(pre_fork_files)
            target._deleted.update(new_deletes)

    return conflicts
