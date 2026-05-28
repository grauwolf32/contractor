"""Patch (de)serialization for :class:`MemoryOverlayFileSystem`.

A *patch* is a deterministic, base-relative description of overlay changes
(``delete_path`` / ``create_dir`` / ``write_file``). :func:`build_overlay_patch`
renders one from the overlay's base-vs-visible state (pure, given byte-read
callbacks — mirrors :mod:`overlay_diff`); the overlay's ``load()`` applies it
back. The base64 / sha256 codec used for file payloads and base-hash guards
lives here too, shared by the patch + snapshot serializers.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Mapping
from typing import Any

Patch = dict[str, Any]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def build_overlay_patch(
    *,
    base_entries: Mapping[str, Any],
    visible_entries: Mapping[str, Any],
    root: str,
    root_marker: str,
    version: int,
    read_base_bytes: Callable[[str], bytes],
    read_effective_bytes: Callable[[str], bytes],
    effective_empty_dir: Callable[[str], bool],
) -> Patch:
    """Build a deterministic base-relative patch.

    ``base_entries`` / ``visible_entries`` map path → info dict (``type`` key).
    The callbacks read a path's bytes from the base filesystem and the effective
    (overlay) view, and test whether an overlay directory is empty. Emits
    ``delete_path`` for paths gone from the visible view, ``create_dir`` for new
    empty overlay dirs, and ``write_file`` for new/modified files (with a
    ``base_hash`` guard when overwriting an existing base file).
    """
    base_paths = set(base_entries)
    visible_paths = set(visible_entries)
    patches: list[Patch] = []

    _base_cache: dict[str, bytes] = {}

    def _read_base_cached(p: str) -> bytes:
        if p not in _base_cache:
            _base_cache[p] = read_base_bytes(p)
        return _base_cache[p]

    # Deletions
    for path in sorted(base_paths - visible_paths):
        if path == root_marker:
            continue

        base_info = base_entries[path]
        entry_type = base_info.get("type", "file")

        patch: Patch = {"op": "delete_path", "path": path, "type": entry_type}
        if entry_type == "file":
            try:
                patch["base_hash"] = sha256_hex(_read_base_cached(path))
            except FileNotFoundError:
                pass
        patches.append(patch)

    # Creates / modifies
    for path in sorted(visible_paths):
        if path == root_marker:
            continue

        visible_info = visible_entries[path]
        visible_type = visible_info.get("type", "file")

        if visible_type == "directory":
            if path not in base_paths and effective_empty_dir(path):
                patches.append({"op": "create_dir", "path": path})
            continue

        current_bytes = read_effective_bytes(path)

        if path not in base_paths:
            patches.append(
                {
                    "op": "write_file",
                    "path": path,
                    "content_b64": b64encode(current_bytes),
                }
            )
            continue

        base_info = base_entries[path]
        if base_info.get("type") != "file":
            raise RuntimeError(f"Type mismatch for {path}: base is not a file")

        base_bytes = _read_base_cached(path)
        if base_bytes != current_bytes:
            patches.append(
                {
                    "op": "write_file",
                    "path": path,
                    "base_hash": sha256_hex(base_bytes),
                    "content_b64": b64encode(current_bytes),
                }
            )

    return {
        "version": version,
        "kind": "overlay_patch",
        "root": root,
        "patches": patches,
    }
