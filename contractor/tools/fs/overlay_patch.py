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
import contextlib
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
            with contextlib.suppress(FileNotFoundError):
                patch["base_hash"] = sha256_hex(_read_base_cached(path))
        patches.append(patch)

    # Creates / modifies (including base→visible type changes)
    for path in sorted(visible_paths):
        if path == root_marker:
            continue

        visible_info = visible_entries[path]
        visible_type = visible_info.get("type", "file")

        base_info = base_entries.get(path)
        base_type = base_info.get("type", "file") if base_info is not None else None

        # A path present in BOTH base and visible but with a different type
        # (dir↔file) must be deleted then recreated. Without this, save() either
        # crashes (dir→file hit "Type mismatch") or silently drops the change
        # (file→dir matched neither the delete nor the create branch) — so the
        # applied patch diverged from render_overlay_diff, which reports it.
        type_changed = base_info is not None and base_type != visible_type
        if type_changed:
            delete_patch: Patch = {
                "op": "delete_path",
                "path": path,
                "type": base_type,
            }
            if base_type == "file":
                with contextlib.suppress(FileNotFoundError):
                    delete_patch["base_hash"] = sha256_hex(_read_base_cached(path))
            patches.append(delete_patch)

        # A type-changed path is treated as brand new from here on.
        is_new = base_info is None or type_changed

        if visible_type == "directory":
            if is_new and effective_empty_dir(path):
                patches.append({"op": "create_dir", "path": path})
            continue

        current_bytes = read_effective_bytes(path)

        if is_new:
            patches.append(
                {
                    "op": "write_file",
                    "path": path,
                    "content_b64": b64encode(current_bytes),
                }
            )
            continue

        # Existing base file, possibly modified in place.
        if base_type != "file":
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
