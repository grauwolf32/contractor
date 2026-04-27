from __future__ import annotations

import base64
import hashlib
import io
import posixpath
import threading
import difflib

from copy import deepcopy
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Iterable, Iterator

from fsspec.spec import AbstractFileSystem

from contractor.tools.fs.models import FsEntry


FileInfo = dict[str, Any]
Patch = dict[str, Any]


def _immediate_child(parent: str, candidate: str, norm_fn) -> str | None:
    """Return the immediate child path of parent that contains candidate, or None."""
    if candidate in {parent, "/"}:
        return None

    rel = posixpath.relpath(candidate, parent)

    if rel in {".", ".."} or rel.startswith("../"):
        return None

    first = rel.split("/", 1)[0]
    if not first or first == ".":
        return None

    if parent == "/":
        return norm_fn("/" + first)

    return norm_fn(posixpath.join(parent, first))


def _safe_base_find(base_fs: AbstractFileSystem, path: str) -> list[str]:
    """
    Best-effort recursive listing of all paths at/under *path* in base_fs.

    Some filesystem implementations have incompatible ``find`` / ``walk``
    signatures so we try several strategies and return whatever we can
    collect.  An empty list is returned when nothing works.
    """
    # Strategy 1: find(withdirs=True, detail=False) — the most common API
    try:
        result = base_fs.find(path, withdirs=True, detail=False)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.keys())
    except Exception:
        pass

    # Strategy 2: find(detail=False) without withdirs
    try:
        result = base_fs.find(path, detail=False)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.keys())
    except Exception:
        pass

    # Strategy 3: iterative walk using ls
    try:
        collected: list[str] = []
        stack = [path]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            collected.append(current)
            try:
                entries = base_fs.ls(current, detail=True)
                for entry in entries:
                    name = entry.get("name", "")
                    if name and name not in visited:
                        if entry.get("type") == "directory":
                            stack.append(name)
                        collected.append(name)
            except Exception:
                pass
        return collected
    except Exception:
        pass

    return []


class MemoryOverlayFileSystem(AbstractFileSystem):
    """
    In-memory overlay over another fsspec filesystem.

    Rules:
    - `base_fs` is treated as read-only from the overlay perspective
    - all writes go only into memory
    - new files/directories can be created in the overlay
    - reads check the overlay first, then fall back to base_fs
    - deleting an overlay file only affects the overlay
    - deleting a base_fs path creates a tombstone in the overlay
    """

    protocol = "overlay"
    root_marker = "/"
    PATCH_VERSION = 1

    def __init__(self, fs: AbstractFileSystem, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_fs = fs
        self._lock = threading.RLock()

        # Overlay state
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = {self.root_marker}
        self._deleted: set[str] = set()

        # Optional in-memory snapshot
        self._snapshot_state: dict[str, Any] | None = None

    def _invalidate_filetype(self, path: str | None = None) -> None:
        """Drop any cached magika identification for *path* on this overlay."""
        FsEntry.invalidate_filetype_cache(self, path)

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------

    @classmethod
    def _strip_protocol(cls, path: str) -> str:
        if not path:
            return cls.root_marker

        if path.startswith("overlay://"):
            path = path[len("overlay://") :]

        if not path.startswith("/"):
            path = "/" + path

        return posixpath.normpath(path)

    def _norm(self, path: str) -> str:
        normalized = self._strip_protocol(path)
        return normalized or self.root_marker

    def _parent(self, path: str) -> str:
        path = self._norm(path)
        if path == self.root_marker:
            return self.root_marker

        parent = posixpath.dirname(path)
        return parent or self.root_marker

    def _ensure_parent_dirs(self, path: str) -> None:
        parts_to_create: list[str] = []
        current = self._parent(path)

        while current != self.root_marker:
            parts_to_create.append(current)
            current = self._parent(current)

        # Reject writes whose parent chain crosses an existing file.
        for ancestor in parts_to_create:
            if ancestor in self._files:
                raise NotADirectoryError(
                    f"Cannot create entry at {path}: {ancestor} is a file"
                )
            if (
                ancestor not in self._dirs
                and not self._is_deleted(ancestor)
                and self._base_isfile(ancestor)
            ):
                raise NotADirectoryError(
                    f"Cannot create entry at {path}: {ancestor} is a file"
                )

        for dir_path in reversed(parts_to_create):
            self._dirs.add(dir_path)
            # Ensure tombstones don't hide newly created parents
            self._deleted.discard(dir_path)

        self._dirs.add(self.root_marker)

    def _check_not_existing_dir(self, path: str) -> None:
        """Raise IsADirectoryError if *path* currently resolves to a directory."""
        if path in self._dirs:
            raise IsADirectoryError(path)
        if (
            path not in self._files
            and not self._is_deleted(path)
            and self._base_isdir(path)
        ):
            raise IsADirectoryError(path)

    def _check_not_existing_file(self, path: str) -> None:
        """Raise FileExistsError if *path* currently resolves to a file."""
        if path in self._files:
            raise FileExistsError(path)
        if (
            path not in self._dirs
            and not self._is_deleted(path)
            and self._base_isfile(path)
        ):
            raise FileExistsError(path)

    # -------------------------------------------------------------------------
    # Overlay visibility / tombstones
    # -------------------------------------------------------------------------

    def _is_deleted(self, path: str) -> bool:
        path = self._norm(path)
        if path in self._deleted:
            return True
        # Walk up ancestors to check for recursive deletes
        current = path
        while current != self.root_marker:
            current = self._parent(current)
            if current in self._deleted:
                return True
        return False

    def _unhide_path(self, path: str) -> None:
        path = self._norm(path)
        prefix = path.rstrip("/") + "/"
        self._deleted = {
            deleted
            for deleted in self._deleted
            if deleted != path and not deleted.startswith(prefix)
        }

    def _overlay_isfile(self, path: str) -> bool:
        return self._norm(path) in self._files

    def _overlay_isdir(self, path: str) -> bool:
        return self._norm(path) in self._dirs

    # -------------------------------------------------------------------------
    # Base FS helpers
    # -------------------------------------------------------------------------

    def _base_exists(self, path: str) -> bool:
        try:
            return bool(self.base_fs.exists(path))
        except FileNotFoundError:
            return False

    def _base_isfile(self, path: str) -> bool:
        try:
            return bool(self.base_fs.isfile(path))
        except FileNotFoundError:
            return False
        except Exception:
            try:
                return self.base_fs.info(path).get("type") == "file"
            except FileNotFoundError:
                return False

    def _base_isdir(self, path: str) -> bool:
        try:
            return bool(self.base_fs.isdir(path))
        except FileNotFoundError:
            return False
        except Exception:
            try:
                return self.base_fs.info(path).get("type") == "directory"
            except FileNotFoundError:
                return False

    def _base_read_bytes(self, path: str) -> bytes:
        path = self._norm(path)
        with self.base_fs.open(path, "rb") as f:
            return f.read()

    def _base_empty_dir(self, path: str) -> bool:
        path = self._norm(path)

        if not self._base_isdir(path):
            return False

        try:
            return len(self.base_fs.ls(path, detail=False)) == 0
        except FileNotFoundError:
            return False

    # -------------------------------------------------------------------------
    # Generic helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _text_encoding(kwargs: dict[str, Any]) -> str:
        return kwargs.get("encoding") or "utf-8"

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _b64encode(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii")

    @staticmethod
    def _b64decode(data: str) -> bytes:
        return base64.b64decode(data.encode("ascii"))

    def _effective_read_bytes(self, path: str) -> bytes:
        return self.cat_file(path)

    def _effective_empty_dir(self, path: str) -> bool:
        path = self._norm(path)

        if not self.isdir(path):
            return False

        try:
            return len(self.ls(path, detail=False)) == 0
        except FileNotFoundError:
            return False

    def _directory_info(self, path: str) -> FileInfo:
        return {
            "name": path,
            "size": 0,
            "type": "directory",
            "created": None,
            "mtime": None,
        }

    def _file_info(self, path: str, data: bytes) -> FileInfo:
        return {
            "name": path,
            "size": len(data),
            "type": "file",
            "created": None,
            "mtime": None,
        }

    # -------------------------------------------------------------------------
    # Tree snapshots
    # -------------------------------------------------------------------------

    def _base_find_detail(self, root: str = "/") -> dict[str, FileInfo]:
        """
        Best-effort recursive listing of base_fs with detail.
        """
        root = self._norm(root)

        try:
            found = self.base_fs.find(root, withdirs=True, detail=True)
            if isinstance(found, dict):
                result: dict[str, FileInfo] = {}

                for path, info in found.items():
                    normalized_path = self._norm(path)
                    normalized_info = dict(info)
                    normalized_info["name"] = normalized_path
                    result[normalized_path] = normalized_info

                if root not in result and self._base_exists(root):
                    try:
                        result[root] = dict(self.base_fs.info(root))
                        result[root]["name"] = root
                    except FileNotFoundError:
                        pass

                return result
        except FileNotFoundError:
            return {}
        except Exception:
            pass

        # Fallback: iterative walk using ls
        result = {}

        try:
            if self._base_exists(root):
                info = dict(self.base_fs.info(root))
                info["name"] = root
                result[root] = info
        except FileNotFoundError:
            return result

        stack = [root]
        visited: set[str] = set()

        while stack:
            current = stack.pop()
            current = self._norm(current)

            if current in visited:
                continue
            visited.add(current)

            if current not in result:
                try:
                    info = dict(self.base_fs.info(current))
                except FileNotFoundError:
                    info = {"name": current, "type": "directory", "size": 0}
                info["name"] = current
                result[current] = info

            try:
                entries = self.base_fs.ls(current, detail=True)
            except FileNotFoundError:
                continue
            except Exception:
                continue

            for entry in entries:
                entry = dict(entry)
                name = self._norm(entry.get("name", ""))
                if not name or name in visited:
                    continue
                entry["name"] = name
                result[name] = entry

                if entry.get("type") == "directory":
                    stack.append(name)

        return result

    def _visible_find_detail(self, root: str = "/") -> dict[str, FileInfo]:
        """
        Recursive listing of the effective tree visible through the overlay.
        """
        root = self._norm(root)

        if not self.exists(root):
            return {}

        if self.isfile(root):
            return {root: self.info(root)}

        result: dict[str, FileInfo] = {root: self.info(root)}
        for path, info in self.find(root, withdirs=True, detail=True).items():
            result[self._norm(path)] = info

        return dict(sorted(result.items()))

    # -------------------------------------------------------------------------
    # Snapshot / state export
    # -------------------------------------------------------------------------

    def _current_overlay_state(self) -> dict[str, Any]:
        return {
            "files": {
                path: self._b64encode(content)
                for path, content in sorted(self._files.items())
            },
            "dirs": sorted(path for path in self._dirs if path != self.root_marker),
            "deleted": sorted(self._deleted),
        }

    def reset_overlay(self) -> None:
        """
        Reset all overlay changes and return to the pure base_fs view.
        """
        with self._lock:
            self._files.clear()
            self._dirs = {self.root_marker}
            self._deleted.clear()
            self._invalidate_filetype()

    def snapshot(self) -> dict[str, Any]:
        """
        Save the current overlay state in memory.
        """
        with self._lock:
            self._snapshot_state = self._current_overlay_state()
            return deepcopy(self._snapshot_state)

    def restore_snapshot(self, snapshot: dict[str, Any] | None = None) -> None:
        """
        Restore overlay state from snapshot().
        """
        with self._lock:
            state = snapshot if snapshot is not None else self._snapshot_state
            if state is None:
                raise ValueError("No snapshot to restore")

            self.reset_overlay()
            self._dirs.update(self._norm(path) for path in state.get("dirs", []))
            self._deleted.update(self._norm(path) for path in state.get("deleted", []))
            self._files = {
                self._norm(path): self._b64decode(content)
                for path, content in state.get("files", {}).items()
            }
            self._invalidate_filetype()

    def export_overlay_state(self) -> dict[str, Any]:
        """
        Serialize the internal overlay state.
        This is not a diff, but a full snapshot of internal state.
        """
        with self._lock:
            return {
                "version": self.PATCH_VERSION,
                "kind": "overlay_state",
                "state": self._current_overlay_state(),
            }

    def import_overlay_state(self, payload: dict[str, Any]) -> None:
        """
        Restore overlay state from export_overlay_state().
        """
        if payload.get("kind") != "overlay_state":
            raise ValueError("Unsupported overlay state payload")

        self.restore_snapshot(payload["state"])

    def restore(self, path: str, *, recursive: bool = True) -> None:
        """
        Restore a path back to the original base_fs view.

        This removes any overlay-only files/directories for ``path`` and clears
        tombstones so the underlay version becomes visible again.  The target
        path must exist in ``base_fs``.
        """
        with self._lock:
            path = self._norm(path)

            if not self._base_exists(path):
                raise FileNotFoundError(f"No base filesystem entry exists for {path}")

            if path == self.root_marker:
                self.reset_overlay()
                return

            prefix = path.rstrip("/") + "/"

            # Drop any overlay materialized files at/under the path.
            for file_path in list(self._files):
                if file_path == path or (recursive and file_path.startswith(prefix)):
                    del self._files[file_path]

            # Drop overlay-created directories at/under the path.
            for dir_path in sorted(
                self._dirs,
                key=lambda item: (item.count("/"), item),
                reverse=True,
            ):
                if dir_path == self.root_marker:
                    continue
                if dir_path == path or (recursive and dir_path.startswith(prefix)):
                    self._dirs.discard(dir_path)

            # Remove tombstones so the base view becomes visible again.
            self._unhide_path(path)

            # Ensure ancestors created in the overlay remain valid.
            self._dirs.add(self.root_marker)

            self._invalidate_filetype(path)

    # -------------------------------------------------------------------------
    # Patch save / load
    # -------------------------------------------------------------------------

    def save(self, root: str = "/") -> Patch:
        """
        Build a deterministic patch relative to base_fs.

        The patch contains the following operations:
        - create_dir
        - write_file
        - delete_path
        """
        with self._lock:
            root = self._norm(root)

            base_entries = self._base_find_detail(root)
            visible_entries = self._visible_find_detail(root)

            base_paths = set(base_entries)
            visible_paths = set(visible_entries)

            patches: list[Patch] = []

            # Cache for base file reads to avoid redundant I/O
            _base_cache: dict[str, bytes] = {}

            def _read_base_cached(p: str) -> bytes:
                if p not in _base_cache:
                    _base_cache[p] = self._base_read_bytes(p)
                return _base_cache[p]

            # Deletions
            for path in sorted(base_paths - visible_paths):
                if path == self.root_marker:
                    continue

                base_info = base_entries[path]
                entry_type = base_info.get("type", "file")

                patch: Patch = {
                    "op": "delete_path",
                    "path": path,
                    "type": entry_type,
                }

                if entry_type == "file":
                    try:
                        patch["base_hash"] = self._sha256_bytes(_read_base_cached(path))
                    except FileNotFoundError:
                        pass

                patches.append(patch)

            # Creates / modifies
            for path in sorted(visible_paths):
                if path == self.root_marker:
                    continue

                visible_info = visible_entries[path]
                visible_type = visible_info.get("type", "file")

                if visible_type == "directory":
                    if path not in base_paths and self._effective_empty_dir(path):
                        patches.append({"op": "create_dir", "path": path})
                    continue

                current_bytes = self._effective_read_bytes(path)

                if path not in base_paths:
                    patches.append(
                        {
                            "op": "write_file",
                            "path": path,
                            "content_b64": self._b64encode(current_bytes),
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
                            "base_hash": self._sha256_bytes(base_bytes),
                            "content_b64": self._b64encode(current_bytes),
                        }
                    )

            return {
                "version": self.PATCH_VERSION,
                "kind": "overlay_patch",
                "root": root,
                "patches": patches,
            }

    def load(self, patch: Patch, *, reset: bool = True) -> None:
        """
        Apply a patch produced by save().

        By default, the overlay is reset first.
        """
        if patch.get("kind") != "overlay_patch":
            raise ValueError("Unsupported patch payload")

        with self._lock:
            if reset:
                self.reset_overlay()

            for item in patch.get("patches", []):
                op = item["op"]
                path = self._norm(item["path"])

                if op == "create_dir":
                    self.makedirs(path, exist_ok=True)
                    continue

                if op == "delete_path":
                    expected_type = item.get("type")

                    if self.exists(path):
                        if expected_type == "directory":
                            self.rm(path, recursive=True)
                        else:
                            self.rm(path)
                    elif self._base_exists(path):
                        self._deleted.add(path)

                    continue

                if op == "write_file":
                    base_hash = item.get("base_hash")
                    if (
                        base_hash is not None
                        and self._base_exists(path)
                        and self._base_isfile(path)
                    ):
                        actual_hash = self._sha256_bytes(self._base_read_bytes(path))
                        if actual_hash != base_hash:
                            raise RuntimeError(
                                f"Base hash mismatch for {path}: "
                                f"expected={base_hash} actual={actual_hash}"
                            )

                    content = self._b64decode(item["content_b64"])
                    self.pipe_file(path, content)
                    continue

                raise ValueError(f"Unknown patch op: {op}")

    # -------------------------------------------------------------------------
    # File API
    # -------------------------------------------------------------------------

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict[str, Any] | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ):
        path = self._norm(path)

        if set(mode) & {"w", "a", "x", "+"}:
            return _OverlayWriteFile(
                self,
                path,
                mode,
                encoding=self._text_encoding(kwargs),
            )

        if "r" not in mode:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        with self._lock:
            if self._is_deleted(path):
                raise FileNotFoundError(path)

            if path in self._files:
                data = self._files[path]
                if "b" in mode:
                    return io.BytesIO(data)
                return io.StringIO(data.decode(self._text_encoding(kwargs)))

        return self.base_fs.open(
            path,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )

    def exists(self, path: str, **kwargs: Any) -> bool:
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                return False

            if path in self._files or path in self._dirs:
                return True

        return self._base_exists(path)

    def lexists(self, path: str, **kwargs: Any) -> bool:
        return self.exists(path, **kwargs)

    def isfile(self, path: str) -> bool:
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                return False
            if path in self._files:
                return True
            if path in self._dirs:
                return False

        return self._base_isfile(path)

    def isdir(self, path: str) -> bool:
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                return False
            if path in self._dirs:
                return True
            if path in self._files:
                return False
            if self._overlay_isdir(path):
                return True

        return self._base_isdir(path)

    def info(self, path: str, **kwargs: Any) -> FileInfo:
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                raise FileNotFoundError(path)

            if path in self._files:
                return self._file_info(path, self._files[path])

            if path in self._dirs or self._overlay_isdir(path):
                return self._directory_info(path)

        return self.base_fs.info(path, **kwargs)

    def size(self, path: str) -> int:
        return int(self.info(path)["size"])

    def created(self, path: str) -> datetime | None:
        try:
            return self.info(path).get("created")
        except FileNotFoundError:
            return None

    def modified(self, path: str) -> datetime | None:
        try:
            info = self.info(path)
            return info.get("mtime") or info.get("modified")
        except FileNotFoundError:
            return None

    # -------------------------------------------------------------------------
    # Write helpers
    # -------------------------------------------------------------------------

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._norm(path)

        with self._lock:
            if path == self.root_marker:
                self._dirs.add(path)
                return

            self._check_not_existing_file(path)

            if not create_parents:
                parent = self._parent(path)
                if not self.exists(parent):
                    raise FileNotFoundError(parent)

            self._ensure_parent_dirs(path)
            self._dirs.add(path)
            self._unhide_path(path)

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        path = self._norm(path)

        if self.exists(path):
            if not exist_ok and not self.isdir(path):
                raise FileExistsError(path)
            return

        self.mkdir(path, create_parents=True)

    def touch(self, path: str, truncate: bool = True, **kwargs: Any) -> None:
        path = self._norm(path)

        with self._lock:
            self._check_not_existing_dir(path)
            self._ensure_parent_dirs(path)
            self._unhide_path(path)

            if not truncate and path in self._files:
                return

            if not truncate and path not in self._files and self._base_exists(path):
                self._files[path] = self._base_read_bytes(path)
                self._invalidate_filetype(path)
                return

            self._files[path] = b""
            self._invalidate_filetype(path)

    def pipe_file(self, path: str, value: bytes | str, **kwargs: Any) -> None:
        path = self._norm(path)

        with self._lock:
            self._check_not_existing_dir(path)
            self._ensure_parent_dirs(path)
            self._unhide_path(path)

            if isinstance(value, str):
                value = value.encode(kwargs.get("encoding") or "utf-8")

            self._files[path] = value
            self._invalidate_filetype(path)

    def write_text(
        self,
        path: str,
        value: str,
        encoding: str = "utf-8",
        errors: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.pipe_file(path, value.encode(encoding, errors or "strict"), **kwargs)

    def write_bytes(self, path: str, value: bytes, **kwargs: Any) -> None:
        self.pipe_file(path, value, **kwargs)

    # -------------------------------------------------------------------------
    # Read helpers
    # -------------------------------------------------------------------------

    def cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes:
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                raise FileNotFoundError(path)

            if path in self._files:
                data = self._files[path]
                if start is not None or end is not None:
                    return data[start:end]
                return data

        data = self.base_fs.cat_file(path, **kwargs)

        if start is not None or end is not None:
            return data[start:end]

        return data

    def cat(
        self,
        path: str | list[str],
        recursive: bool = False,
        on_error: str = "raise",
        **kwargs: Any,
    ):
        if isinstance(path, (list, tuple)):
            result = {}
            for item in path:
                try:
                    result[item] = self.cat_file(item, **kwargs)
                except Exception:
                    if on_error == "raise":
                        raise
                    if on_error == "omit":
                        continue
                    result[item] = None
            return result

        return self.cat_file(path, **kwargs)

    def read_text(
        self,
        path: str,
        encoding: str = "utf-8",
        errors: str | None = None,
        **kwargs: Any,
    ) -> str:
        return self.cat_file(path, **kwargs).decode(encoding, errors or "strict")

    def read_bytes(self, path: str, **kwargs: Any) -> bytes:
        return self.cat_file(path, **kwargs)

    # -------------------------------------------------------------------------
    # Remove / copy / move
    # -------------------------------------------------------------------------

    def rm_file(self, path: str) -> None:
        path = self._norm(path)

        with self._lock:
            removed = False

            if path in self._files:
                del self._files[path]
                removed = True

            if path in self._dirs:
                self._dirs.remove(path)
                removed = True

            if self._base_exists(path):
                self._deleted.add(path)
                removed = True

            if not removed:
                raise FileNotFoundError(path)

            self._invalidate_filetype(path)

    def rmdir(self, path: str) -> None:
        path = self._norm(path)

        if path == self.root_marker:
            raise ValueError("Cannot remove root directory")

        with self._lock:
            prefix = path.rstrip("/") + "/"

            if any(file_path.startswith(prefix) for file_path in self._files):
                raise OSError(f"Directory not empty: {path}")

            if any(
                dir_path.startswith(prefix)
                for dir_path in self._dirs
                if dir_path != path
            ):
                raise OSError(f"Directory not empty: {path}")

            if self._base_exists(path):
                try:
                    base_entries = self.base_fs.ls(path, detail=False)
                except FileNotFoundError:
                    base_entries = []

                visible_entries = [
                    entry
                    for entry in base_entries
                    if not self._is_deleted(self._norm(entry))
                ]
                if visible_entries:
                    raise OSError(f"Directory not empty: {path}")

                self._deleted.add(path)

            self._dirs.discard(path)

    def rm(
        self,
        path: str | list[str],
        recursive: bool = False,
        maxdepth: int | None = None,
    ) -> None:
        if isinstance(path, (list, tuple, set)):
            for item in path:
                self.rm(item, recursive=recursive, maxdepth=maxdepth)
            return

        path = self._norm(path)

        if self.isfile(path):
            self.rm_file(path)
            return

        if self.isdir(path) and not recursive:
            self.rmdir(path)
            return

        if recursive:
            with self._lock:
                prefix = path.rstrip("/") + "/"

                for file_path in list(self._files):
                    if file_path == path or file_path.startswith(prefix):
                        del self._files[file_path]

                for dir_path in list(self._dirs):
                    if dir_path != self.root_marker and (
                        dir_path == path or dir_path.startswith(prefix)
                    ):
                        self._dirs.discard(dir_path)

                if self._base_exists(path):
                    # Adding a tombstone on the path itself is sufficient:
                    # _is_deleted walks up ancestors, so any child of this
                    # path will find this tombstone and be treated as hidden.
                    self._deleted.add(path)

                    # Best-effort: also tombstone individual children so that
                    # _is_deleted can short-circuit via direct set lookup
                    # instead of always walking up.  This is purely an
                    # optimisation; correctness does not depend on it.
                    for base_path in _safe_base_find(self.base_fs, path):
                        self._deleted.add(self._norm(base_path))

                self._invalidate_filetype(path)

            return

        raise FileNotFoundError(path)

    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        self.pipe_file(path2, self.cat_file(path1))

    def copy(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        **kwargs: Any,
    ):
        if recursive and self.isdir(path1):
            source_root = self._norm(path1)
            target_root = self._norm(path2)

            for source in self.find(path1, withdirs=True, detail=False):
                rel_path = posixpath.relpath(source, source_root)
                target = (
                    target_root
                    if rel_path == "."
                    else posixpath.join(target_root, rel_path)
                )

                if self.isdir(source):
                    self.makedirs(target, exist_ok=True)
                else:
                    self.cp_file(source, target)

            return

        return self.cp_file(path1, path2, **kwargs)

    def mv(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ):
        self.copy(path1, path2, recursive=recursive, **kwargs)
        self.rm(path1, recursive=recursive, maxdepth=maxdepth)

    # -------------------------------------------------------------------------
    # Listing / walking
    # -------------------------------------------------------------------------

    def _iter_overlay_children(self, path: str) -> Iterator[FileInfo]:
        path = self._norm(path)
        seen: set[str] = set()

        for dir_path in sorted(self._dirs):
            child = _immediate_child(path, dir_path, self._norm)
            if child is None:
                continue
            if self._is_deleted(child) or child in seen:
                continue

            seen.add(child)
            yield {"name": child, "size": 0, "type": "directory"}

        for file_path, content in sorted(self._files.items()):
            child = _immediate_child(path, file_path, self._norm)
            if child is None:
                continue
            if self._is_deleted(child) or child in seen:
                continue

            seen.add(child)

            if child == self._norm(file_path):
                yield {"name": child, "size": len(content), "type": "file"}
            else:
                yield {"name": child, "size": 0, "type": "directory"}

    def ls(self, path: str = "", detail: bool = True, **kwargs: Any):
        path = self._norm(path)

        with self._lock:
            if self._is_deleted(path):
                raise FileNotFoundError(path)

            merged: dict[str, FileInfo] = {}

            try:
                base_entries = self.base_fs.ls(path, detail=True, **kwargs)
                for entry in base_entries:
                    name = self._norm(entry["name"])
                    if self._is_deleted(name):
                        continue

                    normalized = dict(entry)
                    normalized["name"] = name
                    merged[name] = normalized
            except FileNotFoundError:
                pass

            for entry in self._iter_overlay_children(path):
                name = self._norm(entry["name"])
                if name == path:
                    continue
                merged[name] = entry

            items = sorted(merged.values(), key=lambda item: item["name"])
            if detail:
                return items
            return [item["name"] for item in items]

    def walk(
        self,
        path: str = "",
        maxdepth: int | None = None,
        topdown: bool = True,
        **kwargs: Any,
    ):
        path = self._norm(path)

        if not self.exists(path):
            return

        stack: list[tuple[str, int]] = [(path, 0)]
        visited: set[str] = set()
        deferred: list[tuple[str, list[str], list[str]]] = []

        while stack:
            current, depth = stack.pop()
            current = self._norm(current)

            if current in visited:
                continue
            visited.add(current)

            try:
                entries = self.ls(current, detail=True)
            except FileNotFoundError:
                continue

            dirs: list[str] = []
            files: list[str] = []

            for entry in entries:
                name = self._norm(entry["name"])
                base = posixpath.basename(name)

                if not base or name == current:
                    continue

                if entry["type"] == "directory":
                    dirs.append(base)
                elif entry["type"] == "file":
                    files.append(base)

            dirs = sorted(set(dirs))
            files = sorted(set(files))

            if topdown:
                yield current, dirs, files
            else:
                deferred.append((current, dirs, files))

            if maxdepth is None or depth < maxdepth:
                # Reverse so that left-most dirs are processed first (LIFO stack)
                for dirname in reversed(dirs):
                    child = (
                        posixpath.join(current, dirname)
                        if current != self.root_marker
                        else "/" + dirname
                    )
                    child = self._norm(child)

                    if child not in visited:
                        stack.append((child, depth + 1))

        if not topdown:
            for item in reversed(deferred):
                yield item

    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ):
        path = self._norm(path)
        result: dict[str, FileInfo] = {}

        if self.isfile(path):
            info = self.info(path)
            return {path: info} if detail else [path]

        for root, dirs, files in self.walk(path, maxdepth=maxdepth, **kwargs):
            if withdirs:
                if root != path:
                    result[root] = self.info(root)

                for dirname in dirs:
                    dir_path = (
                        posixpath.join(root, dirname) if root != "/" else "/" + dirname
                    )
                    result[dir_path] = self.info(dir_path)

            for filename in files:
                file_path = (
                    posixpath.join(root, filename) if root != "/" else "/" + filename
                )
                result[file_path] = self.info(file_path)

        if detail:
            return dict(sorted(result.items()))
        return sorted(result)

    def glob(self, path: str, **kwargs: Any):
        pattern = self._norm(path)

        if "**" in pattern:
            search_root = pattern.split("**", 1)[0].rstrip("/") or "/"
            candidates = set()

            if self.exists(search_root):
                candidates.add(search_root)

            candidates.update(self.find(search_root, withdirs=True, detail=False))
        else:
            parts = pattern.strip("/").split("/")
            prefix_parts: list[str] = []

            for part in parts:
                if any(ch in part for ch in "*?["):
                    break
                prefix_parts.append(part)

            search_root = "/" + "/".join(prefix_parts) if prefix_parts else "/"

            candidates = set()
            try:
                for item in self.ls(search_root, detail=True):
                    candidates.add(self._norm(item["name"]))
            except FileNotFoundError:
                return []

        pattern_no_root = pattern.lstrip("/")
        return sorted(
            p for p in candidates if PurePosixPath(p.lstrip("/")).match(pattern_no_root)
        )

    def du(
        self,
        path: str,
        total: bool = True,
        maxdepth: int | None = None,
        withdirs: bool = False,
        **kwargs: Any,
    ):
        entries = self.find(path, maxdepth=maxdepth, withdirs=withdirs, detail=True)
        sizes = {p: int(info.get("size", 0)) for p, info in entries.items()}

        if total:
            return sum(sizes.values())
        return sizes

    def expand_path(
        self,
        path,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: Any,
    ):
        if isinstance(path, (list, tuple, set)):
            result: list[str] = []
            for item in path:
                result.extend(
                    self.expand_path(
                        item,
                        recursive=recursive,
                        maxdepth=maxdepth,
                        **kwargs,
                    )
                )
            return sorted(set(result))

        path = self._norm(path)

        if any(ch in path for ch in "*?["):
            return self.glob(path, **kwargs)

        if recursive and self.isdir(path):
            return [path] + self.find(path, withdirs=True, detail=False)

        return [path]

    def invalidate_cache(self, path: str | None = None) -> None:
        try:
            self.base_fs.invalidate_cache(path)
        except Exception:
            pass

    def changed_paths(self) -> dict[str, list[str]]:
        """
        Path-only inventory of overlay changes relative to ``base_fs``.

        Returns
        -------
        dict
            ``{"added": [...], "modified": [...], "deleted": [...]}``.
            Files whose overlay content equals the base content are not
            reported as modified. Overlay-only empty directories are not
            included; use ``save()`` for the full patch view.
        """
        with self._lock:
            added: list[str] = []
            modified: list[str] = []
            deleted: list[str] = []

            for path, content in self._files.items():
                if not self._base_exists(path):
                    added.append(path)
                    continue

                if self._base_isfile(path):
                    try:
                        base_bytes = self._base_read_bytes(path)
                    except Exception:
                        modified.append(path)
                        continue
                    if base_bytes != content:
                        modified.append(path)
                else:
                    # Base is a directory; overlay file replaces it.
                    modified.append(path)

            for path in self._deleted:
                if self._base_exists(path):
                    deleted.append(path)

            return {
                "added": sorted(added),
                "modified": sorted(modified),
                "deleted": sorted(deleted),
            }

    def diff(
        self,
        root: str = "/",
        *,
        context_lines: int = 3,
        binary_marker: str = "Binary files differ",
    ) -> str:
        """
        Generate a unified-diff-like text representation of all changes
        between ``base_fs`` and the current effective (overlay) view.

        Parameters
        ----------
        root : str
            Subtree to diff.  Defaults to ``"/"``.
        context_lines : int
            Number of unchanged context lines around each change hunk
            (mirrors ``diff -U``).  Defaults to 3.
        binary_marker : str
            Placeholder text emitted when a changed file contains non-text
            (binary) content.

        Returns
        -------
        str
            A string in unified-diff format.  Empty string when there are
            no differences.
        """

        with self._lock:
            root = self._norm(root)

            base_entries = self._base_find_detail(root)
            visible_entries = self._visible_find_detail(root)

            base_paths = set(base_entries)
            visible_paths = set(visible_entries)

            output_lines: list[str] = []

            all_paths = sorted(base_paths | visible_paths)

            # Cache for base file reads
            _base_cache: dict[str, bytes] = {}

            def _read_base_cached(p: str) -> bytes:
                if p not in _base_cache:
                    _base_cache[p] = self._base_read_bytes(p)
                return _base_cache[p]

            def _is_text(data: bytes) -> bool:
                """Heuristic: treat data as text if it decodes as UTF-8
                and contains no null bytes."""
                if b"\x00" in data:
                    return False
                try:
                    data.decode("utf-8")
                    return True
                except (UnicodeDecodeError, ValueError):
                    return False

            def _lines(data: bytes) -> list[str]:
                """Split bytes into text lines suitable for difflib."""
                text = data.decode("utf-8", errors="replace")
                # splitlines(True) keeps line endings; we normalise to \n
                lines = text.splitlines(True)
                # Ensure every line ends with newline for clean diff output
                return [ln if ln.endswith("\n") else ln + "\n" for ln in lines]

            def _emit_diff_header(path: str, status: str) -> None:
                output_lines.append(f"diff --overlay a{path} b{path}")
                output_lines.append(f"{status}")

            for path in all_paths:
                if path == self.root_marker:
                    continue

                in_base = path in base_paths
                in_visible = path in visible_paths

                base_info = base_entries.get(path, {})
                visible_info = visible_entries.get(path, {})

                base_type = base_info.get("type", "file")
                visible_type = visible_info.get("type", "file")

                # --- Deleted paths ---------------------------------------------------
                if in_base and not in_visible:
                    if base_type == "directory":
                        _emit_diff_header(path, "deleted directory")
                        output_lines.append("")
                        continue

                    # Deleted file
                    _emit_diff_header(path, "deleted file")
                    try:
                        base_bytes = _read_base_cached(path)
                    except FileNotFoundError:
                        output_lines.append("")
                        continue

                    if not _is_text(base_bytes):
                        output_lines.append(f"--- a{path}")
                        output_lines.append("+++ /dev/null")
                        output_lines.append(binary_marker)
                        output_lines.append("")
                        continue

                    base_lines = _lines(base_bytes)
                    diff_result = difflib.unified_diff(
                        base_lines,
                        [],
                        fromfile=f"a{path}",
                        tofile="/dev/null",
                        n=context_lines,
                    )
                    output_lines.extend(line.rstrip("\n") for line in diff_result)
                    output_lines.append("")
                    continue

                # --- New paths -------------------------------------------------------
                if not in_base and in_visible:
                    if visible_type == "directory":
                        _emit_diff_header(path, "new directory")
                        output_lines.append("")
                        continue

                    # New file
                    _emit_diff_header(path, "new file")
                    try:
                        current_bytes = self._effective_read_bytes(path)
                    except FileNotFoundError:
                        output_lines.append("")
                        continue

                    if not _is_text(current_bytes):
                        output_lines.append("--- /dev/null")
                        output_lines.append(f"+++ b{path}")
                        output_lines.append(binary_marker)
                        output_lines.append("")
                        continue

                    current_lines = _lines(current_bytes)
                    diff_result = difflib.unified_diff(
                        [],
                        current_lines,
                        fromfile="/dev/null",
                        tofile=f"b{path}",
                        n=context_lines,
                    )
                    output_lines.extend(line.rstrip("\n") for line in diff_result)
                    output_lines.append("")
                    continue

                # --- Both exist – check for modifications ----------------------------
                # Type change (e.g. dir -> file or file -> dir)
                if base_type != visible_type:
                    _emit_diff_header(
                        path, f"type changed: {base_type} -> {visible_type}"
                    )
                    output_lines.append("")
                    continue

                # Directories don't have content to diff
                if visible_type == "directory":
                    continue

                # Compare file contents
                try:
                    base_bytes = _read_base_cached(path)
                except FileNotFoundError:
                    base_bytes = b""

                try:
                    current_bytes = self._effective_read_bytes(path)
                except FileNotFoundError:
                    current_bytes = b""

                if base_bytes == current_bytes:
                    continue  # No change

                _emit_diff_header(path, "modified file")

                base_is_text = _is_text(base_bytes)
                current_is_text = _is_text(current_bytes)

                if not base_is_text or not current_is_text:
                    output_lines.append(f"--- a{path}")
                    output_lines.append(f"+++ b{path}")
                    output_lines.append(binary_marker)
                    output_lines.append("")
                    continue

                base_lines = _lines(base_bytes)
                current_lines = _lines(current_bytes)

                diff_result = difflib.unified_diff(
                    base_lines,
                    current_lines,
                    fromfile=f"a{path}",
                    tofile=f"b{path}",
                    n=context_lines,
                )
                output_lines.extend(line.rstrip("\n") for line in diff_result)
                output_lines.append("")

        # Strip trailing blank lines but keep one final newline
        while output_lines and output_lines[-1] == "":
            output_lines.pop()

        if not output_lines:
            return ""

        return "\n".join(output_lines) + "\n"


class _OverlayWriteFile:
    def __init__(
        self,
        fs: MemoryOverlayFileSystem,
        path: str,
        mode: str,
        encoding: str = "utf-8",
    ) -> None:
        self.fs = fs
        self.path = fs._norm(path)
        self.mode = mode
        self.encoding = encoding
        self.closed = False
        self.binary = "b" in mode

        if "x" in mode and fs.exists(self.path):
            raise FileExistsError(self.path)

        with fs._lock:
            fs._check_not_existing_dir(self.path)

        initial = self._load_initial_bytes()
        self._buf = io.BytesIO()

        if initial:
            self._buf.write(initial)

        if "a" not in mode:
            self._buf.seek(0)
            if "w" in mode:
                self._buf.truncate(0)

        with fs._lock:
            fs._ensure_parent_dirs(self.path)
            fs._unhide_path(self.path)

    def _load_initial_bytes(self) -> bytes:
        if "a" not in self.mode and "+" not in self.mode:
            return b""

        with self.fs._lock:
            if self.path in self.fs._files:
                return self.fs._files[self.path]

        if not self.fs._is_deleted(self.path) and self.fs._base_exists(self.path):
            with self.fs.base_fs.open(self.path, "rb") as f:
                return f.read()

        return b""

    def write(self, data):
        if isinstance(data, str):
            data = data.encode(self.encoding)
        return self._buf.write(data)

    def writelines(self, lines: Iterable[bytes | str]) -> None:
        for line in lines:
            self.write(line)

    def read(self, size: int = -1):
        data = self._buf.read(size)
        if self.binary:
            return data
        return data.decode(self.encoding)

    def readline(self, size: int = -1):
        data = self._buf.readline(size)
        if self.binary:
            return data
        return data.decode(self.encoding)

    def readlines(self, hint: int = -1):
        lines: list[bytes | str] = []
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            if (
                0
                <= hint
                <= sum(
                    len(ln) if isinstance(ln, bytes) else len(ln.encode(self.encoding))
                    for ln in lines
                )
            ):
                break
        return lines

    def truncate(self, size: int | None = None) -> int:
        if size is None:
            size = self._buf.tell()
        self._buf.truncate(size)
        return size

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def flush(self) -> None:
        with self.fs._lock:
            self.fs._files[self.path] = self._buf.getvalue()
            self.fs._invalidate_filetype(self.path)

    def close(self) -> None:
        if self.closed:
            return

        self.flush()
        self.closed = True

    def readable(self) -> bool:
        return "r" in self.mode or "+" in self.mode

    def writable(self) -> bool:
        return any(ch in self.mode for ch in ("w", "a", "x", "+"))

    def seekable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line
