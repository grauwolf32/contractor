from __future__ import annotations

import base64
import hashlib
import io
import posixpath
from copy import deepcopy
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Iterable, Iterator

from fsspec.spec import AbstractFileSystem


FileInfo = dict[str, Any]
Patch = dict[str, Any]


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

    def __init__(self, base_fs: AbstractFileSystem, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_fs = base_fs

        # Overlay state
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = {self.root_marker}
        self._deleted: set[str] = set()

        # Optional in-memory snapshot
        self._snapshot_state: dict[str, Any] | None = None

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
        current = self._parent(path)
        missing: list[str] = []

        while current not in self._dirs and current != self.root_marker:
            missing.append(current)
            current = self._parent(current)

        self._dirs.update(missing)
        self._dirs.add(self.root_marker)

    # -------------------------------------------------------------------------
    # Overlay visibility / tombstones
    # -------------------------------------------------------------------------

    def _is_deleted(self, path: str) -> bool:
        path = self._norm(path)
        return any(
            path == deleted or path.startswith(deleted.rstrip("/") + "/")
            for deleted in self._deleted
        )

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
        path = self._norm(path)

        if path in self._dirs:
            return True

        prefix = path.rstrip("/") + "/"
        has_file_child = any(file_path.startswith(prefix) for file_path in self._files)
        has_dir_child = any(
            dir_path.startswith(prefix) for dir_path in self._dirs if dir_path != path
        )
        return has_file_child or has_dir_child

    # -------------------------------------------------------------------------
    # Base FS helpers
    # -------------------------------------------------------------------------

    def _base_exists(self, path: str) -> bool:
        try:
            return bool(self.base_fs.exists(path))
        except Exception:
            return False

    def _base_isfile(self, path: str) -> bool:
        try:
            return bool(self.base_fs.isfile(path))
        except Exception:
            try:
                return self.base_fs.info(path).get("type") == "file"
            except Exception:
                return False

    def _base_isdir(self, path: str) -> bool:
        try:
            return bool(self.base_fs.isdir(path))
        except Exception:
            try:
                return self.base_fs.info(path).get("type") == "directory"
            except Exception:
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
        except Exception:
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
        except Exception:
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
                    except Exception:
                        pass

                return result
        except Exception:
            pass

        result: dict[str, FileInfo] = {}

        try:
            if self._base_exists(root):
                info = dict(self.base_fs.info(root))
                info["name"] = root
                result[root] = info
        except Exception:
            return result

        try:
            for current, dirs, files in self.base_fs.walk(root):
                current = self._norm(current)

                if current not in result:
                    try:
                        info = dict(self.base_fs.info(current))
                    except Exception:
                        info = {"name": current, "type": "directory", "size": 0}
                    info["name"] = current
                    result[current] = info

                for dirname in dirs:
                    path = self._norm(
                        posixpath.join(current, dirname)
                        if current != "/"
                        else "/" + dirname
                    )
                    try:
                        info = dict(self.base_fs.info(path))
                    except Exception:
                        info = {"name": path, "type": "directory", "size": 0}
                    info["name"] = path
                    result[path] = info

                for filename in files:
                    path = self._norm(
                        posixpath.join(current, filename)
                        if current != "/"
                        else "/" + filename
                    )
                    try:
                        info = dict(self.base_fs.info(path))
                    except Exception:
                        info = {"name": path, "type": "file", "size": 0}
                    info["name"] = path
                    result[path] = info
        except Exception:
            pass

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
        self._files.clear()
        self._dirs = {self.root_marker}
        self._deleted.clear()

    def snapshot(self) -> dict[str, Any]:
        """
        Save the current overlay state in memory.
        """
        self._snapshot_state = self._current_overlay_state()
        return deepcopy(self._snapshot_state)

    def restore_snapshot(self, snapshot: dict[str, Any] | None = None) -> None:
        """
        Restore overlay state from snapshot().
        """
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

    def export_overlay_state(self) -> dict[str, Any]:
        """
        Serialize the internal overlay state.
        This is not a diff, but a full snapshot of internal state.
        """
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
        tombstones so the underlay version becomes visible again. The target
        path must exist in ``base_fs``.
        """
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
            self._dirs, key=lambda item: (item.count("/"), item), reverse=True
        ):
            if dir_path == self.root_marker:
                continue
            if dir_path == path or (recursive and dir_path.startswith(prefix)):
                self._dirs.discard(dir_path)

        # Remove tombstones so the base view becomes visible again.
        self._unhide_path(path)

        # Ensure ancestors created in the overlay remain valid for any sibling paths.
        self._dirs.add(self.root_marker)

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
        root = self._norm(root)

        base_entries = self._base_find_detail(root)
        visible_entries = self._visible_find_detail(root)

        base_paths = set(base_entries)
        visible_paths = set(visible_entries)

        patches: list[Patch] = []

        # Deletions: the path exists in base, but is hidden from the overlay-visible tree
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
                    patch["base_hash"] = self._sha256_bytes(self._base_read_bytes(path))
                except Exception:
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

            base_bytes = self._base_read_bytes(path)
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

        if self._is_deleted(path):
            return False

        if path in self._files or path in self._dirs:
            return True

        return self._base_exists(path)

    def lexists(self, path: str, **kwargs: Any) -> bool:
        return self.exists(path, **kwargs)

    def isfile(self, path: str) -> bool:
        path = self._norm(path)

        if self._is_deleted(path):
            return False
        if path in self._files:
            return True
        if path in self._dirs:
            return False

        return self._base_isfile(path)

    def isdir(self, path: str) -> bool:
        path = self._norm(path)

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
        except Exception:
            return None

    def modified(self, path: str) -> datetime | None:
        try:
            info = self.info(path)
            return info.get("mtime") or info.get("modified")
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Write helpers
    # -------------------------------------------------------------------------

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        path = self._norm(path)

        if path == self.root_marker:
            self._dirs.add(path)
            return

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

        self._ensure_parent_dirs(path)
        self._unhide_path(path)

        if not truncate and path in self._files:
            return

        if not truncate and path not in self._files and self._base_exists(path):
            with self.base_fs.open(path, "rb") as f:
                self._files[path] = f.read()
            return

        self._files[path] = b""

    def pipe_file(self, path: str, value: bytes | str, **kwargs: Any) -> None:
        path = self._norm(path)

        self._ensure_parent_dirs(path)
        self._unhide_path(path)

        if isinstance(value, str):
            value = value.encode(kwargs.get("encoding") or "utf-8")

        self._files[path] = value

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

        if self._is_deleted(path):
            raise FileNotFoundError(path)

        if path in self._files:
            data = self._files[path]
        else:
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

    def rmdir(self, path: str) -> None:
        path = self._norm(path)

        if path == self.root_marker:
            raise ValueError("Cannot remove root directory")

        prefix = path.rstrip("/") + "/"

        if any(file_path.startswith(prefix) for file_path in self._files):
            raise OSError(f"Directory not empty: {path}")

        if any(
            dir_path.startswith(prefix) for dir_path in self._dirs if dir_path != path
        ):
            raise OSError(f"Directory not empty: {path}")

        if self._base_exists(path):
            try:
                base_entries = self.base_fs.ls(path, detail=False)
            except Exception:
                base_entries = []

            visible_entries = [
                entry for entry in base_entries if not self._is_deleted(entry)
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
                self._deleted.add(path)

            # Hide everything under the prefix from the base filesystem
            try:
                for base_path in self.base_fs.find(path, withdirs=True, detail=False):
                    self._deleted.add(self._norm(base_path))
            except Exception:
                pass

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
        prefix = "" if path == self.root_marker else path.rstrip("/") + "/"
        seen: set[str] = set()

        for dir_path in sorted(self._dirs):
            if dir_path in {path, self.root_marker}:
                continue
            if not dir_path.startswith(prefix):
                continue

            rest = dir_path[len(prefix) :]
            if not rest or "/" in rest:
                continue

            child = prefix + rest if prefix else "/" + rest
            if child not in seen and not self._is_deleted(child):
                seen.add(child)
                yield {"name": child, "size": 0, "type": "directory"}

        for file_path, content in sorted(self._files.items()):
            if not file_path.startswith(prefix):
                continue

            rest = file_path[len(prefix) :]
            if not rest:
                continue

            first = rest.split("/", 1)[0]
            child = prefix + first if prefix else "/" + first

            if self._is_deleted(child) or child in seen:
                continue

            if "/" in rest:
                seen.add(child)
                yield {"name": child, "size": 0, "type": "directory"}
            else:
                seen.add(child)
                yield {"name": child, "size": len(content), "type": "file"}

    def ls(self, path: str = "", detail: bool = True, **kwargs: Any):
        path = self._norm(path)

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
        except Exception:
            pass

        for entry in self._iter_overlay_children(path):
            merged[entry["name"]] = entry

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

        def _walk(current: str, depth: int):
            entries = self.ls(current, detail=True)
            dirs = sorted(
                posixpath.basename(entry["name"])
                for entry in entries
                if entry["type"] == "directory"
            )
            files = sorted(
                posixpath.basename(entry["name"])
                for entry in entries
                if entry["type"] == "file"
            )

            if topdown:
                yield current, dirs, files

            if maxdepth is None or depth < maxdepth:
                for dirname in dirs:
                    child = (
                        posixpath.join(current, dirname)
                        if current != "/"
                        else "/" + dirname
                    )
                    yield from _walk(child, depth + 1)

            if not topdown:
                yield current, dirs, files

        yield from _walk(path, 0)

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
            path
            for path in candidates
            if PurePosixPath(path.lstrip("/")).match(pattern_no_root)
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
        sizes = {path: int(info.get("size", 0)) for path, info in entries.items()}

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

    def __getattr__(self, name: str):
        """
        Passthrough for methods/properties that are not overridden here.
        """
        return getattr(self.base_fs, name)


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

        initial = self._load_initial_bytes()
        self._buf = io.BytesIO()

        if initial:
            self._buf.write(initial)

        if "a" not in mode:
            self._buf.seek(0)
            if "w" in mode:
                self._buf.truncate(0)

        fs._ensure_parent_dirs(self.path)
        fs._unhide_path(self.path)

    def _load_initial_bytes(self) -> bytes:
        if "a" not in self.mode and "+" not in self.mode:
            return b""

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

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._buf.seek(offset, whence)

    def tell(self) -> int:
        return self._buf.tell()

    def flush(self) -> None:
        self.fs._files[self.path] = self._buf.getvalue()

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
