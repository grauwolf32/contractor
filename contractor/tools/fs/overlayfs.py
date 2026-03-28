from __future__ import annotations

import fnmatch
import io
import os
import posixpath
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Optional
from pathlib import PurePosixPath
from fsspec.spec import AbstractFileSystem


class MemoryOverlayFileSystem(AbstractFileSystem):
    """
    In-memory overlay over another fsspec filesystem.

    Semantics:
    - base_fs is read-only from the overlay point of view
    - all writes go only into memory
    - new files can be created in memory
    - reads prefer overlay, then fall back to base_fs
    - deleting an overlay-created/overlay-modified file only affects overlay
    """

    protocol = "overlay"
    root_marker = "/"

    def __init__(self, base_fs: AbstractFileSystem, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_fs = base_fs

        # path -> bytes
        self._files: Dict[str, bytes] = {}

        # virtual directories explicitly created in overlay
        self._dirs: set[str] = {self.root_marker}

        # tombstones hide files/dirs that exist only in base_fs
        self._deleted: set[str] = set()

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
        path = self._strip_protocol(path)
        return path if path else self.root_marker

    def _parent(self, path: str) -> str:
        path = self._norm(path)
        if path == self.root_marker:
            return self.root_marker
        parent = posixpath.dirname(path)
        return parent if parent else self.root_marker

    def _ensure_parent_dirs(self, path: str) -> None:
        cur = self._parent(path)
        missing: list[str] = []
        while cur not in self._dirs and cur != self.root_marker:
            missing.append(cur)
            cur = self._parent(cur)
        self._dirs.update(missing)
        self._dirs.add(self.root_marker)

    def _is_deleted(self, path: str) -> bool:
        path = self._norm(path)
        return any(
            path == deleted or path.startswith(deleted.rstrip("/") + "/")
            for deleted in self._deleted
        )

    def _unhide_path(self, path: str) -> None:
        path = self._norm(path)
        prefix = path.rstrip("/") + "/"
        self._deleted = {p for p in self._deleted if p != path and not p.startswith(prefix)}

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

    def _overlay_isfile(self, path: str) -> bool:
        return self._norm(path) in self._files

    def _overlay_isdir(self, path: str) -> bool:
        path = self._norm(path)
        if path in self._dirs:
            return True
        prefix = path.rstrip("/") + "/"
        return any(p.startswith(prefix) for p in self._files) or any(
            d.startswith(prefix) for d in self._dirs if d != path
        )

    def _text_encoding(self, kwargs: dict[str, Any]) -> str:
        return kwargs.get("encoding") or "utf-8"

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
            return _OverlayWriteFile(self, path, mode, encoding=self._text_encoding(kwargs))

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

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        path = self._norm(path)

        if self._is_deleted(path):
            raise FileNotFoundError(path)

        if path in self._files:
            return {
                "name": path,
                "size": len(self._files[path]),
                "type": "file",
                "created": None,
                "mtime": None,
            }

        if path in self._dirs or self._overlay_isdir(path):
            return {
                "name": path,
                "size": 0,
                "type": "directory",
                "created": None,
                "mtime": None,
            }

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

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
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

    def cat(self, path: str | list[str], recursive: bool = False, on_error: str = "raise", **kwargs: Any):
        if isinstance(path, (list, tuple)):
            out = {}
            for p in path:
                try:
                    out[p] = self.cat_file(p, **kwargs)
                except Exception:
                    if on_error == "raise":
                        raise
                    if on_error == "omit":
                        continue
                    out[p] = None
            return out
        return self.cat_file(path, **kwargs)

    def read_text(self, path: str, encoding: str = "utf-8", errors: str | None = None, **kwargs: Any) -> str:
        data = self.cat_file(path, **kwargs)
        return data.decode(encoding, errors or "strict")

    def write_text(
        self,
        path: str,
        value: str,
        encoding: str = "utf-8",
        errors: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.pipe_file(path, value.encode(encoding, errors or "strict"), **kwargs)

    def read_bytes(self, path: str, **kwargs: Any) -> bytes:
        return self.cat_file(path, **kwargs)

    def write_bytes(self, path: str, value: bytes, **kwargs: Any) -> None:
        self.pipe_file(path, value, **kwargs)

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
        if any(p.startswith(prefix) for p in self._files):
            raise OSError(f"Directory not empty: {path}")
        if any(d.startswith(prefix) for d in self._dirs if d != path):
            raise OSError(f"Directory not empty: {path}")

        if self._base_exists(path):
            try:
                base_entries = self.base_fs.ls(path, detail=False)
            except Exception:
                base_entries = []
            visible = [x for x in base_entries if not self._is_deleted(x)]
            if visible:
                raise OSError(f"Directory not empty: {path}")
            self._deleted.add(path)

        self._dirs.discard(path)

    def rm(self, path: str | list[str], recursive: bool = False, maxdepth: int | None = None) -> None:
        if isinstance(path, (list, tuple, set)):
            for p in path:
                self.rm(p, recursive=recursive, maxdepth=maxdepth)
            return

        path = self._norm(path)

        if self.isfile(path):
            self.rm_file(path)
            return

        if not recursive and self.isdir(path):
            self.rmdir(path)
            return

        if recursive:
            prefix = path.rstrip("/") + "/"

            for p in list(self._files):
                if p == path or p.startswith(prefix):
                    del self._files[p]

            for d in list(self._dirs):
                if d != self.root_marker and (d == path or d.startswith(prefix)):
                    self._dirs.discard(d)

            if self._base_exists(path):
                self._deleted.add(path)

            # hide everything from base under prefix
            try:
                for p in self.base_fs.find(path, withdirs=True, detail=False):
                    self._deleted.add(self._norm(p))
            except Exception:
                pass

            return

        raise FileNotFoundError(path)

    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        data = self.cat_file(path1)
        self.pipe_file(path2, data)

    def copy(self, path1: str, path2: str, recursive: bool = False, **kwargs: Any):
        if recursive and self.isdir(path1):
            for src in self.find(path1, withdirs=True, detail=False):
                rel = posixpath.relpath(src, self._norm(path1))
                dst = self._norm(path2) if rel == "." else posixpath.join(self._norm(path2), rel)
                if self.isdir(src):
                    self.makedirs(dst, exist_ok=True)
                else:
                    self.cp_file(src, dst)
            return
        return self.cp_file(path1, path2, **kwargs)

    def mv(self, path1: str, path2: str, recursive: bool = False, maxdepth: int | None = None, **kwargs: Any):
        self.copy(path1, path2, recursive=recursive, **kwargs)
        self.rm(path1, recursive=recursive, maxdepth=maxdepth)

    def _iter_overlay_children(self, path: str) -> Iterator[dict[str, Any]]:
        path = self._norm(path)
        prefix = "" if path == self.root_marker else path.rstrip("/") + "/"
        seen: set[str] = set()

        for d in sorted(self._dirs):
            if d == path or d == self.root_marker:
                continue
            if not d.startswith(prefix):
                continue
            rest = d[len(prefix):]
            if not rest or "/" in rest:
                continue
            child = prefix + rest if prefix else "/" + rest
            if child not in seen and not self._is_deleted(child):
                seen.add(child)
                yield {"name": child, "size": 0, "type": "directory"}

        for fpath, data in sorted(self._files.items()):
            if not fpath.startswith(prefix):
                continue
            rest = fpath[len(prefix):]
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
                yield {"name": child, "size": len(data), "type": "file"}

    def ls(self, path: str = "", detail: bool = True, **kwargs: Any):
        path = self._norm(path)
        if self._is_deleted(path):
            raise FileNotFoundError(path)

        merged: dict[str, dict[str, Any]] = {}

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

        items = sorted(merged.values(), key=lambda x: x["name"])
        if detail:
            return items
        return [x["name"] for x in items]

    def walk(self, path: str = "", maxdepth: int | None = None, topdown: bool = True, **kwargs: Any):
        path = self._norm(path)
        if not self.exists(path):
            return

        def _walk(cur: str, depth: int):
            entries = self.ls(cur, detail=True)
            dirs = sorted([posixpath.basename(e["name"]) for e in entries if e["type"] == "directory"])
            files = sorted([posixpath.basename(e["name"]) for e in entries if e["type"] == "file"])

            if topdown:
                yield cur, dirs, files

            if maxdepth is None or depth < maxdepth:
                for d in dirs:
                    child = posixpath.join(cur, d) if cur != "/" else "/" + d
                    yield from _walk(child, depth + 1)

            if not topdown:
                yield cur, dirs, files

        yield from _walk(path, 0)

    def find(self, path: str, maxdepth: int | None = None, withdirs: bool = False, detail: bool = False, **kwargs: Any):
        path = self._norm(path)
        out: dict[str, dict[str, Any]] = {}

        if self.isfile(path):
            info = self.info(path)
            if detail:
                return {path: info}
            return [path]

        for root, dirs, files in self.walk(path, maxdepth=maxdepth, **kwargs):
            if withdirs:
                if root != path:
                    out[root] = self.info(root)
                for d in dirs:
                    p = posixpath.join(root, d) if root != "/" else "/" + d
                    out[p] = self.info(p)

            for f in files:
                p = posixpath.join(root, f) if root != "/" else "/" + f
                out[p] = self.info(p)

        if detail:
            return dict(sorted(out.items()))
        return sorted(out)

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

        pat = pattern.lstrip("/")
        return sorted(
            p for p in candidates
            if PurePosixPath(p.lstrip("/")).match(pat)
        )

    def du(self, path: str, total: bool = True, maxdepth: int | None = None, withdirs: bool = False, **kwargs: Any):
        entries = self.find(path, maxdepth=maxdepth, withdirs=withdirs, detail=True)
        sizes = {p: int(info.get("size", 0)) for p, info in entries.items()}
        if total:
            return sum(sizes.values())
        return sizes

    def expand_path(self, path, recursive: bool = False, maxdepth: int | None = None, **kwargs: Any):
        if isinstance(path, (list, tuple, set)):
            out: list[str] = []
            for p in path:
                out.extend(self.expand_path(p, recursive=recursive, maxdepth=maxdepth, **kwargs))
            return sorted(set(out))

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
        Passthrough for methods/properties not explicitly overridden.
        """
        return getattr(self.base_fs, name)


class _OverlayWriteFile:
    def __init__(self, fs: MemoryOverlayFileSystem, path: str, mode: str, encoding: str = "utf-8") -> None:
        self.fs = fs
        self.path = fs._norm(path)
        self.mode = mode
        self.encoding = encoding
        self.closed = False
        self.binary = "b" in mode

        if "x" in mode and fs.exists(self.path):
            raise FileExistsError(self.path)

        initial = b""
        if "a" in mode:
            if self.path in fs._files:
                initial = fs._files[self.path]
            elif not fs._is_deleted(self.path) and fs._base_exists(self.path):
                with fs.base_fs.open(self.path, "rb") as f:
                    initial = f.read()
        elif "+" in mode:
            if self.path in fs._files:
                initial = fs._files[self.path]
            elif not fs._is_deleted(self.path) and fs._base_exists(self.path):
                with fs.base_fs.open(self.path, "rb") as f:
                    initial = f.read()

        self._buf = io.BytesIO()
        if initial:
            self._buf.write(initial)
        if "a" not in mode:
            self._buf.seek(0)
            if "w" in mode:
                self._buf.truncate(0)

        fs._ensure_parent_dirs(self.path)
        fs._unhide_path(self.path)

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