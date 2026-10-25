"""Private disposable cache files, accessed only through a pinned directory FD."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path

CACHE_DIRECTORY = ".contractor-archive-cache"
_MARKER_NAME = ".owner"
MARKER = b"contractor-runtime-archive-cache-v1\n"
_FILE_NAME = re.compile(r"[0-9a-f]{64}\.(?:data|meta)(?:\.part)?")
MAX_ENTRY_METADATA_BYTES = 4096


def _private(info: os.stat_result, *, directory: bool = False) -> bool:
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    mode = 0o700 if directory else 0o600
    return (
        kind(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == mode
        and (directory or info.st_nlink == 1)
    )


class CacheFiles:
    """Own only marker-authenticated flat cache state; never recurse or follow links."""

    def __init__(self, root: Path) -> None:
        self.path = root / CACHE_DIRECTORY
        self.fd = -1

    def initialize(self) -> bool:
        try:
            try:
                self.path.mkdir(mode=0o700)
                fresh = True
            except FileExistsError:
                fresh = False
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if not _private(os.fstat(self.fd), directory=True):
                raise OSError("unsafe cache directory")
            if fresh:
                self._write_new(_MARKER_NAME, MARKER)
            elif self._read(_MARKER_NAME, len(MARKER)) != MARKER:
                raise OSError("unowned cache directory")
            # Bound restart work even if a directory was modified while stopped.
            names: list[str] = []
            with os.scandir(self.fd) as children:
                for child in children:
                    if child.name == _MARKER_NAME:
                        continue
                    if len(names) >= 4 * 1024 or _FILE_NAME.fullmatch(child.name) is None:
                        raise OSError("unknown cache state")
                    info = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        raise OSError("unknown cache directory")
                    names.append(child.name)
            for name in names:
                os.unlink(name, dir_fd=self.fd)
            return True
        except OSError:
            self.close()
            return False

    def _read(self, name: str, size: int) -> bytes:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
        with os.fdopen(descriptor, "rb") as file:
            info = os.fstat(file.fileno())
            if not _private(info) or info.st_size != size:
                raise OSError("unsafe cache file")
            data = file.read(size + 1)
            if len(data) != size:
                raise OSError("inconsistent cache file")
            return data

    def read(self, stem: str, size: int, metadata: bytes, digest: str) -> bytes:
        if self._read(stem + ".meta", len(metadata)) != metadata:
            raise OSError("inconsistent cache metadata")
        data = self._read(stem + ".data", size)
        if hashlib.sha256(data).hexdigest() != digest:
            raise OSError("corrupt cache data")
        return data

    def _write_new(self, name: str, data: bytes) -> None:
        descriptor = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd
        )
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()

    def write(self, stem: str, data: bytes, metadata: bytes) -> None:
        if len(metadata) > MAX_ENTRY_METADATA_BYTES:
            raise OSError("cache metadata too large")
        self._write_new(stem + ".data.part", data)
        self._write_new(stem + ".meta.part", metadata)
        for suffix in (".data", ".meta"):
            os.replace(
                stem + suffix + ".part", stem + suffix, src_dir_fd=self.fd, dst_dir_fd=self.fd
            )

    def remove(self, stem: str) -> bool:
        success = True
        for suffix in (".data", ".meta", ".data.part", ".meta.part"):
            try:
                os.unlink(stem + suffix, dir_fd=self.fd)
            except FileNotFoundError:
                pass
            except OSError:
                success = False
        return success

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
