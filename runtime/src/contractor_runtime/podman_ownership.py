"""Private POSIX owner locks and pinned project bind roots. No recursive cleanup."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from contextlib import suppress
from pathlib import Path

from contractor_runtime.sandbox_contracts import SandboxContractError, SandboxErrorCode


def open_directory(path: Path) -> int:
    """Open each component without symlink following, including parents."""
    if not path.is_absolute() or ".." in path.parts:
        raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in path.parts[1:]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd
            )
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


class ServiceOwnerLock:
    def __init__(self, directory: Path, owner: str) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", owner) is None:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        self._directory = directory
        self._name = owner + ".lock"
        self._fd: int | None = None
        self._directory_fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            self.verify()
            return
        directory_fd = None
        lock_fd = None
        try:
            parent = open_directory(self._directory.parent)
            try:
                with suppress(FileExistsError):
                    os.mkdir(self._directory.name, mode=0o700, dir_fd=parent)
            finally:
                os.close(parent)
            directory_fd = open_directory(self._directory)
            info = os.fstat(directory_fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise OSError("unsafe lock directory")
            lock_fd = os.open(
                self._name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory_fd,
            )
            info = os.fstat(lock_fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise OSError("unsafe lock file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd, self._directory_fd = lock_fd, directory_fd
        except (OSError, SandboxContractError):
            if lock_fd is not None:
                os.close(lock_fd)
            if directory_fd is not None:
                os.close(directory_fd)
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE) from None

    def verify(self) -> None:
        if self._fd is None or self._directory_fd is None:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        try:
            current = os.stat(self._name, dir_fd=self._directory_fd, follow_symlinks=False)
            held = os.fstat(self._fd)
            path_fd = open_directory(self._directory)
            try:
                directory = os.fstat(path_fd)
                pinned = os.fstat(self._directory_fd)
                if (
                    (directory.st_dev, directory.st_ino) != (pinned.st_dev, pinned.st_ino)
                    or directory.st_uid != os.geteuid()
                    or stat.S_IMODE(directory.st_mode) != 0o700
                ):
                    raise OSError("lock directory changed")
            finally:
                os.close(path_fd)
            if (
                (current.st_dev, current.st_ino, current.st_nlink) != (held.st_dev, held.st_ino, 1)
                or not stat.S_ISREG(current.st_mode)
                or current.st_uid != os.geteuid()
                or stat.S_IMODE(current.st_mode) != 0o600
            ):
                raise OSError("lock identity changed")
        except OSError:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE) from None

    def release(self) -> None:
        # Never unlink: another owner can already be waiting on this inode.
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._directory_fd is not None:
            os.close(self._directory_fd)
            self._directory_fd = None


class ContentPin:
    def __init__(self, path: Path) -> None:
        text = str(path)
        if path.name != "run_workdir" or any(ord(c) < 32 or c in ',"' for c in text):
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        self.path = path
        try:
            self._fd = open_directory(path)
        except (OSError, ValueError):
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE) from None

    def verify(self) -> None:
        try:
            fd = open_directory(self.path)
            try:
                current, pinned = os.fstat(fd), os.fstat(self._fd)
                if (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                    raise OSError("bind root changed")
            finally:
                os.close(fd)
        except OSError:
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN) from None

    def close(self) -> None:
        os.close(self._fd)
