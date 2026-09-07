"""Persistent root policy prevents disabling/changing the owner around recovery.

This marker is outside the mounted content and is never automatically removed.
Changing a dedicated root's service owner requires an operator migration.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from contractor_runtime.sandbox.contracts import SandboxContractError, SandboxErrorCode
from contractor_runtime.sandbox.podman.ownership import open_directory

MARKER = ".contractor-podman-owner-v1"


def check_root_policy(root: Path, owner: str | None) -> None:
    root = root.expanduser().absolute()
    if owner is None and not os.path.lexists(root / MARKER):
        return
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory = open_directory(root)
    try:
        try:
            descriptor = os.open(
                MARKER, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory
            )
        except FileNotFoundError:
            if owner is None:
                return
            descriptor = os.open(
                MARKER,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            try:
                raw = (owner + "\n").encode("ascii")
                if os.write(descriptor, raw) != len(raw):
                    raise OSError("incomplete root marker")
                os.fsync(descriptor)
                os.fsync(directory)
            finally:
                os.close(descriptor)
            return
        try:
            info = os.fstat(descriptor)
            expected = (owner + "\n").encode("ascii") if owner is not None else None
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
                or info.st_size > 128
                or expected is None
                or os.read(descriptor, 129) != expected
            ):
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)
