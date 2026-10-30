"""The path an open descriptor names, for tests that patch descriptor I/O."""

from __future__ import annotations

import fcntl
import os
import sys

# MAXPATHLEN on macOS, the size F_GETPATH fills.
_DARWIN_PATH_BUFFER = 1024


def descriptor_path(descriptor: int) -> str:
    """The current path of an open descriptor, through /proc on Linux and F_GETPATH on macOS."""

    if sys.platform == "darwin":
        named = fcntl.fcntl(descriptor, fcntl.F_GETPATH, bytes(_DARWIN_PATH_BUFFER))
        return os.fsdecode(named.split(b"\0", 1)[0])
    return os.readlink(f"/proc/self/fd/{descriptor}")
