"""Markers for tests whose behaviour the specification limits to one platform."""

from __future__ import annotations

import sys

import pytest

LINUX_PODMAN_REASON = (
    "OPS-R1: the allocation-scoped Podman backend needs Linux host capabilities "
    "(AF_UNIX SOCK_SEQPACKET socket pairs, pidfd_open)"
)

# Skips a test of the Podman backend's own owner, supervisor or allocation
# protocol on a host other than Linux. Nothing skips it on Linux.
linux_podman = pytest.mark.skipif(sys.platform != "linux", reason=LINUX_PODMAN_REASON)

# The two filesystem fixtures assume Linux name lookup and path-length semantics.
linux_filesystem = pytest.mark.skipif(
    sys.platform != "linux",
    reason="OPS-R1: Linux byte-exact filename/long-path fixture; excluded from macOS acceptance",
)
