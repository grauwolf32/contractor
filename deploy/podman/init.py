"""Inert namespace init; root inside keep-id, no capabilities or control API.

Workloads are exec'd as the host user's nonzero container UID. This process
does not parse workspace files, import project code, or accept workload input.
All completion and liveness authority stays in the host guardian.
"""

import os
import signal


def reap(_signal=None, _frame=None):
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


if os.getpid() != 1 or os.getuid() != 0:
    raise SystemExit(78)

signal.signal(signal.SIGCHLD, reap)
signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
while True:
    signal.pause()
