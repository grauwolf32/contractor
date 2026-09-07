"""Independent host-only cgroup authority. Never imported into the image.

The private inherited socket, cgroup descriptor and pidfd are capabilities, not
tokens. None is inherited by Podman or mounted in the workload namespace.
"""

from __future__ import annotations

import json
import math
import os
import select
import socket
import sys
import time

MAX_LIVENESS_SECONDS = 10.0
POLL_SECONDS = 0.025


def liveness_deadline(lease: object, now: float) -> float:
    if type(lease) not in (int, float) or not math.isfinite(lease) or lease <= now:
        raise ValueError("expired/invalid lease")
    return min(lease, now + MAX_LIVENESS_SECONDS)


class CgroupFence:
    """Pinned exact container scope, including all descendant cgroups."""

    def __init__(self, directory: int, init_pid: int, pidfd: int) -> None:
        self.directory = directory
        self.init_pid = init_pid
        self.pidfd = pidfd

    def read(self, name: str, directory: int | None = None) -> str:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self.directory if directory is None else directory,
        )
        try:
            data = os.read(fd, 65537)
            if len(data) > 65536:
                raise ValueError("oversized cgroup record")
            return data.decode("ascii")
        finally:
            os.close(fd)

    def write(self, name: str, value: bytes) -> None:
        fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=self.directory)
        try:
            if os.write(fd, value) != len(value):
                raise OSError("short cgroup write")
        finally:
            os.close(fd)

    def events(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in (line.split() for line in self.read("cgroup.events").splitlines())
        }

    def init_alive(self) -> bool:
        return not select.select([self.pidfd], [], [], 0)[0]

    def processes(self) -> set[int]:
        # No workload can write its read-only cgroup mount. Recursion still
        # covers OCI runtime subgroups; never rely on process ancestry or pgids.
        budget = 32

        def visit(directory: int) -> set[int]:
            nonlocal budget
            budget -= 1
            if budget < 0:
                raise ValueError("unsupported cgroup tree")
            found = {int(pid) for pid in self.read("cgroup.procs", directory).split()}
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory,
                        )
                        try:
                            found |= visit(child)
                        finally:
                            os.close(child)
            return found

        return visit(self.directory)

    def clean(self, deadline: float) -> bool:
        self.write("cgroup.freeze", b"1")
        while self.events()["frozen"] != 1:
            if time.monotonic() >= deadline:
                return False
            time.sleep(POLL_SECONDS)
        clean = self.init_alive() and self.processes() == {self.init_pid}
        if clean and time.monotonic() < deadline:
            self.write("cgroup.freeze", b"0")
            return True
        # A failed proof intentionally leaves execution frozen until kill.
        return False

    def kill(self) -> None:
        self.write("cgroup.kill", b"1")

    def empty(self) -> bool:
        try:
            return self.events()["populated"] == 0 and not self.init_alive()
        except FileNotFoundError:
            # A pinned, removed cgroup cannot be reused by a later container.
            return not self.init_alive()


def serve(control: socket.socket, fence: CgroupFence, lease: float) -> None:
    """No stdout protocol, filesystem socket, secret or workload UID authority."""
    try:
        control.settimeout(POLL_SECONDS)
        deadline = liveness_deadline(lease, time.monotonic())
        control.send(b'{"status":"ready"}')
        while time.monotonic() < deadline and fence.init_alive():
            readable = select.select(
                [control], [], [], min(POLL_SECONDS, max(0, deadline - time.monotonic()))
            )[0]
            if not readable:
                continue
            packet = control.recv(257)
            if not packet or len(packet) > 256:
                break
            message = json.loads(packet)
            if not isinstance(message, dict):
                break
            if message.get("op") == "renew" and set(message) == {"op", "lease"}:
                # Renewals after expiration never revive a fence, even if queued.
                if time.monotonic() >= deadline:
                    break
                deadline = liveness_deadline(message["lease"], time.monotonic())
                control.send(b'{"status":"renewed"}')
            elif message == {"op": "check"}:
                if not fence.clean(min(deadline, time.monotonic() + 1)):
                    break
                if time.monotonic() >= deadline:
                    break
                control.send(b'{"status":"clean"}')
            else:
                break
    except (OSError, ValueError, KeyError, TypeError):
        pass
    finally:
        # This path has no Podman CLI dependency, including after Runtime death.
        # Never acknowledge cleanup before kernel-confirmed absence of writers.
        try:
            fence.kill()
            until = time.monotonic() + 5
            while not fence.empty():
                if time.monotonic() >= until:
                    # EOF is uncertainty, never a successful stop ack.
                    raise TimeoutError("unconfirmed cgroup termination")
                time.sleep(POLL_SECONDS)
            control.send(b'{"status":"stopped"}')
        except OSError:
            pass


if __name__ == "__main__":
    socket_fd, directory, init_pid, pidfd = map(int, sys.argv[1:5])
    with socket.socket(fileno=socket_fd) as control:
        serve(control, CgroupFence(directory, init_pid, pidfd), float(sys.argv[5]))
