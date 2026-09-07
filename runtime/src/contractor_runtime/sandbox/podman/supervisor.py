"""Private guardian attachment primitive; allocation wiring is deliberately absent."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

from contractor_runtime.sandbox.contracts import (
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
    SandboxIdentity,
)
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.guardian import CgroupFence, liveness_deadline
from contractor_runtime.sandbox.podman.io import local_engine_environment, remaining
from contractor_runtime.sandbox.podman.ownership import open_directory

_START_CLEANUPS: set[asyncio.Task] = set()


def open_pidfd(pid: int) -> int:
    # python-build-standalone may omit os.pidfd_open despite a capable host
    # glibc/kernel. Use the named libc ABI, never architecture-specific syscall
    # numbers or a PID-only liveness fallback.
    function = getattr(os, "pidfd_open", None)
    if function is not None:
        return function(pid)
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.pidfd_open
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    os.set_inheritable(descriptor, False)
    return descriptor


def open_fence(container_id: str, state: dict) -> CgroupFence:
    """Accept only a running, already ownership-verified local inspect record.

    Caller must hold the engine owner and prohibit start/exec until ready.
    This does not substitute for checking the five ownership labels.
    """
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise ValueError("invalid identity")
    path, pid = state["CgroupPath"], state["Pid"]
    prefix = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
    if (
        not isinstance(path, str)
        or not path.startswith(prefix)
        or Path(path).name != f"libpod-{container_id}.scope"
        or ".." in Path(path).parts
        or type(pid) is not int
        or pid <= 1
        or state["Running"] is not True
        or state["Status"] != "running"
    ):
        raise ValueError("unsupported cgroup delegation")
    directory = open_directory(Path("/sys/fs/cgroup") / path.lstrip("/"))
    pidfd = None
    try:
        pidfd = open_pidfd(pid)
        fence = CgroupFence(directory, pid, pidfd)
        membership = Path(f"/proc/{pid}/cgroup").read_text()
        actual = membership.removeprefix("0::").strip()
        if not (actual == path or actual.startswith(path + "/")):
            raise ValueError("init outside owned scope")
        status = Path(f"/proc/{pid}/status").read_text()
        uid = next(line.split()[1] for line in status.splitlines() if line.startswith("Uid:"))
        if int(uid) in {0, os.getuid()} or not fence.init_alive():
            raise ValueError("init must use an isolated subordinate UID")
        # Verify write authority before accepting any command. No broad cgroup
        # parent is opened for writes, and no signal uses a remembered host PID.
        for name in ("cgroup.kill", "cgroup.freeze"):
            fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            os.close(fd)
        return fence
    except BaseException:
        if pidfd is not None:
            os.close(pidfd)
        os.close(directory)
        raise


class GuardianClient:
    """One serialized host-only channel; close/EOF is fatal, never a detach API.

    `lease` is an absolute monotonic conversion of a *confirmed* control lease.
    Runtime must renew more frequently than the guardian's ten-second ceiling.
    An exec transport must never inherit these descriptors.
    """

    def __init__(self, control: socket.socket, process: asyncio.subprocess.Process) -> None:
        self._control = control
        self._process = process
        self._lock = asyncio.Lock()

    @classmethod
    async def start(cls, fence: CgroupFence, *, lease: float) -> GuardianClient:
        liveness_deadline(lease, time.monotonic())
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        parent.setblocking(False)
        # The spawn owner retains duplicate capabilities even if the caller
        # cancels, closes its originals, or the child arrives after its deadline.
        try:
            directory = os.dup(fence.directory)
            try:
                pidfd = os.dup(fence.pidfd)
            except BaseException:
                os.close(directory)
                raise
        except BaseException:
            parent.close()
            child.close()
            raise

        async def spawn():
            try:
                return await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-I",
                    "-B",
                    "-m",
                    "contractor_runtime.sandbox.podman.guardian",
                    str(child.fileno()),
                    str(directory),
                    str(fence.init_pid),
                    str(pidfd),
                    str(lease),
                    pass_fds=(child.fileno(), directory, pidfd),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                    env=local_engine_environment(),
                    cwd="/",
                )
            finally:
                child.close()
                os.close(directory)
                os.close(pidfd)

        task = asyncio.create_task(spawn(), name="podman-guardian-spawn")
        try:
            process = await asyncio.wait_for(
                asyncio.shield(task), remaining(min(lease, time.monotonic() + 3))
            )
            client = cls(parent, process)
            await client._receive("ready", min(lease, time.monotonic() + 3))
            return client
        except BaseException:
            parent.close()

            async def reap_late_spawn():
                try:
                    process = await asyncio.shield(task)
                    await process.wait()
                except (Exception, asyncio.CancelledError):
                    pass

            cleanup = asyncio.create_task(reap_late_spawn(), name="podman-guardian-start-cleanup")
            _START_CLEANUPS.add(cleanup)
            cleanup.add_done_callback(_START_CLEANUPS.discard)
            raise

    async def _receive(self, expected: str, deadline: float) -> None:
        try:
            timeout = remaining(deadline)
            raw = await asyncio.wait_for(
                asyncio.get_running_loop().sock_recv(self._control, 257), timeout
            )
            if len(raw) > 256 or json.loads(raw) != {"status": expected}:
                raise ValueError("unconfirmed guardian response")
        except (ValueError, OSError, TimeoutError):
            self._control.close()
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED) from None

    async def request(self, operation: str, *, deadline: float, lease: float | None = None) -> None:
        try:
            async with asyncio.timeout(remaining(deadline)):
                async with self._lock:
                    if operation not in {"renew", "check", "stop"}:
                        raise ValueError("unknown guardian operation")
                    message: dict = {"op": operation}
                    if operation == "renew":
                        liveness_deadline(lease, time.monotonic())
                        message["lease"] = lease
                    await asyncio.get_running_loop().sock_sendall(
                        self._control, json.dumps(message).encode("ascii")
                    )
                    await self._receive(
                        {"renew": "renewed", "check": "clean", "stop": "stopped"}[operation],
                        deadline,
                    )
        except (OSError, ValueError, TimeoutError):
            # A concurrent reject closes this socket while exec is returning.
            # Preserve the owner's control channel so confirmed stop/removal can
            # still be acknowledged; a transport error is never a clean proof.
            self._control.close()
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED) from None
        except BaseException:
            self._control.close()  # cancellation also irrevocably fences execution
            raise

    async def close(self) -> None:
        self.disconnect()
        await asyncio.wait_for(self._process.wait(), 6)

    def disconnect(self) -> None:
        """Irrevocably request termination; NOT a successful cleanup receipt."""
        self._control.close()


class CompletionGate:
    """Promote a private transport result only with independent cleanup proof.

    The future executor supplies the trusted program status (not parsed command
    stdout), owns the workspace guard and fences/reaps every launch before this
    call. This primitive neither launches commands nor exposes engine authority
    to model tools. Any exception makes the allocation permanently unusable;
    the lifecycle must retain workspace ownership until termination is proved.
    """

    def __init__(
        self, engine: PodmanEngine, identity: SandboxIdentity, guardian: GuardianClient
    ) -> None:
        self._engine = engine
        self._identity = identity
        self._guardian = guardian
        self._failed = False

    async def confirm(self, result: ExecutionResult, *, deadline: float) -> ExecutionResult:
        if self._failed:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        try:
            if result.status != ExecutionStatus.COMPLETED:
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
            await self._guardian.request("check", deadline=deadline)
            state = await self._engine.inspect(self._identity, deadline=deadline)
            if state is None or not state.running or state.status != "running":
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
            remaining(deadline)
            return result
        except BaseException:
            self._failed = True
            self._guardian.disconnect()
            raise
