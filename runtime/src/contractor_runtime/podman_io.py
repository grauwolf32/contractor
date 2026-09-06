"""Bounded local CLI transport and retained asynchronous operation ownership."""

from __future__ import annotations

import asyncio
import math
import os
import pwd
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeVar

from contractor_runtime.sandbox_contracts import SandboxContractError, SandboxErrorCode

T = TypeVar("T")
MAX_CLI_BYTES = 1 << 20


def remaining(deadline: float) -> float:
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise SandboxContractError(SandboxErrorCode.TIMEOUT)
    return deadline - time.monotonic()


class OwnedOperations:
    """Caller timeout/cancellation never releases a running operation's lock.

    An uncooperative transport/kernel operation may remain retained indefinitely;
    subsequent callers still have bounded waits and must not report cleanup.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[object] | None = None

    async def run(self, operation: Callable[[], Awaitable[T]], deadline: float) -> T:
        timeout = remaining(deadline)
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout)
        except TimeoutError:
            raise SandboxContractError(SandboxErrorCode.TIMEOUT) from None

        async def owned() -> T:
            try:
                remaining(deadline)
                return await operation()
            finally:
                self._lock.release()

        task = asyncio.create_task(owned(), name="podman-owned-operation")
        self._task = task
        task.add_done_callback(self._completed)
        try:
            return await asyncio.wait_for(asyncio.shield(task), remaining(deadline))
        except TimeoutError:
            raise SandboxContractError(SandboxErrorCode.TIMEOUT) from None

    def _completed(self, task: asyncio.Task[object]) -> None:
        if not task.cancelled():
            task.exception()
        if self._task is task:
            self._task = None


@dataclass(frozen=True, slots=True)
class CLIResult:
    returncode: int
    stdout: bytes = field(default=b"", repr=False)


class PodmanCLI(Protocol):
    async def run(self, arguments: tuple[str, ...], *, deadline: float) -> CLIResult:
        """Called only by an owned engine operation, never by model tools."""
        ...


def local_engine_environment() -> Mapping[str, str]:
    """Do not inherit Runtime credentials, proxy variables or remote selection."""
    uid = os.getuid()
    if uid == 0 or os.geteuid() != uid:
        raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
    user = pwd.getpwuid(uid)
    runtime_dir = f"/run/user/{uid}"
    return MappingProxyType(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": user.pw_dir,
            "XDG_RUNTIME_DIR": runtime_dir,
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime_dir}/bus",
            "LANG": "C.UTF-8",
        }
    )


class _Capture(asyncio.SubprocessProtocol):
    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.exited: asyncio.Future[None] = loop.create_future()
        self.finished: asyncio.Future[None] = loop.create_future()
        self.transport: asyncio.SubprocessTransport | None = None
        self.stdout = bytearray()
        self.count = 0
        self.pipes = {1, 2}
        self.error: SandboxErrorCode | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.error is not None:
            return
        self.count += len(data)
        if self.count > MAX_CLI_BYTES:
            self.abort(SandboxErrorCode.OUTPUT_LIMIT)
        elif fd == 1:
            self.stdout.extend(data)
        # stderr is counted but never retained or surfaced.

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        self.pipes.discard(fd)
        if exc is not None:
            self.error = SandboxErrorCode.OUTCOME_UNKNOWN
        self._finish()

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)
        self._finish()

    def _finish(self) -> None:
        if self.exited.done() and not self.pipes and not self.finished.done():
            self.finished.set_result(None)

    def abort(self, code: SandboxErrorCode) -> None:
        self.error = self.error or code
        assert self.transport is not None
        # Signal only our CLI child through its owning subprocess transport;
        # never send killpg to a remembered PID after its leader has exited.
        # This is NOT proof that helpers or the container have stopped.
        if self.transport.get_returncode() is None:
            with suppress(ProcessLookupError):
                self.transport.kill()
        for fd in (1, 2):
            pipe = self.transport.get_pipe_transport(fd)
            if pipe is not None:
                pipe.close()


class LocalPodmanCLI:
    def __init__(self, executable: Path = Path("/usr/bin/podman")) -> None:
        if not executable.is_absolute():
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        self._executable = executable
        self._environment = local_engine_environment()

    async def run(self, arguments: tuple[str, ...], *, deadline: float) -> CLIResult:
        remaining(deadline)
        protocol = _Capture()
        try:
            transport, _ = await asyncio.get_running_loop().subprocess_exec(
                lambda: protocol,
                str(self._executable),
                "--remote=false",
                "--log-level=error",
                "--events-backend=none",
                *arguments,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(self._environment),
                cwd="/",
                start_new_session=True,
            )
        except (OSError, ValueError):
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE) from None
        try:
            try:
                await asyncio.wait_for(asyncio.shield(protocol.finished), remaining(deadline))
            except (TimeoutError, SandboxContractError):
                protocol.abort(SandboxErrorCode.TIMEOUT)
            except asyncio.CancelledError:
                protocol.abort(SandboxErrorCode.OUTCOME_UNKNOWN)
                raise
            finally:
                # Keep the engine's owner and service lock if kernel reaping is
                # delayed. Its caller remains bounded by OwnedOperations.
                await asyncio.shield(protocol.exited)
            if protocol.error is not None:
                raise SandboxContractError(protocol.error)
            code = transport.get_returncode()
            assert code is not None
            return CLIResult(code, bytes(protocol.stdout))
        finally:
            transport.close()
