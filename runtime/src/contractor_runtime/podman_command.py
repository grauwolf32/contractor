"""Bounded foreground exec transport, used only by the surviving host owner.

CLI exit 125/126/127 is ambiguous and is never promoted to a program result.
All other candidate exit codes still require the independent completion gate.
No command output is interpreted as a status/guardian protocol.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from contractor_runtime.podman_io import local_engine_environment, remaining
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.sandbox_contracts import SandboxErrorCode, SandboxIdentity


@dataclass(frozen=True, repr=False)
class CommandCapture:
    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_bytes: int
    stderr_bytes: int
    error: SandboxErrorCode | None = None


class _Capture(asyncio.SubprocessProtocol):
    def __init__(self, settings: PodmanSettings):
        loop = asyncio.get_running_loop()
        self.exited = loop.create_future()
        self.finished = loop.create_future()
        self.transport = None
        self.settings = settings
        self.preview = {1: bytearray(), 2: bytearray()}
        self.count = {1: 0, 2: 0}
        self.pipes = {1, 2}
        self.error = None

    def connection_made(self, transport):
        self.transport = transport

    def pipe_data_received(self, fd, data):
        if self.error is not None:
            return
        self.count[fd] += len(data)
        take = max(0, self.settings.preview_bytes - len(self.preview[fd]))
        self.preview[fd].extend(data[:take])
        if sum(self.count.values()) > self.settings.output_max_bytes:
            self.abort(SandboxErrorCode.OUTPUT_LIMIT)

    def pipe_connection_lost(self, fd, exc):
        self.pipes.discard(fd)
        if exc is not None:
            self.abort(SandboxErrorCode.OUTCOME_UNKNOWN)
        self._finish()

    def process_exited(self):
        if not self.exited.done():
            self.exited.set_result(None)
        self._finish()

    def _finish(self):
        if self.exited.done() and not self.pipes and not self.finished.done():
            self.finished.set_result(None)

    def abort(self, code):
        self.error = self.error or code
        if self.transport is not None:
            if self.transport.get_returncode() is None:
                with suppress(ProcessLookupError):
                    self.transport.kill()
            for fd in (1, 2):
                pipe = self.transport.get_pipe_transport(fd)
                if pipe is not None:
                    pipe.close()


class PodmanCommand:
    def __init__(self, settings: PodmanSettings, *, executable: Path = Path("/usr/bin/podman")):
        if not executable.is_absolute():
            raise ValueError("absolute local executable required")
        self.settings = settings
        self.executable = executable

    async def run(
        self, identity: SandboxIdentity, command: str, cwd: str, *, deadline: float
    ) -> CommandCapture:
        remaining(deadline)
        capture = _Capture(self.settings)
        transport = None
        try:
            transport, _ = await asyncio.get_running_loop().subprocess_exec(
                lambda: capture,
                str(self.executable),
                "--remote=false",
                "--log-level=error",
                "--events-backend=none",
                "exec",
                "--interactive=false",
                "--tty=false",
                "--privileged=false",
                "--detach-keys=",
                f"--user={os.getuid()}:{os.getgid()}",
                "--env=PATH=/usr/local/bin:/usr/bin:/bin",
                "--env=HOME=/tmp",
                "--env=LANG=C.UTF-8",
                "--workdir=/workspace" + ("/" + cwd if cwd else ""),
                identity.container_id,
                "/bin/sh",
                "-c",
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=local_engine_environment(),
                cwd="/",
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(capture.finished), max(0, deadline - time.monotonic())
                )
            except TimeoutError:
                capture.abort(SandboxErrorCode.TIMEOUT)
            except asyncio.CancelledError:
                capture.abort(SandboxErrorCode.OUTCOME_UNKNOWN)
                raise
            finally:
                # The owner process retains this task until the exact CLI is
                # reaped; this is not yet proof of container descendant cleanup.
                await asyncio.shield(capture.exited)
            code = transport.get_returncode()
            error = capture.error
            if error is None and (code is None or code < 0 or code in {125, 126, 127}):
                error = SandboxErrorCode.OUTCOME_UNKNOWN
            # Engine diagnostics are not authorized command observations.
            hide = error == SandboxErrorCode.OUTCOME_UNKNOWN
            return CommandCapture(
                code if error is None else None,
                b"" if hide else bytes(capture.preview[1]),
                b"" if hide else bytes(capture.preview[2]),
                capture.count[1],
                capture.count[2],
                error,
            )
        except OSError:
            return CommandCapture(None, b"", b"", 0, 0, SandboxErrorCode.OUTCOME_UNKNOWN)
        finally:
            if transport is not None:
                transport.close()
