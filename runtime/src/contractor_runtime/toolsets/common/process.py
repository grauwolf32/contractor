"""Bounded subprocess pipes with cancellation-safe process-group ownership."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path


class ProcessOutputLimitError(RuntimeError):
    def __init__(self, returncode: int | None, stdout: bytes, stderr: bytes) -> None:
        super().__init__("subprocess output limit exceeded")
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ProcessTimeoutError(subprocess.TimeoutExpired):
    def __init__(
        self,
        command: Sequence[str],
        timeout: float,
        returncode: int | None,
        stdout: bytes,
        stderr: bytes,
    ) -> None:
        super().__init__(command, timeout, output=stdout, stderr=stderr)
        self.returncode = returncode


class _OutputLimit(Exception):
    pass


async def _join_cleanup(task: asyncio.Task[None]) -> bool:
    """Repeated cancellation cannot detach the task which owns the child."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    return cancelled


async def _stop(process: asyncio.subprocess.Process) -> None:
    # Descendants can retain pipes after the leader exits. The complete process
    # group must stop even when the requested command has already returned.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    # A descendant that left the group (setsid, daemons) survives killpg and may
    # hold our pipes open indefinitely. Never wait for its EOF: close our ends,
    # which also lets wait() finish once the killed leader is reaped.
    transport = process._transport  # asyncio exposes no public pipe handle
    for fd in (0, 1, 2):
        pipe = transport.get_pipe_transport(fd)
        if pipe is not None:
            pipe.close()
    await process.wait()


async def run_command(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str],
    input: bytes | None = None,
    timeout: float,
    max_output_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    """Return bounded output only after the leader and its process group stop.

    A descendant that detached into another session is not waited for.
    """
    stdout, stderr = bytearray(), bytearray()
    total = 0
    failure = None

    async def read(stream: asyncio.StreamReader, output: bytearray) -> None:
        nonlocal total
        while chunk := await stream.read(8192):
            remaining = max_output_bytes - total
            output.extend(chunk[:remaining])
            total += min(len(chunk), remaining)
            if len(chunk) > remaining:
                raise _OutputLimit

    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE if input is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        ),
        name="tool-subprocess-spawn",
    )
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:

        async def finish_spawn() -> None:
            with suppress(OSError):
                await _stop(await spawning)

        await _join_cleanup(asyncio.create_task(finish_spawn(), name="tool-subprocess-cleanup"))
        raise

    async def write() -> None:
        assert process.stdin is not None and input is not None
        try:
            process.stdin.write(input)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    assert process.stdout is not None and process.stderr is not None
    pipes = [
        asyncio.create_task(read(process.stdout, stdout)),
        asyncio.create_task(read(process.stderr, stderr)),
    ]
    if input is not None:
        pipes.append(asyncio.create_task(write()))
    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*pipes)
            await process.wait()
    except TimeoutError:
        failure = "timeout"
    except _OutputLimit:
        failure = "output_limit"
    finally:

        async def cleanup() -> None:
            for pipe in pipes:
                pipe.cancel()
            await asyncio.gather(*pipes, return_exceptions=True)
            await _stop(process)

        cancelled = await _join_cleanup(
            asyncio.create_task(cleanup(), name="tool-subprocess-cleanup")
        )
        if cancelled:
            raise asyncio.CancelledError
    if failure == "timeout":
        raise ProcessTimeoutError(
            command, timeout, process.returncode, bytes(stdout), bytes(stderr)
        )
    if failure == "output_limit":
        raise ProcessOutputLimitError(process.returncode, bytes(stdout), bytes(stderr))
    return subprocess.CompletedProcess(command, process.returncode, bytes(stdout), bytes(stderr))
