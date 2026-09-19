"""Bounded scanner subprocesses with cancellation and process-group cleanup."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 1024 * 1024
PREVIEW_BYTES = 32 * 1024


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    error_code: str | None = None
    duration_ms: int = 0

    def observation(self) -> dict:
        return {
            "status": "failed" if self.error_code else "completed",
            "exitCode": self.exit_code,
            "errorCode": self.error_code,
            "stdout": self.stdout[:PREVIEW_BYTES].decode("utf-8", errors="replace"),
            "stderr": self.stderr[:PREVIEW_BYTES].decode("utf-8", errors="replace"),
            "stdoutTruncated": len(self.stdout) > PREVIEW_BYTES,
            "stderrTruncated": len(self.stderr) > PREVIEW_BYTES,
            "outputLimitExceeded": self.error_code == "output_limit_exceeded",
            "durationMs": self.duration_ms,
        }


class _OutputLimit(Exception):
    pass


def child_environment(directory: Path) -> dict[str, str]:
    # Never inherit credentials, proxy settings, scanner config or Python hooks.
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(directory),
        "XDG_CONFIG_HOME": str(directory / "config"),
        "XDG_CACHE_HOME": str(directory / "cache"),
        "TMPDIR": str(directory),
        "LANG": "C.UTF-8",
        "NO_COLOR": "1",
        "CI": "1",
        "DISABLE_NUCLEI_TEMPLATES_PUBLIC_DOWNLOAD": "true",
        "DISABLE_NUCLEI_TEMPLATES_GITHUB_DOWNLOAD": "true",
        "DISABLE_NUCLEI_TEMPLATES_GITLAB_DOWNLOAD": "true",
        "DISABLE_NUCLEI_TEMPLATES_AWS_DOWNLOAD": "true",
        "DISABLE_NUCLEI_TEMPLATES_AZURE_DOWNLOAD": "true",
    }


async def _stop(process: asyncio.subprocess.Process) -> None:
    # Also kill descendants holding pipe descriptors after the leader exits.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.communicate()


async def run_process(command: list[str], directory: Path, timeout: float) -> ProcessResult:
    started = time.monotonic()
    stdout, stderr = bytearray(), bytearray()
    total = 0
    error_code = None

    async def read(stream: asyncio.StreamReader, output: bytearray) -> None:
        nonlocal total
        while chunk := await stream.read(8192):
            remaining = MAX_OUTPUT_BYTES - total
            output.extend(chunk[:remaining])
            total += min(len(chunk), remaining)
            if len(chunk) > remaining:
                raise _OutputLimit

    # A cancellation during process creation must still collect and stop the child.
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            cwd=directory,
            env=child_environment(directory),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    )
    try:
        process = await asyncio.shield(spawning)
    except asyncio.CancelledError:
        with suppress(OSError):
            process = await spawning
            await _stop(process)
        raise
    except OSError:
        return ProcessResult(None, error_code="scanner_unavailable")

    assert process.stdout is not None and process.stderr is not None
    readers = [
        asyncio.create_task(read(process.stdout, stdout)),
        asyncio.create_task(read(process.stderr, stderr)),
    ]
    try:
        async with asyncio.timeout(timeout):
            await asyncio.gather(*readers)
            await process.wait()
    except TimeoutError:
        error_code = "scan_timeout"
    except _OutputLimit:
        error_code = "output_limit_exceeded"
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        # Shield cleanup from repeated cancellation (worker abort + allocation close).
        cleanup = asyncio.create_task(_stop(process))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
    if error_code is None and process.returncode != 0:
        error_code = "scanner_failed"
    return ProcessResult(
        process.returncode,
        bytes(stdout),
        bytes(stderr),
        error_code,
        max(0, int((time.monotonic() - started) * 1000)),
    )
