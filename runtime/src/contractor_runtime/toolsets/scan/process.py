"""Bounded scanner subprocesses with cancellation and process-group cleanup."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from contractor_runtime.toolsets.common.process import (
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_command,
)

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


async def run_process(command: list[str], directory: Path, timeout: float) -> ProcessResult:
    started = time.monotonic()
    error_code = None
    try:
        result = await run_command(
            command,
            cwd=directory,
            env=child_environment(directory),
            timeout=timeout,
            max_output_bytes=MAX_OUTPUT_BYTES,
        )
        exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        if exit_code != 0:
            error_code = "scanner_failed"
    except ProcessTimeoutError as error:
        exit_code, stdout, stderr = error.returncode, error.output, error.stderr
        error_code = "scan_timeout"
    except ProcessOutputLimitError as error:
        exit_code, stdout, stderr = error.returncode, error.stdout, error.stderr
        error_code = "output_limit_exceeded"
    except OSError:
        return ProcessResult(None, error_code="scanner_unavailable")
    return ProcessResult(
        exit_code,
        stdout,
        stderr,
        error_code,
        max(0, int((time.monotonic() - started) * 1000)),
    )
