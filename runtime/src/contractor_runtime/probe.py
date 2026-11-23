"""Bounded, output-free local dependency probes."""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Sequence


async def executable_responds(name: str, arguments: Sequence[str]) -> bool:
    """Return whether one PATH-resolved executable completes successfully.

    The coordinator owns the timeout. This helper owns child cleanup when that
    timeout cancels it and deliberately discards all child output.
    """

    executable = shutil.which(name)
    if executable is None:
        return False
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "CI": "1",
        "NO_COLOR": "1",
        "NO_UPDATE_NOTIFIER": "1",
    }
    process = await asyncio.create_subprocess_exec(
        executable,
        *arguments,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=environment,
    )
    try:
        return await process.wait() == 0
    except asyncio.CancelledError:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
