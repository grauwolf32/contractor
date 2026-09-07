"""Runtime crash/stall fixture. Uses the production owner process and guardian."""

import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fsspec.implementations.local import LocalFileSystem

from contractor_runtime.projectfs import DirectWorkspaceSession
from contractor_runtime.projectfs.provider import ProjectWorkspaceStorage
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings
from contractor_runtime.settings import WorkspaceLimits


async def main():
    config = json.loads(sys.stdin.readline())
    lifecycle = PodmanLifecycle(PodmanSettings(**config["settings"]))
    lease = time.monotonic() + 120
    lifecycle.bind_health(lambda: lease, lambda: None)
    await lifecycle.recover(deadline=time.monotonic() + 30)
    root = Path(config["root"])
    session = DirectWorkspaceSession(
        mode="direct",
        storage=ProjectWorkspaceStorage(
            "local", LocalFileSystem(), str(root.parent), "test", "test"
        ),
        content_root=str(root),
        limits=WorkspaceLimits(100, 100000, 100000, 10000),
        directories=set(),
        text_files={},
        binary_paths=set(),
        stored_binary_paths=set(),
    )
    allocation = lifecycle.allocate("crash-test", session)
    try:
        await allocation.prepare(deadline=datetime.now(UTC) + timedelta(seconds=30))
        print(json.dumps(asdict(allocation.identity)), flush=True)
        await asyncio.Future()
    finally:
        await lifecycle.close(deadline=time.monotonic() + 30)
        await session.close()


asyncio.run(main())
