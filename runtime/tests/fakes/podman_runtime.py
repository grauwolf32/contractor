"""Real crash-gate Runtime-side controller, using the production guardian API."""

import asyncio
import sys
import time

from contractor_runtime.podman_guardian import CgroupFence
from contractor_runtime.podman_supervisor import GuardianClient


async def main():
    directory, init_pid, pidfd = map(int, sys.argv[1:])
    guardian = await GuardianClient.start(
        CgroupFence(directory, init_pid, pidfd), lease=time.monotonic() + 2
    )
    print("ready", flush=True)
    try:
        while True:
            await guardian.request(
                "renew", lease=time.monotonic() + 2, deadline=time.monotonic() + 1
            )
            await asyncio.sleep(0.1)
    finally:
        await guardian.close()


asyncio.run(main())
