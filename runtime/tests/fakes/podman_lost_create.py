"""Crash exactly after real create, before its caller receives any receipt."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from contractor_runtime.podman_engine import PodmanEngine
from contractor_runtime.podman_io import LocalPodmanCLI
from contractor_runtime.podman_settings import PodmanSettings


class LostReply(LocalPodmanCLI):
    async def run(self, arguments, *, deadline):
        result = await super().run(arguments, deadline=deadline)
        if arguments[0] == "create" and result.returncode == 0:
            os._exit(0)  # kernel closes the owner flock; container remains created
        return result


async def main():
    config = json.loads(sys.stdin.readline())
    engine = PodmanEngine(PodmanSettings(**config["settings"]), cli=LostReply())
    await engine.open(deadline=time.monotonic() + 20)
    await engine.create("lost-create", Path(config["root"]), deadline=time.monotonic() + 20)
    raise RuntimeError("the injected lost-create edge was not reached")


asyncio.run(main())
