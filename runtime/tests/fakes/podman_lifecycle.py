"""Real owner protocol/engine with a scripted CLI and kernel-guardian seam."""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

from test_podman_engine import IMAGE, ScriptedEngine

from contractor_runtime.sandbox.contracts import SandboxContractError, SandboxErrorCode
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.lifecycle import OwnerClient, PodmanLifecycle
from contractor_runtime.sandbox.podman.lifecycle_backend import LifecycleBackend
from contractor_runtime.sandbox.podman.owner import serve_owner
from contractor_runtime.sandbox.podman.settings import PodmanSettings


class Guardian:
    def __init__(self):
        self.rejected = False
        self.fail = False
        self.requests = []

    def disconnect(self):
        self.rejected = True

    async def request(self, operation, *, deadline, lease=None):
        self.requests.append((operation, deadline, lease))
        if self.rejected or self.fail or time.monotonic() >= deadline:
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)

    async def close(self):
        self.disconnect()


class Owner:
    def __init__(self, root: Path):
        self.cli = ScriptedEngine()
        self.settings = PodmanSettings(enabled=True, owner="lifecycle-test", image=IMAGE)
        self.engine = PodmanEngine(self.settings, cli=self.cli, owner_directory=root / "owners")
        self.guardians = []
        self.attach_fail = False
        self.backend = LifecycleBackend(self.settings, engine=self.engine, attach=self.attach)
        self.lifecycle = PodmanLifecycle(self.settings, start=self.start)
        self.lease = time.monotonic() + 60
        self.lifecycle.bind_health(lambda: self.lease, lambda: None)
        self.task = None
        self.client = None

    async def attach(self, identity, deadline):
        if self.attach_fail:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        guardian = Guardian()
        self.guardians.append(guardian)
        return None, guardian

    async def start(self, settings):
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        health, child_health = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        for channel in (control, child_control, health, child_health):
            channel.setblocking(False)
        self.task = asyncio.create_task(serve_owner(child_control, child_health, self.backend))
        self.client = OwnerClient(control, health, self)
        return self.client

    def poll(self):
        return 0 if self.task is not None and self.task.done() else None


@asynccontextmanager
async def owner(root):
    fixture = Owner(root)
    try:
        yield fixture
    finally:
        fixture.cli.resume.set()
        fixture.cli.noop_remove = False
        fixture.cli.noop_stop = False
        fixture.cli.exists_error = False
        fixture.cli.list_override = None
        await fixture.lifecycle.close(deadline=time.monotonic() + 5)
        if fixture.task is not None:
            await asyncio.wait_for(fixture.task, 5)
