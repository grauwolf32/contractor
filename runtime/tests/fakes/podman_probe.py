"""Scripted local engine plus explicit kernel/guardian fault seams."""

from __future__ import annotations

import os
import time

from test_podman_engine import ScriptedEngine

from contractor_runtime.podman_command import CommandCapture
from contractor_runtime.podman_io import CLIResult
from contractor_runtime.sandbox_contracts import SandboxContractError, SandboxErrorCode


class ProbeCLI(ScriptedEngine):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure
        self.info_count = 0

    async def run(self, args, *, deadline):
        if args[0] == "info":
            self.calls.append(args)
            self.info_count += 1
            return self.result(
                {
                    "host": {
                        "security": {
                            "rootless": not (self.failure == "rootless" and self.info_count > 1),
                            "seccompEnabled": self.failure != "seccomp",
                        },
                        "cgroupVersion": "v1" if self.failure == "cgroup" else "v2",
                        "cgroupManager": "cgroupfs" if self.failure == "manager" else "systemd",
                    }
                }
            )
        if args[:2] == ("image", "inspect"):
            self.calls.append(args)
            if self.failure == "image":
                return CLIResult(125)
            return self.result(
                [
                    {
                        "Config": {
                            "Labels": {
                                "io.contractor.sandbox.supervisor": "wrong"
                                if self.failure == "image-supervisor"
                                else "cgroup-guardian-v1"
                            }
                        }
                    }
                ]
            )
        return await super().run(args, deadline=deadline)


class Fence:
    def __init__(self, cli, identity, root, settings):
        self.cli, self.identity, self.root, self.settings = cli, identity, root, settings
        self.directory = os.open(root, os.O_DIRECTORY)
        self.pidfd = os.dup(self.directory)
        self.expires = float("inf")
        self.writer = False

    def read(self, name):
        if self.cli.failure == name:
            return "max"
        return {
            "memory.max": str(self.settings.memory_bytes),
            "memory.swap.max": "0",
            "pids.max": str(self.settings.pids),
            "cpu.max": "200000 100000",
        }[name]

    def kill(self):
        row = self.cli.containers.get(self.identity)
        if row is not None:
            row["State"] = {"Running": False, "Status": "exited"}

    def init_alive(self):
        if time.monotonic() >= self.expires and self.cli.failure != "liveness":
            self.kill()
        row = self.cli.containers.get(self.identity)
        return row is not None and row["State"]["Running"]

    def empty(self):
        return not self.init_alive()


class Guardian:
    def __init__(self, fence, lease):
        self.fence = fence
        # Keep expiry tests fast; production's bounded interval is tested live.
        if fence.cli.sequence == 2:
            fence.expires = time.monotonic() + 0.05

    async def request(self, op, *, deadline):
        if self.fence.writer and self.fence.cli.failure != "descendants":
            self.fence.kill()
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)

    def disconnect(self):
        self.fence.kill()

    async def close(self):
        self.disconnect()


class Commands:
    def __init__(self, fences, cli):
        self.fences, self.cli = fences, cli

    async def run(self, identity, command, cwd, *, deadline):
        fence = self.fences[identity.container_id]
        if self.cli.failure == "execution":
            return CommandCapture(2, b"untrusted private output", b"", 24, 0)
        if "from-container" in command:
            (fence.root / "output").write_text("from-container")
        else:
            fence.writer = True
            (fence.root / "writer").write_text("x")
        return CommandCapture(0, b"", b"", 0, 0)


def install(monkeypatch, fixture, root, failure):
    import contractor_runtime.podman_probe as module

    cli = ProbeCLI(failure)
    fixture.cli = cli
    fixture.engine._cli = cli
    fences = {}

    def open_fence(identity, state):
        if failure == "supervisor":
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        fence = Fence(cli, identity, root, fixture.settings)
        fences[identity] = fence
        return fence

    async def start(fence, *, lease):
        return Guardian(fence, lease)

    monkeypatch.setattr(module, "open_fence", open_fence)
    monkeypatch.setattr(module.GuardianClient, "start", start)
    monkeypatch.setattr(module, "PodmanCommand", lambda settings: Commands(fences, cli))
    return cli
