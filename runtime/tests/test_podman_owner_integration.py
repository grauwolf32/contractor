"""Explicit real-rootless owner-process gate; preinstalled image only."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest
from test_podman_supervisor_integration import launch

from contractor_runtime.sandbox.contracts import SandboxContractError, SandboxIdentity
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.io import LocalPodmanCLI, local_engine_environment
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real-rootless owner-process gate",
)


def settings():
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    assert image, "preinstalled digest-pinned CONTRACTOR_TEST_PODMAN_IMAGE is required"
    return PodmanSettings(enabled=True, image=image, owner="owner-gate-" + uuid.uuid4().hex)


def test_real_owner_startup_exclusivity_and_graceful_close():
    async def scenario():
        policy = settings()
        lifecycle = PodmanLifecycle(policy)
        other = PodmanEngine(policy)
        try:
            await lifecycle.recover(deadline=time.monotonic() + 30)
            process = lifecycle._client.process
            assert process.poll() is None
            with pytest.raises(SandboxContractError):
                await other.open(deadline=time.monotonic() + 5)
        finally:
            await lifecycle.close(deadline=time.monotonic() + 30)
        assert process.poll() == 0
        await other.open(deadline=time.monotonic() + 10)
        await other.close(deadline=time.monotonic() + 10)

    asyncio.run(scenario())


@pytest.mark.parametrize("stall", [False, True])
def test_real_runtime_death_or_stall_fences_work_and_owner_recovers(tmp_path, stall):
    async def scenario():
        policy = settings()
        root = tmp_path / "run_workdir"
        root.mkdir()
        sentinel = root / "keep.txt"
        sentinel.write_text("retain until owner confirms removal")
        runtime = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).parent / "fakes/podman_allocation_runtime.py"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=local_engine_environment(),
            start_new_session=True,
        )
        successor = PodmanEngine(policy)
        opened = False
        workload = None
        try:
            runtime.stdin.write(
                json.dumps({"settings": asdict(policy), "root": str(root)}).encode() + b"\n"
            )
            await runtime.stdin.drain()
            raw = await asyncio.wait_for(runtime.stdout.readline(), 35)
            assert raw, (await runtime.stderr.read()).decode()
            identity = SandboxIdentity(**json.loads(raw))
            workload = await launch(
                identity,
                "printf started > /workspace/started; sleep 20; "
                "printf survived > /workspace/survived",
            )
            until = time.monotonic() + 5
            while not (root / "started").exists():
                assert workload.returncode is None and time.monotonic() < until
                await asyncio.sleep(0.02)
            if stall:
                runtime.send_signal(signal.SIGSTOP)
                until = time.monotonic() + 8
                while True:
                    result = await LocalPodmanCLI().run(
                        ("container", "inspect", identity.container_id),
                        deadline=time.monotonic() + 3,
                    )
                    state = json.loads(result.stdout)[0]["State"]
                    if not state["Running"]:
                        break
                    assert time.monotonic() < until, (
                        "guardian did not stop inert PID1 after pulse loss"
                    )
                    await asyncio.sleep(0.1)
                with pytest.raises(SandboxContractError):
                    await successor.open(deadline=time.monotonic() + 3)
            runtime.kill()
            await runtime.wait()
            until = time.monotonic() + 35
            while not opened:
                try:
                    await successor.open(deadline=time.monotonic() + 3)
                    opened = True
                except SandboxContractError:
                    assert time.monotonic() < until, (
                        "owner did not confirm cleanup after Runtime death"
                    )
                    await asyncio.sleep(0.1)
            assert await successor.discover(deadline=time.monotonic() + 5) == ()
            await asyncio.wait_for(workload.communicate(), 5)
            assert workload.returncode != 0 and not (root / "survived").exists()
            assert sentinel.read_text() == "retain until owner confirms removal"
        finally:
            if runtime.returncode is None:
                runtime.kill()
                await runtime.wait()
            if opened:
                for identity in await successor.discover(deadline=time.monotonic() + 10):
                    await successor.remove(identity, deadline=time.monotonic() + 10)
                await successor.close(deadline=time.monotonic() + 10)
            if workload is not None and workload.returncode is None:
                workload.kill()
                await workload.communicate()

    asyncio.run(scenario())
