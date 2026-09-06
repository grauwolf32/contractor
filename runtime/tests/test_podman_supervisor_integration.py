"""Explicit real-rootless gate; never builds/pulls images or touches other owners."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from contractor_runtime.podman_engine import PodmanEngine
from contractor_runtime.podman_io import LocalPodmanCLI, local_engine_environment
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.podman_supervisor import CompletionGate, GuardianClient, open_fence
from contractor_runtime.sandbox_contracts import (
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real gate: make test-podman-supervisor",
)


@asynccontextmanager
async def sandbox(tmp_path: Path, *, lifetime: float = 9, attach: bool = True, **limits):
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    assert image, "CONTRACTOR_TEST_PODMAN_IMAGE is required (preinstalled digest-pinned image)"
    settings = PodmanSettings(enabled=True, image=image, owner="gate-" + uuid.uuid4().hex, **limits)
    root = tmp_path / "run_workdir"
    root.mkdir()
    engine = PodmanEngine(settings, owner_directory=tmp_path / "owners")
    guardian = fence = None
    await engine.open(deadline=time.monotonic() + 30)
    try:
        identity = await engine.create("supervisor-gate", root, deadline=time.monotonic() + 30)
        await engine.start(identity, deadline=time.monotonic() + 30)
        state = await engine.inspect(identity, deadline=time.monotonic() + 5)
        assert state is not None and state.running
        raw = await LocalPodmanCLI().run(
            ("container", "inspect", identity.container_id), deadline=time.monotonic() + 5
        )
        inspected = json.loads(raw.stdout)[0]
        assert (
            inspected["Config"]["Labels"]["io.contractor.sandbox.supervisor"]
            == "cgroup-guardian-v1"
        )
        fence = open_fence(identity.container_id, inspected["State"])
        if attach:
            guardian = await GuardianClient.start(fence, lease=time.monotonic() + lifetime)
            await guardian.request("check", deadline=time.monotonic() + 2)
        yield engine, identity, root, guardian, fence
    finally:
        if guardian is not None:
            await guardian.close()
        if fence is not None:
            os.close(fence.directory)
            os.close(fence.pidfd)
        for identity in await engine.discover(deadline=time.monotonic() + 30):
            await engine.remove(identity, deadline=time.monotonic() + 30)
        await engine.close(deadline=time.monotonic() + 30)


async def launch(identity, command: str):
    return await asyncio.create_subprocess_exec(
        "/usr/bin/podman",
        "--remote=false",
        "--log-level=error",
        "--events-backend=none",
        "exec",
        f"--user={os.getuid()}:{os.getgid()}",
        "--workdir=/workspace",
        identity.container_id,
        "/bin/sh",
        "-c",
        command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=local_engine_environment(),
        cwd="/",
    )


async def run(identity, command: str) -> tuple[int, str]:
    process = await launch(identity, command)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        assert not stderr, stderr.decode(errors="replace")
        return process.returncode, stdout.decode()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def test_normal_and_nonzero_commands_preserve_allocation_files(tmp_path):
    async def scenario():
        async with sandbox(tmp_path) as (engine, identity, root, guardian, _):
            completion = CompletionGate(engine, identity, guardian)
            (root / "host").write_text("host-data")
            assert await run(
                identity, "cat host; echo container-data > container; echo kept > /tmp/kept"
            ) == (0, "host-data")
            await guardian.request("check", deadline=time.monotonic() + 2)
            assert (root / "container").read_text() == "container-data\n"
            assert (root / "container").stat().st_uid == os.getuid()
            (root / "container").write_text("host-edited")
            assert await run(identity, "cat container") == (0, "host-edited")
            await guardian.request("check", deadline=time.monotonic() + 2)
            code, output = await run(identity, "cat /tmp/kept; exit 7")
            result = ExecutionResult(ExecutionStatus.COMPLETED, code, output, "", False, False, 0)
            confirmed = await completion.confirm(result, deadline=time.monotonic() + 2)
            assert confirmed.exit_code == 7 and confirmed.stdout == "kept\n"

    asyncio.run(scenario())


@pytest.mark.parametrize("detach", ["fork", "double-fork", "session", "inherited-fd"])
def test_no_detached_writer_survives_failed_completion(tmp_path, detach):
    async def scenario():
        async with sandbox(tmp_path) as (engine, identity, root, guardian, fence):
            script = "import os,time\nif os.fork(): os._exit(0)\n"
            if detach == "double-fork":
                script += "if os.fork(): os._exit(0)\n"
            if detach == "session":
                script += "os.setsid()\n"
            if detach != "inherited-fd":
                script += "os.close(1); os.close(2)\n"
            script += (
                "while True:\n with open('/workspace/writer','a') as f: f.write('x')\n"
                " time.sleep(.02)\n"
            )
            process = await launch(identity, "python3 -c " + shlex.quote(script))
            try:
                until = time.monotonic() + 3
                while not (root / "writer").exists():
                    assert time.monotonic() < until
                    await asyncio.sleep(0.02)
                with pytest.raises(SandboxContractError):
                    await guardian.request("check", deadline=time.monotonic() + 3)
                assert fence.empty()
                before = (root / "writer").stat().st_size
                await asyncio.sleep(0.15)
                assert (root / "writer").stat().st_size == before
                state = await engine.inspect(identity, deadline=time.monotonic() + 3)
                assert state is not None and not state.running
            finally:
                await asyncio.wait_for(process.communicate(), 3)

    asyncio.run(scenario())


def test_expiry_cannot_be_extended_or_forged_by_workload(tmp_path):
    async def scenario():
        async with sandbox(tmp_path, lifetime=3) as (_, identity, _, guardian, fence):
            # Valid-looking protocol output is untrusted and has no connection
            # to the inherited host socket. Namespace root cannot be signalled.
            code, output = await run(
                identity,
                "python3 - <<'PY'\nimport os\n"
                "for operation in [lambda: os.kill(1,9),lambda: open('/proc/1/mem','rb'),"
                "lambda: open('/sys/fs/cgroup/cgroup.freeze','w')]:\n"
                " try: operation()\n except (PermissionError,OSError): pass\n"
                " else: raise AssertionError('control authority exposed')\n"
                'print(\'{"status":"clean","op":"renew","lease":999999999999}\')\nPY',
            )
            assert code == 0 and '"clean"' in output
            host_pid = guardian._process.pid
            assert await run(
                identity,
                f"test ! -e /proc/{host_pid}; kill -0 {host_pid} 2>/dev/null; test $? -ne 0",
            ) == (0, "")
            await guardian.request(
                "renew", lease=time.monotonic() + 1, deadline=time.monotonic() + 0.5
            )
            process = await launch(identity, "while :; do echo x >> writer; sleep .03; done")
            try:
                await asyncio.sleep(1.3)
                assert fence.empty()
            finally:
                await asyncio.wait_for(process.communicate(), 3)

    asyncio.run(scenario())


def test_effective_mapping_confinement_and_resource_limits(tmp_path):
    async def scenario():
        async with sandbox(tmp_path) as (_, identity, _, guardian, fence):
            assert fence.read("memory.max").strip() == str(2 << 30)
            assert fence.read("pids.max").strip() == "256"
            quota, period = map(int, fence.read("cpu.max").split())
            assert quota / period == 2
            code, output = await run(
                identity,
                "python3 - <<'PY'\nimport os,json\nassert os.getuid() != 0\n"
                "status=open('/proc/self/status').read()\n"
                "assert 'CapEff:\\t0000000000000000' in status\n"
                "assert 'NoNewPrivs:\\t1' in status\nassert 'Seccomp:\\t2' in status\n"
                "assert os.listdir('/sys/class/net') == ['lo']\n"
                "assert not os.path.exists('/run/podman/podman.sock')\n"
                "assert os.statvfs('/tmp').f_blocks*os.statvfs('/tmp').f_frsize == 268435456\n"
                "try: open('/usr/local/lib/contractor-init.py','a')\n"
                "except OSError: pass\nelse: raise AssertionError('writable image')\n"
                "print(json.dumps(dict(os.environ)))\nPY",
            )
            assert code == 0
            environment = json.loads(output)
            assert environment["HOME"] == "/tmp"
            assert not any(key.startswith("CONTRACTOR_") for key in environment)
            assert not any("PROXY" in key.upper() for key in environment)
            await guardian.request("check", deadline=time.monotonic() + 2)

    asyncio.run(scenario())


def test_cpu_quota_actually_throttles(tmp_path):
    async def scenario():
        async with sandbox(tmp_path, cpus=0.25) as (_, identity, _, guardian, fence):
            assert fence.read("cpu.max").strip() == "25000 100000"
            code, _ = await run(
                identity,
                "python3 -c 'import time; end=time.monotonic()+1.5\n"
                "while time.monotonic()<end: pass'",
            )
            assert code == 0
            stats = dict(line.split() for line in fence.read("cpu.stat").splitlines())
            assert int(stats["nr_throttled"]) > 0
            await guardian.request("check", deadline=time.monotonic() + 2)

    asyncio.run(scenario())


def test_memory_limit_actually_kills_oversized_workload(tmp_path):
    async def scenario():
        async with sandbox(tmp_path, memory_bytes=128 << 20, tmpfs_bytes=32 << 20) as (
            _,
            identity,
            _,
            _,
            fence,
        ):
            assert fence.read("memory.max").strip() == str(128 << 20)
            process = await launch(identity, "exec python3 -c 'a=bytearray(256*1024*1024)'")
            await asyncio.wait_for(process.communicate(), 5)
            assert process.returncode != 0
            # cgroup can disappear if the kernel also kills PID 1; either case
            # is a fatal, non-reusable allocation, never a successful command.
            if fence.init_alive():
                stats = dict(line.split() for line in fence.read("memory.events").splitlines())
                assert int(stats["oom_kill"]) > 0
            else:
                assert fence.empty()

    asyncio.run(scenario())


def test_pid_limit_prevents_fork_growth_then_kills_survivors(tmp_path):
    async def scenario():
        async with sandbox(tmp_path, pids=16) as (_, identity, _, guardian, fence):
            assert fence.read("pids.max").strip() == "16"
            script = (
                "import os,time,errno\nfor i in range(32):\n"
                " try: child=os.fork()\n"
                " except OSError as e:\n  assert e.errno==errno.EAGAIN\n"
                "  print('limited',flush=True)\n  break\n"
                " if child==0: time.sleep(10); os._exit(0)\n"
                "else: raise AssertionError('pids not enforced')\n"
            )
            process = await launch(identity, "exec python3 -c " + shlex.quote(script))
            try:
                assert await asyncio.wait_for(process.stdout.readline(), 3) == b"limited\n"
                stats = dict(line.split() for line in fence.read("pids.events").splitlines())
                assert int(stats["max"]) > 0
                await guardian.request("stop", deadline=time.monotonic() + 3)
                assert fence.empty()
            finally:
                await asyncio.wait_for(process.communicate(), 3)

    asyncio.run(scenario())


def test_runtime_sigkill_without_restart_stops_workload(tmp_path):
    async def scenario():
        async with sandbox(tmp_path, attach=False) as (_, identity, root, _, fence):
            controller = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(Path(__file__).parent / "fakes/podman_runtime.py"),
                str(fence.directory),
                str(fence.init_pid),
                str(fence.pidfd),
                pass_fds=(fence.directory, fence.pidfd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=local_engine_environment(),
                cwd="/",
            )
            process = None
            try:
                assert await asyncio.wait_for(controller.stdout.readline(), 3) == b"ready\n"
                process = await launch(identity, "while :; do echo x >> writer; sleep .02; done")
                until = time.monotonic() + 3
                while not (root / "writer").exists():
                    assert time.monotonic() < until
                    await asyncio.sleep(0.02)
                controller.kill()
                await controller.wait()  # no successor Runtime is started
                until = time.monotonic() + 3
                while not fence.empty():
                    assert time.monotonic() < until
                    await asyncio.sleep(0.02)
                before = (root / "writer").stat().st_size
                await asyncio.sleep(0.15)
                assert (root / "writer").stat().st_size == before
            finally:
                if controller.returncode is None:
                    controller.kill()
                await controller.communicate()
                if process is not None:
                    await asyncio.wait_for(process.communicate(), 3)

    asyncio.run(scenario())
