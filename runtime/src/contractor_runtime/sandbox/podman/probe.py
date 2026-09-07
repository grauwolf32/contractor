"""Owner-side, finite destructive self-test on separately owned probe containers.

Only fixed test programs execute. Bind data remains owned by the workspace
provider and cannot be erased until this process confirms exact removal.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
import uuid
from pathlib import Path

from contractor_runtime.sandbox.contracts import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
)
from contractor_runtime.sandbox.podman.command import PodmanCommand
from contractor_runtime.sandbox.podman.io import remaining
from contractor_runtime.sandbox.podman.supervisor import CompletionGate, GuardianClient, open_fence

PROBE_TIMEOUT_SECONDS = 30.0
PROBE_CLEANUP_SECONDS = 8.0
PROBE_FAILURES = frozenset(
    {"prerequisites", "supervisor", "resources", "execution", "descendants", "liveness"}
)

# Parent waits until the double-forked, new-session descendant has opened its
# writer. No inherited output fd delays the CLI exit or substitutes for proof.
WRITER = """import os,time
r,w=os.pipe()
if os.fork():
 os.close(w); assert os.read(r,1)==b'1'; os._exit(0)
os.close(r); os.setsid()
if os.fork(): os._exit(0)
for fd in (0,1,2): os.close(fd)
f=open('/workspace/writer','a'); f.write('x'); f.flush()
os.write(w,b'1'); os.close(w)
while True:
 f.write('x'); f.flush(); time.sleep(.02)
"""


class PodmanProbe:
    def __init__(self, engine, settings):
        self.engine = engine
        self.settings = settings
        self.commands = PodmanCommand(settings)
        self.attempts = []
        self.fences = []
        self.guardians = []
        self.phase = "prerequisites"

    async def run(self, root: Path, *, deadline: float) -> dict:
        work_deadline = deadline - PROBE_CLEANUP_SECONDS
        available = False
        try:
            remaining(work_deadline)
            await self.engine.probe_prerequisites(deadline=work_deadline)
            await self._completion(root, work_deadline)
            await self._expiry(root, work_deadline)
            available = True
        except Exception:
            # The phase is a fixed identifier, never an engine exception string.
            available = False
        finally:
            try:
                await self._cleanup(deadline)
            except BaseException:
                # Do not close the cgroup capabilities or forgive a late create.
                # The owner retains engine ownership for its recovery loop.
                for guardian in self.guardians:
                    guardian.disconnect()
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED) from None
        return {"available": available, "failure": None if available else self.phase}

    async def _start(self, root, deadline, lifetime):
        self.phase = "supervisor"
        allocation = "probe-" + uuid.uuid4().hex
        self.attempts.append(allocation)  # includes a lost create response
        identity = await self.engine.create(allocation, root, deadline=deadline)
        await self.engine.start(identity, deadline=deadline)
        state = await self.engine.supervisor_state(identity, deadline=deadline)
        fence = await asyncio.to_thread(open_fence, identity.container_id, state)
        self.fences.append(fence)
        guardian = await GuardianClient.start(
            fence, lease=min(deadline, time.monotonic() + lifetime)
        )
        self.guardians.append(guardian)
        await guardian.request("check", deadline=min(deadline, time.monotonic() + 1))
        return identity, fence, guardian

    async def _command(self, identity, command, deadline):
        capture = await self.engine.execute(
            identity,
            ExecutionRequest(command),
            deadline=deadline,
            launch_deadline=min(deadline, time.monotonic() + 3),
            transport=self.commands,
        )
        if capture.error is not None or capture.exit_code != 0:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        return ExecutionResult(ExecutionStatus.COMPLETED, 0, "", "", False, False, 0)

    async def _completion(self, root, deadline):
        identity, fence, guardian = await self._start(root, deadline, 8)
        self.phase = "resources"

        def resources():
            memory = int(fence.read("memory.max"))
            swap = int(fence.read("memory.swap.max"))
            pids = int(fence.read("pids.max"))
            quota, period = map(int, fence.read("cpu.max").split())
            if (
                memory != self.settings.memory_bytes
                or not 0 <= swap <= self.settings.memory_bytes
                or pids != self.settings.pids
                or quota <= 0
                or period <= 0
                or abs(quota / period - self.settings.cpus) > 1 / period
            ):
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)

        await asyncio.to_thread(resources)
        self.phase = "execution"
        await asyncio.to_thread((root / "input").write_text, "from-host")
        script = f"""import os
assert os.getuid()=={os.getuid()} and os.getgid()=={os.getgid()} and os.getuid()!=0
status=open('/proc/self/status').read()
assert 'CapEff:\\t0000000000000000' in status
assert 'NoNewPrivs:\\t1' in status and 'Seccomp:\\t2' in status
assert os.listdir('/sys/class/net')==['lo']
assert os.stat('/proc/self/ns/net').st_ino!={os.stat("/proc/self/ns/net").st_ino}
assert not os.path.exists('/run/podman/podman.sock')
assert not any(k.startswith('CONTRACTOR_') or 'PROXY' in k.upper() for k in os.environ)
assert os.statvfs('/').f_flag & os.ST_RDONLY
assert os.statvfs('/sys/fs/cgroup').f_flag & os.ST_RDONLY
v=os.statvfs('/tmp'); assert v.f_blocks*v.f_frsize=={self.settings.tmpfs_bytes}
assert open('/workspace/input').read()=='from-host'
open('/workspace/output','w').write('from-container')
"""
        result = await self._command(identity, "python3 -c " + shlex.quote(script), deadline)
        await CompletionGate(self.engine, identity, guardian).confirm(result, deadline=deadline)
        if await asyncio.to_thread((root / "output").read_text) != "from-container":
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        self.phase = "descendants"
        await self._command(identity, "python3 -c " + shlex.quote(WRITER), deadline)
        # Verify a real live writer before requesting the independent proof.
        if not await asyncio.to_thread(fence.init_alive):
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        if not await asyncio.to_thread((root / "writer").read_bytes):
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        rejected = False
        try:
            await guardian.request("check", deadline=min(deadline, time.monotonic() + 2))
        except SandboxContractError:
            rejected = True
        if not rejected:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        await self._empty(fence, deadline)
        await self._remove(identity.allocation_id, deadline)

    async def _expiry(self, root, deadline):
        identity, fence, _guardian = await self._start(root, deadline, 1.5)
        self.phase = "liveness"
        await self._command(identity, "python3 -c " + shlex.quote(WRITER), deadline)
        if not await asyncio.to_thread(fence.init_alive):
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        # Keep the control socket OPEN. No renew, disconnect, Podman stop or
        # CLI timeout is allowed to substitute for guardian deadline expiry.
        await self._empty(fence, min(deadline, time.monotonic() + 3))
        state = await self.engine.inspect(identity, deadline=deadline)
        if state is None or state.running:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)

    async def _empty(self, fence, deadline):
        while not await asyncio.to_thread(fence.empty):
            remaining(deadline)
            await asyncio.sleep(0.02)

    async def _remove(self, allocation, deadline):
        for identity in await self.engine.discover(deadline=deadline):
            if identity.allocation_id == allocation:
                await self.engine.remove(identity, deadline=deadline)
        await self.engine.confirm_removed(allocation, deadline=deadline)

    async def _cleanup(self, deadline):
        for guardian in self.guardians:
            guardian.disconnect()
        for allocation in self.attempts:
            await self._remove(allocation, deadline)
        for fence in self.fences:
            await self._empty(fence, deadline)
        for guardian in self.guardians:
            await guardian.close()
        for fence in self.fences:
            os.close(fence.directory)
            os.close(fence.pidfd)
