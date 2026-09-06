"""Allocation engine owner, run only in the independent host owner process.

No workspace deletion happens here. This owner survives Runtime socket EOF,
joins in-flight engine operations and retains its service flock until exact
container removal is confirmed. The kernel guardian remains independent too.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

from contractor_runtime.podman_command import PodmanCommand
from contractor_runtime.podman_engine import PodmanEngine
from contractor_runtime.podman_guardian import CgroupFence
from contractor_runtime.podman_ownership import open_directory
from contractor_runtime.podman_probe import PodmanProbe
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.podman_supervisor import CompletionGate, GuardianClient, open_fence
from contractor_runtime.sandbox_contracts import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
    SandboxIdentity,
)


@dataclass(repr=False)
class Entry:
    allocation_id: str
    root: Path
    identity: SandboxIdentity | None = None
    guardian: GuardianClient | None = None
    fence: CgroupFence | None = None
    rejected: bool = False
    prepared: bool = False
    stopped: bool = False


class LifecycleBackend:
    def __init__(
        self, settings: PodmanSettings, *, engine=None, attach=None, commands=None
    ) -> None:
        self.settings = settings
        self.engine = engine if engine is not None else PodmanEngine(settings)
        self.attach = attach or self._attach
        self.commands = commands if commands is not None else PodmanCommand(settings)
        self.entry: Entry | None = None
        self.opened = False
        self.recovered = False
        self.closed = False
        self.lost = False
        self.lease = 0.0
        self.confirmed_lease = 0.0
        self.probe_test: PodmanProbe | None = None
        self._rejected: dict[str, None] = {}

    def pulse(self, lease: float | None) -> None:
        # This deadline belongs to the Runtime, not this surviving process.
        # The owner may not manufacture liveness by renewing itself forever.
        self.lease = min(lease, time.monotonic() + 3) if lease is not None else 0.0
        self.confirmed_lease = lease or 0.0
        if self.entry is not None and self.lease <= time.monotonic():
            self.reject(self.entry.allocation_id)

    def reject(self, allocation_id: str) -> None:
        # One serialized lifecycle RPC can be in flight. Keep recent preflight
        # rejects bounded; completed allocations must not exhaust the service.
        self._rejected[allocation_id] = None
        if len(self._rejected) > 1024:
            del self._rejected[next(iter(self._rejected))]
        if self.entry is not None and self.entry.allocation_id == allocation_id:
            self.entry.rejected = True
            if self.entry.guardian is not None:
                self.entry.guardian.disconnect()

    def disconnect(self) -> None:
        self.lost = True
        if self.probe_test is not None:
            for guardian in self.probe_test.guardians:
                guardian.disconnect()
        if self.entry is not None:
            self.entry.rejected = True
            if self.entry.guardian is not None:
                self.entry.guardian.disconnect()

    async def renew(self) -> None:
        entry = self.entry
        if entry is None or entry.rejected or entry.guardian is None:
            return
        try:
            await entry.guardian.request(
                "renew", lease=self.lease, deadline=min(self.lease, time.monotonic() + 1)
            )
        except (Exception, asyncio.CancelledError):
            self.reject(entry.allocation_id)

    async def recover(self, *, deadline: float) -> None:
        if self.recovered:
            return
        await self.engine.open(deadline=deadline)
        self.opened = True
        for identity in await self.engine.discover(deadline=deadline):
            await self.engine.remove(identity, deadline=deadline)
        self.recovered = True

    async def probe(self, root: Path, *, deadline: float) -> dict:
        if not self.recovered or self.closed or self.lost or self.entry is not None:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        if self.probe_test is not None:
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
        self.probe_test = PodmanProbe(self.engine, self.settings)
        result = await self.probe_test.run(root, deadline=deadline)
        self.probe_test = None  # only a confirmed cleanup releases ownership
        return result

    def _live(self, entry: Entry) -> None:
        if self.lost or entry.rejected or time.monotonic() >= self.lease:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)

    async def prepare(self, allocation_id: str, root: Path, *, deadline: float) -> SandboxIdentity:
        if not self.recovered or self.closed or self.probe_test is not None:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        entry = self.entry
        if entry is None:
            entry = Entry(allocation_id, root, rejected=allocation_id in self._rejected)
            self.entry = entry  # own even an uncertain create before invoking CLI
        if entry.allocation_id != allocation_id or entry.root != root:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        self._live(entry)
        if entry.prepared:
            assert entry.identity is not None
            state = await self.engine.inspect(entry.identity, deadline=deadline)
            if state is None or not state.running or state.status != "running":
                self.reject(allocation_id)
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            return entry.identity
        try:
            entry.identity = await self.engine.create(allocation_id, root, deadline=deadline)
            self._live(entry)
            await self.engine.start(entry.identity, deadline=deadline)
            self._live(entry)
            entry.fence, entry.guardian = await self.attach(
                entry.identity, min(deadline, self.lease)
            )
            self._live(entry)
            await entry.guardian.request("check", deadline=min(deadline, self.lease))
            entry.prepared = True
            return entry.identity
        except BaseException:
            self.reject(allocation_id)
            raise

    async def _attach(self, identity: SandboxIdentity, deadline: float):
        state = await self.engine.supervisor_state(identity, deadline=deadline)
        fence = await asyncio.to_thread(open_fence, identity.container_id, state)
        try:
            guardian = await GuardianClient.start(fence, lease=min(deadline, self.lease))
        except BaseException:
            os.close(fence.directory)
            os.close(fence.pidfd)
            raise
        return fence, guardian

    async def execute(
        self, allocation_id: str, request: ExecutionRequest, *, deadline: float
    ) -> ExecutionResult:
        entry = self.entry
        if entry is None or entry.allocation_id != allocation_id or not entry.prepared:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        self._live(entry)
        started = time.monotonic()
        deadline = min(deadline, self.confirmed_lease, started + self.settings.command_max_seconds)

        # No previous command can still write: its completion check and the
        # Runtime workspace guard precede this RPC. Reject all cwd symlinks.
        def validate_cwd():
            try:
                descriptor = open_directory(entry.root / request.cwd)
                os.close(descriptor)
            except (OSError, ValueError):
                raise SandboxContractError(SandboxErrorCode.INVALID_CWD) from None

        await asyncio.to_thread(validate_cwd)
        try:
            assert entry.identity is not None and entry.guardian is not None
            await entry.guardian.request("check", deadline=min(deadline, self.lease))
            self._live(entry)
            reserve = min(
                self.settings.stop_grace_seconds + 1, max(0, deadline - time.monotonic()) / 2
            )
            capture = await self.engine.execute(
                entry.identity,
                request,
                deadline=deadline,
                launch_deadline=deadline - reserve,
                transport=self.commands,
            )
            if capture.error is None:
                result = ExecutionResult(
                    ExecutionStatus.COMPLETED,
                    capture.exit_code,
                    capture.stdout.decode("utf-8", errors="replace"),
                    capture.stderr.decode("utf-8", errors="replace"),
                    capture.stdout_bytes > len(capture.stdout),
                    capture.stderr_bytes > len(capture.stderr),
                    max(0, int((time.monotonic() - started) * 1000)),
                    stdout_bytes=capture.stdout_bytes,
                    stderr_bytes=capture.stderr_bytes,
                )
                await CompletionGate(self.engine, entry.identity, entry.guardian).confirm(
                    result, deadline=deadline
                )
                self._live(entry)
                return result
            self.reject(allocation_id)
            await self.stop(allocation_id, deadline=deadline)
            status = {
                SandboxErrorCode.TIMEOUT: ExecutionStatus.TIMED_OUT,
                SandboxErrorCode.OUTPUT_LIMIT: ExecutionStatus.OUTPUT_LIMIT_EXCEEDED,
            }.get(capture.error, ExecutionStatus.FAILED)
            return ExecutionResult(
                status,
                None,
                capture.stdout.decode("utf-8", errors="replace"),
                capture.stderr.decode("utf-8", errors="replace"),
                capture.stdout_bytes > len(capture.stdout),
                capture.stderr_bytes > len(capture.stderr),
                max(0, int((time.monotonic() - started) * 1000)),
                capture.error,
                capture.stdout_bytes,
                capture.stderr_bytes,
            )
        except BaseException:
            self.reject(allocation_id)
            await self.stop(allocation_id, deadline=deadline)
            raise

    async def stop(self, allocation_id: str, *, deadline: float) -> None:
        self.reject(allocation_id)
        entry = self.entry
        if entry is None or entry.allocation_id != allocation_id:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        if entry.stopped:
            return
        if entry.identity is not None:
            await self.engine.stop(entry.identity, deadline=deadline)
        # Discovery serializes after an uncertain engine operation, including
        # late create. An absent uncertain attempt still fences engine.close().
        identities = await self.engine.discover(deadline=deadline)
        for identity in identities:
            if identity.allocation_id == allocation_id:
                await self.engine.stop(identity, deadline=deadline)
                if entry.identity is None:
                    entry.identity = identity
        if entry.fence is not None and not await asyncio.to_thread(entry.fence.empty):
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
        entry.stopped = True

    async def remove(self, allocation_id: str, *, deadline: float) -> None:
        entry = self.entry
        if entry is None:
            return
        await self.stop(allocation_id, deadline=deadline)
        if entry.identity is not None:
            await self.engine.remove(entry.identity, deadline=deadline)
        identities = await self.engine.discover(deadline=deadline)
        for identity in identities:
            if identity.allocation_id == allocation_id:
                await self.engine.remove(identity, deadline=deadline)
        # The engine must explicitly confirm this attempt no longer exists;
        # discovery absence alone cannot forgive a lost create response.
        await self.engine.confirm_removed(allocation_id, deadline=deadline)
        if entry.guardian is not None:
            await entry.guardian.close()
        if entry.fence is not None:
            os.close(entry.fence.directory)
            os.close(entry.fence.pidfd)
        self.entry = None

    async def close(self, *, deadline: float) -> None:
        if self.closed:
            return
        self.disconnect()
        if self.entry is not None:
            await self.remove(self.entry.allocation_id, deadline=deadline)
        for identity in await self.engine.discover(deadline=deadline):
            await self.engine.remove(identity, deadline=deadline)
        if self.probe_test is not None:
            await self.probe_test._cleanup(deadline)
            self.probe_test = None
        await self.engine.close(deadline=deadline)
        self.closed = True
