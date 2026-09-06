"""Runtime-side allocation handles for the private, surviving Podman owner."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from contractor_runtime.podman_io import OwnedOperations, local_engine_environment, remaining
from contractor_runtime.podman_owner import encode, receive
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.podman_workroots import check_root_policy
from contractor_runtime.projectfs import DirectWorkspaceSession
from contractor_runtime.sandbox_contracts import (
    SandboxContractError,
    SandboxErrorCode,
    SandboxIdentity,
)

# Popen intentionally has no asyncio transport whose close/destructor could
# kill an owner that must survive Runtime shutdown. Reap with poll, never an
# unbounded executor-thread wait that would block asyncio.run shutdown.
_OWNERS: list[subprocess.Popen] = []


class OwnerClient:
    def __init__(self, control: socket.socket, health: socket.socket, process: subprocess.Popen):
        self.control = control
        self.health = health
        self.process = process
        self.operations = OwnedOperations()
        self.disconnected = asyncio.Event()

    @classmethod
    async def start(cls, settings: PodmanSettings) -> OwnerClient:
        control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        health, child_health = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        control.setblocking(False)
        health.setblocking(False)

        def spawn():
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-m",
                        "contractor_runtime.podman_owner",
                        str(child_control.fileno()),
                        str(child_health.fileno()),
                    ],
                    pass_fds=(child_control.fileno(), child_health.fileno()),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=local_engine_environment(),
                    cwd="/",
                )
                _OWNERS[:] = [item for item in _OWNERS if item.poll() is None]
                _OWNERS.append(process)
                return process
            finally:
                child_control.close()
                child_health.close()

        try:
            process = await asyncio.to_thread(spawn)
            await asyncio.get_running_loop().sock_sendall(
                control, encode({"settings": asdict(settings)})
            )
            return cls(control, health, process)
        except BaseException:
            control.close()
            health.close()
            raise

    def signal(self, message: dict) -> None:
        try:
            raw = encode(message)
            if self.health.send(raw) != len(raw):
                raise OSError("partial health packet")
        except (OSError, ValueError):
            self.disconnect()
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE) from None

    def disconnect(self) -> None:
        self.disconnected.set()
        self.health.close()
        self.control.close()

    async def request(self, op: str, *, deadline: float, **fields):
        async def exchange():
            await asyncio.get_running_loop().sock_sendall(
                self.control, encode({"op": op, "deadline": deadline, **fields})
            )
            return await receive(self.control)

        async def operation():
            if self.disconnected.is_set():
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            response_task = asyncio.create_task(exchange())
            disconnected = asyncio.create_task(self.disconnected.wait())
            try:
                # Retain response ownership on caller cancellation/timeout.
                # A later cleanup cannot consume a late prepare acknowledgement.
                done, _ = await asyncio.wait(
                    {response_task, disconnected}, return_when=asyncio.FIRST_COMPLETED
                )
                if disconnected in done:
                    raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
                response = await response_task
                if set(response) == {"error"}:
                    raise SandboxContractError(SandboxErrorCode(response["error"]))
                if set(response) != {"result"}:
                    raise ValueError("invalid owner response")
                return response["result"]
            except (ValueError, OSError) as error:
                if isinstance(error, SandboxContractError):
                    raise
                self.disconnect()
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN) from None
            finally:
                response_task.cancel()
                disconnected.cancel()
                await asyncio.gather(response_task, disconnected, return_exceptions=True)

        return await self.operations.run(operation, deadline)


class PodmanLifecycle:
    def __init__(self, settings: PodmanSettings, *, start=OwnerClient.start) -> None:
        self.settings = settings
        self._start = start
        self._client: OwnerClient | None = None
        self._operations = OwnedOperations()
        self._heartbeat: asyncio.Task | None = None
        self._lease_source: Callable[[], float | None] = lambda: None
        self._failure: Callable[[], None] = lambda: None
        self._entry: PodmanAllocation | None = None
        self._recovered = False
        self._closed = False
        self._close_confirmed = False
        self._failed = False

    def bind_health(self, lease_source: Callable[[], float | None], failure: Callable[[], None]):
        self._lease_source = lease_source
        self._failure = failure

    def _pulse(self) -> None:
        assert self._client is not None
        self._client.signal({"lease": self._lease_source()})

    async def _keep_alive(self) -> None:
        try:
            while True:
                self._pulse()
                await asyncio.sleep(0.25)
        except Exception:
            self._failed = True
            if self._entry is not None:
                self._entry.reject()
            self._failure()

    async def recover(self, *, deadline: float) -> None:
        async def operation():
            if self._closed or self._failed:
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            if self._recovered:
                return
            if self._client is None:
                self._client = await self._start(self.settings)
                if self._closed:
                    self._client.disconnect()
                    raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
                self._heartbeat = asyncio.create_task(
                    self._keep_alive(), name="podman-runtime-pulse"
                )
            await self._client.request("recover", deadline=deadline)
            self._recovered = True

        await self._operations.run(operation, deadline)

    def allocate(self, allocation_id: str, workspace: DirectWorkspaceSession) -> PodmanAllocation:
        if not self._recovered or self._closed or self._failed:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        if workspace.mode != "direct" or workspace.storage.storage != "local":
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        root = Path(workspace.storage.root) / "run_workdir"
        if self._entry is None:
            self._entry = PodmanAllocation(self, allocation_id, root)
        elif self._entry.allocation_id != allocation_id or self._entry.root != root:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        return self._entry

    async def prepare_root(self, root: Path) -> None:
        await self.recover(deadline=time.monotonic() + self.settings.prepare_max_seconds)
        await asyncio.to_thread(check_root_policy, root, self.settings.owner)

    async def close(self, *, deadline: float) -> None:
        if self._close_confirmed:
            return
        self._closed = True
        if self._entry is not None:
            self._entry.reject()
        try:

            async def operation():
                if self._client is not None:
                    await self._client.request("close", deadline=deadline)
                    while self._client.process.poll() is None:
                        remaining(deadline)
                        await asyncio.sleep(0.01)

            await self._operations.run(operation, deadline)
            self._close_confirmed = True
        finally:
            if self._heartbeat is not None:
                self._heartbeat.cancel()
                await asyncio.gather(self._heartbeat, return_exceptions=True)
            if self._client is not None:
                self._client.disconnect()


class PodmanAllocation:
    def __init__(self, owner: PodmanLifecycle, allocation_id: str, root: Path):
        self.owner = owner
        self.allocation_id = allocation_id
        self.root = root
        self.identity: SandboxIdentity | None = None
        self.rejected = False
        self.removed = False

    def reject(self) -> None:
        self.rejected = True
        if self.owner._client is not None:
            # A closed channel also irrevocably rejects work.
            with suppress(SandboxContractError):
                self.owner._client.signal({"reject": self.allocation_id})

    async def prepare(self, *, deadline: datetime) -> None:
        if self.rejected or self.removed:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        assert self.owner._client is not None
        bounded = min(
            _monotonic(deadline), time.monotonic() + self.owner.settings.prepare_max_seconds
        )
        try:
            self.owner._pulse()
            result = await self.owner._client.request(
                "prepare",
                allocation=self.allocation_id,
                root=str(self.root),
                deadline=bounded,
                lease=self.owner._lease_source() or 0.0,
            )
            identity = SandboxIdentity(**result)
            if (
                identity.allocation_id != self.allocation_id
                or identity.owner != self.owner.settings.owner
            ):
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
            self.identity = identity
        except BaseException:
            self.reject()
            raise

    async def stop(self, *, deadline: datetime) -> None:
        self.reject()
        if not self.removed:
            assert self.owner._client is not None
            await self.owner._client.request(
                "stop", allocation=self.allocation_id, deadline=_monotonic(deadline)
            )

    async def remove(self, *, deadline: datetime) -> None:
        self.reject()
        if not self.removed:
            assert self.owner._client is not None
            await self.owner._client.request(
                "remove", allocation=self.allocation_id, deadline=_monotonic(deadline)
            )
            self.removed = True
            if self.owner._entry is self:
                self.owner._entry = None


def _monotonic(deadline: datetime) -> float:
    return time.monotonic() + (deadline - datetime.now(UTC)).total_seconds()
