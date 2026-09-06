"""Private allocation-owned local engine; not registered as a capability.

All public operations take absolute monotonic deadlines. The caller retains the
workspace guard and content until verified removal. Only trusted hydration code
may supply content_root; this is not a model-selectable host mount API.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from contractor_runtime.podman_command import CommandCapture, PodmanCommand
from contractor_runtime.podman_io import (
    MAX_CLI_BYTES,
    CLIResult,
    LocalPodmanCLI,
    OwnedOperations,
    PodmanCLI,
    remaining,
)
from contractor_runtime.podman_ownership import ContentPin, ServiceOwnerLock
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.sandbox_contracts import (
    ExecutionRequest,
    SandboxContractError,
    SandboxErrorCode,
    SandboxIdentity,
)

LABEL_PREFIX = "io.contractor.sandbox."
MAX_OWNED_CONTAINERS = 1024
_ID = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_STOPPED = frozenset({"created", "configured", "exited", "stopped"})


@dataclass(slots=True, repr=False)
class _Creation:
    allocation_id: str
    creation_id: str
    content: ContentPin
    attempted: bool = False
    identity: SandboxIdentity | None = None


@dataclass(frozen=True, slots=True)
class ContainerState:
    identity: SandboxIdentity = field(repr=False)
    status: str
    running: bool


class PodmanEngine:
    def __init__(
        self,
        settings: PodmanSettings,
        *,
        cli: PodmanCLI | None = None,
        owner_directory: Path | None = None,
    ) -> None:
        if not settings.enabled or settings.owner is None or settings.image is None:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        self._settings = settings
        self._uid, self._gid = os.getuid(), os.getgid()
        if self._uid == 0 or os.geteuid() != self._uid or os.getegid() != self._gid:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        self._cli = cli if cli is not None else LocalPodmanCLI()
        self._owner = settings.owner
        self._incarnation = uuid.uuid4().hex
        # Production services sharing an engine must use this common directory.
        # The override is a private deployment/test seam, never a Workflow field.
        directory = owner_directory or Path(f"/run/user/{self._uid}/contractor-podman-owners")
        self._owner_lock = ServiceOwnerLock(directory, self._owner)
        self._operations = OwnedOperations()
        self._records: dict[str, _Creation] = {}
        self._opened = False
        self._closed = False

    async def open(self, *, deadline: float) -> None:
        async def operation() -> None:
            if self._closed:
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            if self._opened:
                await self._ready()
                return
            await asyncio.to_thread(self._owner_lock.acquire)
            # A failed probe still retains the owner until explicit close. No
            # engine operation or startup recovery may run before this succeeds.
            info = self._json(await self._call(("info", "--format=json"), deadline))
            try:
                valid = info["host"]["security"]["rootless"] is True
            except (TypeError, KeyError):
                valid = False
            if not valid:
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            self._opened = True

        await self._operations.run(operation, self._deadline(deadline))

    async def create(
        self, allocation_id: str, content_root: Path, *, deadline: float
    ) -> SandboxIdentity:
        deadline = self._deadline(deadline, self._settings.prepare_max_seconds)

        async def operation() -> SandboxIdentity:
            await self._ready()
            if not isinstance(allocation_id, str) or _SAFE_ID.fullmatch(allocation_id) is None:
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
            record = self._records.get(allocation_id)
            if record is None:
                if len(self._records) >= MAX_OWNED_CONTAINERS:
                    raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
                content = await asyncio.to_thread(ContentPin, content_root)
                record = _Creation(allocation_id, uuid.uuid4().hex, content)
                self._records[allocation_id] = record
            if record.content.path != content_root:
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
            await asyncio.to_thread(record.content.verify)
            if record.identity is not None:
                if await self._inspect(record.identity, deadline) is None:
                    raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
                return record.identity
            if record.attempted:
                return await self._reconcile(record, deadline)
            remaining(deadline)
            # Persist ownership in memory BEFORE invoking the CLI. Labels are
            # atomic with engine creation and survive a Runtime crash.
            record.attempted = True
            try:
                result = await self._call(self._create_arguments(record), deadline)
                if result.returncode != 0:
                    raise SandboxContractError(SandboxErrorCode.PREPARATION_FAILED)
                container_id = result.stdout.decode("ascii").strip()
                if _ID.fullmatch(container_id) is None:
                    raise ValueError("invalid create response")
                identity = self._identity(record, container_id)
                if await self._inspect(identity, deadline) is None:
                    raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
                record.identity = identity
            except (SandboxContractError, ValueError):
                # A zero/failed/partial response is not proof of non-creation.
                # Exhausted deadline leaves reconciliation for the next caller.
                return await self._reconcile(record, deadline)
            await asyncio.to_thread(record.content.verify)
            return identity

        return await self._operations.run(operation, deadline)

    async def inspect(self, identity: SandboxIdentity, *, deadline: float) -> ContainerState | None:
        async def operation() -> ContainerState | None:
            await self._ready()
            return await self._inspect(identity, deadline)

        deadline = self._deadline(deadline)
        return await self._operations.run(operation, deadline)

    async def supervisor_state(self, identity: SandboxIdentity, *, deadline: float) -> dict:
        """Private verified attachment record; never a tool-visible inspect API."""

        async def operation():
            await self._ready()
            raw = self._json(
                await self._call(("container", "inspect", identity.container_id), deadline)
            )
            if not isinstance(raw, list) or len(raw) != 1:
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
            state = self._parse_inspect(raw[0], identity.container_id)
            if state.identity != identity or state.status != "running" or not state.running:
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
            # Verify again after capture so state from a replaced/foreign
            # resource cannot authorize opening a cgroup.
            if await self._inspect(identity, deadline) != state:
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
            return raw[0]["State"]

        return await self._operations.run(operation, self._deadline(deadline))

    async def confirm_removed(self, allocation_id: str, *, deadline: float) -> None:
        async def operation():
            await self._ready()
            record = self._records.get(allocation_id)
            if record is not None and record.attempted:
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
            if any(item.allocation_id == allocation_id for item in await self._discover(deadline)):
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
            if record is not None:
                # A joined operation that never issued create is unambiguous;
                # release its pin, unlike an absent *attempted* creation.
                await asyncio.to_thread(record.content.close)
                del self._records[allocation_id]

        await self._operations.run(operation, self._deadline(deadline))

    async def start(self, identity: SandboxIdentity, *, deadline: float) -> None:
        deadline = self._deadline(deadline)

        async def operation() -> None:
            await self._ready()
            state = await self._inspect(identity, deadline)
            if state is None:
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            record = self._records.get(identity.allocation_id)
            if record is None or record.identity != identity:
                raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
            await asyncio.to_thread(record.content.verify)
            if state.running and state.status == "running":
                return
            # Do not silently resume/restart an exited or paused workload.
            if state.status not in {"created", "configured"}:
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            await self._call(("start", identity.container_id), deadline)
            state = await self._inspect(identity, deadline)
            if state is None or not state.running or state.status != "running":
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)

        await self._operations.run(operation, deadline)

    async def execute(
        self,
        identity: SandboxIdentity,
        request: ExecutionRequest,
        *,
        deadline: float,
        transport: PodmanCommand,
        launch_deadline: float,
    ) -> CommandCapture:
        """Join every foreground launch before releasing engine ownership."""

        async def operation():
            await self._ready()
            record = self._records.get(identity.allocation_id)
            state = await self._inspect(identity, deadline)
            if (
                record is None
                or record.identity != identity
                or state is None
                or not state.running
                or state.status != "running"
            ):
                raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
            await asyncio.to_thread(record.content.verify)
            remaining(launch_deadline)
            return await transport.run(
                identity, request.command, request.cwd, deadline=launch_deadline
            )

        return await self._operations.run(operation, deadline)

    async def stop(self, identity: SandboxIdentity, *, deadline: float) -> None:
        deadline = self._deadline(deadline)

        async def operation() -> None:
            await self._ready()
            await self._stop(identity, deadline)

        await self._operations.run(operation, deadline)

    async def remove(self, identity: SandboxIdentity, *, deadline: float) -> None:
        deadline = self._deadline(deadline)

        async def operation() -> None:
            await self._ready()
            await self._stop(identity, deadline)
            # Repeat ownership immediately before destructive removal. Never
            # --force, --ignore, --all, --latest, name prefixes or global prune.
            state = await self._inspect(identity, deadline)
            if state is not None:
                if state.running or state.status not in _STOPPED:
                    raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
                await self._call(("rm", identity.container_id), deadline)
                if await self._inspect(identity, deadline) is not None:
                    raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
            record = self._records.get(identity.allocation_id)
            if record is not None and self._identity(record, identity.container_id) == identity:
                await asyncio.to_thread(record.content.close)
                del self._records[identity.allocation_id]

        await self._operations.run(operation, deadline)

    async def discover(self, *, deadline: float) -> tuple[SandboxIdentity, ...]:
        deadline = self._deadline(deadline)

        async def operation() -> tuple[SandboxIdentity, ...]:
            await self._ready()
            return await self._discover(deadline)

        return await self._operations.run(operation, deadline)

    async def close(self, *, deadline: float) -> None:
        deadline = self._deadline(deadline)

        async def operation() -> None:
            if self._closed:
                return
            await asyncio.to_thread(self._owner_lock.verify)
            # Includes predecessor containers and attempts with an unknown ID.
            if any(record.attempted for record in self._records.values()):
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
            if await self._discover(deadline):
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)
            for record in self._records.values():
                await asyncio.to_thread(record.content.close)
            self._records.clear()
            await asyncio.to_thread(self._owner_lock.release)
            self._closed = True

        await self._operations.run(operation, deadline)

    async def _ready(self) -> None:
        if not self._opened or self._closed:
            raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
        await asyncio.to_thread(self._owner_lock.verify)

    async def _call(self, args: tuple[str, ...], deadline: float) -> CLIResult:
        remaining(deadline)
        try:
            result = await self._cli.run(args, deadline=deadline)
            if not isinstance(result, CLIResult) or len(result.stdout) > MAX_CLI_BYTES:
                raise ValueError("invalid CLI result")
            return result
        except (OSError, ValueError) as error:
            if isinstance(error, SandboxContractError):
                raise
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN) from None

    async def _stop(self, identity: SandboxIdentity, deadline: float) -> None:
        state = await self._inspect(identity, deadline)
        if state is None or (not state.running and state.status in _STOPPED):
            return
        grace = min(self._settings.stop_grace_seconds, max(0, int(remaining(deadline)) - 1))
        await self._call(("stop", f"--time={grace}", identity.container_id), deadline)
        state = await self._inspect(identity, deadline)
        if state is not None and (state.running or state.status not in _STOPPED):
            raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)

    async def _inspect(self, identity: SandboxIdentity, deadline: float) -> ContainerState | None:
        if identity.owner != self._owner:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        state = await self._inspect_id(identity.container_id, deadline)
        if state is not None and state.identity != identity:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        return state

    async def _inspect_id(self, container_id: str, deadline: float) -> ContainerState | None:
        if _ID.fullmatch(container_id) is None:
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
        result = await self._call(("container", "inspect", container_id), deadline)
        if result.returncode != 0:
            exists = await self._call(("container", "exists", container_id), deadline)
            if exists.returncode == 1:
                return None
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
        try:
            records = self._json(result)
            if not isinstance(records, list) or len(records) != 1:
                raise ValueError("invalid inspect record")
            return self._parse_inspect(records[0], container_id)
        except (KeyError, TypeError, ValueError):
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE) from None

    def _parse_inspect(self, raw: dict, container_id: str) -> ContainerState:
        try:
            labels = raw["Config"]["Labels"]
            identity = SandboxIdentity(
                owner=labels[LABEL_PREFIX + "owner"],
                incarnation=labels[LABEL_PREFIX + "incarnation"],
                allocation_id=labels[LABEL_PREFIX + "allocation"],
                creation_id=labels[LABEL_PREFIX + "creation"],
                container_id=raw["Id"],
            )
            status, running = raw["State"]["Status"], raw["State"]["Running"]
            if (
                identity.container_id != container_id
                or identity.owner != self._owner
                or labels[LABEL_PREFIX + "managed"] != "1"
                or raw["Name"] != "contractor-" + identity.creation_id
                or status not in _STOPPED | {"running", "paused", "stopping", "removing", "unknown"}
                or type(running) is not bool
            ):
                raise ValueError("invalid container identity/state")
            return ContainerState(identity, status, running)
        except (KeyError, TypeError, ValueError):
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE) from None

    async def _discover(
        self, deadline: float, creation: str | None = None
    ) -> tuple[SandboxIdentity, ...]:
        filters = (
            f"--filter=label={LABEL_PREFIX}managed=1",
            f"--filter=label={LABEL_PREFIX}owner={self._owner}",
        )
        if creation is not None:
            filters += (f"--filter=label={LABEL_PREFIX}creation={creation}",)
        result = await self._call(
            ("ps", "--all", "--no-trunc", "--format={{.ID}}", *filters), deadline
        )
        if result.returncode != 0:
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
        try:
            ids = result.stdout.decode("ascii").splitlines()
        except UnicodeError:
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN) from None
        if len(ids) > MAX_OWNED_CONTAINERS or len(ids) != len(set(ids)):
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
        found = []
        for container_id in ids:
            state = await self._inspect_id(container_id, deadline)
            if state is not None:
                if creation is not None and state.identity.creation_id != creation:
                    raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
                found.append(state.identity)
        return tuple(found)

    async def _reconcile(self, record: _Creation, deadline: float) -> SandboxIdentity:
        found = await self._discover(deadline, record.creation_id)
        if len(found) != 1 or found[0] != self._identity(record, found[0].container_id):
            # Absence after an uncertain create does not authorize another
            # create or releasing the owner. Keep the attempt fenced/owned.
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
        record.identity = found[0]
        await asyncio.to_thread(record.content.verify)
        return found[0]

    def _identity(self, record: _Creation, container_id: str) -> SandboxIdentity:
        return SandboxIdentity(
            self._owner, self._incarnation, record.allocation_id, record.creation_id, container_id
        )

    def _create_arguments(self, record: _Creation) -> tuple[str, ...]:
        settings = self._settings
        assert settings.image is not None
        labels = {
            "managed": "1",
            "owner": self._owner,
            "incarnation": self._incarnation,
            "allocation": record.allocation_id,
            "creation": record.creation_id,
        }
        return (
            "create",
            "--name=contractor-" + record.creation_id,
            *(f"--label={LABEL_PREFIX}{key}={value}" for key, value in labels.items()),
            "--pull=never",
            "--read-only",
            "--read-only-tmpfs=false",
            "--network=none",
            "--ipc=none",
            "--pid=private",
            "--uts=private",
            "--cgroupns=private",
            "--cgroups=enabled",
            "--privileged=false",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--userns=keep-id",
            # Only the inert image PID 1 uses namespace root (a subordinate
            # host UID). Workload exec always selects the nonzero keep-id UID.
            "--user=0:0",
            "--image-volume=ignore",
            "--no-healthcheck",
            "--no-hosts",
            "--log-driver=none",
            "--restart=no",
            "--systemd=false",
            f"--cpus={settings.cpus}",
            f"--memory={settings.memory_bytes}",
            f"--memory-swap={settings.memory_bytes}",
            f"--pids-limit={settings.pids}",
            f"--tmpfs=/tmp:rw,nosuid,nodev,size={settings.tmpfs_bytes},mode=1777",
            f"--mount=type=bind,src={record.content.path},target=/workspace,rw,bind-propagation=rprivate,bind-nonrecursive,relabel=private",
            "--workdir=/workspace",
            "--http-proxy=false",
            "--env-host=false",
            "--unsetenv-all",
            "--env=PATH=/usr/local/bin:/usr/bin:/bin",
            "--env=HOME=/tmp",
            "--env=LANG=C.UTF-8",
            settings.image,
        )

    @staticmethod
    def _json(result: CLIResult):
        try:
            if result.returncode != 0:
                raise ValueError("CLI failed")
            return json.loads(result.stdout)
        except (ValueError, RecursionError):
            raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN) from None

    @staticmethod
    def _deadline(deadline: float, ceiling: int = 120) -> float:
        remaining(deadline)
        return min(deadline, time.monotonic() + ceiling)
