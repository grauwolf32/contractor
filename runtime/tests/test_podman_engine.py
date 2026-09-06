from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from contractor_runtime.podman_engine import LABEL_PREFIX, PodmanEngine
from contractor_runtime.podman_io import CLIResult, LocalPodmanCLI, OwnedOperations
from contractor_runtime.podman_ownership import ServiceOwnerLock
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.sandbox_contracts import SandboxContractError, SandboxErrorCode

IMAGE = "localhost/contractor@sha256:" + "a" * 64


def deadline(seconds: float = 5) -> float:
    return time.monotonic() + seconds


class ScriptedEngine:
    def __init__(self) -> None:
        self.calls = []
        self.containers = {}
        self.sequence = 0
        self.lost_create = False
        self.absent_create = False
        self.noop_stop = False
        self.noop_remove = False
        self.noop_start = False
        self.exists_error = False
        self.rootless = True
        self.list_override = None
        self.block_create = False
        self.entered = asyncio.Event()
        self.resume = asyncio.Event()

    async def run(self, args, *, deadline):
        self.calls.append(args)
        if args[0] == "info":
            return self.result({"host": {"security": {"rootless": self.rootless}}})
        if args[0] == "create":
            self.sequence += 1
            container_id = f"{self.sequence:064x}"
            labels = dict(
                item.removeprefix("--label=").split("=", 1)
                for item in args
                if item.startswith("--label=")
            )
            name = next(item.removeprefix("--name=") for item in args if item.startswith("--name="))
            if not self.absent_create:
                self.containers[container_id] = {
                    "Id": container_id,
                    "Name": name,
                    "Config": {"Labels": labels},
                    "State": {"Status": "created", "Running": False},
                }
            self.entered.set()
            if self.block_create:
                await self.resume.wait()
            if self.lost_create:
                raise SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN)
            return CLIResult(125 if self.absent_create else 0, container_id.encode())
        if args[0] == "ps":
            if self.list_override is not None:
                return CLIResult(0, self.list_override)
            filters = dict(
                item.removeprefix("--filter=label=").split("=", 1)
                for item in args
                if item.startswith("--filter=label=")
            )
            return CLIResult(
                0,
                "\n".join(
                    key
                    for key, value in self.containers.items()
                    if all(value["Config"]["Labels"].get(k) == v for k, v in filters.items())
                ).encode(),
            )
        container_id = args[-1]
        if args[:2] == ("container", "exists"):
            return CLIResult(
                125 if self.exists_error else (0 if container_id in self.containers else 1)
            )
        if args[:2] == ("container", "inspect"):
            if container_id not in self.containers:
                return CLIResult(125)
            return self.result([self.containers[container_id]])
        if args[0] == "start":
            if not self.noop_start:
                self.containers[container_id]["State"] = {"Status": "running", "Running": True}
        elif args[0] == "stop":
            if not self.noop_stop:
                self.containers[container_id]["State"] = {"Status": "exited", "Running": False}
        elif args[0] == "rm":
            if not self.noop_remove:
                self.containers.pop(container_id)
        else:
            raise AssertionError(args)
        return CLIResult(0)

    @staticmethod
    def result(value):
        return CLIResult(0, json.dumps(value).encode())


def setup(tmp_path: Path, fake: ScriptedEngine | None = None, owner="runtime-a"):
    root = tmp_path / "allocation" / "run_workdir"
    root.mkdir(parents=True, exist_ok=True)
    engine = PodmanEngine(
        PodmanSettings(enabled=True, image=IMAGE, owner=owner),
        cli=fake or ScriptedEngine(),
        owner_directory=tmp_path / "owners",
    )
    return engine, root


def test_exact_isolation_policy_idempotency_and_cleanup(tmp_path: Path) -> None:
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation-1", root, deadline=deadline())
        assert await engine.create("allocation-1", root, deadline=deadline()) == identity
        assert fake.sequence == 1
        create = next(args for args in fake.calls if args[0] == "create")
        assert create[-1] == IMAGE
        assert {
            "--pull=never",
            "--read-only",
            "--read-only-tmpfs=false",
            "--network=none",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--userns=keep-id",
            "--ipc=none",
            "--pid=private",
            "--cgroupns=private",
            "--image-volume=ignore",
            "--log-driver=none",
            "--http-proxy=false",
            "--unsetenv-all",
            "--env=HOME=/tmp",
            "--cpus=2.0",
            "--memory=2147483648",
            "--memory-swap=2147483648",
            "--pids-limit=256",
        } <= set(create)
        mounts = [arg for arg in create if arg.startswith("--mount=")]
        assert len(mounts) == 1 and f"src={root},target=/workspace,rw," in mounts[0]
        assert "relabel=private" in mounts[0] and "bind-nonrecursive" in mounts[0]
        assert f"--user={os.getuid()}:{os.getgid()}" in create
        assert not any(
            "label=disable" in arg or "seccomp=unconfined" in arg or "chown" in arg
            for arg in create
        )
        await engine.start(identity, deadline=deadline())
        await engine.start(identity, deadline=deadline())
        assert len([call for call in fake.calls if call[0] == "start"]) == 1
        await engine.stop(identity, deadline=deadline())
        with pytest.raises(SandboxContractError):
            await engine.start(identity, deadline=deadline())
        await engine.remove(identity, deadline=deadline())
        await engine.remove(identity, deadline=deadline())
        for call in fake.calls:
            if call[0] in {"start", "stop", "rm"}:
                assert call[-1] == identity.container_id
                assert not set(call) & {"--all", "--latest", "--force", "--ignore"}
        assert not fake.containers and root.exists()
        await engine.close(deadline=deadline())
        await engine.close(deadline=deadline())
        assert (tmp_path / "owners/runtime-a.lock").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("allocation", ["--all", "a;touch x", "../x", "a\nsecret", "x" * 129])
def test_untrusted_identity_never_reaches_cli(tmp_path: Path, allocation: str) -> None:
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        with pytest.raises(SandboxContractError, match="sandbox_incompatible"):
            await engine.create(allocation, root, deadline=deadline())
        assert not any(call[0] == "create" for call in fake.calls)
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["symlink", "parent-symlink", "broad-root", "mount-injection"])
def test_bind_root_validation(tmp_path: Path, kind: str) -> None:
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        if kind == "symlink":
            link = tmp_path / "run_workdir"
            link.symlink_to(root, target_is_directory=True)
            root = link
        elif kind == "parent-symlink":
            (tmp_path / "alias").symlink_to(root.parent, target_is_directory=True)
            root = tmp_path / "alias/run_workdir"
        elif kind == "broad-root":
            root = tmp_path
        else:
            root = tmp_path / "extra,target=/etc" / "run_workdir"
            root.mkdir(parents=True)
        await engine.open(deadline=deadline())
        with pytest.raises(SandboxContractError):
            await engine.create("allocation", root, deadline=deadline())
        assert not any(call[0] == "create" for call in fake.calls)
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_lost_create_response_reuses_exact_resource(tmp_path: Path) -> None:
    async def scenario():
        fake = ScriptedEngine()
        fake.lost_create = True
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation", root, deadline=deadline())
        assert fake.sequence == 1 and len(fake.containers) == 1
        assert await engine.create("allocation", root, deadline=deadline()) == identity
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [True, False])
def test_create_timeout_or_cancellation_retains_operation_and_owner(tmp_path: Path, cancel: bool):
    async def scenario():
        fake = ScriptedEngine()
        fake.block_create = True
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        task = asyncio.create_task(engine.create("allocation", root, deadline=deadline(0.15)))
        await asyncio.wait_for(fake.entered.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(SandboxContractError, match="sandbox_timeout"):
                await task
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            for _ in range(10):
                await asyncio.sleep(0.002)
                ticks += 1

        async def retry():
            with pytest.raises(SandboxContractError, match="sandbox_timeout"):
                await engine.create("allocation", root, deadline=deadline(0.04))
            with pytest.raises(SandboxContractError):
                await engine.close(deadline=deadline(0.04))

        await asyncio.gather(retry(), heartbeat())
        assert ticks == 10 and fake.sequence == 1
        other, _ = setup(tmp_path, fake)
        with pytest.raises(SandboxContractError, match="sandbox_unavailable"):
            await other.open(deadline=deadline())
        fake.resume.set()
        identity = await engine.create("allocation", root, deadline=deadline())
        assert fake.sequence == 1
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())
        await other.open(deadline=deadline())
        await other.close(deadline=deadline())

    asyncio.run(scenario())


def test_absent_uncertain_create_does_not_retry_or_release_owner(tmp_path: Path):
    async def scenario():
        fake = ScriptedEngine()
        fake.absent_create = True
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        for _ in range(2):
            with pytest.raises(SandboxContractError, match="sandbox_outcome_unknown"):
                await engine.create("allocation", root, deadline=deadline())
        assert fake.sequence == 1
        with pytest.raises(SandboxContractError, match="sandbox_cleanup_failed"):
            await engine.close(deadline=deadline())
        # Model a delayed engine commit, then remove through discovery without
        # first needing a successful create replay.
        args = next(call for call in fake.calls if call[0] == "create")
        fake.absent_create = False
        await fake.run(args, deadline=deadline())
        (identity,) = await engine.discover(deadline=deadline())
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["noop_stop", "noop_remove", "exists_error", "noop_start"])
def test_cli_success_or_ambiguous_absence_is_not_state_confirmation(tmp_path: Path, fault: str):
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation", root, deadline=deadline())
        if fault != "noop_start":
            await engine.start(identity, deadline=deadline())
        setattr(fake, fault, True)
        if fault == "exists_error":
            held = fake.containers.pop(identity.container_id)
        with pytest.raises(SandboxContractError):
            if fault == "noop_start":
                await engine.start(identity, deadline=deadline())
            else:
                await engine.remove(identity, deadline=deadline())
        with pytest.raises(SandboxContractError):
            await engine.close(deadline=deadline())
        setattr(fake, fault, False)
        if fault == "exists_error":
            fake.containers[identity.container_id] = held
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["owner", "creation", "allocation", "incarnation", "managed"])
def test_tampered_ownership_never_reaches_stop_or_remove(tmp_path: Path, field: str):
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation", root, deadline=deadline())
        labels = fake.containers[identity.container_id]["Config"]["Labels"]
        original = labels[LABEL_PREFIX + field]
        labels[LABEL_PREFIX + field] = "other"
        with pytest.raises(SandboxContractError):
            await engine.remove(identity, deadline=deadline())
        assert not any(call[0] in {"rm", "stop"} for call in fake.calls)
        labels[LABEL_PREFIX + field] = original
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_discovery_verifies_filters_and_predecessor_identity(tmp_path: Path):
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation", root, deadline=deadline())
        foreign_id = "f" * 64
        foreign = copy.deepcopy(fake.containers[identity.container_id])
        foreign["Id"] = foreign_id
        foreign["Config"]["Labels"][LABEL_PREFIX + "owner"] = "foreign"
        fake.containers[foreign_id] = foreign
        assert await engine.discover(deadline=deadline()) == (identity,)
        fake.list_override = foreign_id.encode()
        with pytest.raises(SandboxContractError):
            await engine.discover(deadline=deadline())
        fake.list_override = None
        with pytest.raises(SandboxContractError):
            await engine.remove(replace(identity, owner="foreign"), deadline=deadline())
        await engine.remove(identity, deadline=deadline())
        # An owned predecessor can be removed but not started/adopted as live work.
        foreign["Config"]["Labels"][LABEL_PREFIX + "owner"] = "runtime-a"
        foreign["Config"]["Labels"][LABEL_PREFIX + "incarnation"] = "predecessor"
        (predecessor,) = await engine.discover(deadline=deadline())
        with pytest.raises(SandboxContractError):
            await engine.start(predecessor, deadline=deadline())
        await engine.remove(predecessor, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_pinned_bind_root_cannot_be_replaced_before_start(tmp_path: Path):
    async def scenario():
        fake = ScriptedEngine()
        engine, root = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        identity = await engine.create("allocation", root, deadline=deadline())
        root.rename(root.with_name("original"))
        root.mkdir()
        with pytest.raises(SandboxContractError):
            await engine.start(identity, deadline=deadline())
        assert not any(call[0] == "start" for call in fake.calls)
        await engine.remove(identity, deadline=deadline())
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_owner_lock_is_exclusive_persistent_and_rejects_symlinks(tmp_path: Path):
    directory = tmp_path / "owners"
    first, second = ServiceOwnerLock(directory, "service"), ServiceOwnerLock(directory, "service")
    first.acquire()
    with pytest.raises(SandboxContractError):
        second.acquire()
    inode = (directory / "service.lock").stat().st_ino
    first.release()
    second.acquire()
    second.release()
    assert (directory / "service.lock").stat().st_ino == inode
    (directory / "link.lock").symlink_to(directory / "service.lock")
    with pytest.raises(SandboxContractError):
        ServiceOwnerLock(directory, "link").acquire()


def test_owner_lock_excludes_another_process(tmp_path: Path):
    directory = tmp_path / "owners"
    lock = ServiceOwnerLock(directory, "service")
    lock.acquire()
    program = """
import sys
from pathlib import Path
from contractor_runtime.podman_ownership import ServiceOwnerLock
from contractor_runtime.sandbox_contracts import SandboxContractError
lock = ServiceOwnerLock(Path(sys.argv[1]), 'service')
try:
    lock.acquire()
except SandboxContractError:
    sys.exit(17)
lock.release()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, str(directory)], timeout=5, capture_output=True
        )
        assert result.returncode == 17
    finally:
        lock.release()
    result = subprocess.run(
        [sys.executable, "-c", program, str(directory)], timeout=5, capture_output=True
    )
    assert result.returncode == 0


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "permissions", "replacement"])
def test_owner_lock_rejects_unsafe_files(tmp_path: Path, fault: str):
    directory = tmp_path / "owners"
    directory.mkdir(mode=0o700)
    path = directory / "service.lock"
    if fault == "symlink":
        path.symlink_to(tmp_path / "missing")
    elif fault == "hardlink":
        original = tmp_path / "other"
        original.touch(mode=0o600)
        path.hardlink_to(original)
    elif fault == "permissions":
        path.touch(mode=0o644)
    lock = ServiceOwnerLock(directory, "service")
    if fault == "replacement":
        lock.acquire()
        path.rename(directory / "old.lock")
        path.touch(mode=0o600)
        try:
            with pytest.raises(SandboxContractError):
                lock.verify()
        finally:
            lock.release()
    else:
        with pytest.raises(SandboxContractError):
            lock.acquire()


@pytest.mark.parametrize(
    "response", [b"short-id", b"f" * 64 + b"\n" + b"f" * 64, b"\xff", b"x" * ((1 << 20) + 1)]
)
def test_bad_discovery_output_is_safe_and_bounded(tmp_path: Path, response: bytes):
    async def scenario():
        fake = ScriptedEngine()
        engine, _ = setup(tmp_path, fake)
        await engine.open(deadline=deadline())
        fake.list_override = response
        with pytest.raises(SandboxContractError) as failure:
            await engine.discover(deadline=deadline())
        assert str(failure.value) in {"sandbox_incompatible", "sandbox_outcome_unknown"}
        assert not any(call[0] in {"rm", "stop"} for call in fake.calls)
        fake.list_override = None
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_rootless_probe_failure_never_creates(tmp_path: Path):
    async def scenario():
        fake = ScriptedEngine()
        fake.rootless = False
        engine, root = setup(tmp_path, fake)
        with pytest.raises(SandboxContractError):
            await engine.open(deadline=deadline())
        with pytest.raises(SandboxContractError):
            await engine.create("allocation", root, deadline=deadline())
        assert fake.sequence == 0
        await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_root_engine_identity_is_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 0)
    with pytest.raises(SandboxContractError, match="sandbox_unavailable"):
        setup(tmp_path)


def fake_executable(tmp_path: Path) -> Path:
    target = tmp_path / "fake-podman"
    shutil.copyfile(Path(__file__).parent / "fakes/podman_cli.py", target)
    target.chmod(0o700)
    return target


def test_real_cli_transport_preserves_argv_and_does_not_inherit_secrets(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("CONTAINER_HOST", "ssh://private")
    monkeypatch.setenv("HTTP_PROXY", "https://secret")
    monkeypatch.setenv("CONTRACTOR_TOKEN", "private-token")

    async def scenario():
        cli = LocalPodmanCLI(fake_executable(tmp_path))
        hostile = "$(touch /tmp/not-executed); --privileged"
        result = await cli.run(("echo", hostile), deadline=deadline())
        observed = json.loads(result.stdout)
        assert observed["argv"][-1] == hostile
        assert observed["argv"][:3] == [
            "--remote=false",
            "--log-level=error",
            "--events-backend=none",
        ]
        assert not {"CONTAINER_HOST", "HTTP_PROXY", "CONTRACTOR_TOKEN"} & observed["env"].keys()
        assert set(observed["env"]) <= {
            "HOME",
            "PATH",
            "XDG_RUNTIME_DIR",
            "DBUS_SESSION_BUS_ADDRESS",
            "LANG",
            "LC_CTYPE",
        }
        assert observed["env"]["PATH"] == "/usr/bin:/bin"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command,code",
    [
        ("flood", "sandbox_output_limit"),
        ("hang", "sandbox_timeout"),
        ("inherited-pipe", "sandbox_timeout"),
    ],
)
def test_real_cli_output_and_wait_are_bounded(tmp_path: Path, command: str, code: str):
    async def scenario():
        cli = LocalPodmanCLI(fake_executable(tmp_path))
        started = time.monotonic()
        with pytest.raises(SandboxContractError, match=code):
            await cli.run((command,), deadline=deadline(0.15))
        assert time.monotonic() - started < 2

    asyncio.run(scenario())


def test_cancelled_waiter_never_launches_and_operation_remains_owned(tmp_path: Path):
    async def scenario():
        gate = OwnedOperations()
        entered, resume = asyncio.Event(), asyncio.Event()

        async def blocked():
            entered.set()
            await resume.wait()

        task = asyncio.create_task(gate.run(blocked, deadline()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        launched = False

        async def unwanted():
            nonlocal launched
            launched = True

        waiter = asyncio.create_task(gate.run(unwanted, deadline()))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        resume.set()
        await gate.run(lambda: asyncio.sleep(0), deadline())
        assert not launched

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [True, False])
def test_delayed_real_cli_spawn_stays_owned_after_caller_departure(
    tmp_path: Path, monkeypatch, cancel
):
    async def scenario():
        cli = LocalPodmanCLI(fake_executable(tmp_path))
        gate = OwnedOperations()
        loop = asyncio.get_running_loop()
        original = loop.subprocess_exec
        entered, resume = asyncio.Event(), asyncio.Event()
        transports = []

        async def delayed(*args, **kwargs):
            entered.set()
            await resume.wait()
            result = await original(*args, **kwargs)
            transports.append(result[0])
            return result

        monkeypatch.setattr(loop, "subprocess_exec", delayed)
        expiry = deadline(0.15)
        task = asyncio.create_task(gate.run(lambda: cli.run(("hang",), deadline=expiry), expiry))
        await entered.wait()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(SandboxContractError, match="sandbox_timeout"):
                await task
        assert transports == []
        resume.set()
        await gate.run(lambda: asyncio.sleep(0), deadline())
        assert len(transports) == 1 and transports[0].get_returncode() is not None

    asyncio.run(scenario())
