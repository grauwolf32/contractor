from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from contractor_runtime.podman_guardian import CgroupFence, liveness_deadline, serve
from contractor_runtime.podman_supervisor import (
    _START_CLEANUPS,
    CompletionGate,
    GuardianClient,
    open_fence,
    open_pidfd,
)
from contractor_runtime.sandbox_contracts import (
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
)


@pytest.mark.parametrize("lease", [None, True, "200", float("nan"), float("inf"), 0, 99, 100])
def test_lease_rejects_invalid_or_expired_values(lease):
    with pytest.raises(ValueError):
        liveness_deadline(lease, 100)


def test_lease_is_capped_by_both_confirmed_lease_and_liveness_ceiling():
    assert liveness_deadline(104, 100) == 104
    assert liveness_deadline(1000, 100) == 110


def test_closed_guardian_transport_is_typed_cleanup_failure():
    async def scenario():
        control, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        control.setblocking(False)
        guardian = GuardianClient(control, None)
        guardian.disconnect()
        try:
            with pytest.raises(SandboxContractError, match="sandbox_cleanup_failed"):
                await guardian.request("check", deadline=time.monotonic() + 1)
        finally:
            peer.close()

    asyncio.run(scenario())


class FakeFence:
    def __init__(self, clean=True, empty=True):
        self.is_clean = clean
        self.is_empty = empty
        self.killed = threading.Event()
        self.checks = 0

    def init_alive(self):
        return not self.killed.is_set()

    def clean(self, deadline):
        self.checks += 1
        return self.is_clean

    def kill(self):
        self.killed.set()

    def empty(self):
        return self.is_empty


@pytest.mark.parametrize(
    "packet",
    [
        b"",
        b"no-json",
        b"null",
        b"[]",
        b"x" * 257,
        b'{"status":"clean"}',
        b'{"op":"renew","lease":true}',
        b'{"op":"renew","lease":NaN}',
        b'{"op":"stop"}',
        b'{"op":"check","extra":"untrusted"}',
    ],
)
def test_protocol_eof_forgery_and_invalid_renewal_kill(packet):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    fence = FakeFence()
    with parent, child:
        parent.settimeout(2)
        thread = threading.Thread(target=serve, args=(child, fence, time.monotonic() + 1))
        thread.start()
        assert parent.recv(256) == b'{"status":"ready"}'
        if packet:
            parent.send(packet)
            assert parent.recv(256) == b'{"status":"stopped"}'
        else:
            parent.close()
        thread.join(2)
        assert not thread.is_alive() and fence.killed.is_set()


def test_expired_start_and_missing_renewal_kill_without_runtime():
    for lifetime in (-1, 0.05):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        fence = FakeFence()
        with parent, child:
            thread = threading.Thread(
                target=serve, args=(child, fence, time.monotonic() + lifetime)
            )
            thread.start()
            thread.join(1)
            assert not thread.is_alive() and fence.killed.is_set()


def test_trusted_check_and_renewal_then_failed_descendant_proof():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    fence = FakeFence()
    with parent, child:
        parent.settimeout(2)
        thread = threading.Thread(target=serve, args=(child, fence, time.monotonic() + 1))
        thread.start()
        parent.recv(256)
        parent.send(json.dumps({"op": "renew", "lease": time.monotonic() + 0.5}).encode())
        assert parent.recv(256) == b'{"status":"renewed"}'
        parent.send(b'{"op":"check"}')
        assert parent.recv(256) == b'{"status":"clean"}'
        fence.is_clean = False
        parent.send(b'{"op":"check"}')
        assert parent.recv(256) == b'{"status":"stopped"}'
        thread.join(2)
        assert fence.checks == 2 and fence.killed.is_set()


@pytest.mark.parametrize(
    "path", ["/", "/user.slice", "/user.slice/../../", "/tmp/libpod-" + "a" * 64 + ".scope"]
)
def test_never_open_broad_or_unrelated_cgroup(path):
    with pytest.raises(ValueError):
        open_fence(
            "a" * 64, {"CgroupPath": path, "Pid": os.getpid(), "Running": True, "Status": "running"}
        )


def test_pidfd_is_real_close_on_exec_and_detects_live_process():
    fd = open_pidfd(os.getpid())
    try:
        assert not os.get_inheritable(fd)
        assert CgroupFence(-1, os.getpid(), fd).init_alive()
    finally:
        os.close(fd)


def test_cgroup_inventory_includes_recursive_detached_processes_without_links(tmp_path: Path):
    (tmp_path / "cgroup.procs").write_text("100\n")
    child = tmp_path / "container"
    child.mkdir()
    (child / "cgroup.procs").write_text("101\n102\n")
    (tmp_path / "untrusted-link").symlink_to("/sys/fs/cgroup", target_is_directory=True)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert CgroupFence(fd, 100, -1).processes() == {100, 101, 102}
    finally:
        os.close(fd)


def test_failed_completion_never_unfreezes_surviving_writers(monkeypatch):
    fence = CgroupFence(-1, 100, -1)
    writes = []
    monkeypatch.setattr(fence, "write", lambda name, value: writes.append((name, value)))
    monkeypatch.setattr(fence, "events", lambda: {"frozen": 1})
    monkeypatch.setattr(fence, "init_alive", lambda: True)
    monkeypatch.setattr(fence, "processes", lambda: {100, 101})
    assert not fence.clean(time.monotonic() + 1)
    assert writes == [("cgroup.freeze", b"1")]


@pytest.mark.parametrize("running", [True, False])
def test_completion_needs_guardian_and_engine_proof_and_never_revives(running):
    from types import SimpleNamespace

    async def scenario():
        class Guardian:
            disconnected = False

            async def request(self, operation, *, deadline):
                assert operation == "check"

            def disconnect(self):
                self.disconnected = True

        class Engine:
            async def inspect(self, identity, *, deadline):
                return SimpleNamespace(running=running, status="running" if running else "exited")

        guardian = Guardian()
        gate = CompletionGate(Engine(), None, guardian)
        result = ExecutionResult(
            ExecutionStatus.COMPLETED, 7, '{"status":"clean"}', "", False, False, 0
        )
        if running:
            assert await gate.confirm(result, deadline=time.monotonic() + 1) is result
            assert not guardian.disconnected
        else:
            with pytest.raises(SandboxContractError):
                await gate.confirm(result, deadline=time.monotonic() + 1)
            assert guardian.disconnected
            with pytest.raises(SandboxContractError, match="sandbox_unavailable"):
                await gate.confirm(result, deadline=time.monotonic() + 1)

    asyncio.run(scenario())


def test_cancelled_delayed_guardian_spawn_retains_descriptors_and_reaps(monkeypatch, tmp_path):
    async def scenario():
        entered, resume = asyncio.Event(), asyncio.Event()
        reaped = False
        captured = ()

        class Process:
            async def wait(self):
                nonlocal reaped
                reaped = True

        async def delayed_spawn(*args, **kwargs):
            nonlocal captured
            captured = kwargs["pass_fds"]
            assert kwargs["start_new_session"]
            assert not any(key.startswith("CONTRACTOR_") for key in kwargs["env"])
            entered.set()
            await resume.wait()
            for fd in captured:
                os.fstat(fd)
            return Process()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        pidfd = open_pidfd(os.getpid())
        task = asyncio.create_task(
            GuardianClient.start(
                CgroupFence(directory, os.getpid(), pidfd), lease=time.monotonic() + 1
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        os.close(directory)
        os.close(pidfd)
        assert _START_CLEANUPS and not reaped
        for fd in captured:
            os.fstat(fd)
        resume.set()
        await asyncio.gather(*tuple(_START_CLEANUPS))
        assert reaped
        for fd in captured:
            with pytest.raises(OSError):
                os.fstat(fd)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["kill", "empty"])
def test_uncertain_kernel_cleanup_never_emits_stopped_receipt(failure):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    fence = FakeFence()

    def fail():
        raise OSError("kernel failure")

    setattr(fence, failure, fail)

    def guarded():
        with child:
            serve(child, fence, time.monotonic() + 1)

    with parent:
        parent.settimeout(2)
        thread = threading.Thread(target=guarded)
        thread.start()
        assert parent.recv(256) == b'{"status":"ready"}'
        parent.send(b'{"op":"stop"}')
        assert parent.recv(256) == b""
        thread.join(2)
        assert not thread.is_alive()


def test_freeze_timeout_does_not_release_execution(monkeypatch):
    fence = CgroupFence(-1, 100, -1)
    writes = []
    monkeypatch.setattr(fence, "write", lambda name, value: writes.append((name, value)))
    monkeypatch.setattr(fence, "events", lambda: {"frozen": 0})
    assert not fence.clean(time.monotonic() - 1)
    assert writes == [("cgroup.freeze", b"1")]


def test_partial_descriptor_duplication_failure_closes_owned_resources(monkeypatch, tmp_path):
    async def scenario():
        pair = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
        pidfd = open_pidfd(os.getpid())
        duplicates = []
        original_dup = os.dup

        def limited_dup(fd):
            if duplicates:
                raise OSError("descriptor limit")
            result = original_dup(fd)
            duplicates.append(result)
            return result

        monkeypatch.setattr(os, "dup", limited_dup)
        monkeypatch.setattr(socket, "socketpair", lambda *args: pair)
        try:
            with pytest.raises(OSError, match="descriptor limit"):
                await GuardianClient.start(
                    CgroupFence(directory, os.getpid(), pidfd), lease=time.monotonic() + 1
                )
            assert all(sock.fileno() == -1 for sock in pair)
            with pytest.raises(OSError):
                os.fstat(duplicates[0])
            os.fstat(directory)
            os.fstat(pidfd)
        finally:
            os.close(directory)
            os.close(pidfd)
            for sock in pair:
                sock.close()

    asyncio.run(scenario())
