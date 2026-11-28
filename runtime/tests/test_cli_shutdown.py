"""Exercise the real CLI owner with controlled cleanup and a monotonic clock."""

from __future__ import annotations

import asyncio
import dataclasses
import ssl
from pathlib import Path

import pytest
from test_cli import FakeServer, InflightTransport, make_settings
from test_projectfs_storage import local_settings

import contractor_runtime.cli as runtime_cli
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.work_root_lock import hold_work_roots


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ShutdownRuntime:
    def __init__(self, tmp_path, monkeypatch, capabilities, *, podman=True):
        self.settings = dataclasses.replace(
            make_settings(tmp_path),
            workspace=local_settings(tmp_path / "projects"),
            shutdown_grace_seconds=10,
        )
        self.settings = dataclasses.replace(
            self.settings,
            podman=dataclasses.replace(
                self.settings.podman,
                enabled=podman,
                owner="test-shutdown",
                image="localhost/test@sha256:" + "a" * 64,
            ),
        )
        self.clock = Clock(asyncio.get_running_loop().time())
        self.monkeypatch = monkeypatch
        self.steps = []
        self.costs = {}
        self.block = None
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.exited = asyncio.Event()
        self.exits = []
        self.budgets = []
        self.owner_deadlines = []
        self.stop = asyncio.Event()
        self.connections = []
        owner = self

        def force_exit(code):
            self.exits.append((code, self.clock(), tuple(self.steps)))
            self.exited.set()

        deadline = runtime_cli._ShutdownDeadline
        monkeypatch.setattr(
            runtime_cli,
            "_ShutdownDeadline",
            lambda grace: deadline(grace, monotonic=self.clock, force_exit=force_exit),
        )

        class RPCConnections(runtime_cli.RPCConnections):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.phase = "bulk-rpc" if self._bulk else "allocation-rpc"
                owner.connections.append(self)

            async def aclose(self):
                await owner.step(self.phase)
                await super().aclose()

        class Transport(InflightTransport):
            async def aclose(self):
                await owner.step("control-connections")

            async def exchange(self, *args):
                try:
                    return await super().exchange(*args)
                except asyncio.CancelledError:
                    await owner.step("control-loop")
                    raise

        class Watchdog(runtime_cli.LeaseWatchdog):
            async def run(self, stop):
                try:
                    await super().run(stop)
                finally:
                    await owner.step("watchdog")

        class Allocation(runtime_cli.AllocationService):
            async def close(self, remaining):
                owner.budgets.append(remaining)
                await owner.step("allocation")
                await super().close(remaining)

        class Server(FakeServer):
            async def serve(self):
                await super().serve()
                await owner.step("listener")

        class Cache:
            async def close(self):
                await owner.step("cache")

        class Lifecycle:
            def __init__(self, settings):
                self.settings = settings

            async def recover(self, *, deadline):
                pass

            def bind_health(self, *args):
                pass

            async def close(self, *, deadline):
                owner.owner_deadlines.append(deadline)
                await owner.step("podman-owner")

        async def open_cache(*args):
            return Cache()

        async def discover(*args):
            # Exercise process ownership without running external engine probes.
            return capabilities

        self.transport = Transport()
        self.server = Server()
        monkeypatch.setattr(runtime_cli, "RPCConnections", RPCConnections)
        monkeypatch.setattr(runtime_cli, "ControlRPC", lambda *args: self.transport)
        monkeypatch.setattr(runtime_cli, "AllocationService", Allocation)
        monkeypatch.setattr(runtime_cli, "LeaseWatchdog", Watchdog)
        monkeypatch.setattr(runtime_cli, "PodmanLifecycle", Lifecycle)
        monkeypatch.setattr(runtime_cli.ImmutableArchiveCache, "open", open_cache)
        monkeypatch.setattr(runtime_cli, "discover_capabilities", discover)
        monkeypatch.setattr(
            runtime_cli,
            "runtime_agent_client_context",
            lambda **_: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        )
        monkeypatch.setattr(
            runtime_cli,
            "runtime_agent_server_context",
            lambda **_: ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
        )
        monkeypatch.setattr(runtime_cli, "log_runtime_certificate_expiry", lambda *_, **__: None)

    async def step(self, phase):
        self.steps.append(phase)
        if self.block == phase:
            self.blocked.set()
            await self.release.wait()
        self.clock.advance(self.costs.get(phase, 0))

    def start(self, *, server_factory=None):
        return asyncio.create_task(
            runtime_cli.serve(
                self.settings,
                stop_requested=self.stop,
                server_factory=server_factory or (lambda _: self.server),
                install_signal_handlers=False,
            )
        )

    async def ready(self, serving):
        ready = asyncio.create_task(self.transport.heartbeat_started.wait())
        try:
            await asyncio.wait({ready, serving}, return_when=asyncio.FIRST_COMPLETED)
            if serving.done():
                serving.result()
            await ready
        finally:
            ready.cancel()
            await asyncio.gather(ready, return_exceptions=True)
        self.monkeypatch.setattr(asyncio.get_running_loop(), "time", self.clock)


@pytest.mark.parametrize("podman", [False, True])
def test_shutdown_shares_budget_with_outer_resource_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_capabilities: CapabilitySnapshot,
    podman,
):
    async def scenario():
        runtime = ShutdownRuntime(tmp_path, monkeypatch, runtime_capabilities, podman=podman)
        runtime.costs = {
            "control-loop": 2,
            "allocation": 2,
            "listener": 2,
            "allocation-rpc": 1,
            "bulk-rpc": 1,
            "cache": 1,
        }
        serving = runtime.start()
        await runtime.ready(serving)
        started = runtime.clock()
        runtime.stop.set()
        await serving
        assert runtime.budgets == [8]
        if podman:
            assert runtime.owner_deadlines == [started + 10]
        assert set(runtime.steps[:2]) == {"control-loop", "watchdog"}
        assert runtime.steps[2:] == [
            "allocation",
            "listener",
            "allocation-rpc",
            "bulk-rpc",
            "cache",
            *(["podman-owner"] if podman else []),
            "control-connections",
        ]
        assert not runtime.exits
        # A completed shutdown must retire its timer.
        runtime.clock.advance(20)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not runtime.exits

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "phase",
    [
        "control-loop",
        "watchdog",
        "allocation",
        "listener",
        "allocation-rpc",
        "bulk-rpc",
        "cache",
        "podman-owner",
        "control-connections",
    ],
)
def test_shutdown_deadline_fences_even_when_cleanup_await_does_not_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_capabilities: CapabilitySnapshot, phase
):
    async def scenario():
        runtime = ShutdownRuntime(tmp_path, monkeypatch, runtime_capabilities)
        runtime.block = phase
        serving = runtime.start()
        await runtime.ready(serving)
        runtime.stop.set()
        await runtime.blocked.wait()
        runtime.clock.advance(10)
        await runtime.exited.wait()
        assert len(runtime.exits) == 1
        code, when, steps = runtime.exits[0]
        assert (code, when) == (70, runtime.clock())
        assert not serving.done()
        assert tuple(runtime.steps) == steps
        if phase in {"control-loop", "watchdog"}:
            # The other control loop may finish independently; allocation
            # and its dependencies cannot close beneath either one.
            assert set(steps) <= {"control-loop", "watchdog"}
        else:
            assert steps[-1] == phase
        with (
            pytest.raises(RuntimeError, match="scratch work root in use"),
            hold_work_roots(runtime.settings.work_root, runtime.settings.workspace.work_root),
        ):
            pass
        # Only the injected exit returns. Settle its blocked operation for test teardown.
        runtime.release.set()
        with pytest.raises(RuntimeError, match="Runtime shutdown deadline exceeded"):
            await serving

    asyncio.run(scenario())


@pytest.mark.parametrize("trigger", ["signal", "cancellation"])
def test_repeated_cancellation_keeps_the_process_owner_and_work_root_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_capabilities: CapabilitySnapshot,
    trigger,
):
    async def scenario():
        runtime = ShutdownRuntime(tmp_path, monkeypatch, runtime_capabilities)
        runtime.block = "allocation"
        serving = runtime.start()
        await runtime.ready(serving)
        if trigger == "signal":
            runtime.stop.set()
        else:
            serving.cancel()
        await runtime.blocked.wait()
        for _ in range(3):
            serving.cancel()
            await asyncio.sleep(0)
            assert not serving.done()
        assert runtime.steps[-1] == "allocation"
        with (
            pytest.raises(RuntimeError, match="scratch work root in use"),
            hold_work_roots(runtime.settings.work_root, runtime.settings.workspace.work_root),
        ):
            pass
        runtime.release.set()
        with pytest.raises(asyncio.CancelledError):
            await serving
        assert runtime.steps[-1] == "control-connections"
        assert not runtime.exits
        with hold_work_roots(runtime.settings.work_root, runtime.settings.workspace.work_root):
            pass

    asyncio.run(scenario())


def test_listener_construction_failure_closes_connections_and_outer_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime_capabilities: CapabilitySnapshot
):
    async def scenario():
        runtime = ShutdownRuntime(tmp_path, monkeypatch, runtime_capabilities)

        def failed(_):
            raise ValueError("listener construction failed")

        with pytest.raises(ValueError, match="listener construction failed"):
            await runtime.start(server_factory=failed)
        assert runtime.steps == [
            "allocation-rpc",
            "bulk-rpc",
            "cache",
            "podman-owner",
            "control-connections",
        ]
        assert not runtime.exits

    asyncio.run(scenario())
