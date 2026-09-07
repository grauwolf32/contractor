from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fakes.podman_lifecycle import Owner, owner
from test_cli import make_settings
from test_podman_allocation import service_for
from test_podman_engine import deadline
from test_projectfs_storage import local_settings

import contractor_runtime.cli as runtime_cli
from contractor_runtime.capabilities import discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.lease import LeaseWatchdog
from contractor_runtime.projectfs import LocalWorkspaceProvider
from contractor_runtime.sandbox.contracts import SandboxContractError
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.ownership import ContentPin, ServiceOwnerLock
from contractor_runtime.sandbox.podman.workroots import MARKER, check_root_policy


def test_recovery_removes_only_verified_predecessors_before_provider_cleanup(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            root = tmp_path / "run_workdir"
            root.mkdir()
            old = PodmanEngine(
                fixture.settings, cli=fixture.cli, owner_directory=tmp_path / "owners"
            )
            await old.open(deadline=deadline())
            identity = await old.create("old", root, deadline=deadline())
            await old.start(identity, deadline=deadline())
            # Simulate a predecessor whose operations have settled before its
            # process exits. The owner service tests cover unsettled operations.
            old._owner_lock.release()
            for record in old._records.values():
                record.content.close()
            foreign = PodmanEngine(
                replace(fixture.settings, owner="foreign"),
                cli=fixture.cli,
                owner_directory=tmp_path / "owners",
            )
            await foreign.open(deadline=deadline())
            other = await foreign.create("other", root, deadline=deadline())
            try:
                await fixture.lifecycle.recover(deadline=deadline())
                assert identity.container_id not in fixture.cli.containers
                assert other.container_id in fixture.cli.containers
                assert root.exists()
            finally:
                await foreign.remove(other, deadline=deadline())
                await foreign.close(deadline=deadline())

    asyncio.run(scenario())


def test_live_owner_and_uncertain_create_block_successor_until_removal(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, events, _exits = await service_for(tmp_path, fixture)
            fixture.cli.block_create = True
            preparing = asyncio.create_task(service.prepare(spec))
            await asyncio.wait_for(fixture.cli.entered.wait(), 3)
            fixture.client.disconnect()  # Runtime crash, not a cleanup receipt
            await asyncio.sleep(0.02)
            with pytest.raises(SandboxContractError):
                ServiceOwnerLock(tmp_path / "owners", fixture.settings.owner).acquire()
            assert not fixture.task.done() and "files" not in events
            fixture.cli.resume.set()
            with pytest.raises(AllocationError):
                await preparing
            await asyncio.wait_for(fixture.task, 5)
            assert not fixture.cli.containers
            assert not any(call[0] == "start" for call in fixture.cli.calls)
            lock = ServiceOwnerLock(tmp_path / "owners", fixture.settings.owner)
            lock.acquire()
            lock.release()
            # Closed IPC cannot turn a failed Runtime's release into success.
            assert service._context is not None and "files" not in events
            # Fixture close has no live owner left to RPC; mark it reaped.
            fixture.lifecycle._client = None

    from contractor_runtime.allocation import AllocationError

    asyncio.run(scenario())


def test_guardian_lease_never_extends_itself_after_runtime_pulses_stop(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, _events, _exits = await service_for(tmp_path, fixture)
            await service.prepare(spec)
            guardian = fixture.guardians[0]
            fixture.lifecycle._heartbeat.cancel()
            await asyncio.gather(fixture.lifecycle._heartbeat, return_exceptions=True)
            # Compress the Runtime liveness window without waiting three seconds.
            fixture.backend.pulse(time.monotonic() + 0.01)
            await asyncio.sleep(0.03)
            await fixture.backend.renew()
            assert guardian.rejected
            fixture.backend.pulse(time.monotonic() + 60)
            with pytest.raises(SandboxContractError):
                await fixture.backend.prepare(
                    spec.allocation_id, fixture.backend.entry.root, deadline=deadline()
                )
            await service.expire_control_lease(5)

    asyncio.run(scenario())


def test_read_only_confirmed_lease_tracks_acknowledgements_without_revival():
    async def scenario():
        now = [10.0]

        async def expired():
            pass

        watchdog = LeaseWatchdog(expired, monotonic=lambda: now[0])
        assert watchdog.confirmed_deadline is None
        await watchdog.arm(3)
        assert watchdog.confirmed_deadline == 13
        now[0] = 11
        assert await watchdog.acknowledge(1, 5)
        assert watchdog.confirmed_deadline == 16
        assert not await watchdog.acknowledge(1, 100)
        now[0] = 16
        assert watchdog.confirmed_deadline is None
        assert not await watchdog.acknowledge(2, 100)
        assert watchdog.confirmed_deadline is None

    asyncio.run(scenario())


def test_deadline_before_create_issuance_can_release_its_pin(tmp_path, monkeypatch):
    verify = ContentPin.verify

    def slow_verify(self):
        time.sleep(0.05)
        verify(self)

    monkeypatch.setattr(ContentPin, "verify", slow_verify)

    async def scenario():
        fixture = Owner(tmp_path)
        root = tmp_path / "run_workdir"
        root.mkdir()
        await fixture.engine.open(deadline=deadline())
        try:
            with pytest.raises(SandboxContractError):
                await fixture.engine.create("never-issued", root, deadline=deadline(0.01))
            await fixture.engine.confirm_removed("never-issued", deadline=deadline())
            assert not fixture.engine._records
            assert not any(call[0] == "create" for call in fixture.cli.calls)
        finally:
            await fixture.engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_recovery_failure_prevents_all_capability_probes(tmp_path):
    async def scenario():
        fixture = Owner(tmp_path)

        class Unavailable:
            async def recover(self, *, deadline):
                raise RuntimeError("unresolved predecessor")

        factories = built_in_factories(
            tmp_path / "scratch",
            workspace_settings=local_settings(tmp_path / "project"),
            execution_lifecycle=Unavailable(),
        )
        with pytest.raises(RuntimeError, match="unresolved"):
            await discover_capabilities(factories)
        assert not (tmp_path / "scratch").exists() and not (tmp_path / "project").exists()
        assert not fixture.cli.calls

    asyncio.run(scenario())


def test_enabled_profile_is_not_advertised_before_executor_gate(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            capabilities = await discover_capabilities(factories)
            assert "podman@1" not in capabilities.sandbox_profiles
            assert not capabilities.has_toolset("code-execution@1")

    asyncio.run(scenario())


@pytest.mark.parametrize("owner_name", [None, "another-service"])
def test_persistent_root_policy_rejects_disabled_or_changed_owner(tmp_path, owner_name):
    check_root_policy(tmp_path, "service-a")
    check_root_policy(tmp_path, "service-a")
    with pytest.raises(SandboxContractError):
        check_root_policy(tmp_path, owner_name)
    assert (tmp_path / MARKER).read_text() == "service-a\n"


def test_root_policy_rejects_symlink_marker_without_following(tmp_path):
    target = tmp_path / "private"
    target.write_text("unchanged")
    (tmp_path / MARKER).symlink_to(target)
    with pytest.raises(OSError):
        check_root_policy(tmp_path, "service-a")
    assert target.read_text() == "unchanged"


def test_unconfigured_provider_cannot_delete_a_marked_predecessor_root(tmp_path):
    async def scenario():
        provider = LocalWorkspaceProvider(local_settings(tmp_path))
        storage = await provider.create("old")
        path = Path(storage.root)
        check_root_policy(tmp_path, "service-a")
        successor = LocalWorkspaceProvider(local_settings(tmp_path))
        with pytest.raises(SandboxContractError):
            await successor.create("new")
        assert path.exists()
        await provider.cleanup(storage)

    asyncio.run(scenario())


def test_cli_recovery_precedes_scratch_deletion_even_on_startup_failure(tmp_path, monkeypatch):
    async def scenario():
        events = []

        class Unavailable:
            def __init__(self, settings):
                pass

            async def recover(self, *, deadline):
                events.append("recover")
                raise RuntimeError("unresolved")

            async def close(self, *, deadline):
                events.append("close")

        fixture = Owner(tmp_path)
        settings = replace(
            make_settings(tmp_path),
            podman=fixture.settings,
            workspace=local_settings(tmp_path / "project"),
        )
        monkeypatch.setattr(runtime_cli, "PodmanLifecycle", Unavailable)
        monkeypatch.setattr(
            runtime_cli, "cleanup_orphan_workdirs", lambda _: events.append("files")
        )
        with pytest.raises(RuntimeError, match="unresolved"):
            await runtime_cli.serve(settings, install_signal_handlers=False)
        assert events == ["recover", "close"]

    asyncio.run(scenario())


def test_cli_shutdown_drains_active_allocation_before_owner_close(tmp_path, monkeypatch):
    import ssl

    from test_cli import FakeServer, InflightTransport

    from contractor_runtime.state import ProcessState, RuntimeState
    from contractor_runtime.workspace import LocalWorkdirFactory

    class WorkspaceTransport(InflightTransport):
        async def post_json(self, path, payload):
            response = await super().post_json(path, payload)
            return {
                **response,
                "privateProtocolVersion": 2,
                "runtimeAgentId": "a" * 64,
                "labels": [],
                "labelRevision": 1,
            }

    async def scenario():
        async with owner(tmp_path) as fixture:
            service, _state, spec, events, exits = await service_for(tmp_path, fixture)
            state = RuntimeState(instance_id="cli-podman-test")
            service._state = state
            local = LocalWorkdirFactory(tmp_path / "scratch")
            factories = replace(
                service._factories,
                sandbox_profiles={**service._factories.sandbox_profiles, local.ref: local},
            )
            settings = replace(
                make_settings(tmp_path),
                podman=fixture.settings,
                workspace=local_settings(tmp_path / "project"),
            )
            monkeypatch.setattr(runtime_cli, "PodmanLifecycle", lambda _: fixture.lifecycle)
            monkeypatch.setattr(runtime_cli, "built_in_factories", lambda *a, **kw: factories)
            monkeypatch.setattr(runtime_cli, "AllocationService", lambda *a, **kw: service)
            monkeypatch.setattr(
                runtime_cli,
                "runtime_agent_client_context",
                lambda **kw: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
            )
            monkeypatch.setattr(
                runtime_cli,
                "runtime_agent_server_context",
                lambda **kw: ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER),
            )
            stop = asyncio.Event()
            transport, server = WorkspaceTransport(), FakeServer()
            serving = asyncio.create_task(
                runtime_cli.serve(
                    settings,
                    state=state,
                    transport=transport,
                    stop_requested=stop,
                    server_factory=lambda _: server,
                    install_signal_handlers=False,
                )
            )
            try:
                await asyncio.wait_for(transport.heartbeat_started.wait(), 5)
                await service.prepare(spec)
                worker = service._context.worker
                root = Path(service._context.project_workspace.storage.root)
                assert fixture.cli.containers and root.exists()
                stop.set()
                await asyncio.wait_for(serving, 5)
                assert worker.stopped and fixture.backend.closed and not fixture.cli.containers
                assert not root.exists() and events[-1] == "files"
                assert (await state.snapshot()).process_state is ProcessState.STOPPING
                assert server.stopped and not exits
            finally:
                stop.set()
                await asyncio.wait_for(serving, 5)

    asyncio.run(scenario())
