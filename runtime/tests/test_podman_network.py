from __future__ import annotations

import asyncio
import time

import pytest
from fakes.podman_lifecycle import owner
from fakes.podman_probe import install
from test_podman_allocation import finalization, release, service_for
from test_podman_engine import IMAGE, ScriptedEngine, deadline

from contractor_runtime.factories import built_in_factories
from contractor_runtime.sandbox.contracts import (
    PODMAN_NETWORK_PROFILE,
    PODMAN_PROFILE,
    SandboxContractError,
)
from contractor_runtime.sandbox.podman.engine import PodmanEngine
from contractor_runtime.sandbox.podman.settings import PodmanSettings


@pytest.mark.parametrize("enabled,failed_network", [(False, False), (True, False), (True, True)])
def test_network_profile_requires_operator_opt_in_and_its_own_probe(
    tmp_path, monkeypatch, enabled, failed_network
):
    async def scenario():
        async with owner(tmp_path, network_enabled=enabled) as fixture:
            root = tmp_path / "run_workdir"
            root.mkdir()
            cli = install(monkeypatch, fixture, root, None)
            original = fixture.backend.probe

            async def probe(root, *, deadline, network):
                if network == "pasta" and failed_network:
                    return {"available": False, "failure": "execution"}
                return await original(root, deadline=deadline, network=network)

            fixture.backend.probe = probe
            await fixture.lifecycle.recover(deadline=deadline())
            result = await fixture.lifecycle.probe(root, deadline=time.monotonic() + 15)
            assert result == {"available": True, "failure": None}
            assert fixture.lifecycle.probe_available
            assert fixture.lifecycle.network_probe_available == (enabled and not failed_network)
            assert not cli.containers and not fixture.engine._records
            factories = built_in_factories(
                tmp_path / "scratch", execution_lifecycle=fixture.lifecycle
            )
            assert await factories.sandbox_profiles[PODMAN_PROFILE].probe()
            if enabled:
                assert await factories.sandbox_profiles[PODMAN_NETWORK_PROFILE].probe() == (
                    not failed_network
                )
            else:
                assert PODMAN_NETWORK_PROFILE not in factories.sandbox_profiles
            networks = [
                a
                for call in cli.calls
                if call[0] == "create"
                for a in call
                if a.startswith("--network=")
            ]
            assert networks.count("--network=none") == 2
            assert networks.count("--network=pasta") == (2 if enabled and not failed_network else 0)

    asyncio.run(scenario())


def test_engine_pins_network_and_rejects_unapproved_modes_before_create(tmp_path):
    async def scenario():
        cli = ScriptedEngine()
        root = tmp_path / "run_workdir"
        root.mkdir()
        engine = PodmanEngine(
            PodmanSettings(enabled=True, network_enabled=True, image=IMAGE, owner="network-test"),
            cli=cli,
            owner_directory=tmp_path / "owners",
        )
        await engine.open(deadline=deadline())
        try:
            identity = await engine.create("allocation", root, deadline=deadline(), network="pasta")
            assert (
                await engine.create("allocation", root, deadline=deadline(), network="pasta")
                == identity
            )
            for network in ["none", "host", "bridge", "container:other"]:
                with pytest.raises(SandboxContractError):
                    await engine.create("allocation", root, deadline=deadline(), network=network)
            assert len([call for call in cli.calls if call[0] == "create"]) == 1
            create = next(call for call in cli.calls if call[0] == "create")
            assert "--network=pasta" in create and "--network=host" not in create
            assert "--cap-drop=all" in create and "--http-proxy=false" in create
            await engine.remove(identity, deadline=deadline())
        finally:
            await engine.close(deadline=deadline())

    asyncio.run(scenario())


def test_network_operator_setting_is_strict_and_requires_podman():
    assert not PodmanSettings().network_enabled
    for value in [True, 1, "true"]:
        with pytest.raises(ValueError):
            PodmanSettings(network_enabled=value)
    settings = PodmanSettings(enabled=True, image=IMAGE, owner="network-test")
    with pytest.raises(SandboxContractError):
        settings.validate_network("pasta")
    settings.validate_network("none")


def test_allocation_service_selects_networked_execution_and_releases_it(tmp_path):
    async def scenario():
        async with owner(tmp_path, network_enabled=True) as fixture:

            async def verified(root, *, deadline, network):
                return {"available": True, "failure": None}

            fixture.backend.probe = verified
            await fixture.lifecycle.recover(deadline=deadline())
            await fixture.lifecycle.probe(tmp_path / "probe" / "run_workdir", deadline=deadline())
            service, _state, spec, events, exits = await service_for(
                tmp_path, fixture, sandbox_profile="podman-network"
            )
            first = await service.prepare(spec)
            assert await service.prepare(spec) == first
            assert fixture.cli.sequence == 1
            create = next(call for call in fixture.cli.calls if call[0] == "create")
            assert "--network=pasta" in create
            assert "--network=host" not in create
            await service.finalize(finalization(spec))
            await service.release(release(spec))
            assert not fixture.cli.containers and not exits
            assert events == ["hydrate", "worker", "artifacts", "files"]

    asyncio.run(scenario())


@pytest.mark.parametrize("network_enabled", [False, True])
def test_close_joins_health_pulses_before_waiting_for_owner_receipt(tmp_path, network_enabled):
    async def scenario():
        async with owner(tmp_path, network_enabled=network_enabled) as fixture:
            await fixture.lifecycle.recover(deadline=deadline())
            original = fixture.backend.close

            async def close(*, deadline):
                assert fixture.lifecycle._heartbeat.done()
                await original(deadline=deadline)

            fixture.backend.close = close
            await fixture.lifecycle.close(deadline=deadline())
            assert fixture.lifecycle._close_confirmed and fixture.backend.closed

    asyncio.run(scenario())
