from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

import pytest
from fakes.podman_lifecycle import owner
from test_projectfs_storage import local_settings

from contractor_runtime.capabilities import CapabilityDiscoveryError, discover_capabilities
from contractor_runtime.factories import built_in_factories
from contractor_runtime.sandbox.contracts import SandboxContractError, SandboxErrorCode
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.probe import PROBE_FAILURES
from contractor_runtime.sandbox.podman.settings import PodmanSettings


@pytest.mark.parametrize(
    "failure, phase",
    [
        (None, None),
        ("rootless", "prerequisites"),
        ("cgroup", "prerequisites"),
        ("manager", "prerequisites"),
        ("seccomp", "prerequisites"),
        ("image", "prerequisites"),
        ("image-supervisor", "prerequisites"),
        ("supervisor", "supervisor"),
        ("memory.max", "resources"),
        ("memory.swap.max", "resources"),
        ("pids.max", "resources"),
        ("cpu.max", "resources"),
        ("execution", "execution"),
        ("descendants", "descendants"),
        ("liveness", "liveness"),
    ],
)
def test_owner_probe_checks_each_prerequisite_and_cleans_exact_resources(
    tmp_path, monkeypatch, failure, phase
):
    from fakes.podman_probe import install

    async def scenario():
        async with owner(tmp_path) as fixture:
            root = tmp_path / "run_workdir"
            root.mkdir()
            cli = install(monkeypatch, fixture, root, failure)
            await fixture.lifecycle.recover(deadline=time.monotonic() + 3)
            result = await fixture.lifecycle.probe(root, deadline=time.monotonic() + 10)
            assert result == {"available": failure is None, "failure": phase}
            assert not cli.containers and not fixture.engine._records
            assert fixture.backend.probe_test is None
            assert root.exists()  # owner never erases mounted content
            assert not any(args[0] in {"pull", "build", "prune"} for args in cli.calls)
            assert fixture.lifecycle.probe_available == (failure is None)

    asyncio.run(scenario())


def test_owner_retains_uncertain_probe_removal_for_recovery(tmp_path, monkeypatch):
    from fakes.podman_probe import install

    async def scenario():
        async with owner(tmp_path) as fixture:
            root = tmp_path / "run_workdir"
            root.mkdir()
            cli = install(monkeypatch, fixture, root, None)
            cli.noop_remove = True
            await fixture.lifecycle.recover(deadline=time.monotonic() + 3)
            with pytest.raises(SandboxContractError):
                await fixture.lifecycle.probe(root, deadline=time.monotonic() + 10)
            assert fixture.backend.probe_test is not None
            assert root.exists() and fixture.engine._records
            assert not fixture.lifecycle.probe_available
            cli.noop_remove = False

    asyncio.run(scenario())


def test_late_probe_create_is_owned_until_recovery_and_never_advertised(tmp_path, monkeypatch):
    from fakes.podman_probe import install

    import contractor_runtime.sandbox.podman.probe as module

    monkeypatch.setattr(module, "PROBE_CLEANUP_SECONDS", 0.02)

    async def scenario():
        async with owner(tmp_path) as fixture:
            root = tmp_path / "run_workdir"
            root.mkdir()
            cli = install(monkeypatch, fixture, root, None)
            cli.block_create = True
            await fixture.lifecycle.recover(deadline=time.monotonic() + 3)
            probing = asyncio.create_task(
                fixture.lifecycle.probe(root, deadline=time.monotonic() + 0.15)
            )
            await asyncio.wait_for(cli.entered.wait(), 1)
            with pytest.raises(SandboxContractError):
                await probing
            assert root.exists() and not fixture.lifecycle.probe_available
            assert fixture.engine._records and cli.sequence == 1
            cli.resume.set()
            await fixture.lifecycle.close(deadline=time.monotonic() + 3)
            assert not cli.containers and not fixture.engine._records
            assert cli.sequence == 1 and root.exists()

    asyncio.run(scenario())


def test_cancelled_discovery_preserves_probe_bind_and_waits_for_owner(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            entered, resume = asyncio.Event(), asyncio.Event()
            roots = []

            async def probe(root, *, deadline):
                roots.append(root)
                entered.set()
                await resume.wait()
                return {"available": True, "failure": None}

            fixture.backend.probe = probe
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            discovering = asyncio.create_task(discover_capabilities(factories))
            await asyncio.wait_for(entered.wait(), 3)
            discovering.cancel()
            with pytest.raises(asyncio.CancelledError):
                await discovering
            assert roots[0].exists() and not fixture.lifecycle.probe_available
            resume.set()
            await fixture.lifecycle.close(deadline=time.monotonic() + 3)
            assert roots[0].exists()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [None, *sorted(PROBE_FAILURES)])
def test_discovery_pairs_capabilities_only_after_owner_receipt_and_bind_cleanup(
    tmp_path, caplog, failure
):
    async def scenario():
        async with owner(tmp_path) as fixture:
            events = []

            async def probe(root, *, deadline):
                assert fixture.backend.recovered and fixture.backend.entry is None
                assert root.is_dir()
                events.append("probe")
                return {"available": failure is None, "failure": failure}

            fixture.backend.probe = probe
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            cleanup = factories.workspace_provider.cleanup

            async def clean(storage):
                await cleanup(storage)
                events.append("files")

            factories.workspace_provider.cleanup = clean
            assert not await factories.sandbox_profiles["podman@1"].probe()
            result = await discover_capabilities(factories)
            assert ("podman@1" in result.sandbox_profiles) == (failure is None)
            assert result.supports_tools("code-execution@1", ["exec_command"]) == (failure is None)
            assert "local-workdir@1" in result.sandbox_profiles
            assert result.supports_tools("edit-files@1", ["write_file"])
            assert result.workspace.storage == "local"
            assert events[-2:] == ["probe", "files"]
            assert not list((tmp_path / "project").glob("workspace-*"))
            assert str(tmp_path) not in caplog.text
            if failure is None:
                diagnostics = fixture.lifecycle.probe_diagnostics
                assert diagnostics["podmanBindDiskQuotaEnforced"] is False
                assert diagnostics["podmanImageDigest"].startswith("sha256:")
                assert "localhost" not in repr(diagnostics)

    caplog.set_level(logging.INFO, logger="contractor_runtime.capabilities")
    asyncio.run(scenario())


def test_unconfirmed_cleanup_prevents_registration_and_preserves_probe_files(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            roots = []

            async def probe(root, *, deadline):
                roots.append(root)
                raise SandboxContractError(SandboxErrorCode.CLEANUP_FAILED)

            fixture.backend.probe = probe
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            with pytest.raises(CapabilityDiscoveryError, match="cleanup is unconfirmed"):
                await discover_capabilities(factories)
            assert roots[0].exists()
            assert not fixture.lifecycle.probe_available
            assert not await factories.sandbox_profiles["podman@1"].probe()
            assert not await factories.toolsets["code-execution@1"].probe()

    asyncio.run(scenario())


def test_timeout_retains_late_probe_and_never_reuses_its_positive_receipt(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            entered, resume = asyncio.Event(), asyncio.Event()
            roots = []

            async def probe(root, *, deadline):
                roots.append(root)
                entered.set()
                await resume.wait()
                return {"available": True, "failure": None}

            fixture.backend.probe = probe
            await fixture.lifecycle.recover(deadline=time.monotonic() + 3)
            root = tmp_path / "probe"
            root.mkdir()
            task = asyncio.create_task(
                fixture.lifecycle.probe(root, deadline=time.monotonic() + 0.05)
            )
            await entered.wait()
            with pytest.raises(SandboxContractError):
                await task
            assert root.exists() and not fixture.lifecycle.probe_available
            resume.set()
            await fixture.client.operations.run(lambda: asyncio.sleep(0), time.monotonic() + 3)
            assert not fixture.lifecycle.probe_available

    asyncio.run(scenario())


@pytest.mark.parametrize("storage", ["memory", "absent"])
def test_incompatible_workspace_skips_engine_probe_without_losing_ordinary_capacity(
    tmp_path, storage
):
    from dataclasses import replace

    async def scenario():
        async with owner(tmp_path) as fixture:
            settings = (
                replace(local_settings(tmp_path / "unused"), storage="memory", work_root=None)
                if storage == "memory"
                else None
            )
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=settings,
                execution_lifecycle=fixture.lifecycle,
            )
            result = await discover_capabilities(factories)
            assert result.sandbox_profiles == ("local-workdir@1",)
            assert not result.has_toolset("code-execution@1")
            assert fixture.cli.sequence == 0
            if storage == "memory":
                assert result.workspace.storage == "memory"
                assert "overlay" in result.workspace.modes

    asyncio.run(scenario())


def test_short_startup_budget_never_starts_a_probe_without_cleanup_reserve(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            result = await discover_capabilities(factories, podman_timeout_seconds=0.1)
            assert result.sandbox_profiles == ("local-workdir@1",)
            assert not result.has_toolset("code-execution@1") and fixture.cli.sequence == 0

    asyncio.run(scenario())


def test_provider_cleanup_failure_invalidates_positive_probe_receipt(tmp_path):
    async def scenario():
        async with owner(tmp_path) as fixture:

            async def probe(root, *, deadline):
                return {"available": True, "failure": None}

            fixture.backend.probe = probe
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=fixture.lifecycle,
            )
            cleanup = factories.workspace_provider.cleanup

            async def fail_after_probe(storage):
                if fixture.lifecycle.probe_available:
                    raise OSError("private host path")
                await cleanup(storage)

            factories.workspace_provider.cleanup = fail_after_probe
            with pytest.raises(CapabilityDiscoveryError, match="cleanup is unconfirmed"):
                await discover_capabilities(factories)
            assert not fixture.lifecycle.probe_available

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real-rootless startup capability gate",
)
@pytest.mark.parametrize("image_available", [True, False])
def test_real_complete_startup_probe_advertises_only_after_removal(tmp_path, image_available):
    from datetime import UTC, datetime, timedelta

    from test_capabilities import make_settings
    from test_projectfs_zip import archive, workspace_inputs

    from contractor_runtime.projectfs import hydrate_workspace
    from contractor_runtime.sandbox.contracts import ExecutionRequest
    from contractor_runtime.state import RuntimeState

    async def scenario():
        image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
        assert image, "preinstalled digest-pinned CONTRACTOR_TEST_PODMAN_IMAGE required"
        if not image_available:
            image = image.split("@", 1)[0] + "@sha256:" + uuid.uuid4().hex * 2
        lifecycle = PodmanLifecycle(
            PodmanSettings(enabled=True, image=image, owner="probe-gate-" + uuid.uuid4().hex)
        )
        try:
            factories = built_in_factories(
                tmp_path / "scratch",
                workspace_settings=local_settings(tmp_path / "project"),
                execution_lifecycle=lifecycle,
            )
            result = await discover_capabilities(factories)
            assert ("podman@1" in result.sandbox_profiles) == image_available, (
                lifecycle._probe_result
            )
            assert result.supports_tools("code-execution@1", ["exec_command"]) == image_available
            assert not list((tmp_path / "project").glob("workspace-*"))
            state = RuntimeState(instance_id="verified-probe", capabilities=result)
            registration = await state.registration(make_settings(tmp_path))
            assert ("podman@1" in registration.supported_sandbox_profiles) == image_available
            assert (
                any(
                    tool.ref == "code-execution@1" and tool.tools == ["exec_command"]
                    for tool in registration.supported_toolsets
                )
                == image_available
            )
            assert registration == await state.registration(make_settings(tmp_path))
            if image_available:
                # A real allocation must still work after both sacrificial probes.
                lifecycle.bind_health(lambda: time.monotonic() + 60, lambda: None)
                spec, reader = workspace_inputs([("source", "", archive({"source.txt": b"input"}))])
                session = await hydrate_workspace(
                    provider=factories.workspace_provider,
                    spec=spec,
                    artifact_reader=reader,
                    allocation_id="after-probe",
                    timeout_seconds=5,
                )
                handle = lifecycle.allocate("after-probe", session)
                try:
                    await handle.prepare(deadline=datetime.now(UTC) + timedelta(seconds=15))
                    executed = await handle.executor.execute(
                        ExecutionRequest("printf verified > result.txt"),
                        deadline=datetime.now(UTC) + timedelta(seconds=15),
                    )
                    assert executed.exit_code == 0
                    assert await session.read_text("result.txt") == "verified"
                finally:
                    await handle.remove(deadline=datetime.now(UTC) + timedelta(seconds=15))
                    await session.close()
                    await factories.workspace_provider.cleanup(session.storage)
        finally:
            await lifecycle.close(deadline=time.monotonic() + 15)

    asyncio.run(scenario())
