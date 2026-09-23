from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from contractor_runtime import __version__
from contractor_runtime.capabilities import CapabilitySnapshot
from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import (
    ALLOCATION_OWNER_SUFFIX,
    LocalWorkdirFactory,
    cleanup_orphan_workdirs,
)


def test_each_process_state_has_a_fresh_identity() -> None:
    first = RuntimeState()
    second = RuntimeState()
    assert first.instance_id != second.instance_id


def test_idle_is_committed_only_after_registration_ack(
    runtime_capabilities: CapabilitySnapshot,
) -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-fixed", capabilities=runtime_capabilities)
        registration = await state.registration(make_settings())
        assert registration.observed_state.value == "idle"
        assert registration.software_version == __version__
        advertised = {
            capability.ref: set(capability.tools) for capability in registration.supported_toolsets
        }
        expected = {
            capability.ref: set(capability.tools) for capability in runtime_capabilities.toolsets
        }
        assert advertised == expected
        assert registration.supported_runtimes == list(runtime_capabilities.runtimes)
        assert registration.supported_sandbox_profiles == list(
            runtime_capabilities.sandbox_profiles
        )
        assert registration.workspace_capabilities is None
        assert (await state.snapshot()).process_state is ProcessState.STARTING

        await state.mark_registered()
        assert (await state.snapshot()).process_state is ProcessState.IDLE
        heartbeat = await state.heartbeat(1, 0)
        assert heartbeat.instance_id == "runtime-fixed"
        assert heartbeat.observed_state.value == "idle"

        await state.begin_stopping()
        with pytest.raises(RuntimeError):
            await state.heartbeat(2, 1)

    asyncio.run(scenario())


def test_v2_startup_inputs_are_emitted_after_protocol_activation() -> None:
    async def scenario() -> None:
        capabilities = CapabilitySnapshot.create(
            runtimes=["adk@1"],
            toolsets={},
            sandbox_profiles=["local-workdir@1"],
            runtime_adapters=["otlp-http@1"],
        )
        settings = make_settings()
        settings = Settings(
            control_plane_url=settings.control_plane_url,
            advertised_control_url=settings.advertised_control_url,
            advertised_a2a_url=settings.advertised_a2a_url,
            ca_file=settings.ca_file,
            certificate_file=settings.certificate_file,
            private_key_file=settings.private_key_file,
            initial_labels=("debug",),
        )
        registration = await RuntimeState(
            instance_id="runtime-v1-until-v8-004", capabilities=capabilities
        ).registration(settings)
        wire = registration.model_dump(mode="json", by_alias=True)
        assert wire["apiVersion"] == "contractor/v1alpha1"
        assert wire["initialLabels"] == ["debug"]
        assert wire["supportedRuntimeAdapters"] == ["otlp-http@1"]
        assert "runtimeAgentId" not in wire

    asyncio.run(scenario())


def make_settings() -> Settings:
    placeholder = Path("/tmp/contractor-test-placeholder")
    return Settings(
        control_plane_url="https://localhost:8443",
        advertised_control_url="https://localhost:9443",
        advertised_a2a_url="https://localhost:9444",
        ca_file=placeholder,
        certificate_file=placeholder,
        private_key_file=placeholder,
    )


def test_startup_cleanup_removes_only_recognized_allocation_directories(tmp_path: Path) -> None:
    root = tmp_path / "work"
    orphan = asyncio.run(LocalWorkdirFactory(root).prepare()).path
    (orphan / "nested").mkdir()
    (orphan / "nested" / "data").write_text("temporary", encoding="utf-8")
    preserved = root / "operator-notes"
    preserved.mkdir()
    (preserved / "keep").write_text("important", encoding="utf-8")

    cleanup_orphan_workdirs(root)

    assert not orphan.exists()
    assert (preserved / "keep").read_text(encoding="utf-8") == "important"
    assert sorted(entry.name for entry in root.iterdir()) == ["operator-notes"]


def test_startup_cleanup_never_touches_allocation_names_without_owner_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "work"
    root.mkdir()
    unmarked = root / f"allocation-{'a' * 32}"
    unmarked.mkdir()
    (unmarked / "keep").write_text("operator data", encoding="utf-8")
    loose = root / "allocation-notes.txt"
    loose.write_text("operator notes", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = root / f"allocation-{'b' * 32}"
    linked.symlink_to(outside, target_is_directory=True)
    forged = root / f"allocation-{'c' * 32}"
    forged.mkdir()
    (root / f"{forged.name}{ALLOCATION_OWNER_SUFFIX}").write_text("wrong", encoding="ascii")
    # A crash between marker and directory creation leaves only the marker.
    marker_only = asyncio.run(LocalWorkdirFactory(root).prepare()).path
    marker_only.rmdir()

    cleanup_orphan_workdirs(root)

    assert (unmarked / "keep").read_text(encoding="utf-8") == "operator data"
    assert loose.read_text(encoding="utf-8") == "operator notes"
    assert linked.is_symlink() and outside.is_dir()
    assert forged.is_dir()
    assert not (root / f"{marker_only.name}{ALLOCATION_OWNER_SUFFIX}").exists()


def test_allocation_workdir_cleanup_removes_its_owner_marker(tmp_path: Path) -> None:
    async def scenario() -> None:
        factory = LocalWorkdirFactory(tmp_path / "work")
        workspace = await factory.prepare()
        marker = workspace.root / f"{workspace.path.name}{ALLOCATION_OWNER_SUFFIX}"
        assert marker.is_file() and not (workspace.path / marker.name).exists()
        await factory.cleanup(workspace)
        assert list(workspace.root.iterdir()) == []
        await factory.cleanup(workspace)

    asyncio.run(scenario())
