from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from contractor_runtime.settings import Settings
from contractor_runtime.state import ProcessState, RuntimeState
from contractor_runtime.workspace import cleanup_orphan_workdirs


def test_each_process_state_has_a_fresh_identity() -> None:
    first = RuntimeState()
    second = RuntimeState()
    assert first.instance_id != second.instance_id


def test_idle_is_committed_only_after_registration_ack() -> None:
    async def scenario() -> None:
        state = RuntimeState(instance_id="runtime-fixed")
        registration = await state.registration(make_settings())
        assert registration.observed_state.value == "idle"
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
    orphan = root / "allocation-old"
    preserved = root / "operator-notes"
    orphan.mkdir(parents=True)
    (orphan / "data").write_text("temporary", encoding="utf-8")
    preserved.mkdir()
    (preserved / "keep").write_text("important", encoding="utf-8")

    cleanup_orphan_workdirs(root)

    assert not orphan.exists()
    assert (preserved / "keep").read_text(encoding="utf-8") == "important"
