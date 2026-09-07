"""Real rootless command gate using the preinstalled, digest-pinned image."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from test_projectfs_local_direct import workspace

from contractor_runtime.sandbox.contracts import ExecutionRequest, SandboxErrorCode
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real-rootless command execution gate",
)


def deadline():
    return datetime.now(UTC) + timedelta(seconds=30)


@asynccontextmanager
async def allocation(tmp_path, **limits):
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    assert image, "preinstalled digest-pinned image required"
    policy = PodmanSettings(
        enabled=True, image=image, owner="exec-gate-" + uuid.uuid4().hex, **limits
    )
    lifecycle = PodmanLifecycle(policy)
    lifecycle.bind_health(lambda: time.monotonic() + 60, lambda: None)
    async with workspace(tmp_path) as (session, root):
        try:
            await lifecycle.recover(deadline=time.monotonic() + 30)
            handle = lifecycle.allocate("command-gate", session)
            await handle.prepare(deadline=deadline())
            yield handle, session, root
        finally:
            if lifecycle._entry is not None:
                await lifecycle._entry.remove(deadline=deadline())
            await lifecycle.close(deadline=time.monotonic() + 30)


def test_real_exec_uid_shell_cwd_utf8_large_command_and_disk_authority(tmp_path):
    async def scenario():
        async with allocation(tmp_path) as (handle, session, root):
            command = (
                "printf '%s' '; $(literal)' > generated.txt; "
                "id -u; printf '\\377'; printf error >&2; exit 7"
            )
            result = await handle.executor.execute(
                ExecutionRequest(command, "src"), deadline=deadline()
            )
            assert result.exit_code == 7, result.observation()
            assert result.stdout == f"{os.getuid()}\n\ufffd" and result.stderr == "error"
            assert await session.read_text("src/generated.txt") == "; $(literal)"
            await session.write_text("src/generated.txt", "host edit")
            result = await handle.executor.execute(
                ExecutionRequest("cat src/generated.txt; #" + "x" * 65000), deadline=deadline()
            )
            assert result.exit_code == 0 and result.stdout == "host edit"
            assert root.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command, timeout, error",
    [
        ("printf started; sleep 30; touch survived", 2, SandboxErrorCode.TIMEOUT),
        ("while :; do printf xxxxxxxxxxxxxxxxxxxxxxxx; done", 30, SandboxErrorCode.OUTPUT_LIMIT),
        # Guardian rejects the descendant proof conservatively; EOF/stopped is
        # not a trusted explanation of why the clean check failed.
        ("sleep 30 >/dev/null 2>&1 &", 30, SandboxErrorCode.CLEANUP_FAILED),
        ("exit 125", 30, SandboxErrorCode.OUTCOME_UNKNOWN),
    ],
)
def test_real_fatal_commands_cannot_leave_writers(tmp_path, command, timeout, error):
    async def scenario():
        async with allocation(tmp_path, preview_bytes=64, output_max_bytes=4096) as (
            handle,
            session,
            root,
        ):
            result = await handle.executor.execute(
                ExecutionRequest(command, timeout_seconds=timeout), deadline=deadline()
            )
            assert result.exit_code is None
            assert result.error_code == error, result.observation()
            assert handle.rejected and session.execution_guard.fenced
            assert not (root / "survived").exists()
            await handle.remove(deadline=deadline())
            assert handle.stopped.is_set()

    asyncio.run(scenario())
