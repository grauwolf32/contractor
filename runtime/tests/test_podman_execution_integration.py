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

from contractor_runtime.sandbox.contracts import (
    PODMAN_NETWORK_PROFILE,
    PODMAN_PROFILE,
    ExecutionRequest,
    SandboxErrorCode,
)
from contractor_runtime.sandbox.podman.lifecycle import PodmanLifecycle
from contractor_runtime.sandbox.podman.settings import PodmanSettings

pytestmark = pytest.mark.skipif(
    os.environ.get("CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE") != "1",
    reason="explicit real-rootless command execution gate",
)


def deadline():
    return datetime.now(UTC) + timedelta(seconds=30)


@asynccontextmanager
async def allocation(tmp_path, *, sandbox_profile=PODMAN_PROFILE, **limits):
    image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
    assert image, "preinstalled digest-pinned image required"
    policy = PodmanSettings(
        enabled=True,
        network_enabled=sandbox_profile == PODMAN_NETWORK_PROFILE,
        image=image,
        owner="exec-gate-" + uuid.uuid4().hex,
        **limits,
    )
    lifecycle = PodmanLifecycle(policy)
    lifecycle.bind_health(lambda: time.monotonic() + 60, lambda: None)
    async with workspace(tmp_path) as (session, root):
        try:
            await lifecycle.recover(deadline=time.monotonic() + 30)
            if sandbox_profile == PODMAN_NETWORK_PROFILE:
                probe_result = await lifecycle.probe(root, deadline=time.monotonic() + 60)
                assert lifecycle.network_probe_available, (
                    probe_result,
                    lifecycle._network_probe_result,
                )
            handle = lifecycle.allocate("command-gate", session, sandbox_profile=sandbox_profile)
            await handle.prepare(deadline=deadline())
            yield handle, session, root
        finally:
            if lifecycle._entries:
                await next(iter(lifecycle._entries.values())).remove(deadline=deadline())
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


def test_real_network_profile_http_and_isolated_profile_denial(tmp_path):
    import shlex

    async def scenario():
        requests = []

        async def respond(reader, writer):
            try:
                request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
                requests.append(request)
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 14\r\n"
                    b"Connection: close\r\n\r\nnetwork-marker"
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(respond, "0.0.0.0", 0, limit=4096)
        port = server.sockets[0].getsockname()[1]
        script = (
            "import urllib.request\n"
            # Engine deliberately omits the host /etc/hosts file. Use the
            # verified pasta host mapping directly for this local fixture.
            f"url='http://169.254.1.2:{port}/marker'\n"
            "try:\n print(urllib.request.urlopen(url,timeout=5).read().decode())\n"
            "except OSError:\n print('network-denied')\n"
        )
        try:
            for profile in [PODMAN_PROFILE, PODMAN_NETWORK_PROFILE]:
                folder = tmp_path / profile.split("@")[0]
                folder.mkdir()
                async with allocation(folder, sandbox_profile=profile) as (handle, _session, _root):
                    result = await handle.executor.execute(
                        ExecutionRequest("python3 -c " + shlex.quote(script)), deadline=deadline()
                    )
                    assert result.exit_code == 0, result.observation()
                    assert result.stdout == (
                        "network-marker\n"
                        if profile == PODMAN_NETWORK_PROFILE
                        else "network-denied\n"
                    ), result.observation()
            assert len(requests) == 1
        finally:
            server.close()
            await server.wait_closed()

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
@pytest.mark.parametrize("sandbox_profile", [PODMAN_PROFILE, PODMAN_NETWORK_PROFILE])
def test_real_fatal_commands_cannot_leave_writers(
    tmp_path, command, timeout, error, sandbox_profile
):
    async def scenario():
        async with allocation(
            tmp_path, sandbox_profile=sandbox_profile, preview_bytes=64, output_max_bytes=4096
        ) as (
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


@pytest.mark.parametrize("sandbox_profile", [PODMAN_PROFILE, PODMAN_NETWORK_PROFILE])
def test_real_cancelled_command_stops_writers_in_both_profiles(tmp_path, sandbox_profile):
    async def scenario():
        async with allocation(tmp_path, sandbox_profile=sandbox_profile) as (handle, session, root):
            command = asyncio.create_task(
                handle.executor.execute(
                    ExecutionRequest("printf started > cancelling; sleep 30; touch survived"),
                    deadline=deadline(),
                )
            )
            try:
                async with asyncio.timeout(10):
                    while not (root / "cancelling").exists():
                        if command.done():
                            pytest.fail(f"command ended before cancellation: {command.result()}")
                        await asyncio.sleep(0.01)
            finally:
                command.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await command
            assert handle.rejected and session.execution_guard.fenced
            assert not (root / "survived").exists()
            await handle.remove(deadline=deadline())
            assert handle.stopped.is_set()

    asyncio.run(scenario())


def test_real_four_allocations_share_one_owner_and_cleanup_preserves_neighbors(tmp_path):
    from contextlib import AsyncExitStack

    from contractor_runtime.sandbox.contracts import SandboxContractError

    async def scenario():
        image = os.environ.get("CONTRACTOR_TEST_PODMAN_IMAGE")
        assert image
        policy = PodmanSettings(enabled=True, image=image, owner="slots-gate-" + uuid.uuid4().hex)
        lifecycle = PodmanLifecycle(policy)
        lifecycle.bind_health(lambda: time.monotonic() + 60, lambda: None)
        jobs = []
        async with AsyncExitStack() as stack:
            try:
                await lifecycle.recover(deadline=time.monotonic() + 30)
                process = lifecycle._client.process
                handles = []
                roots = []
                for index in range(4):
                    session, root = await stack.enter_async_context(
                        workspace(tmp_path / str(index))
                    )
                    handles.append(lifecycle.allocate(f"allocation-{index}", session))
                    roots.append(root)
                await asyncio.gather(*(handle.prepare(deadline=deadline()) for handle in handles))
                for index, handle in enumerate(handles):
                    jobs.append(
                        asyncio.create_task(
                            handle.executor.execute(
                                ExecutionRequest(
                                    "printf started > started; "
                                    "while [ ! -f release ]; do sleep 0.05; done; "
                                    f"printf allocation-{index}",
                                    "",
                                ),
                                deadline=deadline(),
                            )
                        )
                    )
                async with asyncio.timeout(10):
                    while not all((root / "started").exists() for root in roots):
                        assert all(not job.done() for job in jobs)
                        await asyncio.sleep(0.02)
                assert len(lifecycle._entries) == 4 and lifecycle._client.process is process
                await handles[0].remove(deadline=deadline())
                result = await asyncio.gather(jobs[0], return_exceptions=True)
                assert (
                    isinstance(result[0], SandboxContractError) or result[0].error_code is not None
                )
                assert len(lifecycle._entries) == 3 and process.poll() is None
                for root in roots[1:]:
                    (root / "release").touch()
                results = await asyncio.gather(*jobs[1:])
                assert [result.stdout for result in results] == [
                    f"allocation-{i}" for i in range(1, 4)
                ]
                assert all(result.exit_code == 0 for result in results)
                stale = await handles[0].executor.execute(
                    ExecutionRequest("printf stale > stale.txt", ""), deadline=deadline()
                )
                assert stale.error_code == SandboxErrorCode.UNAVAILABLE
                assert not (roots[0] / "stale.txt").exists()
                session, root = await stack.enter_async_context(workspace(tmp_path / "fifth"))
                fifth = lifecycle.allocate("allocation-fifth", session)
                await fifth.prepare(deadline=deadline())
                result = await fifth.executor.execute(
                    ExecutionRequest("printf fifth", ""), deadline=deadline()
                )
                assert result.stdout == "fifth" and result.exit_code == 0
                assert lifecycle._client.process is process and len(lifecycle._entries) == 4
            finally:
                for root in locals().get("roots", []):
                    (root / "release").touch()
                await lifecycle.close(deadline=time.monotonic() + 30)
                await asyncio.gather(*jobs, return_exceptions=True)
        assert process.poll() == 0 and not lifecycle._entries
        inspector = __import__(
            "contractor_runtime.sandbox.podman.engine", fromlist=["PodmanEngine"]
        ).PodmanEngine(policy)
        await inspector.open(deadline=time.monotonic() + 10)
        try:
            assert await inspector.discover(deadline=time.monotonic() + 10) == ()
        finally:
            await inspector.close(deadline=time.monotonic() + 10)

    asyncio.run(scenario())
