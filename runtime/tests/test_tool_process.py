"""Subprocess ownership covers creation, pipe transfer and repeated cancellation."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.adapters.http_proxy import ProxySubprocessError, ProxySubprocessLauncher
from contractor_runtime.toolsets.common import process as process_module
from contractor_runtime.toolsets.common.process import run_command


def test_bounded_process_transfers_stdin_and_does_not_inherit_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONTRACTOR_TEST_SECRET", "must-not-reach-child")

    async def scenario() -> None:
        result = await run_command(
            [
                sys.executable,
                "-c",
                (
                    "import os,sys; "
                    "assert 'CONTRACTOR_TEST_SECRET' not in os.environ; "
                    "sys.stdout.buffer.write(sys.stdin.buffer.read()); "
                    "sys.stderr.write('diagnostic'); sys.exit(1)"
                ),
            ],
            input=b"source" * 30_000,
            env={"LANG": "C.UTF-8"},
            timeout=3,
            max_output_bytes=200_000,
        )
        assert result.returncode == 1
        assert result.stdout == b"source" * 30_000
        assert result.stderr == b"diagnostic"

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["spawn", "cleanup"])
def test_repeated_cancellation_joins_child_even_during_spawn_or_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    async def scenario() -> None:
        started, finish = asyncio.Event(), asyncio.Event()
        processes: list[asyncio.subprocess.Process] = []
        original_spawn = asyncio.create_subprocess_exec
        original_stop = process_module._stop

        async def spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            processes.append(process)
            if phase == "spawn":
                started.set()
                await finish.wait()
            return process

        async def stop(process):
            if phase == "cleanup":
                started.set()
                await finish.wait()
            await original_stop(process)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(process_module, "_stop", stop)
        task = asyncio.create_task(
            run_command(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                env={},
                cwd=tmp_path,
                timeout=30,
                max_output_bytes=1024,
            )
        )
        try:
            if phase == "cleanup":
                async with asyncio.timeout(3):
                    while not processes:
                        await asyncio.sleep(0.001)
                task.cancel()
            await asyncio.wait_for(started.wait(), 3)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done(), "cancellation detached a still-owned child"
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
            assert len(processes) == 1
            assert processes[0].returncode is not None
            assert not Path(f"/proc/{processes[0].pid}").exists()
        finally:
            finish.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_proxy_async_process_keeps_private_routing_and_rejects_bearer(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://ambient.invalid:9999")
    monkeypatch.setenv("CONTRACTOR_TEST_SECRET", "must-not-reach-child")

    async def scenario() -> None:
        launcher = _launcher(combined_ca_bundle=b"private CA fixture")
        result = await launcher.run_async(
            [
                sys.executable,
                "-c",
                (
                    "import json,os; from pathlib import Path; "
                    "assert 'CONTRACTOR_TEST_SECRET' not in os.environ; "
                    "print(json.dumps({'proxy': os.environ['HTTPS_PROXY'], "
                    "'ca': Path(os.environ['SSL_CERT_FILE']).read_text()}))"
                ),
            ],
        )
        assert json.loads(result.stdout) == {
            "proxy": "http://proxy-user:proxy-password@proxy.invalid:8080",
            "ca": "private CA fixture",
        }
        assert launcher.active_temporary_roots == ()
        await launcher.aclose()
        with pytest.raises(ProxySubprocessError):
            await launcher.run_async([sys.executable, "-c", "print('closed')"])
        bearer = _launcher(bearer_token="private-bearer")
        with pytest.raises(ProxySubprocessError):
            await bearer.run_async([sys.executable, "-c", "print('must not start')"])
        await bearer.aclose()

    asyncio.run(scenario())


def test_proxy_async_close_cancels_child_before_erasing_its_ca(tmp_path: Path) -> None:
    async def scenario() -> None:
        marker = tmp_path / "child-pid"
        launcher = _launcher(combined_ca_bundle=b"private CA fixture")
        task = asyncio.create_task(
            launcher.run_async(
                [
                    sys.executable,
                    "-c",
                    "import os,time; from pathlib import Path; "
                    "assert Path(os.environ['SSL_CERT_FILE']).exists(); "
                    f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)",
                ]
            )
        )
        try:
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            roots = launcher.active_temporary_roots
            assert len(roots) == 1 and roots[0].exists()
            await asyncio.wait_for(launcher.aclose(), 3)
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not Path(f"/proc/{int(marker.read_text())}").exists()
            assert launcher.active_temporary_roots == ()
            assert not roots[0].exists()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await launcher.aclose()

    asyncio.run(scenario())


def _launcher(**settings) -> ProxySubprocessLauncher:
    return ProxySubprocessLauncher(
        proxy_url="http://proxy.invalid:8080",
        basic_auth=None if settings.get("bearer_token") else ("proxy-user", "proxy-password"),
        bearer_token=settings.get("bearer_token"),
        combined_ca_bundle=settings.get("combined_ca_bundle"),
        bypass_hosts=(),
        timeout_seconds=30,
        metrics=RuntimeAdapterMetricsState(),
    )
