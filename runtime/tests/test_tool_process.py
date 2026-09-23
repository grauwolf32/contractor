"""Subprocess ownership covers creation, pipe transfer and repeated cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from contractor_runtime.adapters.host import RuntimeAdapterMetricsState
from contractor_runtime.adapters.http_proxy import ProxySubprocessError, ProxySubprocessLauncher
from contractor_runtime.toolsets.common import process as process_module
from contractor_runtime.toolsets.common.process import (
    ProcessOutputLimitError,
    ProcessTimeoutError,
    run_command,
)
from contractor_runtime.toolsets.likec4 import tools as likec4_module
from contractor_runtime.toolsets.openapi import tools as openapi_module


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


def _detached_holder(marker: Path) -> list[str]:
    # The descendant leaves the process group but keeps the stdout pipe open.
    return [
        "sh",
        "-c",
        f"setsid sh -c 'echo $$ > {marker}; exec sleep 30' & echo started; wait",
    ]


def _kill_detached(marker: Path) -> None:
    if marker.exists():
        with contextlib.suppress(ProcessLookupError, ValueError):
            os.kill(int(marker.read_text()), signal.SIGKILL)


@pytest.mark.parametrize("ending", ["timeout", "cancel"])
def test_detached_descendant_holding_pipes_cannot_stall_stop(tmp_path: Path, ending: str) -> None:
    marker = tmp_path / "holder-pid"

    async def scenario() -> None:
        started = time.monotonic()
        task = asyncio.create_task(
            run_command(
                _detached_holder(marker),
                env={"PATH": "/usr/bin:/bin"},
                timeout=1 if ending == "timeout" else 30,
                max_output_bytes=1024,
            )
        )
        if ending == "timeout":
            with pytest.raises(process_module.ProcessTimeoutError) as raised:
                await asyncio.wait_for(task, 5)
            assert raised.value.output == b"started\n"
        else:
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        assert time.monotonic() - started < 5

    try:
        asyncio.run(scenario())
    finally:
        _kill_detached(marker)


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


def test_proxy_async_child_outcomes_reach_the_caller_unwrapped() -> None:
    async def scenario() -> None:
        launcher = _launcher(combined_ca_bundle=b"private CA fixture")
        with pytest.raises(ProcessTimeoutError) as timed_out:
            await launcher.run_async(
                [sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(60)"],
                timeout=0.3,
            )
        assert timed_out.value.stdout == b"partial\n"
        with pytest.raises(ProcessOutputLimitError):
            await launcher.run_async(
                [sys.executable, "-c", "print('x' * 5000)"], max_output_bytes=1000
            )
        # A non-zero exit is the child's answer (a validator reporting issues),
        # not an adapter failure.
        issues = await launcher.run_async([sys.executable, "-c", "import sys; sys.exit(1)"])
        assert issues.returncode == 1
        assert launcher.active_temporary_roots == ()
        assert launcher._metrics.operations == 3
        assert launcher._metrics.failed_operations == 0
        await launcher.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["timeout", "oversized"])
def test_proxied_validators_report_timeout_and_oversized_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    body = "import time; time.sleep(60)" if outcome == "timeout" else "print('x' * 3_000_000)"
    executable = tmp_path / "validator"
    executable.write_text(f"#!{sys.executable}\n{body}\n")
    executable.chmod(0o700)
    monkeypatch.setattr(openapi_module.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(likec4_module.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(openapi_module, "MAX_VACUUM_OUTPUT_BYTES", 1000)
    monkeypatch.setattr(likec4_module, "MAX_VALIDATOR_OUTPUT_BYTES", 1000)

    async def scenario() -> None:
        launcher = _launcher(timeout_seconds=0.5)
        vacuum = await openapi_module._run_vacuum("openapi: 3.0.3", launcher)
        likec4 = await likec4_module._run_likec4("model {}", tmp_path, launcher)
        await launcher.aclose()
        return vacuum, likec4

    vacuum, likec4 = asyncio.run(scenario())
    if outcome == "timeout":
        assert vacuum["executionError"] == "Vacuum validation timed out"
        assert likec4["executionError"] == "LikeC4 validation timed out"
    else:
        assert vacuum["executionError"] == "Vacuum could not be executed"
        assert likec4["executionError"] == "LikeC4 returned oversized output"


def _launcher(**settings) -> ProxySubprocessLauncher:
    return ProxySubprocessLauncher(
        proxy_url="http://proxy.invalid:8080",
        basic_auth=None if settings.get("bearer_token") else ("proxy-user", "proxy-password"),
        bearer_token=settings.get("bearer_token"),
        combined_ca_bundle=settings.get("combined_ca_bundle"),
        bypass_hosts=(),
        timeout_seconds=settings.get("timeout_seconds", 30),
        metrics=RuntimeAdapterMetricsState(),
    )
