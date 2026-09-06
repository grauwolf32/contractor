from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import replace

import pytest

from contractor_runtime.podman_command import PodmanCommand
from contractor_runtime.podman_settings import PodmanSettings
from contractor_runtime.sandbox_contracts import SandboxErrorCode, SandboxIdentity

IDENTITY = SandboxIdentity("owner", "incarnation", "allocation", "creation", "a" * 64)


async def capture(monkeypatch, script, *, seconds=3, policy=None):
    loop = asyncio.get_running_loop()
    spawn = loop.subprocess_exec
    calls = []

    async def scripted(factory, *args, **kwargs):
        calls.append((args, kwargs))
        # Only this fixed test script is run locally, never the model command.
        return await spawn(factory, sys.executable, "-I", "-c", script, **kwargs)

    monkeypatch.setattr(loop, "subprocess_exec", scripted)
    result = await PodmanCommand(policy or PodmanSettings()).run(
        IDENTITY, "private; $(must_not_run_on_host)", "src", deadline=time.monotonic() + seconds
    )
    return result, calls


@pytest.mark.parametrize("code", [0, 2, 124, 128, 255])
def test_exact_argv_and_independent_program_status(monkeypatch, code):
    async def scenario():
        result, calls = await capture(
            monkeypatch,
            "import os; os.write(1, b'{\"exitCode\":0}\\xff'); os.write(2, b'error'); "
            f"raise SystemExit({code})",
        )
        assert result.exit_code == code and result.error is None
        assert result.stdout.endswith(b"\xff") and result.stderr == b"error"
        argv, kwargs = calls[0]
        assert argv[0] == "/usr/bin/podman" and argv[-4:] == (
            IDENTITY.container_id,
            "/bin/sh",
            "-c",
            "private; $(must_not_run_on_host)",
        )
        assert f"--user={os.getuid()}:{os.getgid()}" in argv
        assert "--workdir=/workspace/src" in argv
        assert "--interactive=false" in argv and "--tty=false" in argv
        assert kwargs["stdin"] == asyncio.subprocess.DEVNULL
        assert "PRIVATE_TOKEN" not in kwargs["env"]

    monkeypatch.setenv("PRIVATE_TOKEN", "sensitive")
    asyncio.run(scenario())


@pytest.mark.parametrize("code", [125, 126, 127])
def test_ambiguous_engine_status_discards_diagnostics(monkeypatch, code):
    async def scenario():
        result, _ = await capture(
            monkeypatch, f"print('private host diagnostic'); raise SystemExit({code})"
        )
        assert result.exit_code is None and result.error == SandboxErrorCode.OUTCOME_UNKNOWN
        assert result.stdout == result.stderr == b""

    asyncio.run(scenario())


def test_concurrent_streams_are_counted_before_preview_truncation(monkeypatch):
    async def scenario():
        policy = replace(PodmanSettings(), preview_bytes=32, output_max_bytes=8192)
        result, _ = await capture(
            monkeypatch,
            "import os, threading; t=threading.Thread(target=lambda: os.write(2, b'e'*2048)); "
            "t.start(); os.write(1, b'o'*4096); t.join()",
            policy=policy,
        )
        assert result.exit_code == 0
        assert (result.stdout_bytes, result.stderr_bytes) == (4096, 2048)
        assert result.stdout == b"o" * 32 and result.stderr == b"e" * 32

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [1, 2])
def test_hard_output_overflow_kills_and_reaps_cli(monkeypatch, stream):
    async def scenario():
        policy = replace(PodmanSettings(), preview_bytes=32, output_max_bytes=4096)
        result, _ = await capture(
            monkeypatch, f"import os\nwhile True: os.write({stream}, b'f' * 4096)", policy=policy
        )
        assert result.error == SandboxErrorCode.OUTPUT_LIMIT and result.exit_code is None
        assert result.stdout_bytes + result.stderr_bytes > 4096
        assert len(result.stdout) <= 32 and len(result.stderr) <= 32

    asyncio.run(scenario())


def test_timeout_preserves_bounded_partial_output_and_reaps_cli(monkeypatch):
    async def scenario():
        started = time.monotonic()
        result, _ = await capture(
            monkeypatch, "import os,time; os.write(1,b'partial'); time.sleep(30)", seconds=0.2
        )
        assert time.monotonic() - started < 2
        assert result.error == SandboxErrorCode.TIMEOUT and result.stdout == b"partial"

    asyncio.run(scenario())
