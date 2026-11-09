"""Transport failure keeps bounded scanner output while joining its process group."""

import asyncio
import sys
from types import SimpleNamespace

import psutil
import pytest

from contractor_runtime.toolsets.scan import tools as scan
from contractor_runtime.toolsets.scan.process import MAX_OUTPUT_BYTES, ProcessResult


def test_completed_scanner_output_survives_simultaneous_bridge_failure(tmp_path, monkeypatch):
    async def scenario():
        bridge = SimpleNamespace(failed=asyncio.Event(), error_code=None)
        completed = ProcessResult(0, b"finding\n", b"diagnostic\n", duration_ms=7)

        async def process(*args, **kwargs):
            bridge.error_code = "scan_target_denied"
            bridge.failed.set()
            return completed

        monkeypatch.setattr(scan, "run_process", process)
        result = await scan._run_controlled_process(bridge, ["scanner"], tmp_path, 30)
        assert result.error_code == "scan_target_denied"
        assert (result.exit_code, result.stdout, result.stderr, result.duration_ms) == (
            0,
            b"finding\n",
            b"diagnostic\n",
            7,
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("overflow", [False, True])
def test_running_scanner_keeps_output_and_reaps_group_on_bridge_failure(
    tmp_path, monkeypatch, overflow
):
    async def scenario():
        bridge = SimpleNamespace(failed=asyncio.Event(), error_code=None)
        original_read = asyncio.StreamReader.read
        original_spawn = asyncio.create_subprocess_exec
        processes = []

        async def spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            processes.append(process)
            return process

        async def read(self, n=-1):
            data = await original_read(self, n)
            if b"diagnostic" in data:
                bridge.error_code = "scan_request_failed"
                bridge.failed.set()
            return data

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(asyncio.StreamReader, "read", read)
        marker = tmp_path / "child-pid"
        source = (
            "import os,time\n"
            "from pathlib import Path\n"
            "child = os.fork()\n"
            "if child == 0:\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            f"Path({str(marker)!r}).write_text(str(child))\n"
            "os.write(1, b'finding\\n')\n"
            "os.write(2, b'diagnostic\\n')\n"
            + (f"os.write(1, b'x' * {MAX_OUTPUT_BYTES * 2})\n" if overflow else "")
            + "time.sleep(60)\n"
        )
        result = await asyncio.wait_for(
            scan._run_controlled_process(bridge, [sys.executable, "-c", source], tmp_path, 30),
            5,
        )
        assert result.error_code == "scan_request_failed"
        assert result.stdout.startswith(b"finding\n")
        assert result.stderr == b"diagnostic\n"
        assert len(result.stdout) + len(result.stderr) <= MAX_OUTPUT_BYTES
        assert processes[0].returncode is not None
        try:
            child = psutil.Process(int(marker.read_text()))
            async with asyncio.timeout(3):
                while child.status() != psutil.STATUS_ZOMBIE:
                    await asyncio.sleep(0.01)
        except psutil.NoSuchProcess:
            pass

    asyncio.run(scenario())
