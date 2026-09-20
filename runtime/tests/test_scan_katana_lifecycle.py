from __future__ import annotations

import asyncio
import json

import pytest
from test_scan_katana import SEED, Artifacts, make_tool, record
from test_scan_toolset import executable

import contractor_runtime.toolsets.scan.tools as scan
from contractor_runtime.toolsets.scan.process import ProcessResult


class BlockedPublication(Artifacts):
    def __init__(self):
        super().__init__()
        self.publish_started = asyncio.Event()
        self.publish_release = asyncio.Event()
        self.publish_cancelled = asyncio.Event()
        self.cleanup_release = asyncio.Event()

    async def write_artifact(self, *args, **kwargs):
        self.publish_started.set()
        try:
            await self.publish_release.wait()
        except asyncio.CancelledError:
            self.publish_cancelled.set()
            await self.cleanup_release.wait()
            raise
        if self.existing:
            raise RuntimeError("create-only output already exists")
        value = await super().write_artifact(*args, **kwargs)
        self.existing = True
        return value


def controlled_scanner(tmp_path, monkeypatch):
    executable(tmp_path, "katana", "raise AssertionError('controlled scanner must be used')")
    monkeypatch.setenv("PATH", str(tmp_path))
    calls = []

    async def execute(command, directory, timeout):
        calls.append(command)
        return ProcessResult(0, stdout=json.dumps(record()).encode())

    monkeypatch.setattr(scan, "run_process", execute)
    return calls


def test_close_cancels_and_joins_blocked_artifact_publication(tmp_path, monkeypatch):
    calls = controlled_scanner(tmp_path, monkeypatch)

    async def scenario():
        artifacts = BlockedPublication()
        tool, state = await make_tool(tmp_path, artifacts)
        invocation = asyncio.create_task(tool(SEED))
        close = None
        try:
            await asyncio.wait_for(artifacts.publish_started.wait(), 1)
            close = asyncio.create_task(tool.close())
            await asyncio.wait_for(artifacts.publish_cancelled.wait(), 1)
            assert not close.done(), "close must wait for publication cancellation cleanup"
            assert not invocation.done()
            artifacts.cleanup_release.set()
            await asyncio.wait_for(close, 1)
            with pytest.raises(asyncio.CancelledError):
                await invocation
            assert not artifacts.writes and len(calls) == 1
            assert not list((tmp_path / "workspace").iterdir())
            assert state.metrics.counters["tool_errors"] == 1
            assert (await tool(SEED))["errorCode"] == "scan_closed"
            assert len(calls) == 1
        finally:
            artifacts.cleanup_release.set()
            invocation.cancel()
            await asyncio.gather(invocation, return_exceptions=True)
            if close is not None:
                await asyncio.gather(close, return_exceptions=True)
            await tool.close()

    asyncio.run(scenario())


def test_second_call_waits_for_artifact_publication_before_collision_check(tmp_path, monkeypatch):
    calls = controlled_scanner(tmp_path, monkeypatch)

    async def scenario():
        artifacts = BlockedPublication()
        tool, state = await make_tool(tmp_path, artifacts)
        first = asyncio.create_task(tool(SEED))
        second = None
        try:
            await asyncio.wait_for(artifacts.publish_started.wait(), 1)
            second_entered = asyncio.Event()

            async def second_call():
                second_entered.set()
                return await tool(SEED)

            second = asyncio.create_task(second_call())
            await asyncio.wait_for(second_entered.wait(), 1)
            await asyncio.sleep(0)
            assert len(calls) == 1 and len(artifacts.reads) == 1
            assert not second.done()
            artifacts.publish_release.set()
            completed, rejected = await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert completed["status"] == "completed"
            assert rejected["errorCode"] == "scan_output_exists"
            assert len(calls) == 1 and len(artifacts.reads) == 2
            assert len(artifacts.writes) == 1
            assert state.metrics.counters["tool_calls"] == 2
            assert state.metrics.counters["tool_errors"] == 1
            assert not list((tmp_path / "workspace").iterdir())
        finally:
            artifacts.cleanup_release.set()
            artifacts.publish_release.set()
            for task in (first, second):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            await tool.close()

    asyncio.run(scenario())
