from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest

import contractor_runtime.toolsets.code_analysis.trailmark_host as host_module
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    TrailmarkChildHost,
    TrailmarkHostError,
)

FAULT_CHILD = Path(__file__).parent / "fixtures" / "trailmark_fault_child.py"


@pytest.mark.parametrize(
    "mode", ["crash", "oom", "oversized", "malformed", "partial", "wrong-id", "bad-result"]
)
def test_child_faults_are_reaped_and_remove_the_mirror(tmp_path: Path, mode: str) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, mode)
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.build(_snapshot())
        assert rejected.value.code in {
            "code_analysis_capacity_exceeded",
            "code_analysis_engine_failed",
        }
        assert host.pid is None
        assert host.mirror_exists is False
        await host.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_hung_build_has_a_hard_deadline_without_blocking_event_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "hang", build_timeout=0.08, stop_timeout=0.05)
        ticks = 0
        stopped = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not stopped.is_set():
                ticks += 1
                await asyncio.sleep(0.005)

        task = asyncio.create_task(ticker())
        try:
            with pytest.raises(TrailmarkHostError) as rejected:
                await host.build(_snapshot())
            assert rejected.value.code == "code_analysis_build_timeout"
        finally:
            stopped.set()
            await task
            await host.close()
        assert ticks >= 8
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_hung_query_has_an_independent_hard_deadline_and_is_reaped(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = TrailmarkChildHost(
            tmp_path,
            child_command=(sys.executable, "-I", str(FAULT_CHILD), "hang-query"),
            build_timeout_seconds=1,
            query_timeout_seconds=0.08,
            stop_timeout_seconds=0.05,
        )
        await host.build(_snapshot())
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.summary()
        assert rejected.value.code == "code_analysis_query_timeout"
        assert host.pid is None
        assert host.mirror_exists is False
        await host.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_rebuild_replaces_one_child_and_concurrent_close_is_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path)
        first = await host.build(_snapshot("4"))
        first_pid = host.pid
        assert first_pid is not None
        second = await host.build(_snapshot("5"))
        second_pid = host.pid
        assert second_pid is not None and second_pid != first_pid
        assert first.snapshot_digest != second.snapshot_digest
        assert not Path(f"/proc/{first_pid}").exists()
        assert len(tuple(tmp_path.glob("code-analysis-mirror-*"))) == 1

        await asyncio.gather(*(host.close() for _ in range(8)))
        await host.close()
        assert host.pid is None
        assert host.mirror_exists is False
        assert not Path(f"/proc/{second_pid}").exists()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_close_kills_a_child_that_ignores_graceful_termination(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "ignore-term", stop_timeout=0.05)
        await host.build(_snapshot())
        pid = host.pid
        assert pid is not None
        await asyncio.gather(host.close(), host.close(), host.close())
        assert not Path(f"/proc/{pid}").exists()
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_unconfirmed_reap_preserves_mirror_and_requires_fenced_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "ignore-term", stop_timeout=0.05)
        await host.build(_snapshot())
        original = host._terminate_process

        async def unconfirmed(_process: asyncio.subprocess.Process) -> bool:
            return False

        monkeypatch.setattr(host, "_terminate_process", unconfirmed)
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.close()
        assert rejected.value.code == "code_analysis_engine_failed"
        assert host.pid is not None
        assert host.mirror_exists is True

        monkeypatch.setattr(host, "_terminate_process", original)
        await host.close()
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_cancellation_waits_for_materialization_then_removes_source_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    original = host_module._materialize_snapshot

    def delayed(snapshot: WorkspaceSnapshot, scratch_root: Path) -> host_module._PreparedMirror:
        started.set()
        assert release.wait(timeout=5)
        return original(snapshot, scratch_root)

    monkeypatch.setattr(host_module, "_materialize_snapshot", delayed)

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path)
        operation = asyncio.create_task(host.build(_snapshot()))
        assert await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await host.close()
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def _fault_host(
    root: Path,
    mode: str,
    *,
    build_timeout: float = 1.0,
    stop_timeout: float = 0.1,
) -> TrailmarkChildHost:
    return TrailmarkChildHost(
        root,
        child_command=(sys.executable, "-I", str(FAULT_CHILD), mode),
        build_timeout_seconds=build_timeout,
        query_timeout_seconds=0.1,
        stop_timeout_seconds=stop_timeout,
    )


def _snapshot(tag: str = "6") -> WorkspaceSnapshot:
    source = "def main():\n    return 1\n"
    return WorkspaceSnapshot(
        directories=(),
        files=(WorkspaceTextFile("app.py", source, len(source.encode("utf-8"))),),
        binary_paths=(),
        digest="sha256:" + tag * 64,
    )
