from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Awaitable
from pathlib import Path

import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock

import contractor_runtime.toolsets.code_analysis.trailmark_host as host_module
from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    TrailmarkChildHost,
    TrailmarkHostError,
)

FAULT_CHILD = Path(__file__).parent / "fixtures" / "trailmark_fault_child.py"
# Budgets no loaded host exhausts while a real child starts, answers or is
# reaped. A deadline a test needs to expire is either short against a child
# that hangs, or the SIGTERM grace that through_term_grace expires by moving the
# loop clock. The safety net only ends a broken scenario.
CHILD_BUDGET_SECONDS = 60.0
SAFETY_NET_SECONDS = 30.0


def term_marker(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where a fault child records SIGTERM, outside the scratch root under test."""
    return tmp_path_factory.mktemp("sigterm") / "ignored"


async def through_term_grace[T](operation: Awaitable[T], marker: Path) -> T:
    """Await an operation that may stop a fault child which ignores SIGTERM.

    The SIGTERM grace is real time no host exhausts, so a child that dies on
    SIGTERM is reaped inside it and the clock never moves. Once the child
    records that it ignored SIGTERM, the grace can only end by expiring: the
    loop clock moves past it once, and the SIGKILL reap that follows keeps a
    fresh budget of its own.
    """

    async def ignored() -> None:
        while not marker.exists():
            await asyncio.sleep(0.005)

    running = asyncio.ensure_future(operation)
    watching = asyncio.ensure_future(ignored())
    try:
        await asyncio.wait(
            {running, watching}, timeout=SAFETY_NET_SECONDS, return_when=asyncio.FIRST_COMPLETED
        )
        if watching.done():
            advance_clock(CHILD_BUDGET_SECONDS)
        return await asyncio.wait_for(running, SAFETY_NET_SECONDS)
    finally:
        watching.cancel()


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


def test_hung_build_has_a_hard_deadline_without_blocking_event_loop(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    marker = term_marker(tmp_path_factory)

    async def scenario() -> None:
        # The child hangs, so this short deadline always expires.
        host = _fault_host(tmp_path, "hang", build_timeout=0.08, marker=marker)
        ticks = 0
        stopped = asyncio.Event()

        async def ticker() -> None:
            # Count loop turns, not elapsed time: a blocked loop takes none.
            nonlocal ticks
            while not stopped.is_set():
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(ticker())
        try:
            with pytest.raises(TrailmarkHostError) as rejected:
                await through_term_grace(host.build(_snapshot()), marker)
            assert rejected.value.code == "code_analysis_build_timeout"
        finally:
            stopped.set()
            await task
            await host.close()
        assert ticks >= 8
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    run_with_advanceable_clock(scenario())


def test_hung_query_has_an_independent_hard_deadline_and_is_reaped(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    marker = term_marker(tmp_path_factory)

    async def scenario() -> None:
        # The child answers the build and hangs on the query, so only the
        # short query deadline expires.
        host = TrailmarkChildHost(
            tmp_path,
            child_command=(sys.executable, "-I", str(FAULT_CHILD), "hang-query", str(marker)),
            build_timeout_seconds=CHILD_BUDGET_SECONDS,
            query_timeout_seconds=0.08,
            stop_timeout_seconds=CHILD_BUDGET_SECONDS,
        )
        await host.build(_snapshot())
        with pytest.raises(TrailmarkHostError) as rejected:
            await through_term_grace(host.summary(), marker)
        assert rejected.value.code == "code_analysis_query_timeout"
        assert host.pid is None
        assert host.mirror_exists is False
        await host.close()
        assert list(tmp_path.iterdir()) == []

    run_with_advanceable_clock(scenario())


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


def test_close_kills_a_child_that_ignores_graceful_termination(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    marker = term_marker(tmp_path_factory)

    async def scenario() -> None:
        host = _fault_host(tmp_path, "ignore-term", marker=marker)
        await host.build(_snapshot())
        pid = host.pid
        assert pid is not None
        await through_term_grace(asyncio.gather(host.close(), host.close(), host.close()), marker)
        assert marker.exists()
        assert not Path(f"/proc/{pid}").exists()
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    run_with_advanceable_clock(scenario())


def test_unconfirmed_reap_preserves_mirror_and_requires_fenced_cleanup(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = term_marker(tmp_path_factory)

    async def scenario() -> None:
        host = _fault_host(tmp_path, "ignore-term", marker=marker)
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
        await through_term_grace(host.close(), marker)
        assert host.pid is None
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    run_with_advanceable_clock(scenario())


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


def test_repeated_cancellation_during_materialization_leaks_no_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original = host_module._materialize_snapshot

    def delayed(snapshot: WorkspaceSnapshot, scratch_root: Path) -> host_module._PreparedMirror:
        started.set()
        assert release.wait(timeout=5)
        try:
            return original(snapshot, scratch_root)
        finally:
            finished.set()

    monkeypatch.setattr(host_module, "_materialize_snapshot", delayed)

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path)
        operation = asyncio.create_task(host.build(_snapshot()))
        assert await asyncio.to_thread(started.wait, 2)
        for _ in range(3):
            operation.cancel()
            await asyncio.sleep(0)
            assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert finished.is_set()
        # Removed as part of the cancelled build, not only by a later close().
        assert list(tmp_path.iterdir()) == []
        await host.close()
        assert host.mirror_exists is False

    asyncio.run(scenario())


def test_cancellation_is_not_replaced_by_a_materialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()

    def failing(snapshot: WorkspaceSnapshot, scratch_root: Path) -> host_module._PreparedMirror:
        started.set()
        assert release.wait(timeout=5)
        raise OSError("scratch is full")

    monkeypatch.setattr(host_module, "_materialize_snapshot", failing)

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path)
        operation = asyncio.create_task(host.build(_snapshot()))
        assert await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await host.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_cancelled_build_keeps_an_unremovable_mirror_fenced_until_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    original = host_module._materialize_snapshot
    original_remove = host_module._remove_mirror

    def delayed(snapshot: WorkspaceSnapshot, scratch_root: Path) -> host_module._PreparedMirror:
        started.set()
        assert release.wait(timeout=5)
        return original(snapshot, scratch_root)

    def refuse(path: Path, scratch_root: Path) -> None:
        raise OSError("busy")

    monkeypatch.setattr(host_module, "_materialize_snapshot", delayed)
    monkeypatch.setattr(host_module, "_remove_mirror", refuse)

    async def scenario() -> None:
        host = TrailmarkChildHost(tmp_path)
        operation = asyncio.create_task(host.build(_snapshot()))
        assert await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert host.mirror_exists is True
        with pytest.raises(TrailmarkHostError) as fenced:
            await host.close()
        assert fenced.value.code == "code_analysis_engine_failed"

        monkeypatch.setattr(host_module, "_remove_mirror", original_remove)
        await host.close()
        assert host.mirror_exists is False
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def _fault_host(
    root: Path,
    mode: str,
    *,
    build_timeout: float = CHILD_BUDGET_SECONDS,
    marker: Path | None = None,
) -> TrailmarkChildHost:
    marker_argument = () if marker is None else (str(marker),)
    return TrailmarkChildHost(
        root,
        child_command=(sys.executable, "-I", str(FAULT_CHILD), mode, *marker_argument),
        build_timeout_seconds=build_timeout,
        query_timeout_seconds=CHILD_BUDGET_SECONDS,
        stop_timeout_seconds=CHILD_BUDGET_SECONDS,
    )


def _snapshot(tag: str = "6") -> WorkspaceSnapshot:
    source = "def main():\n    return 1\n"
    return WorkspaceSnapshot(
        directories=(),
        files=(WorkspaceTextFile("app.py", source, len(source.encode("utf-8"))),),
        binary_paths=(),
        digest="sha256:" + tag * 64,
    )
