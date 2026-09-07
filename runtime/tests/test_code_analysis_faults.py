from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from contractor_runtime.projectfs.storage import WorkspaceSnapshot, WorkspaceTextFile
from contractor_runtime.toolsets.code_analysis.trailmark_host import (
    TrailmarkChildHost,
    TrailmarkHostError,
)

FAULT_CHILD = Path(__file__).parent / "fixtures" / "trailmark_fault_child.py"


def test_definitely_unprocessed_read_only_build_gets_exactly_one_safe_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        marker = tmp_path / "first-child.marker"
        host = TrailmarkChildHost(
            tmp_path / "scratch",
            child_command=(
                sys.executable,
                "-I",
                str(FAULT_CHILD),
                "exit-once-before-read",
                str(marker),
            ),
            build_timeout_seconds=1,
            stop_timeout_seconds=0.1,
        )
        starts = 0
        original_start = host._start_locked

        async def observe_first_exit(mirror: Path) -> None:
            nonlocal starts
            starts += 1
            await original_start(mirror)
            if starts == 1:
                assert host._process is not None
                await host._process.wait()

        monkeypatch.setattr(host, "_start_locked", observe_first_exit)
        result = await host.build(_snapshot())
        assert result.node_count == 1
        assert starts == 2
        # The known-dead child received no bytes; only the one replay has a
        # request ID. There is no retry loop hidden behind this boundary.
        assert host._request_number == 1
        await host.close()
        assert list((tmp_path / "scratch").iterdir()) == []

    asyncio.run(scenario())


def test_crash_after_request_acceptance_is_never_replayed(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "crash")
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.build(_snapshot())
        assert rejected.value.code == "code_analysis_engine_failed"
        assert rejected.value.retryable
        assert host._request_number == 1
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_crash_after_query_acceptance_reaps_child_and_erases_mirror(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "crash-query")
        await host.build(_snapshot())
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.summary()
        assert rejected.value.code == "code_analysis_engine_failed"
        assert rejected.value.retryable
        assert host._request_number == 2
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def test_stderr_flood_is_drained_but_never_retained_as_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        host = _fault_host(tmp_path, "stderr-flood", build_timeout=2)
        result = await host.build(_snapshot())
        assert result.node_count == 1
        assert host.stderr_observed is True
        assert not hasattr(host, "stderr")
        assert "xxxx" not in repr(host.__dict__)
        await host.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("mode", "code", "retryable"),
    [
        ("crash", "code_analysis_engine_failed", True),
        ("oom", "code_analysis_engine_failed", True),
        ("malformed", "code_analysis_engine_failed", True),
        ("partial", "code_analysis_engine_failed", True),
        ("wrong-id", "code_analysis_engine_failed", True),
        ("bad-result", "code_analysis_engine_failed", True),
        ("oversized", "code_analysis_capacity_exceeded", False),
        ("hang", "code_analysis_build_timeout", True),
    ],
)
def test_fault_classification_is_stable_and_scratch_is_immediately_reusable(
    tmp_path: Path,
    mode: str,
    code: str,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        host = _fault_host(
            tmp_path,
            mode,
            build_timeout=0.08 if mode == "hang" else 1,
        )
        with pytest.raises(TrailmarkHostError) as rejected:
            await host.build(_snapshot())
        assert rejected.value.code == code
        assert rejected.value.retryable is retryable
        assert str(rejected.value) == f"Code analysis child failed ({code})"
        assert host.pid is None and not host.mirror_exists
        assert list(tmp_path.iterdir()) == []

        clean = TrailmarkChildHost(tmp_path)
        assert (await clean.build(_snapshot("7"))).node_count >= 1
        await clean.close()
        assert list(tmp_path.iterdir()) == []

    asyncio.run(scenario())


def _fault_host(
    root: Path,
    mode: str,
    *,
    build_timeout: float = 1,
) -> TrailmarkChildHost:
    return TrailmarkChildHost(
        root,
        child_command=(sys.executable, "-I", str(FAULT_CHILD), mode),
        build_timeout_seconds=build_timeout,
        query_timeout_seconds=0.1,
        stop_timeout_seconds=0.05,
    )


def _snapshot(tag: str = "6") -> WorkspaceSnapshot:
    source = "def main():\n    return 1\n"
    return WorkspaceSnapshot(
        directories=(),
        files=(WorkspaceTextFile("app.py", source, len(source.encode("utf-8"))),),
        binary_paths=(),
        digest="sha256:" + tag * 64,
    )
