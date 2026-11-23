"""Process budgets reach physical I/O and retain ownership on expiry."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from fakes.clock import advance_clock, run_with_advanceable_clock
from test_projectfs_zip import archive, workspace_inputs
from test_settings import base_arguments

from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.settings import parse_settings

# Real budgets that physical I/O on a loaded host never exhausts. A test expires
# them by moving the loop clock (fakes.clock) at a point it has gated, never by
# waiting for real time to pass: the configured operation budget, an earlier
# enclosing deadline, and an outer bound that only ends a broken scenario.
IO_BUDGET_SECONDS = 600.0
ENCLOSING_SECONDS = 60.0
SAFETY_NET_SECONDS = 30.0


def test_workspace_timeout_precedence_and_default(tmp_path: Path) -> None:
    args = [
        *base_arguments(tmp_path),
        "--workspace-storage",
        "local",
        "--workspace-work-root",
        str(tmp_path / "project"),
    ]
    environment = {"CONTRACTOR_WORKSPACE_OPERATION_TIMEOUT_SECONDS": "0.25"}
    default = parse_settings(args, {}).workspace
    configured = parse_settings(args, environment).workspace
    overridden = parse_settings(
        [*args, "--workspace-operation-timeout-seconds", "0.5"], environment
    ).workspace
    assert default is not None and default.operation_timeout_seconds == 30.0
    assert configured is not None and configured.operation_timeout_seconds == 0.25
    assert overridden is not None and overridden.operation_timeout_seconds == 0.5


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "invalid", ""])
def test_workspace_timeout_rejects_invalid_values(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        parse_settings(
            [
                *base_arguments(tmp_path),
                "--workspace-storage",
                "local",
                "--workspace-work-root",
                str(tmp_path / "project"),
            ],
            {"CONTRACTOR_WORKSPACE_OPERATION_TIMEOUT_SECONDS": value},
        )


@pytest.mark.parametrize("storage", [None, "memory"])
def test_workspace_timeout_requires_local_storage(tmp_path: Path, storage: str | None) -> None:
    args = base_arguments(tmp_path)
    if storage is not None:
        args += ["--workspace-storage", storage]
    with pytest.raises(SystemExit):
        parse_settings([*args, "--workspace-operation-timeout-seconds", "1"], {})


def test_configured_io_budget_and_shorter_close_deadline_preserve_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        settings = parse_settings(
            [
                *base_arguments(tmp_path),
                "--workspace-storage",
                "local",
                "--workspace-work-root",
                str(tmp_path / "project"),
                "--workspace-operation-timeout-seconds",
                str(IO_BUDGET_SECONDS),
            ],
            {},
        )
        assert settings.workspace is not None
        provider = LocalWorkspaceProvider(settings.workspace)
        spec, reader = workspace_inputs([("source", "", archive({"a.txt": b"before"}))])
        spec.mode = "direct"
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="timeout",
            timeout_seconds=ENCLOSING_SECONDS,
        )
        assert session._local is not None and session._local._filesystem is not None
        local = session._local
        fs = local._filesystem
        started, finish = threading.Event(), threading.Event()
        deadlines: list[float] = []

        def blocked(path: str, *, deadline: float) -> bytes:
            deadlines.append(deadline)
            started.set()
            # Only the test ends this read; the wait merely bounds a broken run.
            assert finish.wait(SAFETY_NET_SECONDS)
            return b"before"

        monkeypatch.setattr(fs, "read", blocked)
        before = time.monotonic()
        operation = asyncio.create_task(session.read_text("a.txt"))
        try:
            assert await asyncio.to_thread(started.wait, SAFETY_NET_SECONDS)
            after = time.monotonic()
            # The configured budget, not the 30 s default, bounds the physical read.
            assert before + IO_BUDGET_SECONDS <= deadlines[0] <= after + IO_BUDGET_SECONDS
            assert not operation.done()
            advance_clock(IO_BUDGET_SECONDS)
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await asyncio.wait_for(operation, SAFETY_NET_SECONDS)
            # The read still runs: the guard fences and keeps owning it.
            assert local.guard.fenced and local.guard._owners and not finish.is_set()
            # The enclosing deadline is earlier than close's own configured budget,
            # so it bounds the close; the physical disposal task survives it.
            closing = asyncio.create_task(
                session.close(deadline=time.monotonic() + ENCLOSING_SECONDS)
            )
            await asyncio.sleep(0)
            retained = local.guard._cleanup
            assert retained is not None and not retained.done() and not closing.done()
            advance_clock(ENCLOSING_SECONDS)
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await asyncio.wait_for(closing, SAFETY_NET_SECONDS)
            assert local.guard._cleanup is retained and not retained.done()
            assert local.guard._owners
            assert Path(session.storage.root).exists()
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await session.read_text("a.txt")
            finish.set()
            await asyncio.wait_for(asyncio.shield(retained), SAFETY_NET_SECONDS)
            await session.close(deadline=time.monotonic() + ENCLOSING_SECONDS)
            assert local.guard._cleanup is retained
            await provider.cleanup(session.storage)
            assert not Path(session.storage.root).exists()
        finally:
            finish.set()
            await asyncio.gather(operation, return_exceptions=True)
            await session.close()
            await provider.cleanup(session.storage)

    run_with_advanceable_clock(scenario())


def test_initialization_uses_earlier_operation_or_enclosing_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contractor_runtime.projectfs.config import WorkspaceLimits
    from contractor_runtime.projectfs.local_direct import LocalDirectWorkspace
    from contractor_runtime.projectfs.local_io import RootedLocalFilesystem

    async def scenario() -> None:
        local = LocalDirectWorkspace(
            str(tmp_path),
            WorkspaceLimits(100, 1000, 1000, 1000),
            operation_timeout_seconds=IO_BUDGET_SECONDS,
        )
        deadlines: list[float] = []
        monkeypatch.setattr(
            RootedLocalFilesystem, "scan", lambda self, *, deadline: deadlines.append(deadline)
        )
        # An enclosing deadline earlier than the operation budget wins exactly.
        enclosing = time.monotonic() + ENCLOSING_SECONDS
        await local.initialize(deadline=enclosing)
        assert deadlines.pop() == enclosing
        # An operation budget earlier than the enclosing deadline wins. The
        # budget starts when initialization does, between the two readings.
        before = time.monotonic()
        await local.initialize(deadline=before + 2 * IO_BUDGET_SECONDS)
        after = time.monotonic()
        assert before + IO_BUDGET_SECONDS <= deadlines.pop() <= after + IO_BUDGET_SECONDS
        await local.close()

    asyncio.run(scenario())
