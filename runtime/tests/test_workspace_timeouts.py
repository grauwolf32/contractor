"""Process budgets reach physical I/O and retain ownership on expiry."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from test_projectfs_zip import archive, workspace_inputs
from test_settings import base_arguments

from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.settings import parse_settings


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
                "0.1",
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
            timeout_seconds=5,
        )
        assert session._local is not None and session._local._filesystem is not None
        local = session._local
        fs = local._filesystem
        started, finish = threading.Event(), threading.Event()
        observed: list[float] = []

        def blocked(path: str, *, deadline: float) -> bytes:
            observed.append(deadline - time.monotonic())
            started.set()
            assert finish.wait(3)
            return b"before"

        monkeypatch.setattr(fs, "read", blocked)
        operation = asyncio.create_task(session.read_text("a.txt"))
        try:
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await asyncio.wait_for(operation, timeout=1)
            assert started.is_set() and 0 < observed[0] <= 0.1
            assert local.guard.fenced and local.guard._owners
            deadline = time.monotonic() + 0.01
            # The short enclosing deadline bounds close even though its own
            # configured budget is longer. Its physical disposal task survives.
            with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
                await session.close(deadline=deadline)
            assert time.monotonic() >= deadline
            retained = local.guard._cleanup
            assert retained is not None and not retained.done()
            assert Path(session.storage.root).exists()
            with pytest.raises(WorkspaceStorageError):
                await session.read_text("a.txt")
            finish.set()
            await asyncio.wait_for(asyncio.shield(retained), timeout=1)
            await session.close(deadline=time.monotonic() + 1)
            assert local.guard._cleanup is retained
            await provider.cleanup(session.storage)
            assert not Path(session.storage.root).exists()
        finally:
            finish.set()
            await asyncio.gather(operation, return_exceptions=True)
            await session.close()
            await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_initialization_uses_earlier_operation_or_enclosing_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contractor_runtime.projectfs.local_direct import LocalDirectWorkspace
    from contractor_runtime.projectfs.local_io import RootedLocalFilesystem
    from contractor_runtime.settings import WorkspaceLimits

    async def scenario() -> None:
        local = LocalDirectWorkspace(
            str(tmp_path), WorkspaceLimits(100, 1000, 1000, 1000), operation_timeout_seconds=0.1
        )
        deadlines: list[float] = []
        monkeypatch.setattr(
            RootedLocalFilesystem, "scan", lambda self, *, deadline: deadlines.append(deadline)
        )
        outer = time.monotonic() + 0.05
        await local.initialize(deadline=outer)
        assert deadlines.pop() == outer
        before = time.monotonic()
        await local.initialize(deadline=before + 1)
        assert before < deadlines.pop() <= time.monotonic() + 0.1
        await local.close()

    asyncio.run(scenario())
