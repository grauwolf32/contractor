"""Linearizability, cancellation and exact-bound gates for WorkspaceSession."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_edit_files_toolset import hydrated_workspace, make_tools

from contractor_runtime.allocation import WorkerState
from contractor_runtime.projectfs import OverlayWorkspaceSession, WorkspaceStorageError


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_concurrent_edits_are_serialized_without_lost_updates(tmp_path: Path, storage: str) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(
            tmp_path, storage, "overlay", f"concurrent-{storage}"
        )
        tools = await make_tools(
            tmp_path, session.writer_view(), WorkerState(), ["append_file", "write_file"]
        )
        await asyncio.wait_for(
            asyncio.gather(
                *(tools["append_file"]("lf.txt", f"line-{index}") for index in range(32))
            ),
            timeout=5,
        )
        content = await session.read_text("lf.txt")
        lines = content.splitlines()
        for index in range(32):
            assert lines.count(f"line-{index}") == 1

        await asyncio.wait_for(
            asyncio.gather(
                *(tools["write_file"](f"generated-{index}.txt", str(index)) for index in range(32))
            ),
            timeout=5,
        )
        snapshot = await session.snapshot()
        assert len([item for item in snapshot.files if item.path.startswith("generated-")]) == 32
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_export_snapshot_rejects_concurrent_change_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "memory", "overlay", "stale-export")
        assert isinstance(session, OverlayWorkspaceSession)
        await session.write_text("lf.txt", "first\n")
        bundle = await session.prepare_export()
        await session.write_text("lf.txt", "second\n")

        with pytest.raises(WorkspaceStorageError, match="export_stale"):
            await session.commit_export(bundle)

        assert await session.changed_paths() == ("lf.txt",)
        fresh = await session.prepare_export()
        await session.commit_export(fresh)
        assert await session.changed_paths() == ()
        assert fresh.snapshot == await session.snapshot()
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_cancelled_waiter_and_rollback_races_leave_complete_snapshots(tmp_path: Path) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "memory", "overlay", "cancel-race")
        assert isinstance(session, OverlayWorkspaceSession)

        await session._lock.acquire()  # type: ignore[attr-defined]
        blocked = asyncio.create_task(session.write_text("lf.txt", "cancelled\n"))
        await asyncio.sleep(0)
        blocked.cancel()
        session._lock.release()  # type: ignore[attr-defined]
        with pytest.raises(asyncio.CancelledError):
            await blocked
        assert "cancelled" not in await session.read_text("lf.txt")

        for iteration in range(40):
            await session.write_text("lf.txt", f"value-{iteration}\n")
            await asyncio.wait_for(
                asyncio.gather(
                    session.rollback_changes("lf.txt"),
                    session.write_text("lf.txt", f"winner-{iteration}\n"),
                ),
                timeout=2,
            )
            value = await session.read_text("lf.txt")
            assert value in {
                "alpha\nbeta\nrecognizable-edit-content-secret\n",
                f"winner-{iteration}\n",
            }
            snapshot = await session.snapshot()
            assert snapshot.digest.startswith("sha256:")

        await session.close()
        with pytest.raises(WorkspaceStorageError, match="not_found"):
            await session.snapshot()
        await provider.cleanup(session.storage)

    asyncio.run(scenario())
