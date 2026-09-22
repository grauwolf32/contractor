from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from test_projectfs_zip import archive, settings, workspace_inputs

from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    OverlayWorkspaceSession,
    WorkspaceStorageError,
    decode_workspace_state,
    hydrate_workspace,
    overlay,
)
from contractor_runtime.settings import WorkspaceLimits


def test_overlay_mutation_state_and_diff_match_across_unchanged_lowers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [
                (
                    "source",
                    "",
                    archive(
                        {
                            "README.md": b"before\n",
                            "old.txt": b"remove me\n",
                            "kind/child.txt": b"child\n",
                            "logo.bin": b"\x00\xff",
                        }
                    ),
                )
            ]
        )
        spec.mode = "overlay"
        local_provider = LocalWorkspaceProvider(settings("local", tmp_path / "local"))
        memory_provider = MemoryWorkspaceProvider(settings("memory"))
        sessions = [
            await hydrate_workspace(
                provider=local_provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="local",
                timeout_seconds=5,
            ),
            await hydrate_workspace(
                provider=memory_provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="memory",
                timeout_seconds=5,
            ),
        ]
        assert all(isinstance(session, OverlayWorkspaceSession) for session in sessions)

        for session in sessions:
            assert (
                session.storage.filesystem.cat(f"{session.storage.root}/run_workdir/README.md")
                == b"before\n"
            )
            await session.write_text("README.md", "after\n")
            await session.delete_path("old.txt")
            await session.make_directory("new/deep", parents=True)
            await session.write_text("new/deep/file.txt", "created\n")
            await session.delete_path("kind", recursive=True)
            await session.write_text("kind", "directory became file\n")
            with pytest.raises(WorkspaceStorageError, match="binary"):
                await session.write_text("logo.bin", "not allowed")

        snapshots = [await session.snapshot() for session in sessions]
        states = [await session.export_state() for session in sessions]  # type: ignore[attr-defined]
        diffs = [await session.diff(max_bytes=65536) for session in sessions]  # type: ignore[attr-defined]
        assert snapshots[0] == snapshots[1]
        assert states[0] == states[1]
        assert diffs[0] == diffs[1]
        assert not diffs[0].truncated
        assert "a/README.md" in diffs[0].text
        assert "b/new/deep/file.txt" in diffs[0].text

        for session in sessions:
            source = session._source  # type: ignore[attr-defined]
            reconstructed = decode_workspace_state(states[0], source, session.limits)
            assert reconstructed.snapshot() == snapshots[0]
            assert (
                session.storage.filesystem.cat(f"{session.storage.root}/run_workdir/README.md")
                == b"before\n"
            )
            assert session.storage.filesystem.exists(f"{session.storage.root}/run_workdir/old.txt")

        await local_provider.cleanup(sessions[0].storage)
        await memory_provider.cleanup(sessions[1].storage)

    asyncio.run(scenario())


def test_overlay_rolls_back_to_checkpoint_and_bounds_diff(tmp_path: Path) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [("source", "", archive({"a.txt": b"one\n", "b.txt": b"two\n"}))]
        )
        spec.mode = "overlay"
        provider = MemoryWorkspaceProvider(settings("memory"))
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="rollback",
            timeout_seconds=5,
        )
        assert isinstance(session, OverlayWorkspaceSession)

        await session.write_text("a.txt", "changed\n")
        await session.write_text("new.txt", "new\n")
        assert await session.changed_paths() == ("a.txt", "new.txt")
        await session.rollback_changes("a.txt")
        assert await session.read_text("a.txt") == "one\n"
        assert await session.changed_paths() == ("new.txt",)
        await session.commit_checkpoint()
        assert await session.changed_paths() == ()

        await session.write_text("new.txt", "x" * 1024)
        bounded = await session.diff(max_bytes=64)
        assert bounded.returned_bytes <= 64
        assert bounded.truncated
        await session.rollback_changes()
        assert await session.read_text("new.txt") == "new\n"
        with pytest.raises(WorkspaceStorageError, match="not_found"):
            await session.rollback_changes("missing")

        await session.close()
        assert session._source.text_files == {}
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_overlay_write_limits_hold_across_writes_without_revalidating_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [("source", "", archive({"a.txt": b"12345", "dir/b.txt": b"12345"}))]
        )
        spec.mode = "overlay"
        limits = WorkspaceLimits(
            max_files=5, max_expanded_bytes=20, max_managed_text_bytes=16, max_file_bytes=10
        )
        provider = MemoryWorkspaceProvider(settings("memory", limits=limits))
        session = await hydrate_workspace(
            provider=provider,
            spec=spec,
            artifact_reader=reader,
            allocation_id="write-limits",
            timeout_seconds=5,
        )
        assert isinstance(session, OverlayWorkspaceSession)

        def unexpected_tree_validation(*_: object) -> None:
            raise AssertionError("text writes must not re-validate the whole tree")

        monkeypatch.setattr(overlay, "_validate_tree", unexpected_tree_validation)

        # Overwrites are charged by their delta: 5 + 5 -> 10 + 5 fits in 16.
        await session.write_text("a.txt", "x" * 10)
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("dir/b.txt", "y" * 7)
        await session.write_text("dir/b.txt", "y" * 6)
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("c.txt", "z")
        # Shrinking releases bytes for later writes; UTF-8 bytes are charged.
        await session.write_text("a.txt", "é")
        await session.write_text("c.txt", "z" * 8)
        assert await session.read_text("a.txt") == "é"
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("d.txt", "w")

        monkeypatch.undo()
        # A replaced tree is recounted before the next incremental write.
        await session.delete_path("c.txt")
        await session.write_text("d.txt", "w" * 8)
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("e.txt", "v")
        await session.rollback_changes()
        monkeypatch.setattr(overlay, "_validate_tree", unexpected_tree_validation)
        await session.write_text("e.txt", "v")
        await session.write_text("f.txt", "u")
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("g.txt", "t")
        await session.write_text("f.txt", "u" * 5)
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            await session.write_text("f.txt", "u" * 6)
        assert await session.changed_paths() == ("e.txt", "f.txt")

        await session.close()
        await provider.cleanup(session.storage)

    asyncio.run(scenario())
