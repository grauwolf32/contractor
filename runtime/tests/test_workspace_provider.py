from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    ProjectWorkspaceStorage,
)
from contractor_runtime.projectfs.provider import LOCAL_DIRECTORY_PREFIX
from contractor_runtime.settings import WorkspaceLimits, WorkspaceSettings


def test_local_provider_removes_only_marker_owned_immediate_stale_children(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        root = tmp_path / "project-workspaces"
        settings = local_settings(root)
        first = LocalWorkspaceProvider(settings)
        stale = await first.create("allocation-stale")
        stale_path = Path(stale.root)

        marker_free = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        marker_free.mkdir()
        (marker_free / "keep").write_text("operator data", encoding="utf-8")
        malformed = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        malformed.mkdir()
        (malformed / ".contractor-workspace-owner").write_text("wrong", encoding="ascii")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep").write_text("outside", encoding="utf-8")
        linked = root / f"{LOCAL_DIRECTORY_PREFIX}{uuid.uuid4().hex}"
        linked.symlink_to(outside, target_is_directory=True)

        replacement = LocalWorkspaceProvider(settings)
        assert await replacement.probe()
        assert not stale_path.exists()
        assert (marker_free / "keep").read_text(encoding="utf-8") == "operator data"
        assert malformed.exists()
        assert linked.is_symlink()
        assert (outside / "keep").read_text(encoding="utf-8") == "outside"

    asyncio.run(scenario())


def test_local_provider_handles_are_private_and_cleanup_is_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "project-workspaces"
        provider = LocalWorkspaceProvider(local_settings(root))
        storage = await provider.create("allocation-1")
        path = Path(storage.root)
        assert path.parent == root
        assert str(root) not in repr(storage)
        (path / "data.txt").write_text("temporary", encoding="utf-8")

        outside = tmp_path / "outside"
        outside.mkdir()
        forged = ProjectWorkspaceStorage(
            storage="local",
            filesystem=storage.filesystem,
            root=str(outside),
            provider_id=storage.provider_id,
            owner_token=storage.owner_token,
        )
        with pytest.raises(ValueError, match="unowned"):
            await provider.cleanup(forged)
        assert outside.exists()

        await provider.cleanup(storage)
        assert not path.exists()
        await provider.cleanup(storage)

    asyncio.run(scenario())


def test_local_provider_rejects_root_replaced_by_symlink(tmp_path: Path) -> None:
    root = tmp_path / "project-workspaces"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.symlink_to(outside, target_is_directory=True)
    provider = LocalWorkspaceProvider(local_settings(root))
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(provider.probe())


def test_memory_provider_isolates_allocations_and_provider_instances() -> None:
    async def scenario() -> None:
        settings = memory_settings()
        first = MemoryWorkspaceProvider(settings)
        second = MemoryWorkspaceProvider(settings)
        left = await first.create("allocation-left")
        right = await first.create("allocation-right")
        foreign = await second.create("allocation-foreign")
        left.filesystem.pipe(f"{left.root}/value.txt", b"left")
        right.filesystem.pipe(f"{right.root}/value.txt", b"right")
        assert left.filesystem.cat(f"{left.root}/value.txt") == b"left"
        assert right.filesystem.cat(f"{right.root}/value.txt") == b"right"
        assert not foreign.filesystem.exists(f"{foreign.root}/value.txt")
        assert not foreign.filesystem.exists(f"{left.root}/value.txt")
        assert left.root != right.root != foreign.root
        assert left.root not in repr(left)

        with pytest.raises(ValueError, match="another provider"):
            await second.cleanup(left)
        await first.cleanup(left)
        assert not left.filesystem.exists(left.root)
        assert right.filesystem.exists(right.root)
        await first.cleanup(right)
        await second.cleanup(foreign)

    asyncio.run(scenario())


def local_settings(root: Path) -> WorkspaceSettings:
    return WorkspaceSettings(storage="local", work_root=root, limits=limits())


def memory_settings() -> WorkspaceSettings:
    return WorkspaceSettings(storage="memory", limits=limits())


def limits() -> WorkspaceLimits:
    return WorkspaceLimits(
        max_files=100,
        max_expanded_bytes=4096,
        max_managed_text_bytes=2048,
        max_file_bytes=1024,
    )
