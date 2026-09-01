"""Adversarial workspace names, archives, physical replacements and redaction."""

from __future__ import annotations

import asyncio
import stat
import unicodedata
import zipfile
from pathlib import Path

import pytest
from test_edit_files_toolset import hydrated_workspace
from test_projectfs_zip import archive, archive_infos, settings, workspace_inputs

from contractor_runtime.projectfs import (
    LocalWorkspaceProvider,
    WorkspacePreparationError,
    WorkspaceStorageError,
    hydrate_workspace,
)
from contractor_runtime.projectfs import storage as storage_module
from contractor_runtime.projectfs.paths import (
    MAX_PROJECT_PATH_BYTES,
    ProjectPathError,
    normalize_project_glob,
    normalize_project_path,
)
from contractor_runtime.settings import WorkspaceLimits


@pytest.mark.parametrize(
    "value",
    [
        "/absolute",
        "../outside",
        "safe/../../outside",
        "safe\\outside",
        "C:/outside",
        "file://outside",
        "safe//child",
        "safe/./child",
        "safe/\x00child",
        "safe/\x1fchild",
        "safe/\x7fchild",
        "a/" * 129 + "leaf",
        "x" * (MAX_PROJECT_PATH_BYTES + 1),
    ],
)
def test_paths_and_globs_reject_every_escape_shape(value: str) -> None:
    with pytest.raises(ProjectPathError):
        normalize_project_path(value, allow_root=False)
    with pytest.raises(ProjectPathError):
        normalize_project_glob(value)


def test_paths_normalize_unicode_but_archives_reject_normalized_aliases(tmp_path: Path) -> None:
    composed = "café/file.txt"
    decomposed = unicodedata.normalize("NFD", composed)
    assert normalize_project_path(decomposed) == composed

    async def scenario() -> None:
        spec, reader = workspace_inputs(
            [("source", "", archive({composed: b"one", decomposed: b"two"}))]
        )
        provider = LocalWorkspaceProvider(settings("local", tmp_path / "unicode"))
        with pytest.raises(WorkspacePreparationError) as rejected:
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="unicode-alias",
                timeout_seconds=5,
            )
        assert rejected.value.code == "workspace_source_invalid"
        assert list((tmp_path / "unicode").iterdir()) == []

    asyncio.run(scenario())


def test_archive_rejects_all_special_entries_and_implicit_parent_limit(tmp_path: Path) -> None:
    async def rejected(payload: bytes, *, max_files: int = 20) -> str:
        limits = WorkspaceLimits(
            max_files=max_files,
            max_expanded_bytes=1 << 20,
            max_managed_text_bytes=1 << 20,
            max_file_bytes=1 << 20,
        )
        root = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
        spec, reader = workspace_inputs([("source", "", payload)])
        provider = LocalWorkspaceProvider(settings("local", root, limits=limits))
        with pytest.raises(WorkspacePreparationError) as failure:
            await hydrate_workspace(
                provider=provider,
                spec=spec,
                artifact_reader=reader,
                allocation_id="adversarial",
                timeout_seconds=5,
            )
        assert list(root.iterdir()) == []
        return failure.value.code

    async def scenario() -> None:
        for file_type in (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK):
            info = zipfile.ZipInfo(f"special-{file_type}")
            info.external_attr = (file_type | 0o600) << 16
            assert await rejected(archive_infos([(info, b"payload")])) == "workspace_source_invalid"

        # One member creates three managed paths through implicit parents. The
        # advertised maxFiles bound applies to the complete tree, not merely
        # zip.infolist().
        assert (
            await rejected(archive({"one/two/file.txt": b"body"}), max_files=2)
            == "workspace_capacity_exceeded"
        )

    asyncio.run(scenario())


def test_direct_local_write_replaces_final_symlink_without_touching_outside(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "local", "direct", "replacement")
        outside = tmp_path / "outside.txt"
        outside.write_text("outside sentinel\n", encoding="utf-8")
        physical = Path(session.storage.root) / "run_workdir" / "lf.txt"
        physical.unlink()
        physical.symlink_to(outside)

        await session.write_text("lf.txt", "private replacement\n")

        assert outside.read_text(encoding="utf-8") == "outside sentinel\n"
        assert not physical.is_symlink()
        assert physical.read_text(encoding="utf-8") == "private replacement\n"
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_direct_local_parent_symlink_fails_closed_without_outside_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "local", "direct", "parent-swap")
        outside = tmp_path / "outside"
        outside.mkdir()
        physical_tree = Path(session.storage.root) / "run_workdir" / "tree"
        original_tree = physical_tree.with_name("tree-original")
        physical_tree.rename(original_tree)
        physical_tree.symlink_to(outside, target_is_directory=True)

        with pytest.raises(WorkspaceStorageError, match="unavailable"):
            await session.write_text("tree/child/file.txt", "must not escape\n")

        assert list(outside.iterdir()) == []
        assert await session.read_text("tree/child/file.txt") == "tree\n"
        physical_tree.unlink()
        original_tree.rename(physical_tree)
        await provider.cleanup(session.storage)

    asyncio.run(scenario())


def test_direct_local_failed_atomic_replace_removes_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        session, provider = await hydrated_workspace(tmp_path, "local", "direct", "replace-fail")
        workdir = Path(session.storage.root) / "run_workdir"

        def reject_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("seeded detail that must not escape")

        monkeypatch.setattr(storage_module.os, "replace", reject_replace)
        with pytest.raises(WorkspaceStorageError, match="workspace_unavailable") as failure:
            await session.write_text("lf.txt", "uncommitted secret\n")

        assert str(failure.value) == "workspace_unavailable"
        assert not list(workdir.rglob(".contractor-write-*"))
        assert (workdir / "lf.txt").read_text(encoding="utf-8").startswith("alpha\n")
        await provider.cleanup(session.storage)

    asyncio.run(scenario())
