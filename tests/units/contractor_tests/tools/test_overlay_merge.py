"""Unit tests for ``contractor.tools.fs.merge``."""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.fs import RootedLocalFileSystem
from contractor.tools.fs.merge import fork_overlay, merge_overlay_forks
from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem


@pytest.fixture()
def base_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("def b(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "shared.py").write_text("def shared(): pass\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def base_fs(base_tree: Path) -> RootedLocalFileSystem:
    return RootedLocalFileSystem(str(base_tree))


@pytest.fixture()
def overlay(base_fs: RootedLocalFileSystem) -> MemoryOverlayFileSystem:
    return MemoryOverlayFileSystem(base_fs)


# ── fork_overlay ──────────────────────────────────────────────────────


class TestForkOverlay:
    def test_empty_patch(self, base_fs):
        fork = fork_overlay(base_fs, None)
        assert fork.read_text("/src/a.py") == "def a(): pass\n"
        assert dict(fork._files) == {}

    def test_fork_from_patch_inherits_writes(self, base_fs, overlay):
        overlay.write_text("/src/a.py", "# @trace\ndef a(): pass\n")
        patch = overlay.save()
        fork = fork_overlay(base_fs, patch)
        assert fork.read_text("/src/a.py") == "# @trace\ndef a(): pass\n"

    def test_forks_are_independent(self, base_fs, overlay):
        patch = overlay.save()
        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)
        fork1.write_text("/src/a.py", "fork1")
        fork2.write_text("/src/a.py", "fork2")
        assert fork1.read_text("/src/a.py") == "fork1"
        assert fork2.read_text("/src/a.py") == "fork2"


# ── merge_overlay_forks ──────────────────────────────────────────────


class TestMergeOverlayForks:
    def test_disjoint_writes_merge_cleanly(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)

        fork1.write_text("/src/a.py", "# @trace\ndef a(): pass\n")
        fork2.write_text("/src/b.py", "# @trace\ndef b(): pass\n")

        conflicts = merge_overlay_forks(overlay, [fork1, fork2], pre)

        assert conflicts == []
        assert overlay.read_text("/src/a.py") == "# @trace\ndef a(): pass\n"
        assert overlay.read_text("/src/b.py") == "# @trace\ndef b(): pass\n"

    def test_conflict_takes_largest(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)

        fork1.write_text("/src/shared.py", "# @trace a\ndef shared(): pass\n")
        fork2.write_text(
            "/src/shared.py",
            "# @trace a\n# @trace b\ndef shared(): pass\n",
        )

        conflicts = merge_overlay_forks(overlay, [fork1, fork2], pre)

        assert conflicts == ["/src/shared.py"]
        assert "# @trace b" in overlay.read_text("/src/shared.py")

    def test_identical_writes_not_counted_as_conflict(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)

        same = "# @trace\ndef shared(): pass\n"
        fork1.write_text("/src/shared.py", same)
        fork2.write_text("/src/shared.py", same)

        conflicts = merge_overlay_forks(overlay, [fork1, fork2], pre)
        assert conflicts == []
        assert overlay.read_text("/src/shared.py") == same

    def test_empty_forks(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()
        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)

        conflicts = merge_overlay_forks(overlay, [fork1, fork2], pre)
        assert conflicts == []
        assert dict(overlay._files) == {}

    def test_pre_fork_files_not_duplicated(self, base_fs, overlay):
        overlay.write_text("/src/a.py", "pre-fork content")
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork2 = fork_overlay(base_fs, patch)
        fork1.write_text("/src/b.py", "new from fork1")

        conflicts = merge_overlay_forks(overlay, [fork1, fork2], pre)
        assert conflicts == []
        assert overlay.read_text("/src/a.py") == "pre-fork content"
        assert overlay.read_text("/src/b.py") == "new from fork1"

    def test_new_file_creation(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork1.write_text("/src/new_file.py", "brand new\n")

        conflicts = merge_overlay_forks(overlay, [fork1], pre)
        assert conflicts == []
        assert overlay.read_text("/src/new_file.py") == "brand new\n"

    def test_directory_merge(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()

        fork1 = fork_overlay(base_fs, patch)
        fork1.makedirs("/src/new_dir", exist_ok=True)
        fork1.write_text("/src/new_dir/f.py", "in new dir\n")

        conflicts = merge_overlay_forks(overlay, [fork1], pre)
        assert conflicts == []
        assert overlay.isdir("/src/new_dir")
        assert overlay.read_text("/src/new_dir/f.py") == "in new dir\n"

    def test_many_forks(self, base_fs, overlay):
        pre = dict(overlay._files)
        patch = overlay.save()
        forks = [fork_overlay(base_fs, patch) for _ in range(5)]

        for i, fork in enumerate(forks):
            fork.write_text(f"/src/op{i}.py", f"# operation {i}\n")

        conflicts = merge_overlay_forks(overlay, forks, pre)
        assert conflicts == []
        for i in range(5):
            assert overlay.read_text(f"/src/op{i}.py") == f"# operation {i}\n"
