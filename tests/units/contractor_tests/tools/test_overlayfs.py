from __future__ import annotations

from pathlib import Path

import pytest

from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem
from contractor.tools.fs.rootfs import RootedLocalFileSystem


@pytest.fixture()
def base_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("print('base a')\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("print('base b')\n", encoding="utf-8")
    (tmp_path / "src" / "nested" / "deep.py").write_text("print('deep')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def base_fs(base_tree: Path) -> RootedLocalFileSystem:
    return RootedLocalFileSystem(str(base_tree))


@pytest.fixture()
def overlay_fs(base_fs: RootedLocalFileSystem) -> MemoryOverlayFileSystem:
    return MemoryOverlayFileSystem(base_fs)


def test_reads_fall_back_to_base_fs(overlay_fs: MemoryOverlayFileSystem):
    assert overlay_fs.read_text("/src/a.py") == "print('base a')\n"


def test_write_existing_file_goes_only_to_overlay(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.write_text("/src/a.py", "print('overlay a')\n", encoding="utf-8")

    assert overlay_fs.read_text("/src/a.py") == "print('overlay a')\n"
    assert (base_tree / "src" / "a.py").read_text(encoding="utf-8") == "print('base a')\n"


def test_can_create_new_file_only_in_memory(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.write_text("/src/new_file.py", "print('new')\n", encoding="utf-8")

    assert overlay_fs.exists("/src/new_file.py")
    assert overlay_fs.read_text("/src/new_file.py") == "print('new')\n"
    assert not (base_tree / "src" / "new_file.py").exists()


def test_open_write_and_read_text_roundtrip(overlay_fs: MemoryOverlayFileSystem):
    with overlay_fs.open("/src/generated.py", "w", encoding="utf-8") as f:
        f.write("x = 1\n")

    with overlay_fs.open("/src/generated.py", "r", encoding="utf-8") as f:
        assert f.read() == "x = 1\n"


def test_open_append_existing_base_file_without_touching_base(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    with overlay_fs.open("/src/b.py", "a", encoding="utf-8") as f:
        f.write("print('overlay tail')\n")

    assert overlay_fs.read_text("/src/b.py") == "print('base b')\nprint('overlay tail')\n"
    assert (base_tree / "src" / "b.py").read_text(encoding="utf-8") == "print('base b')\n"


def test_pipe_file_writes_into_overlay_only(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.pipe_file("/src/piped.txt", "hello from pipe")

    assert overlay_fs.read_text("/src/piped.txt") == "hello from pipe"
    assert not (base_tree / "src" / "piped.txt").exists()


def test_touch_creates_empty_overlay_file_only(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.touch("/src/empty.txt")

    assert overlay_fs.exists("/src/empty.txt")
    assert overlay_fs.read_bytes("/src/empty.txt") == b""
    assert not (base_tree / "src" / "empty.txt").exists()


def test_info_prefers_overlay_version_for_modified_file(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/a.py", "12345\n", encoding="utf-8")

    info = overlay_fs.info("/src/a.py")

    assert info["type"] == "file"
    assert info["size"] == len(b"12345\n")


def test_ls_merges_base_and_overlay_entries(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/new_overlay.py", "print('x')\n", encoding="utf-8")

    entries = overlay_fs.ls("/src", detail=True)
    names = {item["name"] for item in entries}

    assert "/src/a.py" in names
    assert "/src/b.py" in names
    assert "/src/nested" in names
    assert "/src/new_overlay.py" in names


def test_walk_includes_overlay_created_files_and_dirs(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/mem/sub/created.py", "print('created')\n", encoding="utf-8")

    walked = {root: (set(dirs), set(files)) for root, dirs, files in overlay_fs.walk("/src")}

    assert "/src" in walked
    assert "mem" in walked["/src"][0]
    assert "/src/mem" in walked
    assert "sub" in walked["/src/mem"][0]
    assert "/src/mem/sub" in walked
    assert "created.py" in walked["/src/mem/sub"][1]


def test_find_includes_base_and_overlay_files(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/mem/new.py", "print('new')\n", encoding="utf-8")

    found = set(overlay_fs.find("/src", withdirs=False, detail=False))

    assert "/src/a.py" in found
    assert "/src/b.py" in found
    assert "/src/nested/deep.py" in found
    assert "/src/mem/new.py" in found


def test_glob_matches_overlay_and_base_files(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/overlay_extra.py", "print('extra')\n", encoding="utf-8")

    matched = set(overlay_fs.glob("/src/*.py"))

    assert "/src/a.py" in matched
    assert "/src/b.py" in matched
    assert "/src/overlay_extra.py" in matched
    assert "/src/nested/deep.py" not in matched


def test_cp_file_copies_into_overlay_only(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.cp_file("/src/a.py", "/src/copied.py")

    assert overlay_fs.read_text("/src/copied.py") == "print('base a')\n"
    assert not (base_tree / "src" / "copied.py").exists()


def test_mv_moves_overlay_file_without_touching_base(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.write_text("/src/tmp.py", "print('tmp')\n", encoding="utf-8")

    overlay_fs.mv("/src/tmp.py", "/src/moved.py")

    assert not overlay_fs.exists("/src/tmp.py")
    assert overlay_fs.read_text("/src/moved.py") == "print('tmp')\n"
    assert not (base_tree / "src" / "tmp.py").exists()
    assert not (base_tree / "src" / "moved.py").exists()


def test_rm_removes_overlay_only_file(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/to_remove.py", "print('bye')\n", encoding="utf-8")
    assert overlay_fs.exists("/src/to_remove.py")

    overlay_fs.rm("/src/to_remove.py")

    assert not overlay_fs.exists("/src/to_remove.py")


def test_rm_hides_base_file_without_deleting_from_base(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    assert overlay_fs.exists("/src/a.py")

    overlay_fs.rm("/src/a.py")

    assert not overlay_fs.exists("/src/a.py")
    assert (base_tree / "src" / "a.py").exists()
    assert (base_tree / "src" / "a.py").read_text(encoding="utf-8") == "print('base a')\n"


def test_recursive_rm_hides_base_directory_without_touching_base(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.rm("/src/nested", recursive=True)

    assert not overlay_fs.exists("/src/nested")
    assert not overlay_fs.exists("/src/nested/deep.py")
    assert (base_tree / "src" / "nested").exists()
    assert (base_tree / "src" / "nested" / "deep.py").exists()


def test_mkdir_and_makedirs_create_virtual_overlay_dirs_only(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.makedirs("/virtual/path/inside", exist_ok=True)

    assert overlay_fs.isdir("/virtual")
    assert overlay_fs.isdir("/virtual/path")
    assert overlay_fs.isdir("/virtual/path/inside")
    assert not (base_tree / "virtual").exists()


def test_expand_path_supports_recursive_overlay_view(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/generated/out.py", "print('out')\n", encoding="utf-8")

    expanded = set(overlay_fs.expand_path("/src/generated", recursive=True))

    assert "/src/generated" in expanded
    assert "/src/generated/out.py" in expanded


def test_du_counts_overlay_sizes(overlay_fs: MemoryOverlayFileSystem):
    overlay_fs.write_text("/src/size1.txt", "abc", encoding="utf-8")
    overlay_fs.write_text("/src/size2.txt", "hello", encoding="utf-8")

    sizes = overlay_fs.du("/src", total=False, withdirs=False)

    assert sizes["/src/size1.txt"] == 3
    assert sizes["/src/size2.txt"] == 5


def test_x_mode_fails_when_target_already_exists_in_base(overlay_fs: MemoryOverlayFileSystem):
    with pytest.raises(FileExistsError):
        with overlay_fs.open("/src/a.py", "x", encoding="utf-8") as f:
            f.write("should fail\n")


def test_removed_base_file_can_be_recreated_in_overlay(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.rm("/src/a.py")
    assert not overlay_fs.exists("/src/a.py")

    overlay_fs.write_text("/src/a.py", "print('reborn')\n", encoding="utf-8")

    assert overlay_fs.exists("/src/a.py")
    assert overlay_fs.read_text("/src/a.py") == "print('reborn')\n"
    assert (base_tree / "src" / "a.py").read_text(encoding="utf-8") == "print('base a')\n"