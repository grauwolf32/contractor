from __future__ import annotations

from pathlib import Path

import pytest

from contractor.tools.fs.models import FsEntry
from contractor.tools.fs.overlayfs import MemoryOverlayFileSystem
from contractor.tools.fs.rootfs import RootedLocalFileSystem


@pytest.fixture()
def base_tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested").mkdir()
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    (tmp_path / "src" / "a.py").write_text("print('base a')\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("print('base b')\n", encoding="utf-8")
    (tmp_path / "src" / "nested" / "deep.py").write_text(
        "print('deep')\n", encoding="utf-8"
    )
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
    assert (base_tree / "src" / "a.py").read_text(
        encoding="utf-8"
    ) == "print('base a')\n"


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

    assert (
        overlay_fs.read_text("/src/b.py") == "print('base b')\nprint('overlay tail')\n"
    )
    assert (base_tree / "src" / "b.py").read_text(
        encoding="utf-8"
    ) == "print('base b')\n"


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


def test_info_prefers_overlay_version_for_modified_file(
    overlay_fs: MemoryOverlayFileSystem,
):
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


def test_walk_includes_overlay_created_files_and_dirs(
    overlay_fs: MemoryOverlayFileSystem,
):
    overlay_fs.write_text(
        "/src/mem/sub/created.py", "print('created')\n", encoding="utf-8"
    )

    walked = {
        root: (set(dirs), set(files)) for root, dirs, files in overlay_fs.walk("/src")
    }

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
    assert (base_tree / "src" / "a.py").read_text(
        encoding="utf-8"
    ) == "print('base a')\n"


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


def test_expand_path_supports_recursive_overlay_view(
    overlay_fs: MemoryOverlayFileSystem,
):
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


def test_x_mode_fails_when_target_already_exists_in_base(
    overlay_fs: MemoryOverlayFileSystem,
):
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
    assert (base_tree / "src" / "a.py").read_text(
        encoding="utf-8"
    ) == "print('base a')\n"


def test_strip_protocol_handles_empty_overlay_and_relative_paths(
    overlay_fs: MemoryOverlayFileSystem,
):
    assert overlay_fs._strip_protocol("") == "/"
    assert overlay_fs._strip_protocol("overlay://src/a.py") == "/src/a.py"
    assert overlay_fs._strip_protocol("src/a.py") == "/src/a.py"


def test_parent_of_root_is_root(overlay_fs: MemoryOverlayFileSystem):
    assert overlay_fs._parent("/") == "/"


def test_snapshot_and_restore_snapshot_roundtrip(
    overlay_fs: MemoryOverlayFileSystem,
    base_tree: Path,
):
    overlay_fs.write_text("/src/generated.py", "print('generated')\n", encoding="utf-8")
    overlay_fs.rm("/src/a.py")

    snapshot = overlay_fs.snapshot()

    overlay_fs.write_text("/src/generated.py", "print('changed')\n", encoding="utf-8")
    overlay_fs.write_text("/src/another.py", "print('another')\n", encoding="utf-8")
    overlay_fs.write_text("/src/a.py", "print('reborn')\n", encoding="utf-8")

    overlay_fs.restore_snapshot(snapshot)

    assert overlay_fs.read_text("/src/generated.py") == "print('generated')\n"
    assert not overlay_fs.exists("/src/another.py")
    assert not overlay_fs.exists("/src/a.py")
    assert (base_tree / "src" / "a.py").read_text(
        encoding="utf-8"
    ) == "print('base a')\n"


def test_restore_snapshot_without_snapshot_raises(
    overlay_fs: MemoryOverlayFileSystem,
):
    with pytest.raises(ValueError, match="No snapshot to restore"):
        overlay_fs.restore_snapshot()


def test_restore_snapshot_uses_internal_snapshot_when_no_arg_given(
    overlay_fs: MemoryOverlayFileSystem,
):
    overlay_fs.write_text("/src/keep.py", "v1\n", encoding="utf-8")
    overlay_fs.snapshot()

    overlay_fs.write_text("/src/keep.py", "v2\n", encoding="utf-8")
    overlay_fs.restore_snapshot()

    assert overlay_fs.read_text("/src/keep.py") == "v1\n"


def test_getattr_delegates_to_base_fs(
    overlay_fs: MemoryOverlayFileSystem,
    base_fs: RootedLocalFileSystem,
):
    assert overlay_fs.sep == base_fs.sep


def test_open_binary_write_and_read_roundtrip(overlay_fs: MemoryOverlayFileSystem):
    with overlay_fs.open("/src/blob.bin", "wb") as f:
        f.write(b"\x00\x01abc")

    with overlay_fs.open("/src/blob.bin", "rb") as f:
        assert f.read() == b"\x00\x01abc"


def test_open_x_creates_new_overlay_file(overlay_fs: MemoryOverlayFileSystem):
    with overlay_fs.open("/src/brand_new.py", "x", encoding="utf-8") as f:
        f.write("print('new')\n")

    assert overlay_fs.read_text("/src/brand_new.py") == "print('new')\n"


def test_open_append_creates_file_when_missing(overlay_fs: MemoryOverlayFileSystem):
    with overlay_fs.open("/src/appended.txt", "a", encoding="utf-8") as f:
        f.write("hello\n")

    assert overlay_fs.read_text("/src/appended.txt") == "hello\n"


def test_flush_persists_written_content_before_context_exit(
    overlay_fs: MemoryOverlayFileSystem,
):
    f = overlay_fs.open("/src/flushed.txt", "w", encoding="utf-8")
    f.write("hello\n")
    f.flush()

    assert overlay_fs.read_text("/src/flushed.txt") == "hello\n"
    f.close()


def test_reset_overlay_discards_overlay_changes_and_reveals_base(
    overlay_fs: MemoryOverlayFileSystem,
):
    overlay_fs.write_text("/src/a.py", "print('overlay')\n", encoding="utf-8")
    overlay_fs.write_text("/src/new.py", "print('new')\n", encoding="utf-8")
    overlay_fs.rm("/src/b.py")

    overlay_fs.reset_overlay()

    assert overlay_fs.read_text("/src/a.py") == "print('base a')\n"
    assert overlay_fs.exists("/src/b.py")
    assert not overlay_fs.exists("/src/new.py")


def test_writing_a_file_under_an_existing_file_raises(
    overlay_fs: MemoryOverlayFileSystem,
):
    overlay_fs.write_text("/leaf.txt", "leaf", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        overlay_fs.write_text("/leaf.txt/inner.txt", "x", encoding="utf-8")


def test_writing_a_file_over_an_existing_directory_raises(
    overlay_fs: MemoryOverlayFileSystem,
):
    with pytest.raises(IsADirectoryError):
        overlay_fs.write_text("/src", "x", encoding="utf-8")


def test_mkdir_over_existing_file_raises(overlay_fs: MemoryOverlayFileSystem):
    with pytest.raises(FileExistsError):
        overlay_fs.mkdir("/src/a.py")


def test_writing_a_file_under_a_base_file_raises(overlay_fs: MemoryOverlayFileSystem):
    with pytest.raises(NotADirectoryError):
        overlay_fs.write_text("/README.md/inner", "x", encoding="utf-8")


def test_changed_paths_classifies_overlay_state(
    overlay_fs: MemoryOverlayFileSystem,
):
    overlay_fs.write_text("/src/a.py", "print('overlay')\n", encoding="utf-8")
    overlay_fs.write_text("/src/new.py", "print('new')\n", encoding="utf-8")
    overlay_fs.rm("/src/b.py")

    status = overlay_fs.changed_paths()

    assert status["added"] == ["/src/new.py"]
    assert status["modified"] == ["/src/a.py"]
    assert status["deleted"] == ["/src/b.py"]


def test_changed_paths_skips_no_op_writes(overlay_fs: MemoryOverlayFileSystem):
    base_content = overlay_fs.read_text("/src/a.py")
    overlay_fs.write_text("/src/a.py", base_content, encoding="utf-8")

    status = overlay_fs.changed_paths()

    assert status == {"added": [], "modified": [], "deleted": []}


def test_filetype_cache_is_invalidated_on_overlay_write(
    overlay_fs: MemoryOverlayFileSystem,
):
    path = "/src/typed.py"
    python_body = (
        "def hello():\n    return 'world'\n\n"
        "import sys\nfor i in range(10):\n    print(i)\n"
    ) * 10
    overlay_fs.write_text(path, python_body, encoding="utf-8")

    first = FsEntry.from_path(path, overlay_fs, with_types=True)
    assert first is not None and first.filetype is not None
    initial_label = first.filetype.label

    html_body = (
        "<!DOCTYPE html>\n<html><head><title>x</title></head>\n<body>\n"
        "<h1>not python anymore</h1>\n<p>just html now</p>\n</body></html>\n"
    ) * 10
    overlay_fs.write_text(path, html_body, encoding="utf-8")

    second = FsEntry.from_path(path, overlay_fs, with_types=True)
    assert second is not None and second.filetype is not None
    # If the cache survived, second would still report the original label;
    # the invalidation hook must drop the entry on write.
    assert second.filetype.label != initial_label
