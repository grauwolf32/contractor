"""Real-disk containment, mutation and bounded acquisition regressions."""

from __future__ import annotations

import os
import time
import unicodedata
from pathlib import Path

import pytest

from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.local_io import RootedLocalFilesystem
from contractor_runtime.settings import WorkspaceLimits


def disk(root: Path, **overrides: int) -> RootedLocalFilesystem:
    root.mkdir(exist_ok=True)
    limits = dict(
        max_files=100, max_file_bytes=1024, max_expanded_bytes=4096, max_managed_text_bytes=2048
    )
    limits.update(overrides)
    return RootedLocalFilesystem(root, WorkspaceLimits(**limits))


def deadline() -> float:
    return time.monotonic() + 5


def test_read_write_copy_move_remove_use_actual_disk(tmp_path: Path) -> None:
    fs = disk(tmp_path / "work")
    fs.mkdir("src/nested", parents=True, deadline=deadline())
    fs.write("src/code.py", b"first\r\n", deadline=deadline())
    path = tmp_path / "work/src/code.py"
    path.write_bytes(b"later\r\n")
    path.chmod(0o750)
    assert fs.read("src/code.py", deadline=deadline()) == b"later\r\n"
    fs.write("src/code.py", b"edited\r\n", deadline=deadline())
    assert path.stat().st_mode & 0o777 == 0o750
    fs.copy("src", "copy", recursive=True, deadline=deadline())
    fs.move("copy", "moved", deadline=deadline())
    tree = fs.scan(deadline=deadline())
    assert set(tree.texts) == {"src/code.py", "moved/code.py"}
    assert tree.entries["moved/nested"].directory
    fs.remove("moved", recursive=True, deadline=deadline())
    assert not (tmp_path / "work/moved").exists()
    assert path.read_bytes() == b"edited\r\n"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_unsupported_leaf_preflight_never_reads_or_mutates_outside(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside sentinel")
    leaf = root / "bad"
    if kind == "symlink":
        leaf.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, leaf)
    else:
        os.mkfifo(leaf)
    operations = [
        lambda: fs.stat("bad", deadline=deadline()),
        lambda: fs.read("bad", deadline=deadline()),
        lambda: fs.write("bad", b"new", deadline=deadline()),
        lambda: fs.scan(deadline=deadline()),
        lambda: fs.copy("bad", "copied", deadline=deadline()),
        lambda: fs.move("bad", "moved", deadline=deadline()),
        lambda: fs.remove("bad", deadline=deadline()),
    ]
    for operation in operations:
        with pytest.raises(WorkspaceStorageError):
            operation()
    assert outside.read_bytes() == b"outside sentinel"
    assert leaf.lstat()
    assert not (root / "copied").exists()


def test_root_and_parent_replacement_cannot_redirect_access(tmp_path: Path) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"sentinel")
    (root / "parent").symlink_to(outside, target_is_directory=True)
    for operation in [
        lambda: fs.read("parent/secret", deadline=deadline()),
        lambda: fs.write("parent/secret", b"bad", deadline=deadline()),
        lambda: fs.mkdir("parent/sub", deadline=deadline()),
        lambda: fs.remove("parent/secret", deadline=deadline()),
    ]:
        with pytest.raises(WorkspaceStorageError):
            operation()
    root.rename(tmp_path / "original")
    root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceStorageError):
        fs.read("secret", deadline=deadline())
    root.unlink()
    root.mkdir()
    with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
        fs.scan(deadline=deadline())
    assert (outside / "secret").read_bytes() == b"sentinel"


@pytest.mark.parametrize("operation", ["copy", "move", "remove"])
def test_recursive_type_validation_precedes_changes(tmp_path: Path, operation: str) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    (root / "src").mkdir()
    (root / "src/first").write_bytes(b"keep")
    (root / "src/bad").symlink_to(tmp_path / "outside")
    with pytest.raises(WorkspaceStorageError):
        if operation == "remove":
            fs.remove("src", recursive=True, deadline=deadline())
        elif operation == "copy":
            fs.copy("src", "dst", recursive=True, deadline=deadline())
        else:
            fs.move("src", "dst", deadline=deadline())
    assert (root / "src/first").read_bytes() == b"keep"
    assert not (root / "dst").exists()


def test_limits_deadline_names_and_binary_classification(tmp_path: Path) -> None:
    root = tmp_path / "work"
    fs = disk(root, max_files=2, max_file_bytes=5, max_managed_text_bytes=5)
    (root / "a").write_bytes(b"abc")
    (root / "b").write_bytes(b"\x00x")
    tree = fs.scan(deadline=deadline())
    assert tree.texts == {"a": "abc"}
    assert tree.binary_paths == {"b"}
    (root / "c").write_bytes(b"x")
    with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
        fs.scan(deadline=deadline())
    # Scoped reads still work when a complete acquisition exceeds its bound.
    assert fs.read("a", deadline=deadline()) == b"abc"
    (root / "c").unlink()
    (root / "b").write_bytes(b"def")
    with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
        fs.scan(deadline=deadline())
    (root / "a").write_bytes(b"too big")
    with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
        fs.read("a", deadline=deadline())
    with pytest.raises(WorkspaceStorageError, match="workspace_unavailable"):
        fs.scan(deadline=time.monotonic() - 1)
    (root / "a").unlink()
    (root / "b").unlink()
    (root / unicodedata.normalize("NFD", "café")).write_bytes(b"x")
    with pytest.raises(WorkspaceStorageError, match="workspace_path_invalid"):
        fs.scan(deadline=deadline())


def test_growing_or_replaced_file_fails_without_stale_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "work"
    fs = disk(root, max_file_bytes=5)
    path = root / "file"
    path.write_bytes(b"123")
    original = os.read

    def growing(descriptor: int, size: int) -> bytes:
        result = original(descriptor, size)
        with path.open("ab") as output:
            output.write(b"456")
        return result

    monkeypatch.setattr(os, "read", growing)
    with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
        fs.read("file", deadline=deadline())


def test_atomic_write_failure_cleans_temp_and_preserves_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    (root / "file").write_bytes(b"original")

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("secret host detail")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(WorkspaceStorageError) as failure:
        fs.write("file", b"new", deadline=deadline())
    assert str(failure.value) == "workspace_unavailable"
    assert (root / "file").read_bytes() == b"original"
    assert sorted(p.name for p in root.iterdir()) == ["file"]


def test_parent_swapped_between_preflight_and_open_is_not_followed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    (root / "parent").mkdir()
    (root / "parent/file").write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(b"sentinel")
    original = os.open
    swapped = False

    def swap(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "parent" and not swapped:
            swapped = True
            (root / "parent").rename(root / "old")
            (root / "parent").symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    with pytest.raises(WorkspaceStorageError):
        fs.write("parent/file", b"bad", deadline=deadline())
    assert (outside / "file").read_bytes() == b"sentinel"


@pytest.mark.parametrize("change", ["disappear", "replace", "edit"])
def test_scan_does_not_return_stale_contents_after_read_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    path = root / "file"
    path.write_bytes(b"initial")
    original = os.read
    changed = False

    def race(descriptor: int, size: int) -> bytes:
        nonlocal changed
        result = original(descriptor, size)
        if not changed:
            changed = True
            if change == "disappear":
                path.unlink()
            elif change == "replace":
                path.rename(root / "old")
                path.write_bytes(b"different")
            else:
                path.write_bytes(b"edited")
        return result

    monkeypatch.setattr(os, "read", race)
    with pytest.raises(WorkspaceStorageError) as failure:
        fs.scan(deadline=deadline())
    assert str(failure.value) in {"workspace_unavailable", "workspace_type_conflict"}


def test_scan_checks_aggregate_binary_bytes(tmp_path: Path) -> None:
    root = tmp_path / "work"
    fs = disk(root, max_expanded_bytes=5)
    (root / "a").write_bytes(b"\x00ab")
    (root / "b").write_bytes(b"\x00cd")
    for contents in (True, False):
        with pytest.raises(WorkspaceStorageError, match="workspace_limit_exceeded"):
            fs.scan(deadline=deadline(), contents=contents)


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "a/../outside", "a\\b"])
def test_invalid_paths_fail_before_mutation(tmp_path: Path, path: str) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    with pytest.raises(WorkspaceStorageError, match="workspace_path_invalid"):
        fs.write(path, b"new", deadline=deadline())
    assert list(root.iterdir()) == []


def test_mkdir_rechecks_leaf_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "work"
    fs = disk(root)
    original = os.stat
    calls = 0

    def race(path: object, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal calls
        if path == "new":
            calls += 1
            if calls == 2:
                (root / "new").write_bytes(b"external")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", race)
    with pytest.raises(WorkspaceStorageError, match="workspace_type_conflict"):
        fs.mkdir("new", deadline=deadline())
    assert (root / "new").read_bytes() == b"external"
