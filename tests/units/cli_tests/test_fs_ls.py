"""Regression tests for RootedLocalFileSystem.ls with symlinks.

Pre-fix, ls() delegated to fsspec's LocalFileSystem.ls, whose info() called
os.readlink() on a path that _strip_protocol had already symlink-resolved to a
non-link target — raising OSError (EINVAL) and crashing the *entire* listing for
any directory that contained a symlink (node_modules, vendored deps, ...).

Policy: symlinks are never followed and are hidden from listings, matching
walk()/glob().
"""

import os

import pytest

from cli.fs import RootedLocalFileSystem


@pytest.fixture
def fs_with_symlinks(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "sub"))
    with open(os.path.join(root, "real.txt"), "w") as f:
        f.write("hi")
    with open(os.path.join(root, "sub", "inner.txt"), "w") as f:
        f.write("inner")
    # An in-sandbox symlink to a file and one to a directory — both previously
    # crashed ls().
    os.symlink(os.path.join(root, "real.txt"), os.path.join(root, "link.txt"))
    os.symlink(os.path.join(root, "sub"), os.path.join(root, "linkdir"))
    # A symlink escaping the sandbox must never be exposed either.
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    os.symlink(str(outside), os.path.join(root, "escape.txt"))
    return RootedLocalFileSystem(root)


def test_ls_does_not_crash_on_symlinked_entries(fs_with_symlinks):
    # Pre-fix: OSError [Errno 22]. Now: returns cleanly.
    names = fs_with_symlinks.ls("/", detail=False)
    assert "/real.txt" in names
    assert "/sub" in names


def test_ls_hides_symlinks(fs_with_symlinks):
    names = set(fs_with_symlinks.ls("/", detail=False))
    # All three symlinks (in-sandbox file, in-sandbox dir, escaping) are hidden.
    assert "/link.txt" not in names
    assert "/linkdir" not in names
    assert "/escape.txt" not in names
    # Only the real entries remain.
    assert names == {"/real.txt", "/sub"}


def test_ls_detail_returns_info_for_real_entries(fs_with_symlinks):
    detail = fs_with_symlinks.ls("/", detail=True)
    by_name = {e["name"]: e for e in detail}
    assert "/real.txt" in by_name
    assert by_name["/real.txt"]["type"] == "file"
    assert by_name["/sub"]["type"] == "directory"
    # Names are virtual (rooted at "/"), never host paths.
    assert all(e["name"].startswith("/") for e in detail)


def test_ls_on_subdir_still_works(fs_with_symlinks):
    assert fs_with_symlinks.ls("/sub", detail=False) == ["/sub/inner.txt"]


def test_ls_missing_path_returns_empty(fs_with_symlinks):
    # A genuinely absent path is an empty listing, not an error.
    assert fs_with_symlinks.ls("/does-not-exist", detail=False) == []


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses directory read permissions"
)
def test_ls_unreadable_dir_raises_sandbox_clean_error(tmp_path):
    # An existing-but-unreadable directory must surface as a tool error, not a
    # silently-empty (looks-like-success) listing that the agent reads as "empty".
    root = str(tmp_path)
    locked = os.path.join(root, "locked")
    os.makedirs(locked)
    fs = RootedLocalFileSystem(root)
    os.chmod(locked, 0o000)
    try:
        with pytest.raises(OSError) as excinfo:
            fs.ls("/locked", detail=False)
    finally:
        os.chmod(locked, 0o755)  # restore so tmp_path teardown can clean up
    msg = str(excinfo.value)
    assert "/locked" in msg
    assert root not in msg  # the host path never leaks to the LLM-facing error
