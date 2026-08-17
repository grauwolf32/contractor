"""Regression tests for base→visible type changes in build_overlay_patch.

Pre-fix, build_overlay_patch (the save() path) had no type-change branch that
render_overlay_diff (the diff() path) has:
  * dir → file raised RuntimeError("Type mismatch") — crashing artifact save.
  * file → dir silently emitted no op — the applied patch diverged from the diff.
"""
from __future__ import annotations

from collections.abc import Callable

from contractor.tools.fs.overlay_patch import build_overlay_patch


def _reader(mapping: dict[str, bytes]) -> Callable[[str], bytes]:
    def read(path: str) -> bytes:
        if path in mapping:
            return mapping[path]
        raise FileNotFoundError(path)

    return read


def _build(base_entries, visible_entries, base_bytes, eff_bytes, empty_dirs=()):
    return build_overlay_patch(
        base_entries=base_entries,
        visible_entries=visible_entries,
        root="/",
        root_marker="/",
        version=1,
        read_base_bytes=_reader(base_bytes),
        read_effective_bytes=_reader(eff_bytes),
        effective_empty_dir=lambda p: p in empty_dirs,
    )


def test_dir_to_file_does_not_crash_and_recreates():
    # /d was a directory (with /d/x); now /d is a file.
    patch = _build(
        base_entries={"/d": {"type": "directory"}, "/d/x": {"type": "file"}},
        visible_entries={"/d": {"type": "file"}},
        base_bytes={"/d/x": b"old"},
        eff_bytes={"/d": b"now a file"},
    )
    ops = patch["patches"]

    # The stale directory entry is deleted...
    assert {"op": "delete_path", "path": "/d", "type": "directory"} in ops
    # ...and /d is recreated as a file.
    writes = [o for o in ops if o["op"] == "write_file" and o["path"] == "/d"]
    assert len(writes) == 1
    # The orphaned child is deleted too.
    assert any(o["op"] == "delete_path" and o["path"] == "/d/x" for o in ops)
    # delete-before-create ordering so apply() never writes a file over a dir.
    del_idx = next(i for i, o in enumerate(ops) if o["path"] == "/d" and o["op"] == "delete_path")
    write_idx = next(i for i, o in enumerate(ops) if o["path"] == "/d" and o["op"] == "write_file")
    assert del_idx < write_idx


def test_file_to_dir_emits_delete_and_children():
    # /f was a file; now /f is a directory containing /f/inner.
    patch = _build(
        base_entries={"/f": {"type": "file"}},
        visible_entries={"/f": {"type": "directory"}, "/f/inner": {"type": "file"}},
        base_bytes={"/f": b"was file"},
        eff_bytes={"/f/inner": b"inner"},
    )
    ops = patch["patches"]

    # The stale base file is deleted (previously: silently dropped).
    deletes = [o for o in ops if o["op"] == "delete_path" and o["path"] == "/f"]
    assert len(deletes) == 1
    assert deletes[0]["type"] == "file"
    assert "base_hash" in deletes[0]  # guarded against concurrent base change
    # The new child is written.
    assert any(o["op"] == "write_file" and o["path"] == "/f/inner" for o in ops)


def test_file_to_empty_dir_creates_dir():
    # /f was a file; now /f is an empty directory.
    patch = _build(
        base_entries={"/f": {"type": "file"}},
        visible_entries={"/f": {"type": "directory"}},
        base_bytes={"/f": b"was file"},
        eff_bytes={},
        empty_dirs=("/f",),
    )
    ops = patch["patches"]
    assert any(o["op"] == "delete_path" and o["path"] == "/f" for o in ops)
    assert {"op": "create_dir", "path": "/f"} in ops


def test_same_type_modify_still_works():
    # Sanity: an ordinary in-place file edit is unaffected by the new branch.
    patch = _build(
        base_entries={"/a": {"type": "file"}},
        visible_entries={"/a": {"type": "file"}},
        base_bytes={"/a": b"old"},
        eff_bytes={"/a": b"new"},
    )
    ops = patch["patches"]
    writes = [o for o in ops if o["op"] == "write_file" and o["path"] == "/a"]
    assert len(writes) == 1
    assert "base_hash" in writes[0]
    assert not any(o["op"] == "delete_path" for o in ops)
