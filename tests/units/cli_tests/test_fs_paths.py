from __future__ import annotations

from pathlib import Path

from cli.fs import RootedLocalFileSystem


def test_relative_path_is_rooted_at_sandbox_when_cwd_is_descendant(
    tmp_path: Path,
    monkeypatch,
):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "target.txt").write_text("root", encoding="utf-8")
    (nested / "target.txt").write_text("cwd", encoding="utf-8")
    monkeypatch.chdir(nested)

    fs = RootedLocalFileSystem(str(tmp_path))

    with fs.open("target.txt", "rb") as stream:
        assert stream.read() == b"root"
