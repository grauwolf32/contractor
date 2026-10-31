from __future__ import annotations

import time

import pytest

from contractor_runtime.toolsets.scan.nuclei_package import PackageError, snapshot_package


@pytest.mark.parametrize("metadata", ["directory", "file", "symlink", "dangling_symlink"])
def test_root_git_metadata_is_neither_hashed_nor_copied(tmp_path, metadata):
    source = tmp_path / "installed"
    source.mkdir()
    (source / "template.yaml").write_text("id: fixture\n")
    (source / ".nuclei-ignore").write_text("ignored-template.yaml\n")
    reference = snapshot_package(source, tmp_path / "archive", time.monotonic() + 10)
    git = source / ".git"
    if metadata == "directory":
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n")
        (git / "hidden.yaml").write_text("not a scanner template")
        (git / "outside").symlink_to(tmp_path / "absent")
    elif metadata == "file":
        git.write_text("gitdir: /outside/worktree/metadata\n")
    else:
        outside = tmp_path / "external-metadata"
        if metadata == "symlink":
            outside.mkdir()
            (outside / "HEAD").write_text("external repository")
        git.symlink_to(outside, target_is_directory=True)

    snapshot = snapshot_package(source, tmp_path / "private", time.monotonic() + 10)
    assert snapshot.digest == reference.digest
    assert snapshot.files == reference.files
    assert not (snapshot.root / ".git").exists()
    assert not (snapshot.root / ".git").is_symlink()
    assert (snapshot.root / ".nuclei-ignore").read_text() == "ignored-template.yaml\n"

    if metadata == "directory":
        (git / "HEAD").write_text("a different checkout\n")
    elif metadata == "file":
        git.write_text("gitdir: /another/worktree/metadata\n")
    repeated = snapshot_package(source, tmp_path / "repeated", time.monotonic() + 10)
    assert repeated.digest == reference.digest


def test_nested_git_named_content_still_participates_in_the_pin(tmp_path):
    source = tmp_path / "installed"
    (source / "helpers").mkdir(parents=True)
    (source / "template.yaml").write_text("id: fixture\n")
    helper = source / "helpers/.git"
    helper.write_text("first resource")
    first = snapshot_package(source, tmp_path / "first", time.monotonic() + 10)
    helper.write_text("changed resource")
    second = snapshot_package(source, tmp_path / "second", time.monotonic() + 10)
    assert first.digest != second.digest
    assert (second.root / "helpers/.git").read_text() == "changed resource"


def test_git_metadata_alone_is_not_a_package(tmp_path):
    source = tmp_path / "installed"
    (source / ".git").mkdir(parents=True)
    (source / ".git/HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(PackageError, match=r"^nuclei_templates_unavailable$"):
        snapshot_package(source, tmp_path / "private", time.monotonic() + 10)
