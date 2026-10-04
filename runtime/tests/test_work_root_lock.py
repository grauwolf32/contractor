from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from contractor_runtime.projectfs import LocalWorkspaceProvider
from contractor_runtime.projectfs.provider import cleanup_stale_local_workspaces
from contractor_runtime.sandbox.contracts import SandboxErrorCode
from contractor_runtime.sandbox.podman.ownership import (
    OwnerLockHeldError,
    OwnerLockUnsafeError,
    ServiceOwnerLock,
)
from contractor_runtime.settings import LOCAL_WORKSPACE_DEFAULTS, WorkspaceLimits, WorkspaceSettings
from contractor_runtime.work_root_lock import WorkRootLockError, hold_work_roots
from contractor_runtime.workspace import LocalWorkdirFactory, cleanup_orphan_workdirs


def _workspace_settings(root: Path) -> WorkspaceSettings:
    return WorkspaceSettings(
        storage="local", work_root=root, limits=WorkspaceLimits(*LOCAL_WORKSPACE_DEFAULTS)
    )


@pytest.mark.parametrize("shared_root", ["scratch", "project"])
def test_second_runtime_process_cannot_clean_another_runtime_root(
    tmp_path: Path, shared_root: str
) -> None:
    scratch = tmp_path / "scratch"
    project = tmp_path / "project"
    allocation = asyncio.run(LocalWorkdirFactory(scratch).prepare())
    workspace = asyncio.run(LocalWorkspaceProvider(_workspace_settings(project)).create("live"))
    allocation_marker = scratch / f"{allocation.path.name}.contractor-owner"
    workspace_marker = Path(workspace.root) / ".contractor-workspace-owner"
    assert allocation.path.is_dir() and allocation_marker.is_file()
    assert Path(workspace.root).is_dir() and workspace_marker.is_file()

    contender_scratch = scratch if shared_root == "scratch" else tmp_path / "other-scratch"
    contender_project = project if shared_root == "project" else tmp_path / "other-project"
    program = """
import asyncio
import sys
from pathlib import Path
from contractor_runtime.cli import serve
from contractor_runtime.settings import Settings, WorkspaceLimits, WorkspaceSettings
scratch, project = map(Path, sys.argv[1:])
settings = Settings(
    control_plane_url='https://localhost:8443',
    advertised_control_url='https://localhost:9443',
    advertised_a2a_url='https://localhost:9444',
    ca_file=Path('unused'), certificate_file=Path('unused'), private_key_file=Path('unused'),
    work_root=scratch,
    workspace=WorkspaceSettings(
        storage='local', work_root=project, limits=WorkspaceLimits(10, 1024, 1024, 1024)
    ),
)
try:
    asyncio.run(serve(settings, install_signal_handlers=False))
except RuntimeError as error:
    if 'work root in use' in str(error):
        sys.exit(17)
    raise
"""
    with hold_work_roots(scratch, project):
        result = subprocess.run(
            [sys.executable, "-c", program, str(contender_scratch), str(contender_project)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 17, result.stderr
        assert allocation.path.is_dir() and allocation_marker.is_file()
        assert Path(workspace.root).is_dir() and workspace_marker.is_file()

    with hold_work_roots(scratch, project):
        cleanup_orphan_workdirs(scratch)
        cleanup_stale_local_workspaces(project)
    assert not allocation.path.exists() and not allocation_marker.exists()
    assert not Path(workspace.root).exists()


def test_runtime_work_root_lock_released_after_process_crash(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    project = tmp_path / "project"
    program = """
import sys
from pathlib import Path
from contractor_runtime.work_root_lock import hold_work_roots
with hold_work_roots(Path(sys.argv[1]), Path(sys.argv[2])):
    print('ready', flush=True)
    sys.stdin.buffer.read()
"""
    holder = subprocess.Popen(
        [sys.executable, "-c", program, str(scratch), str(project)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "ready"
        with (
            pytest.raises(RuntimeError, match="scratch work root in use"),
            hold_work_roots(scratch, project),
        ):
            pass
    finally:
        holder.kill()
        holder.communicate(timeout=10)
    with hold_work_roots(scratch, project):
        pass


def test_work_root_errors_tell_contention_from_an_unsafe_root_mode(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    project = tmp_path / "project"
    with (
        hold_work_roots(scratch, None),
        pytest.raises(WorkRootLockError) as contention,
        hold_work_roots(scratch, None),
    ):
        pass
    assert str(contention.value) == f"scratch work root in use by another process: {scratch}"

    # An operator-created root with a default umask is not private.
    project.mkdir()
    project.chmod(0o755)
    with pytest.raises(WorkRootLockError) as unsafe, hold_work_roots(scratch, project):
        pass
    uid = os.geteuid()
    assert str(unsafe.value) == (
        f"project workspace work root unsafe: {project}: directory must be owned by "
        f"uid {uid} with mode 0700, found uid {uid} mode 0755"
    )
    project.chmod(0o700)
    with hold_work_roots(scratch, project):
        pass


@pytest.mark.parametrize("fault", ["mode", "hard-link", "symlink", "directory"])
def test_unsafe_lock_file_is_reported_as_unsafe_not_in_use(tmp_path: Path, fault: str) -> None:
    root = tmp_path / "scratch"
    root.mkdir()
    root.chmod(0o700)
    lock = root / "contractor-runtime.lock"
    if fault == "directory":
        lock.mkdir()
    elif fault == "symlink":
        (tmp_path / "target").touch(mode=0o600)
        lock.symlink_to(tmp_path / "target")
    else:
        lock.touch(mode=0o600)
        if fault == "mode":
            lock.chmod(0o644)
        else:
            os.link(lock, tmp_path / "second-link")
    with (
        pytest.raises(
            WorkRootLockError, match=rf"^scratch work root unsafe: {re.escape(str(root))}: "
        ),
        hold_work_roots(root, None),
    ):
        pass


def test_owner_lock_errors_keep_the_sandbox_unavailable_code(tmp_path: Path) -> None:
    directory = tmp_path / "owners"
    first = ServiceOwnerLock(directory, "service")
    first.acquire()
    try:
        with pytest.raises(OwnerLockHeldError) as held:
            ServiceOwnerLock(directory, "service").acquire()
    finally:
        first.release()
    directory.chmod(0o750)
    with pytest.raises(OwnerLockUnsafeError) as unsafe:
        ServiceOwnerLock(directory, "service").acquire()
    # Podman engine callers keep treating both as an unavailable sandbox.
    assert held.value.code is unsafe.value.code is SandboxErrorCode.UNAVAILABLE
