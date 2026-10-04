"""Exclusive ownership of local Runtime roots before startup cleanup."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path


class WorkRootLockError(RuntimeError):
    """A Runtime root cannot be safely owned by this process."""


def validate_distinct_work_roots(scratch: Path, project: Path | None) -> None:
    """Do not allow either cleanup walker to reach the other's root."""

    if project is None:
        return
    scratch = scratch.expanduser().resolve()
    project = project.expanduser().resolve()
    if scratch == project or scratch.is_relative_to(project) or project.is_relative_to(scratch):
        raise ValueError("--work-root and --workspace-work-root must be separate, non-nested roots")


@contextmanager
def hold_work_roots(scratch: Path, project: Path | None) -> Iterator[None]:
    """Hold persistent, non-blocking locks until all Runtime work has stopped.

    The existing owner lock checks a regular, single-link, user-owned 0600
    file beneath a private 0700 directory and never removes the lock inode.
    """

    # Import after settings has initialized: sandbox contracts load projectfs,
    # whose provider imports settings while defining the local workspace type.
    from contractor_runtime.sandbox.contracts import SandboxContractError
    from contractor_runtime.sandbox.podman.ownership import ServiceOwnerLock

    validate_distinct_work_roots(scratch, project)
    roots = [("scratch", scratch)]
    if project is not None:
        roots.append(("project workspace", project))
    with ExitStack() as stack:
        for label, root in roots:
            if not root.is_absolute() or root == Path(root.anchor):
                raise ValueError(f"{label} work root must be an absolute, non-root directory")
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            lock = ServiceOwnerLock(root, "contractor-runtime")
            try:
                lock.acquire()
            except SandboxContractError:
                raise WorkRootLockError(f"{label} work root in use or unsafe: {root}") from None
            stack.callback(lock.release)
        yield
