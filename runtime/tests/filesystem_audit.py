"""Audited filesystem operations below chosen paths, with the thread of each.

Thread identity, not elapsed time, proves that blocking filesystem work stays
off the event-loop thread: a worker thread can hold the GIL, so loop gaps are
not deterministic.
"""

from __future__ import annotations

import os
import sys
import threading

_EVENTS = frozenset(
    {"open", "os.chmod", "os.mkdir", "os.remove", "os.rmdir", "shutil.rmtree", "tempfile.mkdtemp"}
)
_recorders: list[FilesystemAudit] = []


def _audit(event: str, args: tuple[object, ...]) -> None:
    if not _recorders or event not in _EVENTS or not args:
        return
    path = args[0]
    if isinstance(path, str | bytes | os.PathLike):
        for recorder in tuple(_recorders):
            recorder.record(event, os.fsdecode(os.fspath(path)))


# Audit hooks cannot be removed; this one is inert while no recorder is active.
sys.addaudithook(_audit)


class FilesystemAudit:
    """Record operations on absolute paths that contain one of ``fragments``."""

    def __init__(self, *fragments: str) -> None:
        self._fragments = fragments
        self._lock = threading.Lock()
        self.operations: list[tuple[str, str, int]] = []

    def __enter__(self) -> FilesystemAudit:
        _recorders.append(self)
        return self

    def __exit__(self, *_exception: object) -> None:
        _recorders.remove(self)

    def record(self, event: str, path: str) -> None:
        if any(fragment in path for fragment in self._fragments):
            with self._lock:
                self.operations.append((event, path, threading.get_ident()))

    def on_thread(self, thread: int) -> list[tuple[str, str]]:
        with self._lock:
            return [(event, path) for event, path, ident in self.operations if ident == thread]

    def events(self) -> list[tuple[str, str]]:
        with self._lock:
            return [(event, path) for event, path, _ in self.operations]
