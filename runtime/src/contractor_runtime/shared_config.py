"""Process-owned shared policy, installed from Control Plane before preparation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SharedConfigSnapshot:
    version: int
    digest: str
    artifact_payload_max_bytes: int


_snapshot: SharedConfigSnapshot | None = None


def install_shared_config(snapshot: SharedConfigSnapshot) -> None:
    global _snapshot
    if _snapshot is not None and _snapshot != snapshot:
        raise RuntimeError("shared config changed; restart the Runtime Agent")
    _snapshot = snapshot


def artifact_payload_max_bytes() -> int:
    if _snapshot is None:
        raise RuntimeError("shared config has not been loaded")
    return _snapshot.artifact_payload_max_bytes
