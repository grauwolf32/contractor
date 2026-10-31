"""Process-owned shared policy, installed from Control Plane before preparation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SharedConfigSnapshot:
    version: int
    digest: str
    artifact_payload_max_bytes: int
    public_page_max_items: int
    artifact_list_max_items: int
    worker_result_max_bytes: int
    worker_completion_max_bytes: int
    worker_failure_message_max_bytes: int
    worker_result_max_artifacts: int
    worker_observation_max_tools: int
    worker_workspace_max_files_read: int
    worker_state_max_bytes: int
    worker_state_workspace_max_paths: int
    worker_state_workspace_paths_max_bytes: int


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


def artifact_list_max_items() -> int:
    if _snapshot is None:
        raise RuntimeError("shared config has not been loaded")
    return _snapshot.artifact_list_max_items


def get_shared_config() -> SharedConfigSnapshot:
    if _snapshot is None:
        raise RuntimeError("shared config has not been loaded")
    return _snapshot
