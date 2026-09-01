"""Allocation-private project workspace storage primitives."""

from contractor_runtime.projectfs.hydrate import WorkspacePreparationError, hydrate_workspace
from contractor_runtime.projectfs.overlay import (
    WORKSPACE_OVERLAY_MEDIA_TYPE,
    OverlayOperation,
    OverlayWorkspaceSession,
    WorkspaceStateError,
    canonical_overlay_operations,
    decode_workspace_state,
    encode_workspace_state,
)
from contractor_runtime.projectfs.provider import (
    LocalWorkspaceProvider,
    MemoryWorkspaceProvider,
    ProjectWorkspaceStorage,
    WorkspaceCapabilitySnapshot,
    WorkspaceProvider,
    build_workspace_provider,
)
from contractor_runtime.projectfs.storage import (
    DirectWorkspaceSession,
    ManagedWorkspaceTree,
    WorkspaceDiff,
    WorkspaceReader,
    WorkspaceSnapshot,
    WorkspaceStorageError,
    WorkspaceTextFile,
    WorkspaceWriter,
)

__all__ = [
    "WORKSPACE_OVERLAY_MEDIA_TYPE",
    "DirectWorkspaceSession",
    "LocalWorkspaceProvider",
    "ManagedWorkspaceTree",
    "MemoryWorkspaceProvider",
    "OverlayOperation",
    "OverlayWorkspaceSession",
    "ProjectWorkspaceStorage",
    "WorkspaceCapabilitySnapshot",
    "WorkspaceDiff",
    "WorkspacePreparationError",
    "WorkspaceProvider",
    "WorkspaceReader",
    "WorkspaceSnapshot",
    "WorkspaceStateError",
    "WorkspaceStorageError",
    "WorkspaceTextFile",
    "WorkspaceWriter",
    "build_workspace_provider",
    "canonical_overlay_operations",
    "decode_workspace_state",
    "encode_workspace_state",
    "hydrate_workspace",
]
