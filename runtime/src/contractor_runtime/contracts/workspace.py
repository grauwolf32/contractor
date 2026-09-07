"""Private Runtime protocol workspace models and validation."""

from __future__ import annotations

from typing import Self

from pydantic import (
    Field,
    model_validator,
)

from contractor_runtime.contracts.artifacts import ArtifactRef
from contractor_runtime.contracts.base import (
    ID_PATTERN,
    MAX_UINT64,
    MAX_WORKER_FILES_READ,
    WireModel,
    WorkspaceModeV2,
    WorkspaceStorageV2,
    _require_sorted_unique,
    _require_workspace_target,
)


class WorkspaceObservationSummary(WireModel):
    scoped_files: int = Field(ge=0, le=MAX_UINT64)
    scope_complete: bool
    discovered_files: int = Field(ge=0, le=MAX_UINT64)
    read_files: int = Field(ge=0, le=MAX_UINT64)
    matched_files: int = Field(ge=0, le=MAX_UINT64)
    modified_files: int = Field(ge=0, le=MAX_UINT64)
    detail_complete: bool
    unread_files: int | None = Field(default=None, ge=0, le=MAX_UINT64)
    files_read: list[str] = Field(max_length=MAX_WORKER_FILES_READ)
    files_read_truncated: bool

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        seen: set[str] = set()
        for path in self.files_read:
            if not path:
                raise ValueError("Worker observed workspace path must not be empty")
            _require_workspace_target(path)
            if path in seen:
                raise ValueError("Worker filesRead paths must be unique")
            seen.add(path)
        if len(self.files_read) > self.read_files:
            raise ValueError("Worker filesRead detail exceeds readFiles")
        if self.files_read_truncated != (len(self.files_read) < self.read_files):
            raise ValueError("Worker filesRead truncation is inconsistent")
        coverage_complete = self.scope_complete and self.detail_complete
        if coverage_complete != (self.unread_files is not None):
            raise ValueError("Worker unreadFiles completeness is inconsistent")
        if self.unread_files is not None and self.unread_files > self.scoped_files:
            raise ValueError("Worker unreadFiles exceeds scopedFiles")
        return self


class WorkspaceLimitsV2(WireModel):
    max_files: int = Field(gt=0)
    max_expanded_bytes: int = Field(gt=0)
    max_managed_text_bytes: int = Field(gt=0)
    max_file_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if (
            self.max_file_bytes > self.max_expanded_bytes
            or self.max_managed_text_bytes > self.max_expanded_bytes
        ):
            raise ValueError("workspace limits are invalid")
        return self


class AllocationWorkspaceSourceV2(WireModel):
    artifact: ArtifactRef
    target: str

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        self.artifact.require_exact()
        _require_workspace_target(self.target)
        return self


class AllocationWorkspaceStateV2(WireModel):
    artifact: ArtifactRef

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        self.artifact.require_exact()
        return self


class AllocationWorkspaceExportV2(WireModel):
    state: str = Field(pattern=ID_PATTERN.pattern)
    diff: str = Field(pattern=ID_PATTERN.pattern)

    @model_validator(mode="after")
    def validate_export(self) -> Self:
        if self.state == self.diff:
            raise ValueError("workspace export slots must be distinct")
        return self


class WorkspaceCapabilitiesV2(WireModel):
    storage: WorkspaceStorageV2
    modes: list[WorkspaceModeV2] = Field(min_length=1, max_length=2)
    limits: WorkspaceLimitsV2

    @model_validator(mode="after")
    def validate_capabilities(self) -> Self:
        _require_sorted_unique("workspace capability modes", self.modes, maximum=2)
        return self


class AllocationWorkspaceSpecV2(WireModel):
    mode: WorkspaceModeV2
    sources: list[AllocationWorkspaceSourceV2] = Field(min_length=1, max_length=32)
    state: AllocationWorkspaceStateV2 | None = None
    export: AllocationWorkspaceExportV2 | None = None

    @model_validator(mode="after")
    def validate_workspace(self) -> Self:
        targets = [source.target for source in self.sources]
        for index, target in enumerate(targets):
            for other_index, other in enumerate(targets):
                if index == other_index:
                    continue
                if target == other or target == "" or other.startswith(f"{target}/"):
                    raise ValueError("workspace source targets must be unique and non-overlapping")
        if self.export is not None and self.mode != "overlay":
            raise ValueError("workspace export requires overlay mode")
        return self
