"""Process-level project workspace storage settings and their resource limits."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from contractor_runtime.contracts import WorkspaceStorage

DEFAULT_WORKSPACE_OPERATION_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    max_files: int
    max_expanded_bytes: int
    max_managed_text_bytes: int
    max_file_bytes: int


@dataclass(frozen=True, slots=True)
class WorkspaceSettings:
    storage: WorkspaceStorage
    limits: WorkspaceLimits
    work_root: Path | None = field(default=None, repr=False)
    operation_timeout_seconds: float = DEFAULT_WORKSPACE_OPERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not math.isfinite(self.operation_timeout_seconds) or self.operation_timeout_seconds <= 0:
            raise ValueError("workspace operation timeout must be finite and positive")
