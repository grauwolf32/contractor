"""Private allocation lifecycle hooks, separate from scratch and tool execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from contractor_runtime.sandbox_contracts import SandboxErrorCode, SandboxExecutor

if TYPE_CHECKING:
    from contractor_runtime.projectfs import DirectWorkspaceSession


class PreparedExecution(Protocol):
    @property
    def executor(self) -> SandboxExecutor: ...

    def bind_failure(self, callback: Callable[[SandboxErrorCode], None]) -> None: ...

    def reject(self) -> None: ...

    async def prepare(self, *, deadline: datetime) -> None: ...

    async def stop(self, *, deadline: datetime) -> None: ...

    async def remove(self, *, deadline: datetime) -> None: ...


class ExecutionLifecycle(Protocol):
    async def recover(self, *, deadline: float) -> None: ...

    async def prepare_root(self, root: Path) -> None: ...

    def allocate(
        self, allocation_id: str, workspace: DirectWorkspaceSession
    ) -> PreparedExecution: ...

    async def close(self, *, deadline: float) -> None: ...
