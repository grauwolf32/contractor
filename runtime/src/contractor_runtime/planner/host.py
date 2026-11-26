"""Execution authority required by a Planner, independent of its deployment."""

from typing import Any, Protocol

from contractor_runtime.contracts.planner import PlannerOperationResult


class PlannerHost(Protocol):
    async def execute(self, name: str, arguments: dict[str, Any]) -> PlannerOperationResult:
        """Apply one ordered operation and return its confirmed outcome."""
        ...


class PlannerStoppedError(RuntimeError):
    """A classified execution failure that retains no request content."""

    def __init__(self, code: str, message: str, retryable: bool) -> None:
        self.code, self.retryable = code, retryable
        super().__init__(message)
