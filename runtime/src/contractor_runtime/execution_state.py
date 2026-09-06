"""Content-free fatal execution state, never part of model-editable State."""

from __future__ import annotations

from contractor_runtime.sandbox_contracts import SandboxErrorCode


class SandboxExecutionFailed(RuntimeError):
    def __init__(self, code: SandboxErrorCode):
        self.code = code
        super().__init__(code.value)


class WorkerExecutionState:
    def __init__(self):
        self.failure: SandboxErrorCode | None = None

    def fail(self, code: SandboxErrorCode) -> None:
        self.failure = self.failure or code

    def check(self) -> None:
        if self.failure is not None:
            raise SandboxExecutionFailed(self.failure)
