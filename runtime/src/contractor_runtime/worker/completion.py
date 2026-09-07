"""Trusted Worker completion decisions and lifecycle boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from contractor_runtime.contracts import (
    StageContentRequest,
    WorkerCompletionContract,
    WorkerCompletionDiagnostics,
    WorkerFailure,
    WorkerResult,
    WorkerSummarizerConfig,
)

MAX_COMPLETION_REMINDER_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class ContinueCompletion:
    reminder: str
    kind: Literal["continue"] = "continue"

    def __post_init__(self):
        if (
            not isinstance(self.reminder, str)
            or not self.reminder
            or len(self.reminder.encode()) > MAX_COMPLETION_REMINDER_BYTES
            or self.kind != "continue"
        ):
            raise ValueError("completion reminder must be bounded Runtime-authored text")


@dataclass(frozen=True, slots=True)
class CompleteCompletion:
    result: WorkerResult
    kind: Literal["complete"] = "complete"

    def __post_init__(self):
        if self.kind != "complete" or not isinstance(self.result, WorkerResult):
            raise ValueError("complete boundary requires a trusted WorkerResult")


@dataclass(frozen=True, slots=True)
class FailCompletion:
    failure: WorkerFailure
    kind: Literal["fail"] = "fail"

    def __post_init__(self):
        if self.kind != "fail" or not isinstance(self.failure, WorkerFailure):
            raise ValueError("failed boundary requires a bounded WorkerFailure")


CompletionDecision = ContinueCompletion | CompleteCompletion | FailCompletion


class WorkerCompletionError(ValueError):
    """Implementation-owned failure with a bounded code and retryability."""

    code: str
    retryable: bool = False


class PreparedWorkerCompletion(ABC):
    """Trusted allocation binding invoked outside model-controlled tool calls."""

    contract: WorkerCompletionContract
    required_tools: frozenset[str]
    diagnostics: WorkerCompletionDiagnostics | None
    diagnostics_sink: Callable[[WorkerCompletionDiagnostics], None] | None
    phase_sink: Callable[[WorkerCompletionDiagnostics], Awaitable[None]] | None

    @abstractmethod
    def reset_diagnostics(self) -> None: ...

    @abstractmethod
    async def record_phase(self, phase: str, failure_code: str | None = None) -> None: ...

    @abstractmethod
    def begin(self, invocation_id: str) -> None: ...

    @abstractmethod
    async def end(self) -> None: ...

    @abstractmethod
    def validate_request(self, request: StageContentRequest) -> WorkerFailure | None: ...

    @abstractmethod
    def failure(self, error: Exception) -> WorkerFailure: ...

    @abstractmethod
    async def finish(
        self,
        *,
        request: StageContentRequest,
        deadline: float,
        check_active: Callable[[], None],
    ) -> CompletionDecision: ...


def bind_worker_completion(
    *,
    tools: Mapping[str, Any],
    contract: WorkerCompletionContract | None,
    summarizer: WorkerSummarizerConfig | None,
) -> PreparedWorkerCompletion | None:
    bindings = [
        binding
        for tool in tools.values()
        if (binding := getattr(tool, "completion_binding", None)) is not None
    ]
    if contract is None:
        if bindings:
            raise ValueError("Completion tools require a trusted completion contract")
        return None
    if not bindings or summarizer is not None:
        raise ValueError("Worker completion requires its trusted prepared tool binding")
    binding = bindings[0]
    if (
        not isinstance(binding, PreparedWorkerCompletion)
        or binding.contract != contract
        or not binding.required_tools
        or any(value is not binding for value in bindings)
        or any(
            getattr(tools.get(name), "completion_binding", None) is not binding
            for name in binding.required_tools
        )
    ):
        raise ValueError("Worker completion requires its trusted prepared tool binding")
    return binding
