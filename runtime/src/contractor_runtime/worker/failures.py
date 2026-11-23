"""Worker failure classification shared by execution and completion."""

from __future__ import annotations

from contractor_runtime.contracts import (
    WorkerFailure,
)
from contractor_runtime.llm.openai import (
    GatewayModelError,
)
from contractor_runtime.worker.compaction import (
    WorkerCompactionError,
)


def _chained_error[E: BaseException](error: BaseException, kind: type[E]) -> E | None:
    """Find a Runtime signal or failure of ``kind`` through ADK's callback wrappers.

    ADK re-raises plugin and callback failures inside its own exceptions, so the
    walk follows the explicit cause first, then the implicit context, and stops
    at the first repeated exception.
    """

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, kind):
            return current
        current = current.__cause__ or current.__context__
    return None


def _compaction_failure(error: WorkerCompactionError) -> WorkerFailure:
    if error.code == "worker_context_limit_exceeded":
        return _failure(error.code, f"Worker context limit exceeded ({error.cause})", False)
    return _failure(
        error.code, f"Worker context compaction failed ({error.cause})", error.retryable
    )


def _failure(code: str, message: str, retryable: bool) -> WorkerFailure:
    return WorkerFailure(code=code, message=message, retryable=retryable)


def _summarizer_failure(cause: str, *, retryable: bool = True) -> WorkerFailure:
    return _failure(
        "worker_summarization_failed",
        f"Worker terminal summarization failed ({cause})",
        retryable,
    )


def _summarizer_result_failure_code(code: str) -> str:
    return {
        "worker_result_missing": "result_missing",
        "worker_result_invalid": "result_invalid",
        "worker_result_subtask_mismatch": "result_subtask_mismatch",
        "worker_result_too_large": "result_too_large",
        "unsafe_worker_result": "unsafe_result",
        "invalid_worker_result_binding": "invalid_result_binding",
    }.get(code, "result_rejected")


def _gateway_worker_failure(error: GatewayModelError) -> WorkerFailure:
    if error.request_invalid:
        return _failure(
            "worker_gateway_invalid_request",
            "Worker LLM Gateway request is invalid",
            False,
        )
    return _failure(
        "worker_gateway_unavailable",
        "Worker LLM Gateway request failed",
        error.retryable,
    )
