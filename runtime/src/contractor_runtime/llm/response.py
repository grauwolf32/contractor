"""Content-free classification of model response errors and limits."""

from typing import Any


def output_limit_reached(response: Any) -> bool:
    if bool(getattr(response, "partial", False)):
        return False
    reason = getattr(response, "finish_reason", None)
    return (
        getattr(reason, "value", reason) == "MAX_TOKENS"
        or getattr(response, "error_code", None) == "MAX_TOKENS"
    )


def model_response_failed(response: Any) -> bool:
    """Distinguish an actual error response from empty content and output limits."""
    return getattr(response, "error_code", None) not in (
        None,
        "MODEL_RETURNED_NO_CONTENT",
    ) and not output_limit_reached(response)


class WorkerResponseRecovery(RuntimeError):
    """Content-free control signal after a charged, unusable model response."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__("Worker response requires recovery")


def empty_stop_response(response: Any) -> bool:
    reason = getattr(response, "finish_reason", None)
    if getattr(reason, "value", reason) != "STOP" or getattr(response, "partial", False):
        return False
    if model_response_failed(response):
        return False
    parts = getattr(getattr(response, "content", None), "parts", None) or ()
    return not any(
        getattr(part, "function_call", None) is not None
        or (not getattr(part, "thought", False) and (getattr(part, "text", None) or "").strip())
        for part in parts
    )
