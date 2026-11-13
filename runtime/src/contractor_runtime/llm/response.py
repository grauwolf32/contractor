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
