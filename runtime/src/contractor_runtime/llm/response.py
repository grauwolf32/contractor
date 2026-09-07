"""Content-free classification of terminal model response limits."""

from typing import Any


def output_limit_reached(response: Any) -> bool:
    if bool(getattr(response, "partial", False)):
        return False
    reason = getattr(response, "finish_reason", None)
    return (
        getattr(reason, "value", reason) == "MAX_TOKENS"
        or getattr(response, "error_code", None) == "MAX_TOKENS"
    )
