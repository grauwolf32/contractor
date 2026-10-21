"""Content-free HTTP failures shared with allocation preparation."""

from types import MappingProxyType

_ERROR_RETRYABILITY = MappingProxyType(
    {
        "http_request_invalid": False,
        "http_target_denied": False,
        "http_request_failed": True,
        "http_response_invalid": False,
        "http_response_too_large": False,
        "http_body_not_found": False,
        "http_cookie_state_invalid": False,
        "http_cookie_state_conflict": False,
        "http_cookie_storage_unavailable": True,
        "http_cookie_write_denied": False,
    }
)


class HTTPToolError(RuntimeError):
    """Stable content-free model-facing HTTP failure."""

    def __init__(self, code: str) -> None:
        normalized = code if code in _ERROR_RETRYABILITY else "http_request_failed"
        self.code = normalized
        self.retryable = _ERROR_RETRYABILITY[normalized]
        super().__init__(f"HTTP operation failed ({normalized})")
