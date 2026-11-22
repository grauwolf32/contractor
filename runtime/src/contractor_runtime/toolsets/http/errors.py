"""Content-free HTTP failures shared with allocation preparation."""

from types import MappingProxyType

from contractor_runtime.core.input_errors import ToolInputError

_ERROR_RETRYABILITY = MappingProxyType(
    {
        "http_request_invalid": False,
        "http_credential_unavailable": False,
        "http_target_denied": False,
        "http_request_failed": True,
        "http_dns_failed": False,
        "http_tls_failed": False,
        "http_connection_failed": False,
        "http_timeout": False,
        "http_response_invalid": False,
        "http_response_too_large": False,
        "http_body_not_found": False,
        "http_cookie_state_invalid": False,
        "http_cookie_state_conflict": False,
        "http_cookie_storage_unavailable": True,
        "http_cookie_write_denied": False,
    }
)


_MESSAGES = {
    "http_request_invalid": (
        "Invalid HTTP arguments. Check the URL, method, headers, body and timeout."
    ),
    "http_target_denied": "Target denied by the authorized HTTP scope.",
    "http_credential_unavailable": (
        "Credential unavailable. Select an exact ID from http_credentials."
    ),
    "http_dns_failed": "DNS resolution failed.",
    "http_tls_failed": "TLS connection failed.",
    "http_connection_failed": "Connection failed or was interrupted.",
    "http_timeout": "HTTP request timed out.",
    "http_response_invalid": "The target returned an invalid HTTP response.",
}


class HTTPToolError(RuntimeError):
    """Stable content-free model-facing HTTP failure."""

    def __init__(self, code: str) -> None:
        normalized = code if code in _ERROR_RETRYABILITY else "http_request_failed"
        self.code = normalized
        self.retryable = _ERROR_RETRYABILITY[normalized]
        super().__init__(f"HTTP operation failed ({normalized})")


class HTTPArgumentError(HTTPToolError, ToolInputError):
    def __init__(self, field: str, message: str):
        HTTPToolError.__init__(self, "http_request_invalid")
        self.field = field
        self.message = message

    @property
    def diagnostic_message(self) -> str:
        return f"{self.field}: {self.message}"


def controlled_http_message(code: str) -> str:
    """Safe controlled-tool text, independent of ordinary HTTP error handling."""
    return _MESSAGES.get(code, "HTTP request failed.")
