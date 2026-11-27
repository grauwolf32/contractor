"""Normalize provider failures without retaining response bodies or credentials.

Status-based rules belong to the openai-compatible@1 protocol and live here.
Provider-specific text lives in the Gateway's declared failure signatures (or
the protocol default) and is matched exactly, never by substring or pattern.
"""

from contextlib import suppress
from dataclasses import dataclass

import httpx
from google.genai import types

from contractor_runtime.contracts import GatewayFailureSignatures

MAX_ERROR_CLASSIFICATION_BYTES = 16 * 1024

# LiteLLM wraps the upstream OpenAI-compatible error as a Python dict repr.
# Only this observed wrapper is recognized; the wrapped text still has to be
# declared as a signature.
_LITELLM_WRAPPER_PREFIX = "litellm.BadRequestError: OpenAIException - Error code: 400 - "
_LITELLM_WRAPPER_SUFFIX = ". Received Model Group="


@dataclass(frozen=True, slots=True)
class GatewayFailure:
    code: str
    retryable: bool
    status: int | None = None


class GatewayModelError(RuntimeError):
    """Secret-free boundary error for failures below the model adapter.

    ``response_received`` marks a Gateway response the adapter rejected (for
    example invalid tool-call JSON): the provider already spent tokens, so the
    parsed numeric ``usage_metadata`` (or ``None`` when unavailable) must still
    be accounted. No response content is retained.
    """

    def __init__(
        self,
        provider_error_type: str,
        *,
        retryable: bool = True,
        failure: GatewayFailure | None = None,
        response_received: bool = False,
        usage_metadata: types.GenerateContentResponseUsageMetadata | None = None,
        request_invalid: bool = False,
        empty_stop: bool = False,
        response_metadata: dict[str, str | bool] | None = None,
    ) -> None:
        self.provider_error_type = provider_error_type
        self.retryable = retryable
        self.failure = failure
        self.response_received = response_received
        self.usage_metadata = usage_metadata if response_received else None
        self.request_invalid = request_invalid
        self.empty_stop = empty_stop
        self.response_metadata = response_metadata if response_received else None
        super().__init__(f"LLM gateway call failed ({provider_error_type})")


class _AdapterFailure(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def classify_response(
    response: httpx.Response, signatures: GatewayFailureSignatures
) -> GatewayFailure:
    status = response.status_code
    body = None
    if len(response.content) <= MAX_ERROR_CLASSIFICATION_BYTES:
        with suppress(ValueError, RecursionError):
            body = response.json()
    error = body.get("error", body) if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and code in signatures.permanent_codes:
        return GatewayFailure(code, False, status)
    message = error.get("message") if isinstance(error, dict) else error
    if _model_unavailable(status, message, signatures):
        return GatewayFailure("model_unavailable", True, status)
    if status in {401, 403}:
        return GatewayFailure("gateway_access_denied", False, status)
    if status == 429:
        return GatewayFailure("gateway_rate_limited", True, status)
    if status in {408, 504}:
        return GatewayFailure("gateway_timeout", True, status)
    if status == 409 or 500 <= status < 600:
        return GatewayFailure("gateway_unavailable", True, status)
    # An explicit gateway hint may permit a transport retry. Permanent semantic
    # classifications above always take precedence over this hint.
    retryable = response.headers.get("x-should-retry") == "true"
    return GatewayFailure(
        "gateway_unavailable" if retryable else "gateway_request_rejected", retryable, status
    )


def _model_unavailable(status: int, message: object, signatures: GatewayFailureSignatures) -> bool:
    if not isinstance(message, str):
        return False
    for signature in signatures.model_unavailable:
        if signature.status != status:
            continue
        if signature.message_equals is not None and message == signature.message_equals:
            return True
        if signature.litellm_wrapped is not None and _litellm_wrapped(
            message, signature.litellm_wrapped
        ):
            return True
    return False


def _litellm_wrapped(message: str, upstream_text: str) -> bool:
    if not message.startswith(_LITELLM_WRAPPER_PREFIX):
        return False
    upstream = message.removeprefix(_LITELLM_WRAPPER_PREFIX)
    wrapped = repr({"error": upstream_text})
    return upstream == wrapped or upstream.startswith(wrapped + _LITELLM_WRAPPER_SUFFIX)
