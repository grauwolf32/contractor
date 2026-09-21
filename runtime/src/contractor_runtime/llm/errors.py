"""Normalize provider failures without retaining response bodies or credentials.

Status-based rules belong to the openai-compatible@1 protocol and live here.
Provider-specific text lives in the Gateway's declared failure signatures (or
the protocol default) and is matched exactly, never by substring or pattern.
"""

from contextlib import suppress
from dataclasses import dataclass

import httpx

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


def classify_response(
    response: httpx.Response, signatures: GatewayFailureSignatures
) -> GatewayFailure:
    status = response.status_code
    body = None
    if len(response.content) <= MAX_ERROR_CLASSIFICATION_BYTES:
        with suppress(ValueError):
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
