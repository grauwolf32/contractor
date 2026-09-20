"""Normalize provider failures without retaining response bodies or credentials."""

from contextlib import suppress
from dataclasses import dataclass

import httpx

# LM Studio responses observed through LiteLLM during the 2026-09-20 outage.
# These exact messages describe dependency availability, despite HTTP 400.
MODEL_UNLOADED_MESSAGES = frozenset(
    {"Model unloaded by user or API request.", "Model is unloaded."}
)
PERMANENT_PROVIDER_CODES = frozenset(
    {"insufficient_quota", "budget_exceeded", "context_length_exceeded"}
)
MAX_ERROR_CLASSIFICATION_BYTES = 16 * 1024


@dataclass(frozen=True, slots=True)
class GatewayFailure:
    code: str
    retryable: bool
    status: int | None = None


def classify_response(response: httpx.Response) -> GatewayFailure:
    status = response.status_code
    body = None
    if len(response.content) <= MAX_ERROR_CLASSIFICATION_BYTES:
        with suppress(ValueError):
            body = response.json()
    error = body.get("error", body) if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and code in PERMANENT_PROVIDER_CODES:
        return GatewayFailure(code, False, status)
    message = error.get("message") if isinstance(error, dict) else error
    if status == 400 and _model_unloaded(message):
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


def _model_unloaded(message: object) -> bool:
    if not isinstance(message, str):
        return False
    if message in MODEL_UNLOADED_MESSAGES:
        return True
    # LiteLLM wraps the upstream OpenAI-compatible error as a Python dict repr.
    # Accept its observed wrapper only; never scan arbitrary request error text.
    prefix = "litellm.BadRequestError: OpenAIException - Error code: 400 - "
    if not message.startswith(prefix):
        return False
    upstream = message.removeprefix(prefix)
    return any(
        upstream == repr({"error": text})
        or upstream.startswith(repr({"error": text}) + ". Received Model Group=")
        for text in MODEL_UNLOADED_MESSAGES
    )
