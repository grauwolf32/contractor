"""Allocation-local OpenAI-compatible LLM Gateway client construction."""

from __future__ import annotations

import asyncio
import uuid
from asyncio import sleep
from dataclasses import dataclass, field
from email.utils import mktime_tz, parsedate_tz
from math import isfinite
from random import random
from time import time
from typing import TYPE_CHECKING, Any

import httpx

from contractor_runtime.adapters.http_proxy import ProxyHTTPClient
from contractor_runtime.contracts.settings import (
    GatewayFailureSignatures,
    default_gateway_failure_signatures,
)
from contractor_runtime.llm.errors import GatewayFailure, classify_response
from contractor_runtime.llm.recovery import GatewayRecoveryClient, RecoveryStoppedError

if TYPE_CHECKING:
    from contractor_runtime.factories import WorkerBuildContext

GATEWAY_MAX_RETRIES = 3
GATEWAY_RETRY_GRACE_SECONDS = 60.0
# Bound on the best-effort probe release sent when a granted model call is
# abandoned; the Server's own probe timeout covers a release that is lost.
RECOVERY_RELEASE_TIMEOUT_SECONDS = 5.0
_MAX_RETRY_AFTER_SECONDS = 120.0
_STATUS_ERROR_TYPES = {
    400: "BadRequestError",
    401: "AuthenticationError",
    403: "PermissionDeniedError",
    404: "NotFoundError",
    409: "ConflictError",
    422: "UnprocessableEntityError",
    429: "RateLimitError",
}


class GatewayClientClosedError(RuntimeError):
    """Raised without connection details when an allocation client is closed."""


class GatewayRequestError(RuntimeError):
    """Retain only the classified failure, never provider headers or bodies."""

    def __init__(
        self,
        provider_error_type: str,
        *,
        retryable: bool,
        failure: GatewayFailure | None = None,
        retry_after_seconds: float = 0,
    ) -> None:
        self.provider_error_type = provider_error_type
        self.retryable = retryable
        self.transport_retry_allowed = True
        self.failure = failure
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"LLM gateway request failed ({provider_error_type})")


@dataclass(slots=True)
class GatewayClientHandle:
    """Send non-streaming Chat Completions over one allocation's HTTP route."""

    _client: httpx.AsyncClient | None
    _completion_url: httpx.URL | None
    _api_key: str
    _owns_http_client: bool
    request_timeout_seconds: float
    operation_timeout_seconds: float
    max_retries: int = GATEWAY_MAX_RETRIES
    recovery: GatewayRecoveryClient | None = None
    failure_signatures: GatewayFailureSignatures = field(
        default_factory=default_gateway_failure_signatures
    )

    @property
    def closed(self) -> bool:
        return self._client is None

    def __repr__(self) -> str:
        return (
            "GatewayClientHandle("
            f"closed={self.closed!r}, owns_http_client={self._owns_http_client!r}, "
            f"operation_timeout_seconds={self.operation_timeout_seconds!r})"
        )

    async def complete(self, payload: dict[str, Any]) -> Any:
        """Return one response after a bounded series of physical attempts."""

        if self.recovery is not None:
            return await self._complete_with_recovery(payload)
        return await self._complete_transport(
            payload, self.max_retries, self.request_timeout_seconds
        )

    async def _complete_transport(self, payload, max_retries, request_timeout_seconds):
        for attempt in range(max_retries + 1):
            client = self._client
            url = self._completion_url
            if client is None or url is None:
                raise GatewayClientClosedError("LLM Gateway client is closed")
            # Build before the transport boundary: an invalid local payload is
            # not a connection failure and must not be sent repeatedly.
            request = client.build_request(
                "POST",
                url,
                json=payload,
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "accept": "application/json",
                    "x-stainless-retry-count": str(attempt),
                    "x-stainless-read-timeout": str(request_timeout_seconds),
                },
                timeout=request_timeout_seconds,
            )
            response: httpx.Response | None = None
            error: GatewayRequestError | None = None
            should_retry = True
            retry_after: float | None = None
            try:
                response = await client.send(request)
            except httpx.TimeoutException:
                error = GatewayRequestError(
                    "APITimeoutError",
                    retryable=True,
                    failure=GatewayFailure("gateway_timeout", True),
                )
            except Exception:
                # Includes allocation proxy failures; cancellation is a
                # BaseException and propagates without another attempt.
                error = GatewayRequestError(
                    "APIConnectionError",
                    retryable=True,
                    failure=GatewayFailure("gateway_unavailable", True),
                )
            if response is not None:
                try:
                    if response.is_success:
                        try:
                            return response.json()
                        except ValueError:
                            error = GatewayRequestError("InvalidGatewayResponse", retryable=False)
                        should_retry = False
                    else:
                        error = _status_error(response, self.failure_signatures)
                        retry_after = _retry_after_seconds(response.headers)
                        should_retry = _should_retry(response, retry_after, self.failure_signatures)
                        if response.headers.get("x-should-retry") == "false":
                            error.transport_retry_allowed = False
                        if retry_after is not None and isfinite(retry_after):
                            error.retry_after_seconds = max(0, retry_after)
                finally:
                    await response.aclose()
            # Release request/response buffers before a retry delay or failure.
            del request, response
            if error is None:
                raise RuntimeError("LLM gateway attempt produced no response")
            if not should_retry or attempt >= max_retries:
                # Outside the except suite so a provider exception cannot keep
                # credentials or response data alive through __context__.
                raise error from None
            await sleep(_retry_delay(attempt, retry_after))
        raise RuntimeError("LLM gateway retry limit is invalid")

    async def _complete_with_recovery(self, payload: dict[str, Any]) -> Any:
        """Keep one logical model call alive; previously executed tools never replay."""
        recovery = self.recovery
        if recovery is None:
            raise RuntimeError("model recovery requires an authority client")
        model = payload["model"]
        request_id = uuid.uuid4().hex
        try:
            while True:
                decision = await recovery.update(model, request_id, "acquire")
                if not decision.allowed:
                    await sleep(decision.retry_after_seconds)
                    continue
                error = None
                try:
                    async with asyncio.timeout(decision.request_timeout_seconds):
                        result = await self._complete_transport(
                            payload, 0, decision.request_timeout_seconds
                        )
                except GatewayRequestError as caught:
                    error = caught
                except TimeoutError:
                    error = GatewayRequestError(
                        "APITimeoutError",
                        retryable=True,
                        failure=GatewayFailure("gateway_timeout", True),
                    )
                except (asyncio.CancelledError, Exception):
                    # Cancellation or an unexpected failure must not leave the
                    # granted probe held until the Server times it out.
                    await asyncio.shield(_release_probe(recovery, model, request_id))
                    raise
                if error is None:
                    await recovery.update(model, request_id, "succeeded")
                    return result
                if not error.retryable or not error.transport_retry_allowed:
                    await recovery.update(model, request_id, "finished")
                    raise error from None
                failure = error.failure or GatewayFailure("gateway_unavailable", True)
                await recovery.update(
                    model, request_id, "failed", failure.code, error.retry_after_seconds
                )
                request_id = uuid.uuid4().hex
        except RecoveryStoppedError:
            # Losing the authority is not a provider failure: report it under
            # its own permanent code instead of the exception's class name.
            raise GatewayRequestError(
                "RecoveryAuthorityUnavailable",
                retryable=False,
                failure=GatewayFailure("recovery_authority_unavailable", False),
            ) from None

    async def close(self) -> None:
        """Erase the credential and close only a directly owned HTTP client."""

        client = self._client
        owns_http_client = self._owns_http_client
        self._client = None
        self._completion_url = None
        self._api_key = ""
        self._owns_http_client = False
        if client is None:
            return
        if owns_http_client:
            await client.aclose()


async def _release_probe(recovery: GatewayRecoveryClient, model: str, request_id: str) -> None:
    try:
        async with asyncio.timeout(RECOVERY_RELEASE_TIMEOUT_SECONDS):
            await recovery.update(model, request_id, "finished")
    except Exception:
        # Best effort: transport loss, a stopped authority or the bound
        # expiring must not replace the error that abandoned the call.
        pass


def new_gateway_client(
    *,
    base_url: str,
    api_key: str | None,
    timeout_seconds: float,
    http_client: httpx.AsyncClient | None = None,
    recovery: GatewayRecoveryClient | None = None,
    failure_signatures: GatewayFailureSignatures | None = None,
) -> GatewayClientHandle:
    """Construct a bounded-retry client for one immutable allocation route."""

    base = httpx.URL(base_url)
    path, separator, query = base.raw_path.partition(b"?")
    completion_url = base.copy_with(
        raw_path=path.rstrip(b"/") + b"/chat/completions" + separator + query
    )
    owns_http_client = http_client is None
    if http_client is None:
        # Process-global HTTP(S)_PROXY and trust-store variables are not an
        # allocation configuration channel. Explicit Runtime adapters supply
        # their own client when proxy routing is selected.
        http_client = httpx.AsyncClient(timeout=timeout_seconds, trust_env=False)
    return GatewayClientHandle(
        recovery=recovery,
        failure_signatures=(
            failure_signatures
            if failure_signatures is not None
            else default_gateway_failure_signatures()
        ),
        _client=http_client,
        _completion_url=completion_url,
        _api_key=api_key or "contractor-no-token",
        _owns_http_client=owns_http_client,
        request_timeout_seconds=timeout_seconds,
        operation_timeout_seconds=timeout_seconds + GATEWAY_RETRY_GRACE_SECONDS,
    )


def _status_error(
    response: httpx.Response, signatures: GatewayFailureSignatures
) -> GatewayRequestError:
    failure = classify_response(response, signatures)
    provider_error_type = _STATUS_ERROR_TYPES.get(
        response.status_code,
        "InternalServerError" if response.status_code >= 500 else "APIStatusError",
    )
    return GatewayRequestError(provider_error_type, retryable=failure.retryable, failure=failure)


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    try:
        return float(headers["retry-after-ms"]) / 1000
    except (KeyError, ValueError):
        pass
    retry_after = headers.get("retry-after")
    try:
        return float(retry_after)
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_tz(retry_after)
        return mktime_tz(parsed) - time() if parsed is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _should_retry(
    response: httpx.Response, retry_after: float | None, signatures: GatewayFailureSignatures
) -> bool:
    failure = classify_response(response, signatures)
    if not failure.retryable:
        return False
    if retry_after is not None and isfinite(retry_after) and retry_after > _MAX_RETRY_AFTER_SECONDS:
        return False
    override = response.headers.get("x-should-retry")
    if override in {"true", "false"}:
        return override == "true"
    return failure.retryable


def _retry_delay(attempt: int, retry_after: float | None) -> float:
    if (
        retry_after is not None
        and isfinite(retry_after)
        and 0 < retry_after <= _MAX_RETRY_AFTER_SECONDS
    ):
        return retry_after
    return min(0.5 * 2 ** min(attempt, 4), 8.0) * (1 - 0.25 * random())


def build_gateway_client(context: WorkerBuildContext) -> GatewayClientHandle:
    """Resolve the exact allocation settings into an isolated client handle."""

    settings = context.runtime_settings
    token = settings.llm_gateway_token
    token_value = token.get_secret_value() if token is not None else None
    timeout = float(settings.request_timeout_seconds)
    signatures = settings.llm_gateway_failure_signatures
    adapter_handle = context.adapter_handles.model_http
    if adapter_handle is None:
        return new_gateway_client(
            base_url=settings.llm_gateway_url,
            api_key=token_value,
            timeout_seconds=timeout,
            recovery=context.gateway_recovery,
            failure_signatures=signatures,
        )
    if not isinstance(adapter_handle, ProxyHTTPClient):
        raise TypeError("llm-gateway proxy handle has an invalid type")
    return new_gateway_client(
        base_url=settings.llm_gateway_url,
        api_key=token_value,
        timeout_seconds=timeout,
        http_client=adapter_handle.async_client,
        recovery=context.gateway_recovery,
        failure_signatures=signatures,
    )
