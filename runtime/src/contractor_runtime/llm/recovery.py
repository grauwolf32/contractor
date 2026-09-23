"""Allocation-authenticated access to the Server's shared model recovery gate."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import Field, ValidationError

from contractor_runtime.backoff import bounded_backoff
from contractor_runtime.contracts.base import WireModel

if TYPE_CHECKING:
    from contractor_runtime.artifacts import ArtifactTransport

logger = logging.getLogger(__name__)

# A lost Control Plane response must never authorize inference, so a transport
# outage is waited out until the allocation lease itself expires; model retry
# timing is returned by Server. The reconnect cadence only observes the
# authority and backs off with jitter so many Runtimes do not reconnect in
# lockstep after one Server restart.
AUTHORITY_RECONNECT_CEILING_SECONDS = 5.0
# A reachable authority that keeps answering 5xx is not covered by the lease
# watchdog (heartbeats still succeed), so that condition has its own bound.
AUTHORITY_ERROR_DEADLINE_SECONDS = 120.0
AUTHORITY_RETRY_LOG_INTERVAL_SECONDS = 30.0
MAX_RECOVERY_RESPONSE_BYTES = 4096


class RecoveryStoppedError(RuntimeError):
    """The allocation no longer has authority to continue this invocation."""


class RecoveryDecision(WireModel):
    """The Server's gatewayrecovery.Decision, decoded strictly and completely.

    ``code`` and ``requires_retry`` are status for the Run page; a Runtime (like
    the Server's own Planner participant) acts only on ``allowed`` and the two
    timings. They stay declared because the strict wire model forbids unknown
    members and the Server always sends ``requiresRetry``: dropping them would
    make every decision undecodable and stop model recovery.
    """

    allowed: bool = Field(strict=True)
    code: str | None = None
    retry_after_seconds: float = Field(ge=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    requires_retry: bool = Field(strict=True)


class GatewayRecoveryClient:
    def __init__(
        self,
        allocation_id: str,
        transport: ArtifactTransport,
        *,
        on_retry: Callable[[str], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._transport = transport
        self._path = f"/allocations/{allocation_id}/gateway-recovery"
        self._on_retry = on_retry
        self._monotonic = monotonic
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._jitter = jitter

    async def update(
        self,
        model: str,
        request_id: str,
        action: str,
        code: str | None = None,
        retry_after_seconds: float = 0,
    ) -> RecoveryDecision:
        from contractor_runtime.artifacts import ArtifactTransportError

        message = {"model": model, "requestId": request_id, "action": action}
        if code is not None:
            message["code"] = code
        body = json.dumps({**message, "retryAfterSeconds": retry_after_seconds}).encode()
        attempts = 0
        started: float | None = None
        error_started: float | None = None
        last_logged: float | None = None
        while True:
            cause: str
            try:
                response = await self._transport.request(
                    "POST",
                    self._path,
                    headers={"Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=MAX_RECOVERY_RESPONSE_BYTES,
                )
            except ArtifactTransportError:
                cause = "transport"
                # Transport loss is bounded by the allocation lease, not here.
                error_started = None
            else:
                if response.status_code == 200:
                    return _decode_decision(response.body)
                if 400 <= response.status_code < 500:
                    raise RecoveryStoppedError("Model recovery allocation is no longer available")
                cause = f"http_{response.status_code}"
                now = self._monotonic()
                if error_started is None:
                    error_started = now
                elif now - error_started >= AUTHORITY_ERROR_DEADLINE_SECONDS:
                    raise RecoveryStoppedError(
                        "Model recovery authority kept failing beyond its deadline"
                    )
            attempts += 1
            now = self._monotonic()
            if started is None:
                started = now
            if self._on_retry is not None:
                self._on_retry(cause)
            if last_logged is None or now - last_logged >= AUTHORITY_RETRY_LOG_INTERVAL_SECONDS:
                last_logged = now
                logger.warning(
                    "Model recovery authority unavailable (action=%s cause=%s attempts=%d"
                    " elapsedSeconds=%d)",
                    action,
                    cause,
                    attempts,
                    int(now - started),
                )
            await self._sleep(
                bounded_backoff(attempts, AUTHORITY_RECONNECT_CEILING_SECONDS, self._jitter)
            )


def _decode_decision(body: bytes) -> RecoveryDecision:
    # A trusted Server that answers with an undecodable decision has a defect;
    # retrying cannot fix it and guessing must never authorize inference.
    try:
        return RecoveryDecision.model_validate_json(body)
    except (ValidationError, ValueError):
        raise RecoveryStoppedError(
            "Model recovery authority returned an invalid decision"
        ) from None
