"""Allocation-authenticated access to the Server's shared model recovery gate."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Self

from pydantic import Field, ValidationError, model_validator

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

# The participant protocol of the Server's recovery gate (SCH-R105 to SCH-R112):
# the updates a participant sends, the terminal ones that relinquish its probe
# and wait, and the codes a failed update may carry. The Server's
# gatewayrecovery.Request applies the same rules;
# api/testdata/v1alpha1/gateway-recovery-participant-cases.json pins both.
RECOVERY_ACTIONS = frozenset({"acquire", "failed", "succeeded", "finished", "released"})
TERMINAL_RECOVERY_ACTIONS = frozenset({"succeeded", "finished", "released"})
RECOVERY_FAILURE_CODES = frozenset(
    {"gateway_rate_limited", "gateway_timeout", "gateway_unavailable", "model_unavailable"}
)
MAX_RECOVERY_IDENTIFIER_BYTES = 256


class RecoveryStoppedError(RuntimeError):
    """The allocation no longer has authority to continue this invocation."""


class RecoveryUpdate(WireModel):
    """One participant update, held to the rules of the Server's gatewayrecovery.Request."""

    model: str
    request_id: str
    action: str
    code: str | None = None
    retry_after_seconds: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        for value in (self.model, self.request_id):
            if not 1 <= len(value.encode("utf-8")) <= MAX_RECOVERY_IDENTIFIER_BYTES:
                raise ValueError("model recovery identifiers must have 1 to 256 bytes")
        if self.action not in RECOVERY_ACTIONS:
            raise ValueError("model recovery action is unknown")
        if self.action == "failed" and self.code not in RECOVERY_FAILURE_CODES:
            raise ValueError("a failed model recovery update needs a retryable failure code")
        return self


class RecoveryDecision(WireModel):
    """The Server's gatewayrecovery.Decision, decoded strictly and completely.

    ``code`` and ``requires_retry`` are status for the Run page; a Runtime Planner
    or Worker acts only on ``allowed`` and the retry timing, and bounds each
    model request by its own allocation timeout. They
    stay declared because the strict wire model forbids unknown members and the
    Server always sends ``requiresRetry``: dropping them would make every
    decision undecodable and stop model recovery.
    """

    allowed: bool = Field(strict=True)
    code: str | None = None
    retry_after_seconds: float = Field(ge=0, allow_inf_nan=False)
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
        self._allocation_id = allocation_id
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
        from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransportError

        body = RecoveryUpdate(
            model=model,
            requestId=request_id,
            action=action,
            **({"code": code} if code is not None else {}),
            retryAfterSeconds=retry_after_seconds,
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        attempts = 0
        started: float | None = None
        error_started: float | None = None
        last_logged: float | None = None
        while True:
            cause: str
            try:
                response = await self._transport.gateway_recovery(self._allocation_id, body)
                return _decode_decision(response)
            except ArtifactTransportError:
                cause = "transport"
                # Transport loss is bounded by the allocation lease, not here.
                error_started = None
            except ArtifactAPIError as error:
                if not error.retryable:
                    if error.code in {"invalid_response", "invalid_error_response"}:
                        raise RecoveryStoppedError(
                            "Model recovery authority returned an invalid decision"
                        ) from None
                    raise RecoveryStoppedError(
                        "Model recovery allocation is no longer available"
                    ) from None
                cause = error.code
                now = self._monotonic()
                if error_started is None:
                    error_started = now
                elif now - error_started >= AUTHORITY_ERROR_DEADLINE_SECONDS:
                    raise RecoveryStoppedError(
                        "Model recovery authority kept failing beyond its deadline"
                    ) from None
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


def _decode_decision(body: dict) -> RecoveryDecision:
    # A trusted Server that answers with an undecodable decision has a defect;
    # retrying cannot fix it and guessing must never authorize inference.
    try:
        return RecoveryDecision.model_validate_json(
            json.dumps(body, ensure_ascii=False, allow_nan=False)
        )
    except (ValidationError, ValueError):
        raise RecoveryStoppedError(
            "Model recovery authority returned an invalid decision"
        ) from None
