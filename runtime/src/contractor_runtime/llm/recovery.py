"""Allocation-authenticated access to the Server's shared model recovery gate."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from pydantic import Field

from contractor_runtime.contracts.base import WireModel

if TYPE_CHECKING:
    from contractor_runtime.artifacts import ArtifactTransport

# A lost Control Plane response must never authorize inference. This reconnect
# cadence only observes the authority; model retry timing is returned by Server.
AUTHORITY_RECONNECT_SECONDS = 1.0
MAX_RECOVERY_RESPONSE_BYTES = 4096


class RecoveryStoppedError(RuntimeError):
    """The allocation no longer has authority to continue this invocation."""


class RecoveryDecision(WireModel):
    allowed: bool = Field(strict=True)
    code: str | None = None
    retry_after_seconds: float = Field(ge=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
    requires_retry: bool = Field(strict=True)


class GatewayRecoveryClient:
    def __init__(self, allocation_id: str, transport: ArtifactTransport) -> None:
        self._transport = transport
        self._path = f"/allocations/{allocation_id}/gateway-recovery"

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
        while True:
            try:
                response = await self._transport.request(
                    "POST",
                    self._path,
                    headers={"Content-Type": "application/json"},
                    body=body,
                    max_response_bytes=MAX_RECOVERY_RESPONSE_BYTES,
                )
            except ArtifactTransportError:
                await asyncio.sleep(AUTHORITY_RECONNECT_SECONDS)
                continue
            if response.status_code == 200:
                return RecoveryDecision.model_validate_json(response.body)
            if 400 <= response.status_code < 500:
                raise RecoveryStoppedError("Model recovery allocation is no longer available")
            await asyncio.sleep(AUTHORITY_RECONNECT_SECONDS)
