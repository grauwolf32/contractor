"""Server host adapter over the existing allocation-authenticated transport."""

import asyncio
from datetime import UTC, datetime

from contractor_runtime.artifacts import ArtifactTransport, ArtifactTransportError, allocation_path
from contractor_runtime.contracts import API_VERSION, PrivateProtocolDecodeError, decode_private
from contractor_runtime.contracts.planner import (
    MAX_PLANNER_OPERATION_BYTES,
    PlannerOperation,
    PlannerOperationResult,
)
from contractor_runtime.planner.host import PlannerStoppedError


class ServerPlannerHost:
    def __init__(
        self, allocation_id: str, transport: ArtifactTransport, *, deadline: datetime
    ) -> None:
        self._path = allocation_path(allocation_id, "planner-operations")
        self._transport = transport
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._deadline = deadline

    async def execute(self, name, arguments) -> PlannerOperationResult:
        async with self._lock:
            self._sequence += 1
            try:
                request = PlannerOperation(
                    apiVersion=API_VERSION, sequence=self._sequence, name=name, arguments=arguments
                )
                body = request.model_dump_json(by_alias=True).encode()
            except ValueError:
                raise PlannerStoppedError(
                    "planner_operation_invalid", "Planner operation exceeds its contract", False
                ) from None
            # One identical retry is safe because the current Server host keeps
            # the last ordered operation and its response until the next request.
            for attempt in range(2):
                remaining = (self._deadline - datetime.now(UTC)).total_seconds()
                if remaining <= 0:
                    raise PlannerStoppedError(
                        "planner_deadline_exceeded", "Planner execution deadline expired", True
                    )
                try:
                    response = await self._transport.request(
                        "POST",
                        self._path,
                        headers={"Content-Type": "application/json"},
                        body=body,
                        max_response_bytes=MAX_PLANNER_OPERATION_BYTES,
                        timeout_seconds=remaining,
                    )
                    break
                except ArtifactTransportError:
                    if attempt == 1:
                        raise PlannerStoppedError(
                            "planner_execution_failed",
                            "Planner execution host is unavailable",
                            True,
                        ) from None
            if response.status_code != 200:
                retryable = response.status_code >= 500
                raise PlannerStoppedError(
                    "planner_execution_failed" if retryable else "planner_operation_invalid",
                    "Planner execution host is unavailable",
                    retryable,
                )
            try:
                return decode_private(PlannerOperationResult, response.body)
            except PrivateProtocolDecodeError:
                raise PlannerStoppedError(
                    "planner_operation_invalid", "Planner execution host reply is invalid", False
                ) from None
