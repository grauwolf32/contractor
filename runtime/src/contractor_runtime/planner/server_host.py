"""Server host adapter over the existing allocation-authenticated transport."""

import asyncio
import json
from datetime import UTC, datetime

from contractor_runtime.artifacts import ArtifactAPIError, ArtifactTransport, ArtifactTransportError
from contractor_runtime.contracts import API_VERSION, PrivateProtocolDecodeError, decode_private
from contractor_runtime.contracts.planner import (
    PlannerOperation,
    PlannerOperationResult,
)
from contractor_runtime.planner.host import PlannerStoppedError
from contractor_runtime.threads import to_thread_until_done


class ServerPlannerHost:
    def __init__(
        self, allocation_id: str, transport: ArtifactTransport, *, deadline: datetime
    ) -> None:
        self._allocation_id = allocation_id
        self._transport = transport
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._deadline = deadline

    async def execute(self, name, arguments) -> PlannerOperationResult:
        async with self._lock:
            self._sequence += 1
            try:
                body = await to_thread_until_done(
                    lambda: PlannerOperation(
                        apiVersion=API_VERSION,
                        sequence=self._sequence,
                        name=name,
                        arguments=arguments,
                    ).model_dump(mode="json", by_alias=True, exclude_none=True),
                    name="planner-operation-encoding",
                )
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
                    response = await self._transport.planner_operation(
                        self._allocation_id, body, timeout_seconds=remaining
                    )
                    break
                except ArtifactAPIError as error:
                    raise PlannerStoppedError(
                        "planner_execution_failed"
                        if error.retryable
                        else "planner_operation_invalid",
                        "Planner execution host rejected the operation",
                        error.retryable,
                    ) from None
                except ArtifactTransportError:
                    if attempt == 1:
                        raise PlannerStoppedError(
                            "planner_execution_failed",
                            "Planner execution host is unavailable",
                            True,
                        ) from None
            try:
                return await to_thread_until_done(
                    decode_private,
                    PlannerOperationResult,
                    json.dumps(response, ensure_ascii=False, allow_nan=False),
                    name="planner-operation-decoding",
                )
            except PrivateProtocolDecodeError:
                raise PlannerStoppedError(
                    "planner_operation_invalid", "Planner execution host reply is invalid", False
                ) from None
