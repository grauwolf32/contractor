"""Narrow command capability; workspace ownership survives its awaiting caller."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from contractor_runtime.projectfs.errors import WorkspaceStorageError
from contractor_runtime.projectfs.operation_guard import WorkspaceOperationGuard
from contractor_runtime.sandbox.contracts import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SandboxContractError,
    SandboxErrorCode,
)

if TYPE_CHECKING:
    from contractor_runtime.sandbox.podman.lifecycle import PodmanAllocation


def failed(code: SandboxErrorCode, started: float) -> ExecutionResult:
    status = {
        SandboxErrorCode.TIMEOUT: ExecutionStatus.TIMED_OUT,
        SandboxErrorCode.OUTPUT_LIMIT: ExecutionStatus.OUTPUT_LIMIT_EXCEEDED,
    }.get(code, ExecutionStatus.FAILED)
    return ExecutionResult(
        status, None, "", "", False, False, max(0, int((time.monotonic() - started) * 1000)), code
    )


class PodmanExecutor:
    def __init__(self, allocation: PodmanAllocation, guard: WorkspaceOperationGuard):
        self._allocation = allocation
        self._guard = guard
        self._cleanup: asyncio.Task | None = None

    def _reject(self, code: SandboxErrorCode) -> None:
        self._guard.fence()
        self._allocation.reject()
        self._allocation.failure(code)

    def _stop(self) -> None:
        if self._cleanup is None:

            async def cleanup():
                # No receipt means no permission to release workspace ownership.
                # Allocation teardown can confirm removal independently.
                with suppress(Exception):
                    await self._allocation.stop(
                        deadline=datetime.now(UTC)
                        + timedelta(seconds=self._allocation.owner.settings.stop_grace_seconds + 5)
                    )

            self._cleanup = asyncio.create_task(cleanup(), name="sandbox-command-stop")

    async def execute(self, request: ExecutionRequest, *, deadline: datetime) -> ExecutionResult:
        started = time.monotonic()
        allocation = self._allocation
        owner = allocation.owner
        end = min(
            started + request.timeout_seconds,
            started + owner.settings.command_max_seconds,
            started + (deadline - datetime.now(UTC)).total_seconds(),
            owner._lease_source() or 0.0,
        )

        async def operation():
            try:
                if allocation.rejected or allocation.removed or owner._failed:
                    raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
                assert owner._client is not None
                raw = await owner._client.request(
                    "execute",
                    allocation=allocation.allocation_id,
                    request=asdict(request),
                    lease=owner._lease_source() or 0.0,
                    deadline=end,
                )
                raw["status"] = ExecutionStatus(raw["status"])
                raw["error_code"] = (
                    SandboxErrorCode(raw["error_code"]) if raw["error_code"] is not None else None
                )
                result = ExecutionResult(**raw)
                if result.error_code is not None:
                    self._reject(result.error_code)
                    # The owner returns a failed result only after confirmed stop.
                    allocation.stopped.set()
                return result
            except SandboxContractError as error:
                code = error.code
                if code in {SandboxErrorCode.INVALID_CWD, SandboxErrorCode.INVALID_COMMAND}:
                    return failed(code, started)
            except Exception:
                code = SandboxErrorCode.OUTCOME_UNKNOWN
            self._reject(code)
            self._stop()
            await allocation.stopped.wait()
            return failed(code, started)

        try:
            result = await self._guard.run_async(operation, deadline=end)
            return replace(result, duration_ms=max(0, int((time.monotonic() - started) * 1000)))
        except asyncio.CancelledError:
            self._reject(SandboxErrorCode.OUTCOME_UNKNOWN)
            self._stop()
            raise
        except WorkspaceStorageError:
            code = (
                SandboxErrorCode.TIMEOUT
                if time.monotonic() >= end
                else SandboxErrorCode.UNAVAILABLE
            )
            self._reject(code)
            self._stop()
            return failed(code, started)
