"""Explicit, bounded container execution; no host adapter or lifecycle authority."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from contractor_runtime.sandbox.contracts import (
    EXECUTION_CHANNELS,
    EXECUTION_TOOLS,
    EXECUTION_TOOLSET,
    ExecutionRequest,
    SandboxContractError,
    SandboxErrorCode,
    selected_executor,
)
from contractor_runtime.sandbox.podman.executor import failed
from contractor_runtime.sandbox.podman.settings import PODMAN_LIMIT_CEILINGS, PodmanSettings


class CodeExecutionToolsetFactory:
    ref = EXECUTION_TOOLSET
    exported_tools = EXECUTION_TOOLS
    infrastructure_channels = EXECUTION_CHANNELS
    requires_workspace = True

    def __init__(
        self,
        *,
        available=lambda: False,
        settings: Callable[[], PodmanSettings] | None = None,
    ):
        self._available = available
        # Read lazily: the operator policy belongs to the execution lifecycle.
        self._settings = settings

    async def probe(self) -> frozenset[str]:
        return self.exported_tools if self._available() else frozenset()

    async def create_selected(
        self,
        *,
        selected,
        allocation_id,
        run_id,
        namespace,
        runtime_settings,
        workspace,
        state,
        adapter_handles=None,
        project_workspace=None,
        sandbox_executor=None,
    ):
        del allocation_id, run_id, namespace, runtime_settings, workspace, adapter_handles
        del project_workspace
        if not selected:
            return {}
        executor = selected_executor(
            self.ref, tuple(selected), self.infrastructure_channels, sandbox_executor
        )
        settings = self._settings() if self._settings is not None else None
        return {name: ExecCommandTool(executor, state, settings=settings) for name in selected}


_DESCRIPTION = """Run a shell command in the allocation container.

    Background processes are forbidden. Timeout or output overflow terminates the
    allocation; partial file changes are not rolled back.

    Args:
        command: Shell command to execute in the container.
        cwd: Workspace-relative working directory; empty means the workspace root.
        timeout_seconds: Wall-clock limit from 1 to {maximum} seconds; defaults to 60.
            Larger values are rejected. The limit also covers waiting for the
            workspace and {reserve} kept for cleanup, so the command itself may be
            stopped that much earlier.

    Returns:
        Status, exitCode, stdout and stderr previews, truncation flags, durationMs
        and errorCode when execution fails.
    """


class ExecCommandTool:
    name = "exec_command"

    def __init__(self, executor, state, *, settings: PodmanSettings | None = None):
        self._executor = executor
        self._state = state
        self._closed = False
        # The operator maximum is the real ceiling: advertise it and reject more
        # instead of silently shortening the requested limit.
        self._maximum = (
            settings.command_max_seconds
            if settings is not None
            else PODMAN_LIMIT_CEILINGS["command_max_seconds"]
        )
        reserve = (
            f"up to {settings.command_cleanup_reserve_seconds} seconds"
            if settings is not None
            else "a short reserve"
        )
        self.description = _DESCRIPTION.format(maximum=self._maximum, reserve=reserve)
        self.__name__ = self.name
        self.__doc__ = self.description

    def _request(self, command, cwd="", timeout_seconds=60) -> ExecutionRequest:
        request = ExecutionRequest(command, cwd, timeout_seconds)
        if request.timeout_seconds > self._maximum:
            raise SandboxContractError(SandboxErrorCode.INVALID_COMMAND)
        return request

    async def close(self) -> None:
        # Container teardown belongs to AllocationService, never to a tool.
        self._closed = True

    def contractor_raw_argument_error(self, args):
        error = None
        try:
            if not isinstance(args, dict) or set(args) - {"command", "cwd", "timeout_seconds"}:
                raise SandboxContractError(SandboxErrorCode.INVALID_COMMAND)
            self._request(**args)
        except SandboxContractError as rejection:
            error = rejection
        except TypeError:
            error = SandboxContractError(SandboxErrorCode.INVALID_COMMAND)
        if error is not None:
            # Record before ADK's fallback can summarize arbitrary unknown keys.
            self._state.metrics.record_tool_call(
                self.name, arguments={}, error=error, duration_ms=0
            )
        return error

    async def __call__(self, command: str, cwd: str = "", timeout_seconds: int = 60) -> dict:
        self._state.execution.check()
        started = time.monotonic()
        if self._closed:
            return failed(SandboxErrorCode.UNAVAILABLE, started).observation()
        try:
            request = self._request(command, cwd, timeout_seconds)
            result = await self._executor.execute(
                request, deadline=datetime.now(UTC) + timedelta(seconds=timeout_seconds)
            )
        except SandboxContractError as error:
            result = failed(error.code, started)
        except asyncio.CancelledError:
            self._state.execution.fail(SandboxErrorCode.OUTCOME_UNKNOWN)
            self._state.metrics.record_tool_call(
                self.name,
                arguments={},
                error=SandboxContractError(SandboxErrorCode.OUTCOME_UNKNOWN),
                duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            )
            raise
        if result.error_code not in {
            None,
            SandboxErrorCode.INVALID_COMMAND,
            SandboxErrorCode.INVALID_CWD,
        }:
            self._state.execution.fail(result.error_code)
        self._state.metrics.record_sandbox_execution(
            stdout_bytes=result.stdout_bytes,
            stderr_bytes=result.stderr_bytes,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            exit_code=result.exit_code,
        )
        self._state.metrics.record_tool_call(
            self.name,
            arguments={},
            result={
                "status": result.status.value,
                "stdout_bytes": result.stdout_bytes,
                "stderr_bytes": result.stderr_bytes,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            },
            error=SandboxContractError(result.error_code) if result.error_code else None,
            result_size_bytes=result.stdout_bytes + result.stderr_bytes,
            duration_ms=result.duration_ms,
        )
        return result.observation()
