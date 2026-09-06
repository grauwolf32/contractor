"""Private execution contracts; no engine, factory registration or host fallback.

Only allocation lifecycle code will own AllocationSandbox. Selected execution
tools receive SandboxExecutor, never lifecycle/engine or host adapter authority.
The implementation must hold the workspace guard until all writers are stopped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from contractor_runtime.contracts import AllocationSpec, AllocationSpecV2
from contractor_runtime.projectfs.paths import ProjectPathError, normalize_project_path

PODMAN_PROFILE = "podman@1"
EXECUTION_TOOLSET = "code-execution@1"
EXECUTION_TOOLS = frozenset({"exec_command"})
EXECUTION_CHANNELS = MappingProxyType({"exec_command": frozenset({"sandbox-execution"})})
MAX_COMMAND_BYTES = 64 << 10
DEFAULT_COMMAND_SECONDS = 60


@dataclass(frozen=True, slots=True)
class SandboxRequirements:
    workspace_mode: str
    workspace_storage: str


SANDBOX_REQUIREMENTS = MappingProxyType({PODMAN_PROFILE: SandboxRequirements("direct", "local")})
TOOLSET_SANDBOX_REQUIREMENTS = MappingProxyType({EXECUTION_TOOLSET: PODMAN_PROFILE})


class SandboxErrorCode(StrEnum):
    INVALID_COMMAND = "sandbox_invalid_command"
    INVALID_CWD = "sandbox_invalid_cwd"
    UNAVAILABLE = "sandbox_unavailable"
    PREPARATION_FAILED = "sandbox_preparation_failed"
    TIMEOUT = "sandbox_timeout"
    OUTPUT_LIMIT = "sandbox_output_limit"
    BACKGROUND_EXECUTION = "sandbox_background_execution"
    CLEANUP_FAILED = "sandbox_cleanup_failed"
    OUTCOME_UNKNOWN = "sandbox_outcome_unknown"
    INCOMPATIBLE = "sandbox_incompatible"


class SandboxContractError(ValueError):
    def __init__(self, code: SandboxErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def validate_sandbox_selection(spec: AllocationSpec, storage: str | None) -> None:
    profile = spec.agent_template.sandbox_profile
    ref = f"{profile.sandbox_profile_id}@{profile.version}"
    for selected in spec.agent_template.toolsets:
        toolset = f"{selected.ref.toolset_id}@{selected.ref.version}"
        required = TOOLSET_SANDBOX_REQUIREMENTS.get(toolset)
        if required is not None and (
            ref != required or not selected.tools or not set(selected.tools) <= EXECUTION_TOOLS
        ):
            raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
    requirement = SANDBOX_REQUIREMENTS.get(ref)
    if requirement is not None and (
        not isinstance(spec, AllocationSpecV2)
        or spec.workspace is None
        or spec.workspace.mode != requirement.workspace_mode
        or storage != requirement.workspace_storage
    ):
        raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    command: str = field(repr=False)
    cwd: str = field(default="", repr=False)
    timeout_seconds: int = DEFAULT_COMMAND_SECONDS

    def __post_init__(self) -> None:
        try:
            valid = (
                isinstance(self.command, str)
                and 0 < len(self.command) <= MAX_COMMAND_BYTES
                and 0 < len(self.command.encode("utf-8")) <= MAX_COMMAND_BYTES
                and "\x00" not in self.command
            )
        except UnicodeError:
            valid = False
        if (
            not valid
            or type(self.timeout_seconds) is not int
            or not 0 < self.timeout_seconds <= 3600
        ):
            raise SandboxContractError(SandboxErrorCode.INVALID_COMMAND)
        try:
            object.__setattr__(self, "cwd", normalize_project_path(self.cwd))
        except ProjectPathError:
            raise SandboxContractError(SandboxErrorCode.INVALID_CWD) from None


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    OUTPUT_LIMIT_EXCEEDED = "output_limit_exceeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: ExecutionStatus
    exit_code: int | None
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    stdout_truncated: bool
    stderr_truncated: bool
    duration_ms: int
    error_code: SandboxErrorCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus) or (
            self.error_code is not None and not isinstance(self.error_code, SandboxErrorCode)
        ):
            raise ValueError("invalid sandbox result category")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("invalid sandbox duration")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("invalid sandbox exit code")
        if self.status == ExecutionStatus.COMPLETED:
            if self.exit_code is None or self.error_code is not None:
                raise ValueError("completed sandbox result requires a trusted exit code")
        elif self.exit_code is not None or self.error_code is None:
            raise ValueError("failed sandbox result requires a stable error code and no exit code")
        if type(self.stdout_truncated) is not bool or type(self.stderr_truncated) is not bool:
            raise ValueError("invalid sandbox truncation flags")
        for value in (self.stdout, self.stderr):
            if not isinstance(value, str) or len(value.encode("utf-8")) > 3 << 20:
                raise ValueError("invalid sandbox output preview")

    def observation(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "exitCode": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdoutTruncated": self.stdout_truncated,
            "stderrTruncated": self.stderr_truncated,
            "durationMs": self.duration_ms,
            "errorCode": self.error_code.value if self.error_code is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SandboxIdentity:
    """Private ownership tuple; never use user names alone as cleanup authority."""

    owner: str = field(repr=False)
    incarnation: str = field(repr=False)
    allocation_id: str = field(repr=False)
    creation_id: str = field(repr=False)
    container_id: str = field(repr=False)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.container_id) is None:
            raise ValueError("sandbox ownership requires a full engine ID")
        for value in (self.owner, self.incarnation, self.allocation_id, self.creation_id):
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
            ):
                raise ValueError("invalid sandbox ownership identity")


class SandboxExecutor(Protocol):
    async def execute(self, request: ExecutionRequest, *, deadline: datetime) -> ExecutionResult:
        """Includes guard wait, command and confirmed descendant cleanup; no replay."""
        ...


class AllocationSandbox(Protocol):
    @property
    def identity(self) -> SandboxIdentity: ...

    @property
    def executor(self) -> SandboxExecutor: ...

    async def renew(self, *, deadline: datetime) -> None:
        """Renew only up to the confirmed control lease; workload has no access."""
        ...

    async def stop(self, *, deadline: datetime) -> None: ...

    async def remove(self, *, deadline: datetime) -> None:
        """Return only after verified removal; uncertainty retains ownership."""
        ...


def selected_executor(
    toolset_ref: str,
    tools: tuple[str, ...],
    channels: Mapping[str, frozenset[str]],
    executor: SandboxExecutor | None,
) -> SandboxExecutor:
    """Narrow factory handoff, to be wired by the allocation lifecycle task."""
    if toolset_ref != EXECUTION_TOOLSET or not tools or not set(tools) <= EXECUTION_TOOLS:
        raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
    if any(channels.get(tool) != EXECUTION_CHANNELS[tool] for tool in tools):
        raise SandboxContractError(SandboxErrorCode.INCOMPATIBLE)
    if executor is None:
        raise SandboxContractError(SandboxErrorCode.UNAVAILABLE)
    return executor
